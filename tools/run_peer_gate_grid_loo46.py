#!/usr/bin/env python3
"""Paired 46-parcel exploratory LOOCV of Reference26 geological gate variants.

Uses original teammate's 26-feature private bundle and ColdExperts/full_fit.
Every arm tests EXACTLY the first 46 rows in train138.parquet (137/1 splits).

  python tools/run_peer_gate_grid_loo46.py --check
  python tools/run_peer_gate_grid_loo46.py --smoke --jobs 1
  python tools/run_peer_gate_grid_loo46.py --run --jobs 12
  python tools/run_peer_gate_grid_loo46.py --summary
"""
from __future__ import annotations

import os

# Respect user's process count while limiting threads *inside* each process.
for name in (
    "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
):
    os.environ[name] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


from run_selected_champions_loo import (  # noqa: E402
    BUNDLE,
    ID,
    PEER_REPO,
    ROOT,
    TARGET,
    digest,
    fold_record,
    load_inputs,
    write_csv,
)
from sklearn.metrics import r2_score  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

from geocebada.evaluation.peer_gate_grid import (  # noqa: E402
    GATE_GRIDS,
    VARIANTS,
    configured_gate,
    native_reference_gate,
)

COUNT = 46
OUT = ROOT / "reports/peer_gate_grid_loo46"
ORIGINAL = "Reference26_original_KMeans"
CONTROL = "Compact14_original"
PREDICTIONS = {
    "original": ORIGINAL,
    "kmeans_expanded": "Reference26_KMeans_2_4_6_8_12",
    "dbscan": "Reference26_DBSCAN_geology",
    "support": "Reference26_geological_support",
    "global": "Reference26_global_weight",
}
_DATA = None


def variant_list(raw: str) -> tuple[str, ...]:
    chosen = tuple(s.strip() for s in raw.split(","))
    if not chosen or len(chosen) != len(set(chosen)) or any(
        v not in VARIANTS for v in chosen
    ):
        raise ValueError(f"Specify distinct gate variants from {VARIANTS}.")
    if "original" not in chosen:
        raise ValueError("Original KMeans is required as a fixed paired control.")
    return tuple(v for v in VARIANTS if v in chosen)


def model_columns(variants: tuple[str, ...]) -> tuple[str, ...]:
    return (CONTROL, *(PREDICTIONS[v] for v in variants))


def validate_fold(path: Path, held_id: str, y: float, variants: tuple[str, ...]):
    row = pd.read_csv(path, dtype={ID: str})
    if len(row) != 1 or set(row.columns) != {ID, "observed", *model_columns(variants)}:
        raise ValueError(f"Unexpected 46-fold result schema at {path}.")
    current = row.iloc[0]
    if current[ID] != held_id or abs(float(current["observed"]) - y) > 1e-10:
        raise ValueError(f"Unexpected held-out parcel or yield at {path}.")
    if not np.isfinite(row[list(model_columns(variants))].to_numpy(float)).all():
        raise ValueError(f"Nonfinite model predictions at {path}.")
    return row


def spec(bundle: Path, peer: Path, variants: tuple[str, ...]) -> dict:
    return {
        "schema": 1,
        "protocol": "First 46 rows of peer train138.parquet, 137/1 outer LOOCV",
        "outer_positions_zero_based": list(range(COUNT)),
        "inner": "same four-fold train-only KFold(seed=20261018) for every model",
        "baseline_and_variants": list(variants),
        "gate_grids": {
            name: [list(v) if isinstance(v, tuple) else v for v in values]
            for name, values in GATE_GRIDS.items() if name in variants
        },
        "model_columns": list(model_columns(variants)),
        "bundle_manifest_sha256": digest(bundle / "manifest.json"),
        "peer_source_sha256": digest(peer / "src/current_champions.py"),
        "gate_implementation_sha256": digest(
            ROOT / "src/geocebada/evaluation/peer_gate_grid.py"
        ),
        "dbscan_implementation_sha256": digest(
            ROOT / "src/geocebada/evaluation/peer_geology_dbscan.py"
        ),
        "fold_protocol_sha256": digest(
            ROOT / "tools/run_selected_champions_loo.py"
        ),
        "runner_sha256": digest(Path(__file__).resolve()),
    }


def lock(out: Path, data: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    path = out / "run_spec.json"
    if path.is_file():
        if json.loads(path.read_text(encoding="utf-8")) != data:
            raise ValueError("Inputs or code changed. Select a NEW --output directory.")
    else:
        if list(out.glob("fold_*.csv")):
            raise ValueError("Found old results without a provenance manifest.")
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def initialize(bundle: str, peer: str) -> None:
    global _DATA
    _DATA = load_inputs(Path(bundle), Path(peer))


def fit_one(position: int, out_string: str, variants: tuple[str, ...]):
    """Fit every gate independently using the SAME original 137 train rows."""
    if _DATA is None:
        raise RuntimeError("Missing initialized private peer bundle.")
    frame, groups, distance, manifest, ids, ColdExperts, predict_saved = _DATA
    held_id = ids[position]
    target = float(frame.iloc[position][TARGET])
    output = Path(out_string)
    pred_path = output / f"fold_{position + 1:03d}.csv"
    if pred_path.is_file():
        validate_fold(pred_path, held_id, target, variants)
        return position + 1, "cached", 0.
    start = time.perf_counter()
    tr, inner = fold_record(position)
    masked = frame.copy(deep=True)
    masked.loc[masked.index[position], TARGET] = np.nan
    if not np.isnan(float(masked.iloc[position][TARGET])):
        raise RuntimeError("Failed to mask held-out target before fitting.")
    if not np.isfinite(masked.iloc[tr][TARGET].to_numpy(float)).all():
        raise RuntimeError("Missing training labels.")

    values = {ID: held_id, "observed": target}
    diagnostics = {}
    compact_control = None
    with threadpool_limits(limits=1):
        for variant in variants:
            with configured_gate(variant):
                factory = ColdExperts(
                    masked, groups, distance, manifest["geology"], "random",
                    tr, inner, output / f"cache/fold_{position + 1:03d}/{variant}",
                )
                fit = factory.full_fit(tr)
                if set(fit["train_positions"]) != set(tr):
                    raise RuntimeError("Source model used unexpected training positions.")
                pack = {
                    "outer_experts": fit,
                    "train_positions": tr,
                    "validation_positions": np.array([position], dtype=int),
                    "B1_weight": 0.0,
                }
                predictions = predict_saved(
                    pack, np.array([position], dtype=int), masked, distance
                )
                reference = float(np.asarray(predictions["Reference26"]).item())
                compact = float(np.asarray(predictions["Compact14"]).item())
                if not np.isfinite([reference, compact]).all():
                    raise RuntimeError(f"Nonfinite prediction for {variant}.")
                if compact_control is None:
                    compact_control = compact
                    values[CONTROL] = compact
                elif not np.isclose(
                    compact, compact_control, rtol=0.0, atol=1e-7
                ):
                    raise RuntimeError(
                        "Compact14 changed when only the Reference26 gate was "
                        f"modified for {variant}: {compact} versus {compact_control}."
                    )
                values[PREDICTIONS[variant]] = reference
                diagnostics[variant] = native_reference_gate(fit)
    result = pd.DataFrame([values])
    write_csv(pred_path, result)
    audit = output / f"gate_{position + 1:03d}.json"
    audit.write_text(json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8")
    return position + 1, "trained", time.perf_counter() - start


def summarize(out: Path, n: int, variants: tuple[str, ...], *, complete: bool):
    paths = [out / f"fold_{i:03d}.csv" for i in range(1, n + 1)]
    if any(not p.is_file() for p in paths):
        raise ValueError(f"Cannot summarize: not all first {n} folds are present.")
    predictions = pd.concat(
        [pd.read_csv(path, dtype={ID: str}) for path in paths],
        ignore_index=True,
    )
    if len(predictions) != n or predictions[ID].duplicated().any():
        raise ValueError("Missing or duplicated held-out parcel.")
    for col in ("observed", *model_columns(variants)):
        if not np.isfinite(predictions[col].to_numpy(float)).all():
            raise ValueError(f"Nonfinite entries in {col}.")
    actual = predictions["observed"].to_numpy(float)
    scores = []
    for name in model_columns(variants):
        residual = predictions[name].to_numpy(float) - actual
        scores.append({
            "model": name,
            "heldout": n,
            "rmse": float(np.sqrt(np.mean(residual**2))),
            "mae": float(np.mean(np.abs(residual))),
            "bias": float(np.mean(residual)),
            "r2": float(r2_score(actual, predictions[name])) if n > 1 else np.nan,
            "delta_mse_vs_original": (
                float(np.mean(residual**2))
                - float(np.mean((predictions[ORIGINAL].to_numpy(float) - actual)**2))
            ),
        })
    metrics = pd.DataFrame(scores).sort_values("rmse", kind="mergesort")
    prefix = "loo46" if complete else "smoke"
    write_csv(out / f"{prefix}_predictions.csv", predictions)
    write_csv(out / f"{prefix}_scores.csv", metrics)
    print(f"[{prefix.upper()}] Same first {n} held-out parcels:\n"
          + metrics.to_string(index=False), flush=True)
    if not complete:
        print("[SMOKE] Descriptive only; not the final 46-fold scores.", flush=True)
    else:
        print(
            "[LOO46] This fixed first-third comparison is exploratory, NOT "
            "a 138-fold LOOCV or independent confirmation. Selecting the best "
            "variant using its 46 outer errors makes the reported winning "
            "score optimistic.",
            flush=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ("check", "smoke", "run", "summary"):
        mode.add_argument(f"--{name}", action="store_true")
    parser.add_argument("--bundle", type=Path, default=BUNDLE)
    parser.add_argument("--peer-repo", type=Path, default=PEER_REPO)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument(
        "--variants", default=",".join(VARIANTS),
        help="Comma-separated subset, always include original; default all gates.",
    )
    parser.add_argument("--smoke-folds", type=int, default=1)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be a positive integer (no artificial maximum).")
    if not 1 <= args.smoke_folds <= COUNT:
        parser.error(f"--smoke-folds must be from 1 to {COUNT}.")
    try:
        variants = variant_list(args.variants)
    except ValueError as exc:
        parser.error(str(exc))

    bundle = args.bundle.resolve()
    peer = args.peer_repo.resolve()
    out = (args.output / "smoke" if args.smoke else args.output).resolve()
    frame, _groups, _distance, _manifest, ids, _class, _predict = load_inputs(
        bundle, peer
    )
    if len(ids) != 138:
        raise ValueError("The exported bundle did not contain 138 rows.")
    print(
        f"[GATE] Original bundle: {bundle}\n"
        f"[GATE] Outer test positions: 1 through {COUNT} (bundle order); "
        f"all strategies evaluated on exactly these same parcels.\n"
        f"[GATE] Gate strategies: {', '.join(variants)}",
        flush=True,
    )
    if args.check:
        print("[GATE CHECK] Original source and private inputs validated.", flush=True)
        return 0

    lock(out, spec(bundle, peer, variants))
    if args.summary:
        summarize(out, COUNT, variants, complete=True)
        return 0
    n = args.smoke_folds if args.smoke else COUNT
    todo = []
    for i in range(n):
        path = out / f"fold_{i + 1:03d}.csv"
        if path.is_file():
            validate_fold(path, ids[i], float(frame.iloc[i][TARGET]), variants)
            print(f"[GATE {i + 1:03d}/{n}] cached", flush=True)
        else:
            todo.append(i)
    if todo:
        print(
            f"[GATE] Running {len(todo)} outer folds with "
            f"{min(args.jobs, len(todo))} user-requested worker process(es).",
            flush=True,
        )
        with ProcessPoolExecutor(
            max_workers=min(args.jobs, len(todo)),
            initializer=initialize,
            initargs=(str(bundle), str(peer)),
        ) as pool:
            tasks = {
                pool.submit(fit_one, position, str(out), variants): position
                for position in todo
            }
            for future in as_completed(tasks):
                fold, status, elapsed = future.result()
                print(
                    f"[GATE {fold:03d}/{n}] {status} ({elapsed:.1f}s)",
                    flush=True,
                )
    summarize(out, n, variants, complete=not args.smoke)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
