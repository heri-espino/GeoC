#!/usr/bin/env python3
"""Reference26 fixed-KMeans geological-cluster count comparison.

Only Reference26 is retrained. Baseline LOOCV Reference26 predictions are
read from saved original peer output and matched against all held-out IDs.

Run the same first 46 parcels for each requested k, using 137 training labels,
original ColdExperts.full_fit with nested R3 selection, and no target leakage.
An optional 138-fold run is supported once candidates have been shortlisted.

  python tools/run_reference26_kmeans_grid.py --check
  python tools/run_reference26_kmeans_grid.py --smoke --jobs 1
  python tools/run_reference26_kmeans_grid.py --run --jobs 12
  python tools/run_reference26_kmeans_grid.py --summary

  # Optional full LOOCV on shortlisted k:
  python tools/run_reference26_kmeans_grid.py --run --folds 138 --ks 14,26 \
      --output reports/reference26_kmeans_fixed_138 --jobs 12
"""
from __future__ import annotations

import os

for key in (
    "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
):
    os.environ[key] = "1"

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

from geocebada.evaluation.peer_reference26_fixed_kmeans import (  # noqa: E402
    DEFAULT_K,
    fixed_geological_kmeans,
    gate_diagnostics,
)

BASELINE = ROOT / "reports/selected_champions_loo/loo_predictions.csv"
OUTPUT = ROOT / "reports/reference26_kmeans_fixed_46"
BASE_NAME = "Reference26_original"
_DATA = None


def parse_ks(value: str) -> tuple[int, ...]:
    try:
        vals = tuple(int(x.strip()) for x in value.split(","))
    except ValueError as exc:
        raise ValueError("Use comma-separated integer k values, e.g. 4,8,14,26.") from exc
    if not vals or any(k < 1 for k in vals) or len(set(vals)) != len(vals):
        raise ValueError("k must contain distinct positive integers.")
    return tuple(sorted(vals))


def k_name(k: int) -> str:
    return f"Reference26_KMeans_k{k}"


def expected_columns(ks: tuple[int, ...]) -> set[str]:
    return {ID, "observed", *(k_name(k) for k in ks)}


def validate(df: pd.DataFrame, ks: tuple[int, ...], count: int):
    if len(df) != count or set(df.columns) != expected_columns(ks):
        raise ValueError("Saved fixed-KMeans fold is incomplete or has wrong columns.")
    if not df[ID].astype(str).is_unique:
        raise ValueError("Duplicate held-out parcel IDs.")
    for column in expected_columns(ks) - {ID}:
        if not np.isfinite(df[column].to_numpy(float)).all():
            raise ValueError(f"Nonfinite {column} in fixed-KMeans predictions.")


def reference_predictions(
    path: Path, ids: list[str], y: np.ndarray, folds: int,
) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Original 138-fold predictions not found: {path}. "
            "Recover them via run_selected_champions_loo.py --summary."
        )
    original = pd.read_csv(path, dtype={ID: str})
    if (
        len(original) != 138 or not original[ID].is_unique
        or not {ID, "observed", "Reference26"}.issubset(original.columns)
    ):
        raise ValueError("Original Reference26 results must contain 138 unique parcels.")
    if set(original[ID].astype(str)) != set(ids):
        raise ValueError("Original Reference26 LOOCV parcel IDs differ from bundle.")
    ordered = original.set_index(ID).reindex(ids).reset_index()
    if (
        ordered["observed"].isna().any()
        or not np.allclose(
            ordered["observed"].to_numpy(float), y,
            rtol=0, atol=1e-9,
        )
        or not np.isfinite(ordered["Reference26"].to_numpy(float)).all()
    ):
        raise ValueError("Original Reference26 yields/predictions disagree with bundle.")
    return ordered.iloc[:folds][[ID, "observed", "Reference26"]].rename(
        columns={"Reference26": BASE_NAME}
    )


def spec(
    bundle: Path, peer: Path, ks: tuple[int, ...], folds: int,
) -> dict:
    return {
        "protocol": f"first {folds} teammate bundle rows, 137 train / 1 heldout",
        "folds": folds,
        "fixed_cluster_counts": list(ks),
        "inner_cv": "Original ColdExperts full_fit R3, 4 folds seed 20261018",
        "baseline_from_saved_Reference26": True,
        "teammate_manifest_sha256": digest(bundle / "manifest.json"),
        "teammate_code_sha256": digest(peer / "src/current_champions.py"),
        "fixed_kmeans_source_sha256": digest(
            ROOT / "src/geocebada/evaluation/peer_reference26_fixed_kmeans.py"
        ),
        "fold_protocol_sha256": digest(
            ROOT / "tools/run_selected_champions_loo.py"
        ),
        "runner_sha256": digest(Path(__file__).resolve()),
    }


def locked(folder: Path, provenance: dict):
    folder.mkdir(parents=True, exist_ok=True)
    file = folder / "run_spec.json"
    if file.is_file():
        if json.loads(file.read_text(encoding="utf-8")) != provenance:
            raise ValueError("Experiment settings or model code changed. Use a fresh --output.")
    elif list(folder.glob("fold_*.csv")):
        raise ValueError("Fold predictions exist without a provenance manifest.")
    else:
        file.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")


def worker_init(bundle: str, peer: str):
    global _DATA
    _DATA = load_inputs(Path(bundle), Path(peer))


def fit_one(position: int, outstr: str, ks: tuple[int, ...]):
    if _DATA is None:
        raise RuntimeError("Worker private data was not initialized.")
    frame, groups, distance, manifest, ids, ColdExperts, predict_saved = _DATA
    out = Path(outstr)
    held = ids[position]
    y = float(frame.iloc[position][TARGET])
    path = out / f"fold_{position + 1:03d}.csv"
    if path.is_file():
        row = pd.read_csv(path, dtype={ID: str})
        validate(row, ks, 1)
        if row.iloc[0][ID] != held or abs(row.iloc[0]["observed"] - y) > 1e-10:
            raise ValueError("Cached predictions correspond to different parcel.")
        return position + 1, "cached", 0.0

    tr, inner = fold_record(position)
    masked = frame.copy(deep=True)
    masked.loc[masked.index[position], TARGET] = np.nan
    if (
        not np.isnan(masked.iloc[position][TARGET])
        or not np.isfinite(masked.iloc[tr][TARGET].to_numpy(float)).all()
    ):
        raise RuntimeError("Outer held-out yield masking failed.")

    row = {ID: held, "observed": y}
    gates = {}
    started = time.perf_counter()
    with threadpool_limits(limits=1):
        for k in ks:
            with fixed_geological_kmeans(k):
                factory = ColdExperts(
                    masked, groups, distance, manifest["geology"], "random",
                    tr, inner, out / f"cache/fold_{position + 1:03d}/k{k}",
                )
                fitted = factory.full_fit(tr)
                if set(fitted["train_positions"]) != set(tr):
                    raise RuntimeError("Original model trained on incorrect parcels.")
                prediction = predict_saved({
                    "outer_experts": fitted,
                    "train_positions": tr,
                    "validation_positions": np.asarray([position], dtype=int),
                    "B1_weight": 0.0,
                }, np.asarray([position], dtype=int), masked, distance)
                value = float(np.asarray(prediction["Reference26"]).item())
                if not np.isfinite(value):
                    raise ValueError(f"Nonfinite Reference26 prediction for k={k}.")
                row[k_name(k)] = value
                gates[k_name(k)] = gate_diagnostics(fitted)

    values = pd.DataFrame([row])
    validate(values, ks, 1)
    write_csv(path, values)
    (out / f"kmeans_{position + 1:03d}.json").write_text(
        json.dumps(gates, indent=2) + "\n", encoding="utf-8"
    )
    return position + 1, "trained", time.perf_counter() - started


def summaries(
    output: Path, original: pd.DataFrame, ks: tuple[int, ...], count: int,
    *,
    complete: bool,
):
    files = [output / f"fold_{i:03d}.csv" for i in range(1, count + 1)]
    if any(not file.is_file() for file in files):
        raise ValueError(f"Need all {count} held-out parcels for this result.")
    experiment = pd.concat(
        [pd.read_csv(file, dtype={ID: str}) for file in files],
        ignore_index=True,
    )
    validate(experiment, ks, count)
    paired = original.merge(
        experiment, on=ID, how="left", suffixes=("", "_new"),
        validate="one_to_one",
    )
    if (
        paired["observed_new"].isna().any()
        or not np.allclose(
            paired["observed"], paired["observed_new"], rtol=0, atol=1e-9
        )
    ):
        raise ValueError("Paired Reference26 fold IDs or observed yields differ.")
    paired = paired.drop(columns=["observed_new"])
    real = paired["observed"].to_numpy(float)
    rows = []
    original_mse = np.mean((paired[BASE_NAME].to_numpy(float) - real) ** 2)
    for name in (BASE_NAME, *(k_name(k) for k in ks)):
        residual = paired[name].to_numpy(float) - real
        mse = float(np.mean(residual**2))
        rows.append({
            "model": name,
            "n_heldout_parcels": count,
            "training_labels_per_fold": 137,
            "rmse": float(np.sqrt(mse)),
            "mae": float(np.mean(abs(residual))),
            "bias": float(np.mean(residual)),
            "r2": float(r2_score(real, paired[name]))
            if count > 1 else np.nan,
            "delta_mse_vs_original": mse - original_mse,
        })
    result = pd.DataFrame(rows).sort_values(
        "rmse", kind="stable"
    ).reset_index(drop=True)
    prefix = f"loo{count}" if complete else "smoke"
    write_csv(output / f"{prefix}_predictions_private.csv", paired)
    write_csv(output / f"{prefix}_summary.csv", result)
    print(f"[REFERENCE26 FIXED KMEANS] Same first {count} parcels:\n"
          + result.to_string(index=False), flush=True)
    print(
        "Fixed k models are individual experiments. The original Reference26 "
        "uses its own train-only choice between k=4 and 8. Choosing a fixed k "
        "from held-out scores creates selection optimism.",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ("check", "smoke", "run", "summary"):
        mode.add_argument(f"--{name}", action="store_true")
    parser.add_argument("--bundle", type=Path, default=BUNDLE)
    parser.add_argument("--peer-repo", type=Path, default=PEER_REPO)
    parser.add_argument("--original-results", type=Path, default=BASELINE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--ks", default=",".join(str(v) for v in DEFAULT_K))
    parser.add_argument("--folds", type=int, choices=(46, 138), default=46)
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive; no artificial maximum.")
    try:
        ks = parse_ks(args.ks)
    except ValueError as exc:
        parser.error(str(exc))
    if args.folds == 138 and args.output == OUTPUT:
        parser.error(
            "For --folds 138 provide an explicit different --output path, "
            "to preserve all first-46 results."
        )

    bundle = args.bundle.resolve()
    peer = args.peer_repo.resolve()
    out = (args.output / "smoke" if args.smoke else args.output).resolve()
    frame, _groups, _distance, _manifest, ids, _factory, _predict = load_inputs(
        bundle, peer
    )
    actual = frame[TARGET].to_numpy(float)
    n = 1 if args.smoke else args.folds
    baseline = reference_predictions(
        args.original_results.resolve(), ids, actual, n
    )
    print(
        "[REFERENCE26] Original model 26 features unchanged; "
        f"fixed geological KMeans counts: {ks}; "
        f"first {args.folds} common held-out parcels.",
        flush=True,
    )
    if args.check:
        print("[REFERENCE26 CHECK] Native peer bundle and original LOOCV baseline PASS.")
        return 0
    locked(out, spec(bundle, peer, ks, args.folds))
    if args.summary:
        summaries(out, baseline, ks, args.folds, complete=True)
        return 0

    missing = []
    for position in range(n):
        path = out / f"fold_{position + 1:03d}.csv"
        if path.is_file():
            row = pd.read_csv(path, dtype={ID: str})
            validate(row, ks, 1)
            if (
                row.iloc[0][ID] != ids[position]
                or abs(row.iloc[0]["observed"] - actual[position]) > 1e-10
            ):
                raise ValueError(f"Stale cached fixed-k fold {position + 1}.")
        else:
            missing.append(position)
    if missing:
        count = min(args.jobs, len(missing))
        print(
            f"[REFERENCE26] {len(missing)} missing folds, "
            f"{count} CPU process(es), {len(ks)} separate KMeans k values.",
            flush=True,
        )
        with ProcessPoolExecutor(
            max_workers=count, initializer=worker_init,
            initargs=(str(bundle), str(peer)),
        ) as executor:
            futures = {
                executor.submit(fit_one, position, str(out), ks): position
                for position in missing
            }
            for future in as_completed(futures):
                idx, state, seconds = future.result()
                print(
                    f"[REFERENCE26 {idx:03d}/{n}] {state} {seconds:.1f}s",
                    flush=True,
                )

    summaries(out, baseline, ks, n, complete=not args.smoke)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
