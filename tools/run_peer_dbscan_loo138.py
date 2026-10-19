#!/usr/bin/env python3
"""Full 138-fold LOOCV for Reference26+DBSCAN; compare with saved champions.

No repeated original Reference26/Compact14 fitting. Source teammate model
ColdExperts.full_fit and original DBSCAN gate tuning on train-only inner folds.
Use the exact 138 shared held-outs; this is not the earlier first-46 subset.

  python tools/run_peer_dbscan_loo138.py --check
  python tools/run_peer_dbscan_loo138.py --smoke
  python tools/run_peer_dbscan_loo138.py --run --jobs 12
  python tools/run_peer_dbscan_loo138.py --summary
  python tools/run_peer_dbscan_loo138.py --compare
"""
from __future__ import annotations

import os

# Cap threads per process, but not the user's requested number of processes.
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

from geocebada.evaluation.peer_gate_grid import configured_gate, native_reference_gate  # noqa: E402

N = 138
MODEL = "Reference26_DBSCAN"
OUTPUT = ROOT / "reports/peer_dbscan_loo138"
ORIGINAL = ROOT / "reports/selected_champions_loo/loo_predictions.csv"
MEMBERSHIP = ROOT / "reports/checkpoint_08/confirmation_membership.csv"
_DATA = None


def verify_rows(frame: pd.DataFrame, required: set[str], *, n: int = N) -> None:
    if (len(frame) != n or set(frame.columns) != required
            or not frame[ID].astype(str).is_unique):
        raise ValueError("Incomplete/malformed 138-parcel LOOCV table.")
    for key in required - {ID}:
        if not np.isfinite(frame[key].to_numpy(float)).all():
            raise ValueError(f"Invalid numbers in {key}.")


def provenance(bundle: Path, peer: Path) -> dict:
    return {
        "protocol": "all 138 peer bundle rows, 137 fit, one masked heldout",
        "inner": "4-fold KFold shuffled 20261018; DBSCAN selected train-only",
        "models": [MODEL],
        "bundle_manifest_sha256": digest(bundle / "manifest.json"),
        "peer_code_sha256": digest(peer / "src/current_champions.py"),
        "gate_code_sha256": digest(ROOT / "src/geocebada/evaluation/peer_gate_grid.py"),
        "dbscan_code_sha256": digest(
            ROOT / "src/geocebada/evaluation/peer_geology_dbscan.py"
        ),
        "fold_code_sha256": digest(ROOT / "tools/run_selected_champions_loo.py"),
        "runner_sha256": digest(Path(__file__).resolve()),
    }


def lock(folder: Path, value: dict) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    file = folder / "run_spec.json"
    if file.exists():
        if json.loads(file.read_text(encoding="utf-8")) != value:
            raise ValueError(
                "Code or inputs changed. Use a fresh --output directory "
                "rather than mixing completed folds."
            )
    elif list(folder.glob("fold_*.csv")):
        raise ValueError("Untracked previously completed folds exist.")
    else:
        file.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def init_worker(bundle: str, peer: str) -> None:
    global _DATA
    _DATA = load_inputs(Path(bundle), Path(peer))


def fit_one(position: int, out_string: str):
    if _DATA is None:
        raise RuntimeError("Worker bundle not initialized.")
    frame, groups, distance, manifest, ids, ColdExperts, predict_saved = _DATA
    output = Path(out_string)
    held_id = ids[position]
    actual = float(frame.iloc[position][TARGET])
    destination = output / f"fold_{position + 1:03d}.csv"
    if destination.is_file():
        cached = pd.read_csv(destination, dtype={ID: str})
        verify_rows(cached, {ID, "observed", MODEL}, n=1)
        if cached.iloc[0][ID] != held_id or abs(cached.iloc[0]["observed"] - actual) > 1e-10:
            raise ValueError("Stale cached DBSCAN held-out parcel.")
        return position + 1, "cached", 0.

    train, inner = fold_record(position)
    masked = frame.copy(deep=True)
    masked.loc[masked.index[position], TARGET] = np.nan
    if (not np.isnan(masked.iloc[position][TARGET])
            or not np.isfinite(masked.iloc[train][TARGET].to_numpy(float)).all()):
        raise ValueError("Held-out yield isolation failed.")
    started = time.perf_counter()
    with threadpool_limits(limits=1), configured_gate("dbscan"):
        factory = ColdExperts(
            masked, groups, distance, manifest["geology"], "random",
            train, inner, output / f"cache/fold_{position + 1:03d}",
        )
        trained = factory.full_fit(train)
        if set(trained["train_positions"]) != set(train):
            raise ValueError("Peer fitted wrong training rows.")
        prediction = predict_saved({
            "outer_experts": trained,
            "train_positions": train,
            "validation_positions": np.array([position]),
            "B1_weight": 0.0,
        }, np.array([position]), masked, distance)
        predicted = float(np.asarray(prediction["Reference26"]).item())
        gate = native_reference_gate(trained)

    row = pd.DataFrame([{ID: held_id, "observed": actual, MODEL: predicted}])
    verify_rows(row, {ID, "observed", MODEL}, n=1)
    write_csv(destination, row)
    gate_path = output / f"gate_{position + 1:03d}.json"
    gate_path.write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
    return position + 1, "trained", time.perf_counter() - started


def read_dbscan(output: Path) -> pd.DataFrame:
    files = [output / f"fold_{i:03d}.csv" for i in range(1, N + 1)]
    if any(not path.is_file() for path in files):
        raise ValueError("DBSCAN needs all 138 saved folds before comparison.")
    data = pd.concat(
        [pd.read_csv(f, dtype={ID: str}) for f in files], ignore_index=True
    )
    verify_rows(data, {ID, "observed", MODEL})
    return data


def score(values: pd.DataFrame, cols: tuple[str, ...]) -> pd.DataFrame:
    actual = values["observed"].to_numpy(float)
    reference_error = values["Reference26"].to_numpy(float) - actual if (
        "Reference26" in values
    ) else None
    rows = []
    for name in cols:
        residual = values[name].to_numpy(float) - actual
        item = {
            "model": name, "n_holdouts": len(values),
            "training_labels_per_fit": 137,
            "rmse": float(np.sqrt(np.mean(residual**2))),
            "mae": float(np.mean(abs(residual))),
            "bias": float(np.mean(residual)),
            "r2": float(r2_score(actual, values[name])) if len(values) > 1 else np.nan,
        }
        if reference_error is not None:
            item["delta_mse_vs_Reference26"] = float(
                np.mean(residual**2) - np.mean(reference_error**2)
            )
        rows.append(item)
    return pd.DataFrame(rows).sort_values("rmse", kind="stable").reset_index(drop=True)


def local04d_predictions(expected: pd.DataFrame, source: Path | None) -> np.ndarray:
    if source is not None:
        existing = pd.read_csv(source, dtype={ID: str})
        cols = set(existing.columns)
        if cols != {ID, "observed", "Local04D"}:
            raise ValueError(
                "External Local04D predictions must contain exactly "
                "ID_POLIGONO,observed,Local04D (138 rows)."
            )
        verify_rows(existing, {ID, "observed", "Local04D"})
        aligned = existing.set_index(ID).reindex(expected[ID].astype(str))
        if aligned.isna().any().any():
            raise ValueError("Local04D file has mismatched/missing parcel IDs.")
        if not np.allclose(
            aligned["observed"].to_numpy(float),
            expected["observed"].to_numpy(float), atol=1e-9, rtol=0,
        ):
            raise ValueError("Local04D file has different observed yields.")
        return aligned["Local04D"].to_numpy(float)

    # Refit the *original Local04D*, not another PCA. This requires
    # Checkpoint 04A saved C4 PCA and a strict 41-historical-forecast parity pass.
    from geocebada.evaluation.checkpoint08_local import FROZEN, predict_old
    from geocebada.evaluation.peer_frozen_local04d import load_frozen_local04d

    historical = load_frozen_local04d(ROOT, MEMBERSHIP)
    if set(historical.y.index.astype(str)) != set(expected[ID].astype(str)):
        raise ValueError("Peer training IDs differ from GeoC Local04D training IDs.")
    observed = historical.y.reindex(expected[ID].astype(str)).to_numpy(float)
    if not np.allclose(
        observed, expected["observed"].to_numpy(float), atol=1e-9, rtol=0
    ):
        raise ValueError("GeoC historical observed yields differ from peer targets.")

    known = historical.y.index.astype(str).tolist()
    preds = []
    with threadpool_limits(limits=1):
        for parcel in expected[ID].astype(str):
            train = [name for name in known if name != parcel]
            preds.append(float(predict_old(historical, train, [parcel], FROZEN)[0]))
    predictions = np.asarray(preds, float)
    if not np.isfinite(predictions).all():
        raise ValueError("Local04D refits produced nonfinite predictions.")
    return predictions


def compare(output: Path, originals: Path, local_path: Path | None) -> None:
    if not (output / "run_spec.json").is_file():
        raise ValueError("Missing DBSCAN run provenance.")
    dbscan = read_dbscan(output)
    if not originals.is_file():
        raise FileNotFoundError(
            f"Need original 138-fold Reference26+Compact14 results: {originals}. "
            "Run python tools/run_selected_champions_loo.py --run --jobs 2 first."
        )
    original = pd.read_csv(originals, dtype={ID: str})
    verify_rows(original, {ID, "observed", "Reference26", "Compact14"})
    if set(dbscan[ID]) != set(original[ID]):
        raise ValueError("The DBSCAN and original model holdout parcel IDs differ.")
    joined = original.merge(
        dbscan[[ID, "observed", MODEL]], on=ID, how="left", validate="one_to_one",
        suffixes=("", "_dbscan"),
    )
    if not np.allclose(
        joined["observed"].to_numpy(float),
        joined["observed_dbscan"].to_numpy(float), atol=1e-9, rtol=0,
    ):
        raise ValueError("DBSCAN and original model held-out yields differ.")
    joined = joined.drop(columns="observed_dbscan")
    joined["Local04D"] = local04d_predictions(joined, local_path)
    cols = ("Reference26", MODEL, "Compact14", "Local04D")
    verify_rows(joined, {ID, "observed", *cols})
    metrics = score(joined, cols)
    write_csv(output / "four_model_predictions_private.csv", joined)
    write_csv(output / "four_model_summary.csv", metrics)
    print("\n[138 PAIRED LOOCV] The same 138 held-out parcels and 137 labels/fold:")
    print(metrics.to_string(index=False), flush=True)
    print(
        "All model selection previously used these yields; treat this as "
        "exploratory validation, not an independent test.",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ("check", "smoke", "run", "summary", "compare"):
        mode.add_argument(f"--{name}", action="store_true")
    parser.add_argument("--bundle", type=Path, default=BUNDLE)
    parser.add_argument("--peer-repo", type=Path, default=PEER_REPO)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--original-results", type=Path, default=ORIGINAL)
    parser.add_argument("--local04d-file", type=Path, default=None)
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be at least 1 (no fixed upper cap).")

    out = (args.output / "smoke" if args.smoke else args.output).resolve()
    if args.summary:
        if not (out / "run_spec.json").is_file():
            raise ValueError("No DBSCAN run provenance manifest.")
        values = read_dbscan(out)
        metrics = score(values, (MODEL,))
        write_csv(out / "loo_predictions.csv", values)
        write_csv(out / "loo_summary.csv", metrics)
        print(metrics.to_string(index=False), flush=True)
        return 0
    if args.compare:
        compare(
            out, args.original_results.resolve(),
            args.local04d_file.resolve() if args.local04d_file else None,
        )
        return 0

    bundle, peer = args.bundle.resolve(), args.peer_repo.resolve()
    frame, _groups, _distance, _manifest, ids, _cls, _prediction = load_inputs(
        bundle, peer
    )
    print(
        f"[DBSCAN 138] Bundle {bundle}\n"
        f"[DBSCAN 138] Original peer recipe with inner-tuned geological DBSCAN, "
        f"all {N} held-out parcels.",
        flush=True,
    )
    if args.check:
        return 0
    lock(out, provenance(bundle, peer))
    number = 1 if args.smoke else N
    missing = []
    for i in range(number):
        path = out / f"fold_{i + 1:03d}.csv"
        if path.exists():
            data = pd.read_csv(path, dtype={ID: str})
            verify_rows(data, {ID, "observed", MODEL}, n=1)
            if (str(data.iloc[0][ID]) != ids[i]
                    or abs(float(data.iloc[0]["observed"])
                           - float(frame.iloc[i][TARGET])) > 1e-10):
                raise ValueError(f"Incorrect cached held-out fold {i+1}.")
        else:
            missing.append(i)
    if missing:
        print(
            f"[DBSCAN 138] Training {len(missing)} folds with "
            f"{min(args.jobs, len(missing))} CPU processes.",
            flush=True,
        )
        with ProcessPoolExecutor(
            max_workers=min(args.jobs, len(missing)),
            initializer=init_worker,
            initargs=(str(bundle), str(peer)),
        ) as pool:
            tasks = {
                pool.submit(fit_one, pos, str(out)): pos for pos in missing
            }
            for future in as_completed(tasks):
                fold, state, elapsed = future.result()
                print(f"[DBSCAN {fold}/{N}] {state}, {elapsed:.1f}s", flush=True)
    if args.smoke:
        print("[SMOKE] One-fold training complete, not a LOOCV performance score.")
    else:
        values = read_dbscan(out)
        metrics = score(values, (MODEL,))
        write_csv(out / "loo_predictions.csv", values)
        write_csv(out / "loo_summary.csv", metrics)
        print(metrics.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
