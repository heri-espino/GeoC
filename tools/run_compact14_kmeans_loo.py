#!/usr/bin/env python3
"""138-fold LOOCV: Compact14 + geological KMeans (k=4/8).

Reference26 original LOOCV is reused, not retrained. Existing teammate source
is untouched; only its Compact14 global/local mixing is replaced. We use the
first 138 original peer bundle rows, excluding one observed yield each time.

  python tools/run_compact14_kmeans_loo.py --check
  python tools/run_compact14_kmeans_loo.py --smoke
  python tools/run_compact14_kmeans_loo.py --run --jobs 12
  python tools/run_compact14_kmeans_loo.py --compare
"""
from __future__ import annotations

import os

# Independent folds use independent processes. One numerical thread per worker.
for key in (
    "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
):
    os.environ[key] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
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

from geocebada.evaluation.peer_compact14_kmeans import (  # noqa: E402
    compact_kmeans_factory,
    fit_compact_only,
    predict_compact_kmeans,
    selected_k,
)

N = 138
MODEL = "Compact14_KMeans"
OUTPUT = ROOT / "reports/compact14_kmeans_loo138"
ORIGINAL = ROOT / "reports/selected_champions_loo/loo_predictions.csv"
_DATA = None


def validate(frame: pd.DataFrame, cols: set[str], n: int = N):
    if len(frame) != n or set(frame.columns) != cols:
        raise ValueError("Incorrect saved LOOCV shape or columns.")
    if not frame[ID].astype(str).is_unique:
        raise ValueError("Duplicate held-out parcel IDs.")
    for key in cols - {ID}:
        if not np.isfinite(frame[key].to_numpy(float)).all():
            raise ValueError(f"Nonfinite values in {key}.")


def load_original(path: Path, ids: list[str], yields: np.ndarray) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing saved original 138-fold models: {path}. "
            "Run selected-champions --summary to regenerate from completed folds."
        )
    original = pd.read_csv(path, dtype={ID: str})
    validate(original, {ID, "observed", "Reference26", "Compact14"})
    if set(original[ID]) != set(ids):
        raise ValueError("Original 138-fold model parcel IDs do not match the bundle.")
    aligned = original.set_index(ID).loc[ids].reset_index()
    if not np.allclose(
        aligned["observed"].to_numpy(float),
        yields, atol=1e-9, rtol=0,
    ):
        raise ValueError("Original and current observed yields differ.")
    return aligned


def fingerprint(bundle: Path, peer: Path) -> dict:
    return {
        "protocol": "138 x leave-one-parcel-out; first 138 bundle rows",
        "training_labels_per_fold": 137,
        "inner": "4-fold train-only CV; seed 20261018; geological KMeans k 4 or 8",
        "model": MODEL,
        "original_reference_model": "Original Reference26 from saved 138-fold CSV",
        "original_bundle_manifest": digest(bundle / "manifest.json"),
        "original_teammate_model": digest(peer / "src/current_champions.py"),
        "original_teammate_compact": digest(peer / "src/pruned_global_architecture.py"),
        "new_compact_gate": digest(
            ROOT / "src/geocebada/evaluation/peer_compact14_kmeans.py"
        ),
        "fold_definition": digest(
            ROOT / "tools/run_selected_champions_loo.py"
        ),
        "runner": digest(Path(__file__).resolve()),
    }


def lock(folder: Path, spec: dict) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "run_spec.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != spec:
            raise ValueError(
                "Inputs or code changed since the previous run; use a new "
                "--output directory, not a mix of incompatible results."
            )
    elif list(folder.glob("fold_*.csv")):
        raise ValueError("Existing fold files without their provenance record.")
    else:
        path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")


def init_worker(bundle: str, peer: str):
    global _DATA
    _DATA = load_inputs(Path(bundle), Path(peer))


def fit_one(pos: int, output: str) -> tuple[int, str, float]:
    if _DATA is None:
        raise RuntimeError("Peer worker has not loaded data.")
    frame, groups, distance, manifest, ids, factory_type, _native_predict = _DATA
    held_id = ids[pos]
    observed = float(frame.iloc[pos][TARGET])
    folder = Path(output)
    file = folder / f"fold_{pos + 1:03d}.csv"
    if file.is_file():
        row = pd.read_csv(file, dtype={ID: str})
        validate(row, {ID, "observed", MODEL}, 1)
        if row.iloc[0][ID] != held_id or abs(row.iloc[0]["observed"] - observed) > 1e-10:
            raise ValueError("Cached fold was generated for another parcel/target.")
        return pos + 1, "cached", 0.0

    tr, inner = fold_record(pos)
    masked = frame.copy(deep=True)
    masked.loc[masked.index[pos], TARGET] = np.nan
    if (not np.isnan(masked.iloc[pos][TARGET])
            or not np.isfinite(masked.iloc[tr][TARGET].to_numpy(float)).all()):
        raise RuntimeError("Outer parcel label masking failed.")

    start = time.perf_counter()
    with threadpool_limits(limits=1), compact_kmeans_factory():
        factory = factory_type(
            masked, groups, distance, manifest["geology"], "random",
            tr, inner, folder / f"cache/fold_{pos + 1:03d}",
        )
        fitted = fit_compact_only(factory, tr)
        if set(fitted["train_positions"]) != set(tr):
            raise RuntimeError("Compact14 KMeans trained on unexpected parcels.")
        prediction = predict_compact_kmeans(
            fitted, np.asarray([pos], int), masked, distance
        )
        diagnostic = selected_k(fitted["outer_experts"])

    prediction = float(np.asarray(prediction).item())
    row = pd.DataFrame([{ID: held_id, "observed": observed, MODEL: prediction}])
    validate(row, {ID, "observed", MODEL}, 1)
    write_csv(file, row)
    (folder / f"kmeans_{pos + 1:03d}.json").write_text(
        json.dumps(diagnostic, indent=2) + "\n", encoding="utf-8"
    )
    return pos + 1, "trained", time.perf_counter() - start


def aggregate(folder: Path) -> pd.DataFrame:
    files = [folder / f"fold_{i:03d}.csv" for i in range(1, N + 1)]
    if any(not f.is_file() for f in files):
        raise ValueError("Need all 138 saved Compact14 + KMeans folds.")
    rows = pd.concat(
        [pd.read_csv(f, dtype={ID: str}) for f in files],
        ignore_index=True,
    )
    validate(rows, {ID, "observed", MODEL})
    return rows


def scores(comparison: pd.DataFrame) -> pd.DataFrame:
    observed = comparison["observed"].to_numpy(float)
    names = [v for v in ("Reference26", "Compact14", MODEL) if v in comparison.columns]
    output = []
    for name in names:
        err = comparison[name].to_numpy(float) - observed
        output.append({
            "model": name,
            "n_holdouts": len(comparison),
            "training_labels_per_fold": 137,
            "rmse": float(np.sqrt(np.mean(err * err))),
            "mae": float(np.mean(abs(err))),
            "bias": float(np.mean(err)),
            "r2": float(r2_score(observed, comparison[name]))
            if len(comparison) >= 2 else np.nan,
        })
    return pd.DataFrame(output).sort_values("rmse", kind="stable").reset_index(drop=True)


def compare(folder: Path, saved_original: Path, ids: list[str], y: np.ndarray):
    if not (folder / "run_spec.json").is_file():
        raise ValueError("No matching KMeans result provenance.")
    original = load_original(saved_original, ids, y)
    new = aggregate(folder)
    joined = original.merge(
        new, on=ID, how="left", validate="one_to_one", suffixes=("", "_km")
    )
    if joined[MODEL].isna().any() or not np.allclose(
        joined["observed"].to_numpy(float),
        joined["observed_km"].to_numpy(float), rtol=0, atol=1e-9,
    ):
        raise ValueError("The KMeans and original model heldouts differ.")
    joined = joined.drop(columns="observed_km")
    write_csv(folder / "paired_predictions_private.csv", joined)
    summary = scores(joined)
    write_csv(folder / "paired_summary.csv", summary)
    frequencies = Counter()
    for i in range(1, N + 1):
        path = folder / f"kmeans_{i:03d}.json"
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            frequencies[str(data.get("k", "fallback"))] += 1
    frequency_rows = pd.DataFrame([
        {"selected_k": key, "folds": value}
        for key, value in sorted(frequencies.items())
    ], columns=["selected_k", "folds"])
    write_csv(folder / "selected_k_counts.csv", frequency_rows)
    print("[COMPACT14 + KMEANS] Exact paired original 138 LOOCV predictions:\n"
          + summary.to_string(index=False), flush=True)
    print("[COMPACT14 + KMEANS] Selected k by outer fold:\n"
          + frequency_rows.to_string(index=False), flush=True)
    print("The additional geological gate means this is not a strictly 14-feature model.")


def verify_native_compact_parity(
    bundle: Path, peer: Path, originals: Path, out: Path,
) -> None:
    """Smoke a copied compact-only R3 path against a REAL saved native prediction."""
    frame, groups, distance, manifest, ids, factory_type, _ = load_inputs(bundle, peer)
    original = load_original(originals, ids, frame[TARGET].to_numpy(float))
    pos = 0
    tr, inner = fold_record(pos)
    masked = frame.copy(deep=True)
    masked.loc[masked.index[pos], TARGET] = np.nan
    with threadpool_limits(limits=1):
        factory = factory_type(
            masked, groups, distance, manifest["geology"], "random",
            tr, inner, out / "native_parity",
        )
        fit = fit_compact_only(factory, tr)
        predicted = float(predict_compact_kmeans(
            fit, np.asarray([pos]), masked, distance,
        )[0])
    saved = float(original.iloc[pos]["Compact14"])
    error = abs(predicted - saved)
    print(f"[NATIVE PARITY] first held-out Compact14 prediction, "
          f"abs diff={error:.3g} t/ha", flush=True)
    if error > 1e-7:
        raise RuntimeError(
            "Compact14-only R3 branch does not reproduce the saved original "
            f"model ({error:.6g} t/ha). Stop before running 138 KMeans folds."
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    options = parser.add_mutually_exclusive_group(required=True)
    for flag in ("check", "smoke", "run", "summary", "compare"):
        options.add_argument("--" + flag, action="store_true")
    parser.add_argument("--peer-repo", type=Path, default=PEER_REPO)
    parser.add_argument("--bundle", type=Path, default=BUNDLE)
    parser.add_argument("--original-results", type=Path, default=ORIGINAL)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive; there is no arbitrary maximum.")

    bundle = args.bundle.resolve()
    peer = args.peer_repo.resolve()
    output = (args.output / "smoke" if args.smoke else args.output).resolve()
    if args.summary:
        rows = aggregate(output)
        results = scores(rows)
        write_csv(output / "loo_predictions.csv", rows)
        write_csv(output / "loo_summary.csv", results)
        print(results.to_string(index=False), flush=True)
        return 0

    frame, _groups, _distance, _manifest, ids, _, _ = load_inputs(bundle, peer)
    print("[COMPACT14 + KMEANS] Original peer 138-row dataset; "
          "train 137/test 1; geological k=4 or 8; original Reference26 reused.",
          flush=True)
    original = args.original_results.resolve()
    if args.check:
        load_original(original, ids, frame[TARGET].to_numpy(float))
        print("[CHECK] Original 138-fold Reference26/Compact14 IDs and y match.")
        return 0
    if args.compare:
        compare(
            output, original, ids, frame[TARGET].to_numpy(float)
        )
        return 0

    lock(output, fingerprint(bundle, peer))
    if args.smoke:
        verify_native_compact_parity(bundle, peer, original, output)
    n = 1 if args.smoke else N
    pending = []
    for pos in range(n):
        file = output / f"fold_{pos + 1:03d}.csv"
        if file.exists():
            saved = pd.read_csv(file, dtype={ID: str})
            validate(saved, {ID, "observed", MODEL}, 1)
            if (
                saved.iloc[0][ID] != ids[pos]
                or abs(saved.iloc[0]["observed"] - frame.iloc[pos][TARGET]) > 1e-10
            ):
                raise ValueError(f"Stale results in fold {pos + 1}.")
        else:
            pending.append(pos)
    if pending:
        print(f"[COMPACT14+KMEANS] {len(pending)} missing folds; "
              f"{min(args.jobs, len(pending))} CPU processes.", flush=True)
        with ProcessPoolExecutor(
            max_workers=min(args.jobs, len(pending)),
            initializer=init_worker,
            initargs=(str(bundle), str(peer)),
        ) as executor:
            futures = {
                executor.submit(fit_one, i, str(output)): i for i in pending
            }
            for task in as_completed(futures):
                fold, state, elapsed = task.result()
                print(f"[KMEANS {fold:03d}/{N}] {state} {elapsed:.1f}s", flush=True)
    if args.smoke:
        print("[SMOKE] One-fold run passed; this is not an LOOCV RMSE.")
    else:
        compare(
            output, original, ids, frame[TARGET].to_numpy(float)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
