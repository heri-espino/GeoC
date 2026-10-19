#!/usr/bin/env python3
"""Compare original median vs TRAIN-ONLY arithmetic-mean X imputation for Reference26.

Frozen Reference26 architecture / parameters and same 138 LOOCV heldouts.
Only numerical preprocessing missing-value fallbacks change from median
to arithmetic mean; fitted from each nested train subset, never from a test
yield. Original median predictions are read from the prior completed LOOCV.

  python tools/run_reference26_mean_imputation_loo.py --check
  python tools/run_reference26_mean_imputation_loo.py --smoke --jobs 1
  python tools/run_reference26_mean_imputation_loo.py --run --jobs 12
  python tools/run_reference26_mean_imputation_loo.py --summary
"""
from __future__ import annotations

import os

for name in (
    "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
):
    os.environ[name] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
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

from geocebada.evaluation.peer_mean_imputation import (  # noqa: E402
    ATTRIBUTES,
    missing_numeric_inputs,
    use_training_means,
)

N = 138
NEW = "Reference26_mean"
BASE = "Reference26_median"
OUT = ROOT / "reports/reference26_mean_imputation_loo138"
ORIGINAL = ROOT / "reports/selected_champions_loo/loo_predictions.csv"
_INPUTS = None


def load_baseline(path: Path, frame: pd.DataFrame) -> pd.DataFrame:
    """Read and align original 138 saved Reference26 predictions, without fitting."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Saved original 138-fold Reference26 predictions not found: {path}. "
            "Recover them with run_selected_champions_loo.py --summary."
        )
    saved = pd.read_csv(path, dtype={ID: str})
    ids = frame[ID].astype(str).tolist()
    if (
        len(saved) != N or not saved[ID].astype(str).is_unique
        or not {ID, "observed", "Reference26"}.issubset(saved.columns)
        or set(saved[ID].astype(str)) != set(ids)
    ):
        raise ValueError("Original LOOCV must contain exactly 138 matching IDs.")
    matched = saved.set_index(ID).loc[ids].reset_index()
    actual = frame[TARGET].to_numpy(float)
    if (
        not np.isfinite(matched[["observed", "Reference26"]].to_numpy(float)).all()
        or not np.allclose(
            matched["observed"].to_numpy(float), actual,
            rtol=0.0, atol=1e-9,
        )
    ):
        raise ValueError("Saved original Reference26 yields differ from peer bundle.")
    return matched[[ID, "observed", "Reference26"]].rename(
        columns={"Reference26": BASE}
    )


def run_spec(bundle: Path, peer: Path, original: Path) -> dict:
    teammate_files = sorted({
        "src/current_champions.py",
        "src/family_4/conditional_response.py",
        "src/geoc_signal_research.py",
        "src/physical_similarity.py",
        "src/indicator_neighborhoods.py",
        "src/temporal_indicator_weights.py",
    })
    return {
        "schema": 1,
        "design": "original Reference26, median vs inner-training arithmetic mean",
        "outer_cv": "138 heldouts in peer bundle order; 137 labeled training per fold",
        "inner_cv": "teammate native 4-fold nested selection, seed 20261018",
        "preprocessors": [list(row) for row in ATTRIBUTES],
        "median_original_from_saved_predictions": True,
        "bundle_manifest_sha256": digest(bundle / "manifest.json"),
        "baseline_predictions_sha256": digest(original),
        "peer_sources_sha256": {
            path: digest(peer / path) for path in teammate_files
        },
        "mean_override_sha256": digest(
            ROOT / "src/geocebada/evaluation/peer_mean_imputation.py"
        ),
        "original_fold_protocol_sha256": digest(
            ROOT / "tools/run_selected_champions_loo.py"
        ),
        "runner_sha256": digest(Path(__file__).resolve()),
    }


def lock(folder: Path, spec: dict) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    file = folder / "run_spec.json"
    if file.is_file():
        if json.loads(file.read_text(encoding="utf-8")) != spec:
            raise ValueError(
                "Inputs, original baseline, or mean imputation code changed. "
                "Use a different --output directory, not mixed old folds."
            )
    elif list(folder.glob("fold_*.csv")):
        raise ValueError("Found private fold results without a provenance manifest.")
    else:
        file.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")


def fold_file(folder: Path, position: int) -> Path:
    return folder / f"fold_{position + 1:03d}.csv"


def validate(
    path: Path, position: int, ids: list[str], actual: np.ndarray,
) -> float:
    row = pd.read_csv(path, dtype={ID: str})
    if (
        len(row) != 1 or set(row.columns) != {ID, "observed", NEW}
        or row.iloc[0][ID] != ids[position]
        or abs(float(row.iloc[0]["observed"]) - float(actual[position])) > 1e-10
        or not np.isfinite(row[NEW].to_numpy(float)).all()
    ):
        raise ValueError(f"Invalid or stale Reference26 mean LOOCV fold: {path}")
    return float(row.iloc[0][NEW])


def init_worker(bundle: str, peer: str) -> None:
    global _INPUTS
    _INPUTS = load_inputs(Path(bundle), Path(peer))


def fit_one(pos: int, folder_string: str) -> tuple[int, str, float]:
    if _INPUTS is None:
        raise RuntimeError("Worker private source inputs were not loaded.")
    frame, groups, distance, manifest, ids, ColdExperts, predict_saved = _INPUTS
    folder = Path(folder_string)
    output = fold_file(folder, pos)
    if output.is_file():
        validate(
            output, pos, ids, frame[TARGET].to_numpy(float)
        )
        return pos + 1, "cached", 0.0

    train, inner = fold_record(pos)
    masked = frame.copy(deep=True)
    masked.loc[masked.index[pos], TARGET] = np.nan
    if (
        len(train) != N - 1 or pos in train or len(inner) != 4
        or not np.isnan(float(masked.iloc[pos][TARGET]))
        or not np.isfinite(masked.iloc[train][TARGET].to_numpy(float)).all()
    ):
        raise ValueError("Invalid outer heldout y-mask or nested training fold.")
    temp = folder / "_training_cache" / f"fold_{pos + 1:03d}"
    started = time.perf_counter()
    try:
        with threadpool_limits(limits=1), use_training_means():
            factory = ColdExperts(
                masked, groups, distance, manifest["geology"],
                "random", train, inner, temp,
            )
            fit = factory.full_fit(train)
            if set(fit["train_positions"]) != set(train):
                raise ValueError("Native Reference26 fit used unexpected labels.")
            predictions = predict_saved(
                {
                    "outer_experts": fit,
                    "train_positions": train,
                    "validation_positions": np.asarray([pos], int),
                    "B1_weight": 0.0,
                },
                np.asarray([pos], int), masked, distance,
            )
        predicted = float(np.asarray(predictions["Reference26"], float).item())
        if not np.isfinite(predicted):
            raise ValueError("Mean-imputed Reference26 produced a nonfinite yield.")
        write_csv(output, pd.DataFrame([{
            ID: ids[pos],
            "observed": float(frame.iloc[pos][TARGET]),
            NEW: predicted,
        }]))
    finally:
        shutil.rmtree(temp, ignore_errors=True)
    return pos + 1, "trained", time.perf_counter() - started


def summary(folder: Path, baseline: pd.DataFrame, n: int) -> None:
    ids = baseline[ID].astype(str).tolist()
    actual = baseline["observed"].to_numpy(float)
    values = [validate(fold_file(folder, i), i, ids, actual) for i in range(n)]
    together = baseline.iloc[:n].copy()
    together[NEW] = values
    y = together["observed"].to_numpy(float)
    mse0 = float(np.mean((together[BASE].to_numpy(float) - y)**2))
    rows = []
    for name in (BASE, NEW):
        err = together[name].to_numpy(float) - y
        mse = float(np.mean(err**2))
        rows.append({
            "imputation": "median" if name == BASE else "mean",
            "model": name,
            "n_heldout": n,
            "train_labels_per_fit": N - 1,
            "rmse": float(np.sqrt(mse)),
            "mae": float(np.mean(abs(err))),
            "bias": float(np.mean(err)),
            "r2": float(r2_score(y, together[name])) if n > 1 else np.nan,
            "delta_mse_vs_median": mse - mse0,
        })
    table = pd.DataFrame(rows)
    prefix = "loo138" if n == N else "smoke"
    write_csv(folder / f"{prefix}_summary.csv", table)
    write_csv(folder / f"{prefix}_predictions_private.csv", together)
    print(
        f"[REFERENCE26 MEAN] Paired {n} heldout parcels (lower RMSE is better):\n"
        + table.to_string(index=False),
        flush=True,
    )
    if n != N:
        print("[SMOKE] Not a full LOOCV performance estimate.", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    for verb in ("check", "smoke", "run", "summary"):
        group.add_argument(f"--{verb}", action="store_true")
    parser.add_argument("--bundle", type=Path, default=BUNDLE)
    parser.add_argument("--peer-repo", type=Path, default=PEER_REPO)
    parser.add_argument("--baseline", type=Path, default=ORIGINAL)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be a positive integer, without artificial cap.")

    bundle = args.bundle.resolve()
    peer = args.peer_repo.resolve()
    original = args.baseline.resolve()
    folder = (args.output / "smoke" if args.smoke else args.output).resolve()
    frame, _groups, _distance, manifest, ids, _class, _predict = load_inputs(
        bundle, peer
    )
    baseline = load_baseline(original, frame)
    audit = missing_numeric_inputs(frame, manifest["features"])
    total = int(audit["missing_cells"].sum())
    print(
        "[MEAN] Original Reference26, 138 common outer holdouts, "
        "137 training labels per fit. Original model architecture unchanged.",
        flush=True,
    )
    print(
        f"[MEAN] Missing input cells in the 26-feature bundle: {total} "
        f"across {len(audit)} numeric features.",
        flush=True,
    )
    if not audit.empty:
        print(audit.to_string(index=False), flush=True)
    if args.check:
        return 0
    lock(folder, run_spec(bundle, peer, original))
    write_csv(folder / "missing_x_audit_aggregate.csv", audit)
    if args.summary:
        summary(folder, baseline, N)
        return 0

    n = 1 if args.smoke else N
    if total == 0:
        # No missing X anywhere. All five mean/median preprocessors receive
        # identical inputs in every inner/outer refit. Avoid 138 wasted fits.
        print("[MEAN] No missing X: mean and median are identical by design.", flush=True)
        for i in range(n):
            output = fold_file(folder, i)
            if output.is_file():
                validate(output, i, ids, frame[TARGET].to_numpy(float))
                continue
            write_csv(output, pd.DataFrame([{
                ID: ids[i], "observed": float(frame.iloc[i][TARGET]),
                NEW: float(baseline.iloc[i][BASE]),
            }]))
        summary(folder, baseline, n)
        return 0

    pending = []
    for i in range(n):
        path = fold_file(folder, i)
        if path.is_file():
            validate(path, i, ids, frame[TARGET].to_numpy(float))
            print(f"[MEAN {i + 1:03d}/{n}] cached", flush=True)
        else:
            pending.append(i)
    if pending:
        print(
            f"[MEAN] {len(pending)} folds; "
            f"{min(args.jobs, len(pending))} worker processes.",
            flush=True,
        )
        with ProcessPoolExecutor(
            max_workers=min(args.jobs, len(pending)),
            initializer=init_worker,
            initargs=(str(bundle), str(peer)),
        ) as pool:
            futures = {
                pool.submit(fit_one, i, str(folder)): i for i in pending
            }
            for task in as_completed(futures):
                i, state, elapsed = task.result()
                print(
                    f"[MEAN {i:03d}/{n}] {state} ({elapsed:.1f}s)",
                    flush=True,
                )
    summary(folder, baseline, n)
    if n == N:
        print(
            "[MEAN] Comparing two candidate imputations on the same 138 "
            "labels is exploratory; selection can inflate the observed gain.",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
