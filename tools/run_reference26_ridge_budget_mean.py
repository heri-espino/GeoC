#!/usr/bin/env python3
"""Reference26 mean-imputed staged Ridge search: 25 x 12 -> 8 x 46 -> 3 x 138.

This experiment changes only the five numerical X imputers to training-fold
means and then searches geographic k and Ridge alpha in the original
Reference26 recipe. All imputations are fitted separately inside native
nested CV. The saved old MEDIAN grid is never imported as a training cache.

  python tools/run_reference26_ridge_budget_mean.py --check
  python tools/run_reference26_ridge_budget_mean.py --smoke --jobs 1
  python tools/run_reference26_ridge_budget_mean.py --screen --jobs 12
  python tools/run_reference26_ridge_budget_mean.py --advance --jobs 12
  python tools/run_reference26_ridge_budget_mean.py --final --jobs 12
  python tools/run_reference26_ridge_budget_mean.py --summary
"""
from __future__ import annotations

import os

for name in (
    "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
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
import run_reference26_ridge250_loo as original_grid  # noqa: E402
import run_reference26_ridge_budget as budget  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

from geocebada.evaluation.peer_mean_imputation import (  # noqa: E402
    use_training_means,
)

ROOT = original_grid.ROOT
BUNDLE = original_grid.BUNDLE
PEER_REPO = original_grid.PEER_REPO
BASELINE = original_grid.BASELINE
OUTPUT = ROOT / "reports/reference26_ridge_budget_mean"
SCREEN = budget.SCREEN
STAGES = budget.STAGES
DEFAULT = original_grid.DEFAULT_PAIR
ID = original_grid.ID
TARGET = original_grid.TARGET


def provenance(bundle: Path, peer: Path, original: Path) -> dict:
    """The *method*, original y, script and five mean preprocessors are frozen."""
    from run_reference26_mean_imputation_loo import run_spec

    native = run_spec(bundle, peer, original)
    return {
        "schema": 1,
        "imputation": "arithmetic mean, fit only on each train subset",
        "protocol": "first 12, then 46, then 138 matched LOOCV folds",
        "search": [[k, a] for k, a in SCREEN],
        "stages": {k: list(v) for k, v in STAGES.items()},
        "peer_mean_protocol": native,
        "ridge_runner_sha256": original_grid.digest(
            ROOT / "tools/run_reference26_ridge250_loo.py"
        ),
        "budget_selection_sha256": original_grid.digest(
            ROOT / "tools/run_reference26_ridge_budget.py"
        ),
        "this_runner_sha256": original_grid.digest(Path(__file__).resolve()),
    }


def fold_path(output: Path, pos: int, pair: tuple[int, float]) -> Path:
    return original_grid.fold_file(output, pos, pair)


def run_fold_batch(pos: int, pairs: tuple, output_name: str):
    """All original fitted Reference26 components, fold-local mean imputers."""
    if original_grid._WORKER is None:
        raise RuntimeError("The native worker was not initialized.")
    output = Path(output_name)
    frame, _groups, _distance, _meta, ids, _cls, _predict = original_grid._WORKER
    y = frame[TARGET].to_numpy(float)
    trained = cached = 0
    start = time.perf_counter()
    from geocebada.evaluation.peer_reference26_ridge_grid import (
        share_climate_fits_between_grid_candidates,
    )

    with threadpool_limits(limits=1), use_training_means(), share_climate_fits_between_grid_candidates():
        for pair in pairs:
            path = fold_path(output, pos, pair)
            if path.is_file():
                original_grid.validate_row(path, pos, ids, y, pair)
                cached += 1
                continue
            prediction, temp = original_grid.fit_one(pos, pair, output)
            try:
                original_grid.write_csv(path, pd.DataFrame([{
                    ID: ids[pos],
                    "observed": float(y[pos]),
                    original_grid.candidate_name(*pair): prediction,
                }]))
            finally:
                shutil.rmtree(temp, ignore_errors=True)
            trained += 1
    return pos + 1, trained, cached, time.perf_counter() - start


def results(output: Path, reference: pd.DataFrame, stage: str, pairs: tuple):
    """All rows use exactly the same heldout IDs; median is comparison only."""
    count = STAGES[stage][0]
    ids = reference[ID].astype(str).tolist()
    y_all = reference["observed"].to_numpy(float)
    y = y_all[:count]
    old_predictions = reference["Reference26"].to_numpy(float)[:count]
    old_mse = float(np.mean((old_predictions-y)**2))
    predictions = {}
    rows = []
    for k, alpha in pairs:
        pair = (k, alpha)
        name = original_grid.candidate_name(k, alpha)
        pred = np.asarray([
            original_grid.validate_row(
                fold_path(output, pos, pair), pos, ids, y_all, pair
            ) for pos in range(count)
        ], float)
        predictions[name] = pred
        residual = pred - y
        mse = float(np.mean(residual**2))
        rows.append({
            "model": name,
            "imputation": "mean",
            "k": k, "alpha": alpha,
            "n_common_holdouts": count,
            "rmse": float(np.sqrt(mse)),
            "mae": float(np.mean(abs(residual))),
            "mean_error": float(np.mean(residual)),
            "delta_mse_vs_original_median": mse-old_mse,
        })
    summary = pd.DataFrame(rows).sort_values(
        ["rmse", "k", "alpha"], kind="stable"
    ).reset_index(drop=True)
    original_grid.write_csv(output / f"{stage}_scores.csv", summary)
    (output / f"{stage}_selection.json").write_text(
        json.dumps({
            "stage": stage, "imputation": "mean",
            "n_common_holdouts": count,
            "evaluated_pairs": [[k, a] for k, a in pairs],
        }, indent=2) + "\n", encoding="utf-8",
    )
    median_rmse = float(np.sqrt(old_mse))
    mean_anchor = next(row["rmse"] for row in rows
                       if (row["k"], row["alpha"]) == DEFAULT)
    print(f"[MEAN {stage.upper()}] Same first {count} heldouts; "
          f"original Reference26 median RMSE={median_rmse:.6f}; "
          f"Reference26 mean k24/a30 RMSE={mean_anchor:.6f}.", flush=True)
    print(summary.to_string(index=False), flush=True)
    print("[METHOD] Hyperparameters selected from staged heldout errors; "
          "scores are exploratory, not an independent unbiased estimate.",
          flush=True)


def followup_pairs(output: Path, stage: str):
    return budget.selected_pairs(output, stage)


def check_saved(output: Path, spec: dict):
    """Never reuse median outputs or mean outputs from changed source code."""
    budget.write_manifest(output, spec)


def run_stage(stage: str, output: Path, reference: pd.DataFrame, args):
    pairs = followup_pairs(output, stage)
    count = STAGES[stage][0]
    allids = reference[ID].astype(str).tolist()
    ally = reference["observed"].to_numpy(float)
    missing = []
    for pos in range(count):
        pair_candidates = []
        for pair in pairs:
            path = fold_path(output, pos, pair)
            if path.is_file():
                original_grid.validate_row(path, pos, allids, ally, pair)
            else:
                pair_candidates.append(pair)
        for i in range(0, len(pair_candidates), args.batch_size):
            missing.append((pos, tuple(pair_candidates[i:i+args.batch_size])))
    print(f"[MEAN {stage.upper()}] {len(pairs)} models × {count} parcels, "
          f"{len(missing)} remaining batches; {args.jobs} requested workers.",
          flush=True)
    if missing:
        with ProcessPoolExecutor(
            max_workers=min(args.jobs, len(missing)),
            initializer=original_grid.worker_init,
            initargs=(str(args.bundle.resolve()), str(args.peer_repo.resolve())),
        ) as pool:
            futures = [
                pool.submit(run_fold_batch, pos, pairs_, str(output))
                for pos, pairs_ in missing
            ]
            for i, future in enumerate(as_completed(futures), 1):
                held, new, cached, elapsed = future.result()
                print(f"[MEAN {stage} {i}/{len(missing)}] "
                      f"fold={held:03d}: {new} new + {cached} cached; "
                      f"{elapsed:.1f}s", flush=True)
    results(output, reference, stage, pairs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    for verb in ("check", "smoke", "screen", "advance", "final", "summary"):
        mode.add_argument(f"--{verb}", action="store_true")
    parser.add_argument("--bundle", type=Path, default=BUNDLE)
    parser.add_argument("--peer-repo", type=Path, default=PEER_REPO)
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--jobs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=5)
    args = parser.parse_args()
    if args.jobs < 1 or args.batch_size < 1:
        parser.error("--jobs and --batch-size must be positive.")

    bundle = args.bundle.resolve()
    peer = args.peer_repo.resolve()
    original = args.baseline.resolve()
    output = (args.output / "smoke" if args.smoke else args.output).resolve()
    frame, _groups, _distance, _meta, ids, _cls, _pred = (
        original_grid.load_inputs(bundle, peer)
    )
    baseline = original_grid.original_results(
        original, ids, frame[TARGET].to_numpy(float)
    )
    print("[REFERENCE26] Mean-imputed 25→8→3 local Ridge search "
          "on the original 138 parcel bundle.", flush=True)
    if args.check:
        print("[CHECK] Mean imputation sources and 138 original outcomes loaded; "
              "no fits or median grid reuse.", flush=True)
        return 0

    check_saved(output, provenance(bundle, peer, original))
    if args.summary:
        for stage in STAGES:
            file = output / f"{stage}_scores.csv"
            if file.is_file():
                print(f"\n[{stage.upper()}]\n"
                      + pd.read_csv(file).head(15).to_string(index=False))
        return 0
    if args.smoke:
        # Do not call original_grid.parity_baseline: its frozen baseline
        # uses MEDIAN imputers and is not a valid parity target for MEAN.
        original_grid._WORKER = original_grid.load_inputs(bundle, peer)
        fold, new, cached, elapsed = run_fold_batch(0, (DEFAULT,), str(output))
        print(f"[MEAN SMOKE] fold {fold}: {new} fitted, {cached} cached; "
              f"elapsed {elapsed:.1f}s; not a full LOOCV score.", flush=True)
        return 0
    if args.screen:
        stage = "screen"
    elif args.advance:
        stage = "advance"
    else:
        stage = "final"
    run_stage(stage, output, baseline, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
