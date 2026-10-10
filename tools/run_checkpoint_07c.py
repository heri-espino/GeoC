#!/usr/bin/env python3
"""Checkpoint 07C nested high-compute tuning: X highdim -> models -> Local04D.

Run from the installed GeoCebada environment, in repository root:

    python tools/run_checkpoint_07c.py --preflight --device gpu
    python tools/run_checkpoint_07c.py --smoke --device cpu
    python tools/run_checkpoint_07c.py --device gpu --trials 48
    python tools/run_checkpoint_07c.py --stage summarize

Outer 16 x 97/41 splits; each outer model is tuned only inside the 97
pseudo-training labels. Local04D prediction and 10/25/50% blends are frozen,
not trained on pseudo-target labels. No official 59-target predictions.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from geocebada.data.targets import load_yield_split
from geocebada.evaluation.checkpoint07c import (
    FAMILIES,
    aggregate_07c_results,
    load_development,
    run_nested_search,
)

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("evaluate", "summarize"), default="evaluate"
    )
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--families", default=",".join(FAMILIES))
    parser.add_argument("--trials", type=int, default=48)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--max-splits", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--features", type=Path,
        default=ROOT / "reports/checkpoint_07/highdim/expanded_parcel_features.csv",
    )
    parser.add_argument(
        "--targets", type=Path,
        default=ROOT / "data/source/tabular/"
        "ID_area_rendimiento_70_30_Reto_AgroCebada.csv",
    )
    parser.add_argument(
        "--membership", type=Path,
        default=ROOT / "reports/checkpoint_04b/pseudo_competition_splits.csv",
    )
    parser.add_argument(
        "--baseline", type=Path,
        default=ROOT / "reports/checkpoint_06/parcel_oof_residuals.csv",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "reports/checkpoint_07c",
    )
    return parser.parse_args()


def _preflight(args: argparse.Namespace, families: tuple[str, ...]) -> dict:
    for path in (args.features, args.targets, args.membership, args.baseline):
        if not path.is_file():
            raise FileNotFoundError(f"Missing 07C input: {path}")
        with path.open("rb") as stream:
            if stream.read(48).startswith(b"version https://git-lfs"):
                raise RuntimeError(f"Git LFS pointer, not usable source: {path}")
    if args.trials < 1 or args.inner_folds < 2 or args.threads < 1:
        raise ValueError("trials, inner-folds and threads must be positive.")
    if args.max_splits < 0:
        raise ValueError("max-splits cannot be negative.")
    required = {
        "optuna", *(
            {"catboost"} if "catboost" in families else set()
        ), *(
            {"xgboost"} if "xgboost" in families else set()
        ), *(
            {"lightgbm"} if "lightgbm" in families else set()
        ),
    }
    missing = sorted(x for x in required if importlib.util.find_spec(x) is None)
    if missing:
        raise RuntimeError(
            f"Missing 07C dependencies {missing}. "
            'Run: python -m pip install -e ".[dev,models,checkpoint07]"'
        )
    if args.device == "gpu" and any(
        f in families for f in ("catboost", "xgboost")
    ):
        try:
            subprocess.run(
                ["nvidia-smi", "-L"], stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, timeout=10, check=True,
            )
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            raise RuntimeError(
                "No NVIDIA GPU detected. GPU is the default for CatBoost/XGBoost. "
                "Use --device cpu explicitly to authorize CPU fallback."
            ) from e
    feature_columns = pd.read_csv(args.features, nrows=0).columns
    if len(feature_columns) < 100:
        raise ValueError("Unexpectedly small full 07B feature table.")
    return {
        "n_columns_including_id": len(feature_columns),
        "families": list(families),
        "device": args.device,
        "lightgbm_and_sklearn_device": "cpu",
        "outer_splits_total": 16,
        "inner_folds": args.inner_folds,
        "trials_per_outer_split_and_family": args.trials,
        "max_splits": args.max_splits,
        "validation": "97/41 nested development; 16 frozen overlapping splits",
        "canonical_local04d_modified": False,
        "hidden_y_accessed": False,
    }


def _commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            capture_output=True, text=True, timeout=8,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def main() -> int:
    args = parse_args()
    families = tuple(x.strip().lower() for x in args.families.split(",") if x.strip())
    if not families or len(set(families)) != len(families):
        raise ValueError("Choose at least one family without duplicates.")
    if set(families) - set(FAMILIES):
        raise ValueError(
            f"Unknown families {sorted(set(families) - set(FAMILIES))}"
        )
    if args.stage == "summarize":
        out = args.output.resolve()
        if args.smoke:
            out = out / "smoke"
        pred, summary = aggregate_07c_results(out)
        pred.to_csv(out / "development_predictions.csv", index=False)
        summary.to_csv(out / "development_summary.csv", index=False)
        print(summary.head(20).to_string(index=False), flush=True)
        print(f"07C partial or full development summary: {out}")
        return 0

    status = _preflight(args, families)
    print("[07C PREFLIGHT] " + json.dumps(status, indent=2), flush=True)
    if args.preflight:
        return 0
    output = args.output.resolve()
    if args.smoke:
        output = output / "smoke"
    output.mkdir(parents=True, exist_ok=True)
    # The full 07B feature table is built X-only; no synthetic smoke matrix.
    x = pd.read_csv(args.features, low_memory=False)
    y = load_yield_split(path=args.targets, validate=True)
    membership = pd.read_csv(args.membership, low_memory=False)
    incumbent = pd.read_csv(args.baseline, low_memory=False)
    data = load_development(x, y, membership, incumbent)
    split_ids = sorted(data.splits)
    if args.smoke:
        split_ids = split_ids[:1]
        families = families[:2]
        max_trials = min(args.trials, 2)
    else:
        if args.max_splits:
            split_ids = split_ids[:args.max_splits]
        max_trials = args.trials
    print(
        f"[07C] {len(split_ids)} splits; {len(families)} families; "
        f"{max_trials} trials per split/family; "
        f"{args.inner_folds} inner folds.",
        flush=True,
    )
    for index, split_id in enumerate(split_ids, 1):
        for family in families:
            prediction, meta = run_nested_search(
                data, split_id, family, output=output, trials=max_trials,
                inner_folds=args.inner_folds, seed=args.seed,
                device="cpu" if args.smoke else args.device,
                threads=args.threads,
            )
            print(
                f"[07C {index}/{len(split_ids)}] {split_id} {family} "
                f"model RMSE={meta['outer_rmse_model']:.5f}, "
                f"Local04D RMSE={meta['outer_rmse_local04d']:.5f}, "
                f"rows={len(prediction)}.",
                flush=True,
            )
    pred, summary = aggregate_07c_results(output)
    pred.to_csv(output / "development_predictions.csv", index=False)
    summary.to_csv(output / "development_summary.csv", index=False)
    report = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "git_commit": _commit(),
        "families": list(families),
        "device": "cpu" if args.smoke else args.device,
        "smoke": args.smoke,
        "max_trials": max_trials,
        "inner_folds": args.inner_folds,
        "n_outer_splits_completed": len(split_ids),
        "nested_cv": True,
        "selection_status": "development only; no model promoted",
        "canonical_local04d_modified": False,
    }
    (output / "run_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print("[07C DEVELOPMENT] " + summary.head(15).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
