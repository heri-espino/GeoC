#!/usr/bin/env python3
"""Evaluate four frozen geographic climate models using 138-fold LOOCV.

Reuse target-free climate matrices already built in the earlier 16-fold run.

  python tools/run_area_weighted_climate_loo.py --check
  python tools/run_area_weighted_climate_loo.py --smoke
  python tools/run_area_weighted_climate_loo.py --run
  python tools/run_area_weighted_climate_loo.py --summary
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from geocebada.evaluation.area_weighted_climate_comparison import make_features, predict_local
from geocebada.evaluation.checkpoint08_local import ID, atomic_json, digest
from geocebada.evaluation.checkpoint10_pca import load_pca_data
from geocebada.evaluation.climate_leave_one_out import (
    MODELS,
    leave_one_out_predictions,
    leave_one_out_summary,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports/area_weighted_climate_experiment"
OUT = ROOT / "reports/area_weighted_climate_loo"
MEMBERSHIPS = ROOT / "reports/checkpoint_08/confirmation_membership.csv"
CLIMATE_FILES = ("weighted_monthly.csv", "touched_monthly.csv", "weighted_temporal.csv")
IMPLEMENTATION = (
    "tools/run_area_weighted_climate_loo.py",
    "src/geocebada/evaluation/climate_leave_one_out.py",
    "src/geocebada/evaluation/area_weighted_climate_comparison.py",
    "src/geocebada/evaluation/checkpoint10_pca.py",
    "src/geocebada/evaluation/checkpoint04d1.py",
)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    temp = path.with_suffix(".csv.tmp")
    frame.to_csv(temp, index=False, float_format="%.16g")
    temp.replace(path)


def _load(source: Path):
    if not (source / "run_spec.json").is_file():
        raise FileNotFoundError(
            f"Missing original climate run_spec.json in {source}. "
            "Run the original --build command before LOOCV."
        )
    for name in CLIMATE_FILES:
        if not (source / name).is_file():
            raise FileNotFoundError(f"Missing previously built climate data: {source / name}")
    baseline = load_pca_data(ROOT, MEMBERSHIPS, max_components=160)
    tables = [pd.read_csv(source / name, dtype={ID: str}) for name in CLIMATE_FILES]
    features = make_features(baseline, tables[0], tables[1], tables[2])
    if len(features.baseline.y) != 138 or len(features.baseline.ids) != 197:
        raise ValueError("Expected 138 visible and 59 hidden parcel yields.")
    if len(features.baseline.folds) != 16:
        raise ValueError("Expected exactly 16 historical reference folds.")
    return features


def _fingerprint(source: Path, output: Path) -> None:
    official = {
        "source_climate_provenance": source / "run_spec.json",
        "historical_membership": MEMBERSHIPS,
        "observed_yields": (
            ROOT / "data/source/tabular/ID_area_rendimiento_70_30_Reto_AgroCebada.csv"
        ),
        "base_features": ROOT / "data/processed/features_v1/parcel_features_all.csv",
        "agronomic_features": (
            ROOT / "data/processed/agronomic_features_v1/parcel_agronomic_features.csv"
        ),
        "empirical_features": (
            ROOT / "data/processed/empirical_features_v1/parcel_empirical_features.csv"
        ),
        **{name: source / name for name in CLIMATE_FILES},
    }
    all_inputs = {**official, **{name: ROOT / name for name in IMPLEMENTATION}}
    for name, path in all_inputs.items():
        if not path.is_file():
            raise FileNotFoundError(f"Cannot fingerprint {name}: {path}")
    meta = {
        "schema": 1,
        "protocol": "138_leave_one_parcel_out_frozen_parameters",
        "train_size": 137,
        "holdout_size": 1,
        "n_folds": 138,
        "models": list(MODELS),
        "transductive_x": "original 197-parcel C4 PCA and climate PCA (no target y)",
        "paths_and_sha256": {
            name: {"path": str(path.relative_to(ROOT)), "sha256": digest(path)}
            for name, path in all_inputs.items()
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    spec = output / "run_spec.json"
    if spec.is_file():
        if json.loads(spec.read_text(encoding="utf-8")) != meta:
            raise ValueError(
                "LOOCV inputs or code changed. Use a NEW --output directory "
                "rather than mixing old and new predictions."
            )
    else:
        if (output / "loo_predictions.csv").exists():
            raise ValueError("Existing LOOCV output has no matching provenance spec.")
        atomic_json(spec, meta)


def _summary(output: Path) -> None:
    path = output / "loo_predictions.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing complete LOOCV predictions: {path}")
    scores, paired = leave_one_out_summary(pd.read_csv(path, dtype={ID: str}))
    _write_csv(output / "loo_summary.csv", scores)
    _write_csv(output / "loo_paired_differences.csv", paired)
    print("[LOO] Leave-one-parcel-out: 138 outer folds, 137 train + 1 test each.")
    print("[LOO] Scores (lower RMSE/MAE, higher R2 better):\n"
          + scores.to_string(index=False), flush=True)
    print("[LOO] Paired RMSE/MSE differences vs frozen geographic Ridge:\n"
          + paired.to_string(index=False), flush=True)
    print(
        "[LOO] Scores are NOT directly comparable with historical 16 x 41, "
        "97-train RMSE 0.481936. Spatial autocorrelation and model-selection "
        "history make this exploratory.",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    for command in ("check", "smoke", "run", "summary"):
        mode.add_argument(f"--{command}", action="store_true")
    parser.add_argument("--features-dir", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    source = args.features_dir.resolve()
    output = args.output.resolve()
    _fingerprint(source, output)
    if args.summary:
        _summary(output)
        return 0
    f = _load(source)
    if args.check:
        print(
            "[LOO CHECK] 197 X rows, 138 observed yields, 59 unobserved yields, "
            "all four frozen climate feature arms PASS.", flush=True,
        )
        return 0
    if args.smoke:
        target = str(f.baseline.y.index[0])
        train = [pid for pid in f.baseline.y.index if pid != target]
        before = {model: predict_local(f, train, [target], model) for model in MODELS}
        old = float(f.baseline.y.loc[target])
        f.baseline.y.loc[target] = old + 100000.
        try:
            after = {model: predict_local(f, train, [target], model) for model in MODELS}
            for model in MODELS:
                np.testing.assert_allclose(before[model], after[model], rtol=0, atol=0)
        finally:
            f.baseline.y.loc[target] = old
        print("[LOO SMOKE] All four models exclude the held-out target label: PASS.")
        return 0
    prediction_path = output / "loo_predictions.csv"
    if prediction_path.exists():
        print("[LOO] Existing output found; verifying saved predictions.", flush=True)
        leave_one_out_summary(pd.read_csv(prediction_path, dtype={ID: str}))
    else:
        result = leave_one_out_predictions(f)
        leave_one_out_summary(result)  # refuse incomplete data before atomic write
        _write_csv(prediction_path, result)
        print("[LOO] Saved 138 x 4 paired out-of-fold predictions.", flush=True)
    _summary(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
