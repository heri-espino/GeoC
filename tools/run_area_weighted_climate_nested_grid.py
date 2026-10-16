#!/usr/bin/env python3
"""Nested LOOCV grid search, with frozen geographic Ridge reference.

Requires the already built area-weighted climate feature tables and the previous
frozen LOOCV output. Never rebuilds rasters. No 59 hidden yields are touched.

  python tools/run_area_weighted_climate_nested_grid.py --smoke
  python tools/run_area_weighted_climate_nested_grid.py --run
  python tools/run_area_weighted_climate_nested_grid.py --summary

Run a broader grid separately with --wide (different output folder).
Results are written atomically per outer parcel for restart safety.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from run_area_weighted_climate_loo import MEMBERSHIPS, ROOT, SOURCE, _load

from geocebada.evaluation.checkpoint08_local import ID, atomic_json, digest
from geocebada.evaluation.climate_leave_one_out import MODELS
from geocebada.evaluation.climate_nested_grid import (
    FROZEN,
    TUNED_SUFFIX,
    grid_candidates,
    summarize_nested_grid,
    tune_one_outer_parcel,
)

FROZEN_LOO = ROOT / "reports/area_weighted_climate_loo/loo_predictions.csv"
FOCUSED_OUT = ROOT / "reports/area_weighted_climate_nested_grid"
WIDE_OUT = ROOT / "reports/area_weighted_climate_nested_grid_wide"
SOURCE_CODE = (
    "src/geocebada/evaluation/climate_nested_grid.py",
    "src/geocebada/evaluation/climate_leave_one_out.py",
    "src/geocebada/evaluation/area_weighted_climate_comparison.py",
    "src/geocebada/evaluation/checkpoint04d1.py",
    "src/geocebada/evaluation/checkpoint10_pca.py",
    "tools/run_area_weighted_climate_nested_grid.py",
    "tools/run_area_weighted_climate_loo.py",
)


def _write(path: Path, frame: pd.DataFrame) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False, float_format="%.16g")
    temp.replace(path)


def _fingerprint(output: Path, source: Path, wide: bool) -> None:
    inputs = {
        "previous_loo_predictions": FROZEN_LOO,
        "climate_run_spec": source / "run_spec.json",
        "weighted_monthly": source / "weighted_monthly.csv",
        "touched_monthly": source / "touched_monthly.csv",
        "weighted_temporal": source / "weighted_temporal.csv",
        "targets": ROOT / "data/source/tabular/ID_area_rendimiento_70_30_Reto_AgroCebada.csv",
        "membership": MEMBERSHIPS,
        "base": ROOT / "data/processed/features_v1/parcel_features_all.csv",
        "agronomic": ROOT / "data/processed/agronomic_features_v1/parcel_agronomic_features.csv",
        "empirical": ROOT / "data/processed/empirical_features_v1/parcel_empirical_features.csv",
        **{p: ROOT / p for p in SOURCE_CODE},
    }
    for name, path in inputs.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing nested CV input {name}: {path}")
    spec = {
        "schema": 1,
        "protocol": "outer_leave_one_out_inner_fivefold_full_grid",
        "outer_folds": 138,
        "outer_train": 137,
        "inner_folds": 5,
        "grid_type": "wide" if wide else "focused",
        "candidates": [list(p) for p in grid_candidates(wide=wide)],
        "frozen": list(FROZEN),
        "models": list(MODELS),
        "inputs": {
            key: {"path": str(path.relative_to(ROOT)), "sha256": digest(path)}
            for key, path in inputs.items()
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "run_spec.json"
    if destination.is_file():
        if json.loads(destination.read_text(encoding="utf-8")) != spec:
            raise ValueError(
                "Grid, inputs, or code changed. Use a NEW --output folder, "
                "not a partially completed previous run."
            )
    else:
        if list(output.glob("outer_*__predictions.csv")):
            raise ValueError("Unidentified prior predictions without a fingerprint.")
        atomic_json(destination, spec)


def _collect(output: Path, ids: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    predicted = []
    selected = []
    for i, pid in enumerate(ids, start=1):
        predictions_path = output / f"outer_{i:03d}__predictions.csv"
        selections_path = output / f"outer_{i:03d}__selections.csv"
        if not predictions_path.is_file() or not selections_path.is_file():
            raise ValueError(f"Missing completed nested outer fold {i}/138.")
        current_p = pd.read_csv(predictions_path, dtype={ID: str})
        current_s = pd.read_csv(selections_path, dtype={ID: str})
        if (set(current_p[ID]) != {pid} or set(current_s[ID]) != {pid}
                or len(current_p) != 5 or len(current_s) != 4):
            raise ValueError(f"Corrupted or misordered nested fold {i}.")
        predicted.append(current_p)
        selected.append(current_s)
    return (
        pd.concat(predicted, ignore_index=True),
        pd.concat(selected, ignore_index=True),
    )


def _reference_parity(predictions: pd.DataFrame, ids: list[str]) -> None:
    historic = pd.read_csv(FROZEN_LOO, dtype={ID: str})
    original = historic.loc[historic["model"].eq(MODELS[0])]
    if (len(original) != 138 or set(original[ID]) != set(ids)
            or original[ID].duplicated().any()):
        raise ValueError("Original LOOCV reference does not contain all 138 parcels.")
    ref = original.set_index(ID).reindex(ids)
    new = predictions.loc[predictions["model"].eq(MODELS[0])].set_index(ID).reindex(ids)
    for column in ("observed", "predicted"):
        delta = float(np.max(np.abs(
            new[column].to_numpy(float) - ref[column].to_numpy(float)
        )))
        if not np.isfinite(delta) or delta > 1e-8:
            raise ValueError(f"Frozen baseline parity FAILED for {column}: {delta}")
    print("[GRID] Original LOOCV reference predictions: exact parity PASS.", flush=True)


def _summary(output: Path, ids: list[str]) -> None:
    predictions, selections = _collect(output, ids)
    scores, deltas, counts = summarize_nested_grid(predictions, selections)
    _reference_parity(predictions, ids)
    for name, frame in (
        ("nested_predictions.csv", predictions),
        ("nested_selections.csv", selections),
        ("nested_summary.csv", scores),
        ("nested_paired_differences.csv", deltas),
        ("nested_parameter_counts.csv", counts),
    ):
        _write(output / name, frame)
    print("[GRID] 138 outer leave-one-out folds x 5 matched model procedures.")
    print("[GRID] RMSE/MAE/R2, with selection strictly inside each outer fold:\n"
          + scores.to_string(index=False), flush=True)
    print("[GRID] Paired delta RMSE vs exact frozen geographic Ridge:\n"
          + deltas.to_string(index=False), flush=True)
    print(
        "[GRID] Descriptive exploratory scores only; overlapping geographies "
        "and extensive prior model search limit independence. "
        "No promotion or 59-label predictions.",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    for command in ("smoke", "run", "summary"):
        mode.add_argument(f"--{command}", action="store_true")
    parser.add_argument("--wide", action="store_true", help="Broader 168-candidate grid.")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--features-dir", type=Path, default=SOURCE)
    parser.add_argument("--from-fold", type=int, default=1)
    parser.add_argument("--through-fold", type=int, default=138)
    args = parser.parse_args()
    output = (args.output or (WIDE_OUT if args.wide else FOCUSED_OUT)).resolve()
    source = args.features_dir.resolve()
    if not (1 <= args.from_fold <= args.through_fold <= 138):
        parser.error("Require 1 <= --from-fold <= --through-fold <= 138.")
    _fingerprint(output, source, args.wide)
    features = _load(source)
    ids = features.baseline.y.index.astype(str).tolist()
    if args.summary:
        _summary(output, ids)
        return 0
    if args.smoke:
        held = ids[0]
        fixed = FROZEN
        alternatives = (12, 10.0, 1.)
        predictions, selected = tune_one_outer_parcel(
            features, held, candidates=[fixed, alternatives],
        )
        assert len(predictions) == 5 and len(selected) == 4
        print(
            "[GRID SMOKE] Matched fivefold nested selection and one heldout "
            "prediction for each of four feature arms: PASS. No full run.",
            flush=True,
        )
        return 0
    candidates = grid_candidates(wide=args.wide)
    print(
        f"[GRID] {len(candidates)} candidates x 4 model feature arms, "
        "five inner folds per each of 138 outer holdouts.", flush=True,
    )
    for idx in range(args.from_fold, args.through_fold + 1):
        held = ids[idx - 1]
        predicted = output / f"outer_{idx:03d}__predictions.csv"
        selected = output / f"outer_{idx:03d}__selections.csv"
        if predicted.is_file() and selected.is_file():
            oldp = pd.read_csv(predicted, dtype={ID: str})
            olds = pd.read_csv(selected, dtype={ID: str})
            if (len(oldp) != 5 or len(olds) != 4
                    or set(oldp[ID]) != {held} or set(olds[ID]) != {held}
                    or set(oldp["model"]) != {
                        MODELS[0], *(m + TUNED_SUFFIX for m in MODELS)
                    } or set(olds["model"]) != set(MODELS)):
                raise ValueError(f"Damaged saved fold {idx}.")
            print(f"[GRID {idx}/138] existing result validated.", flush=True)
            continue
        if predicted.exists() != selected.exists():
            raise ValueError(f"Incomplete fold {idx}; inspect both saved files.")
        result, settings = tune_one_outer_parcel(features, held, candidates=candidates)
        _write(predicted, result)
        _write(selected, settings)
        print(
            f"[GRID {idx}/138] four independently tuned climate arms complete.",
            flush=True,
        )
    if args.from_fold == 1 and args.through_fold == 138:
        _summary(output, ids)
    else:
        print("[GRID] Partial fold range saved. Run remaining folds, then --summary.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
