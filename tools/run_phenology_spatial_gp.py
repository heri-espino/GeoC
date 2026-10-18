#!/usr/bin/env python3
"""Run a paired, fold-safe phenology-conditioned spatial-GP experiment.

PowerShell:
  python tools/run_phenology_spatial_gp.py --check
  python tools/run_phenology_spatial_gp.py --smoke
  python tools/run_phenology_spatial_gp.py --run
  python tools/run_phenology_spatial_gp.py --summary
  python tools/run_phenology_spatial_gp.py --predict-targets

Requires previously processed HLS/SMAP and area-weighted monthly weather.
No raw satellite download, hidden label access, or final-model promotion.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from geocebada.evaluation.checkpoint04a import haversine_distance_matrix
from geocebada.evaluation.checkpoint08_local import ID, atomic_json, digest
from geocebada.evaluation.checkpoint10_pca import inputs, load_pca_data
from geocebada.evaluation.checkpoint10_pca import predict as predict_incumbent
from geocebada.evaluation.phenology_spatial_gp import (
    CANDIDATE,
    SpatialGPData,
    evaluate_split,
    inner_select,
    predict_spatial_gp,
    summarize_matched,
)
from geocebada.features.phenology_spatial import build_phenology_weather_features

ROOT = Path(__file__).resolve().parents[1]
MEMBERSHIPS = ROOT / "reports/checkpoint_08/confirmation_membership.csv"
CLIMATE = ROOT / "reports/area_weighted_climate_experiment/weighted_monthly.csv"
HLS = ROOT / "data/processed/checkpoint_07/hls_parcel_observations.csv"
SMAP = ROOT / "data/processed/checkpoint_07/smap_parcel_daily.csv"
OUT = ROOT / "reports/phenology_spatial_gp"
CODE = (
    "tools/run_phenology_spatial_gp.py",
    "src/geocebada/features/phenology_spatial.py",
    "src/geocebada/evaluation/phenology_spatial_gp.py",
    "src/geocebada/evaluation/checkpoint10_pca.py",
    "src/geocebada/evaluation/checkpoint04d1.py",
    "src/geocebada/evaluation/checkpoint04a.py",
    "src/geocebada/evaluation/checkpoint08_local.py",
)


def _write_csv(path: Path, table: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    table.to_csv(temp, index=False, float_format="%.16g")
    temp.replace(path)


def _source_paths(climate: Path) -> dict[str, Path]:
    sources = {**inputs(ROOT, MEMBERSHIPS), "climate_weighted": climate,
               "hls_panel": HLS, "smap_panel": SMAP}
    sources.update({f"code:{name}": ROOT / name for name in CODE})
    return sources


def _lock_run(output: Path, climate: Path) -> None:
    """Reject resumptions with changed X, labels, splits or implementation."""
    files = _source_paths(climate)
    missing = [name for name, path in files.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing experiment inputs: {missing}. See docs/PHENOLOGY_SPATIAL_GP.md"
        )
    spec = {
        "experiment": "phenology_spatial_gp_v1",
        "seed": 20261018,
        "protocol": "16 historically reused 97/41 masks; inner 3fold CV",
        "checksums": {name: digest(path) for name, path in sorted(files.items())},
        "no_hidden_y": True,
        "no_promotion": True,
    }
    output.mkdir(parents=True, exist_ok=True)
    target = output / "run_spec.json"
    if target.exists():
        old = json.loads(target.read_text(encoding="utf-8"))
        if old != spec:
            raise ValueError(
                "Run provenance changed; choose a new --output directory. "
                "Never combine completed folds from different data/code."
            )
    else:
        atomic_json(target, spec)


def _load(climate: Path) -> SpatialGPData:
    """Build 197 X rows and use the official C4 numerical-parity gate."""
    baseline = load_pca_data(ROOT, MEMBERSHIPS, max_components=20)
    weighted = pd.read_csv(climate, dtype={ID: str})
    hls = pd.read_csv(HLS, dtype={ID: str}, low_memory=False)
    smap = pd.read_csv(SMAP, dtype={ID: str}, low_memory=False)
    physical = build_phenology_weather_features(weighted, hls, smap, baseline.ids)
    coords = pd.read_csv(
        ROOT / "data/processed/features_v1/parcel_features_all.csv",
        usecols=[ID, "base_centroid_lat", "base_centroid_lon"],
        dtype={ID: str},
    )
    if not coords[ID].is_unique or set(coords[ID]) != set(baseline.ids):
        raise ValueError("Centroids do not match all 197 parcel IDs.")
    coords = coords.set_index(ID).reindex(baseline.ids)
    if coords.isna().any().any():
        raise ValueError("Incomplete centroid coordinates.")
    distance = haversine_distance_matrix(
        coords["base_centroid_lat"], coords["base_centroid_lon"]
    )
    data = SpatialGPData(baseline, physical, distance)
    data.validate()
    return data


def _coverage(data: SpatialGPData) -> dict:
    """Diagnostics only: no response/yield enters feature creation."""
    x = data.physical.drop(columns=ID)
    climate_cols = ["prec_total", "temp_mean", "gdd5_monthly_proxy"]
    return {
        "rows": len(x), "columns": x.shape[1],
        "nonmissing_per_feature": x.notna().sum().astype(int).to_dict(),
        "complete_hls_ndvi_parcels": int(x["ndvi_n_days"].gt(0).sum()),
        "complete_smap_parcels": int(x["smap_n_days"].gt(0).sum()),
        "peak_month_observed_parcels": int(x["peak_month"].notna().sum()),
        "unique_coarse_weather_profiles": int(
            x[climate_cols].drop_duplicates().shape[0]
        ),
        "all_numeric_columns_finite_or_nan": bool(
            np.isfinite(x.to_numpy(float)[~np.isnan(x.to_numpy(float))]).all()
        ),
    }


def _evaluate(data: SpatialGPData, output: Path, *, smoke: bool) -> None:
    splits = sorted(data.baseline.folds)
    if smoke:
        splits = splits[:1]
    for j, split_id in enumerate(splits, start=1):
        slug = f"fold_{j:02d}"
        pred_file = output / f"{slug}_predictions.csv"
        score_file = output / f"{slug}_inner_scores.csv"
        meta_file = output / f"{slug}_selection.json"
        if all(p.is_file() for p in (pred_file, score_file, meta_file)):
            print(f"[{j}/{len(splits)}] {split_id}: resumed", flush=True)
            continue
        predictions, scores, meta = evaluate_split(data, split_id, smoke=smoke)
        _write_csv(pred_file, predictions)
        _write_csv(score_file, scores)
        atomic_json(meta_file, meta)
        print(
            f"[{j}/{len(splits)}] {split_id}: "
            f"selected spatial_share={meta['selected']['spatial_share']}, "
            f"inner RMSE={meta['selected_inner_rmse']:.5f}", flush=True
        )


def _summary(output: Path) -> None:
    expected = [output / f"fold_{i:02d}_predictions.csv" for i in range(1, 17)]
    missing = [p.name for p in expected if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Only complete 16-fold runs can be summarized; missing {missing}."
        )
    combined = pd.concat([pd.read_csv(path, dtype={ID: str}) for path in expected])
    if combined["split_id"].nunique() != 16 or len(combined) != 16 * 41 * 2:
        raise ValueError("Incomplete or duplicated matched pseudo-target bank.")
    summary, paired = summarize_matched(combined)
    _write_csv(output / "comparison_summary.csv", summary)
    _write_csv(output / "paired_delta_mse.csv", paired)
    print(summary.to_string(index=False), flush=True)
    print(
        "Paired mean delta-MSE (candidate minus Local04D): "
        f"{paired['delta_mse'].mean():.6f}; "
        "historical masks reused, not independent confirmation.",
        flush=True,
    )


def _predict_targets(data: SpatialGPData, output: Path) -> None:
    """Generate *candidate-only* 59 forecasts after inner CV on all 138 y."""
    known = data.baseline.y.index.astype(str).tolist()
    hidden = [pid for pid in data.baseline.ids if pid not in set(known)]
    if len(known) != 138 or len(hidden) != 59:
        raise ValueError("Hidden-label/parcel-count contract failed.")
    selected, scores = inner_select(data, known, smoke=False)
    candidate = predict_spatial_gp(data, known, hidden, selected)
    frozen = predict_incumbent(
        data.baseline, known, hidden,
        components=20, k=24, alpha=30.0, power=1.0,
    )
    out = pd.DataFrame({
        ID: hidden, CANDIDATE: candidate,
        "GeoPCA20_GeographicWeightedRidge": frozen,
    })
    if not np.isfinite(out.iloc[:, 1:].to_numpy(float)).all():
        raise ValueError("Nonfinite final candidate predictions.")
    _write_csv(output / "candidate_59_predictions_NOT_PROMOTED.csv", out)
    _write_csv(output / "final_138label_inner_cv.csv", scores)
    atomic_json(output / "candidate_59_selection.json", {
        "selected": vars(selected), "inner_cv_rmse": float(scores.iloc[0]["inner_rmse"]),
        "training_labels": 138, "hidden_labels_accessed": False,
        "promoted": False, "warning": "Candidate only; not independently confirmed.",
    })
    print("Wrote 59 candidate-only forecasts; official Local04D unchanged.", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check", action="store_true")
    action.add_argument("--smoke", action="store_true")
    action.add_argument("--run", action="store_true")
    action.add_argument("--summary", action="store_true")
    action.add_argument("--predict-targets", action="store_true")
    parser.add_argument("--climate", type=Path, default=CLIMATE)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    climate = args.climate if args.climate.is_absolute() else ROOT / args.climate
    if args.summary:
        _summary(output)
        return 0
    destination = output / "smoke" if args.smoke else output
    _lock_run(destination, climate)
    data = _load(climate)
    report = _coverage(data)
    atomic_json(destination / "x_only_coverage.json", report)
    if args.check:
        print(json.dumps(report, indent=2, sort_keys=True))
    elif args.predict_targets:
        _predict_targets(data, destination)
    else:
        _evaluate(data, destination, smoke=args.smoke)
        if args.run:
            _summary(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
