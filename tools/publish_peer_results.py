#!/usr/bin/env python3
"""Publish only audited aggregate peer-model CV scores, never parcel-level data.

  python tools/publish_peer_results.py --export
  git add docs/PEER_LOOCV_RECORDED_RESULTS.md results/peer_models/
  git commit -m "Archive aggregate peer model LOOCV results"
  git push origin experiment/nuria-champions-loocv

Only copies metric columns from recognized 46/138-fold summary files.
Source directories in reports/ remain ignored and are never unignored.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/peer_models"

# (source private CSV, export file, expected holdouts, allowed names)
SOURCES = (
    (
        "reports/selected_champions_loo/loo_summary.csv",
        "reference_compact_loo138.csv", 138,
        {"Reference26", "Compact14"},
    ),
    (
        "reports/area_weighted_climate_loo/loo_summary.csv",
        "historic_local04d_climate_loo138.csv", 138,
        {
            "GeoPCA20_GeographicWeightedRidge",
            "GeoPCA20_PlusTouchedPixelClimate",
            "GeoPCA20_PlusAreaWeightedClimate",
            "GeoPCA20_PlusAreaWeightedTemporalClimate",
        },
    ),
    (
        "reports/peer_gate_grid_loo46/loo46_scores.csv",
        "geological_gates_loo46.csv", 46,
        {
            "Reference26_DBSCAN_geology",
            "Reference26_global_weight",
            "Reference26_original_KMeans",
            "Reference26_KMeans_2_4_6_8_12",
            "Reference26_geological_support",
            "Compact14_original",
        },
    ),
    (
        "reports/peer_dbscan_loo138/loo_summary.csv",
        "dbscan_loo138.csv", 138,
        {"Reference26_DBSCAN"},
    ),
    (
        "reports/peer_dbscan_loo138/four_model_summary.csv",
        "four_model_comparison_loo138.csv", 138,
        {"Reference26", "Reference26_DBSCAN", "Compact14", "Local04D"},
    ),
    (
        "reports/peer_champions_loo/loo_summary.csv",
        "legacy_five_model_loo138.csv", 138,
        {"GeoC_Local04D", "CT3", "Reference26", "Compact14", "B1"},
    ),
)

NUMERIC = (
    "rmse", "mae", "r2", "bias", "mean_error",
    "delta_mse_vs_original", "delta_mse_vs_Reference26",
)
COUNT_FIELDS = (
    "held_out_parcels", "heldout", "unique_heldout_parcels",
    "n_holdouts", "heldout_parcels",
)
TRAIN_FIELDS = ("train_labels_per_fold", "training_labels_per_fit")


def _hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def audited_scores(
    source: Path, *, count: int, allowed: set[str],
) -> pd.DataFrame:
    scores = pd.read_csv(source)
    if (not len(scores) or "model" not in scores.columns
            or "rmse" not in scores.columns or "r2" not in scores.columns):
        raise ValueError(f"Not a valid model-level aggregate summary: {source}")
    if (scores["model"].duplicated().any()
            or not set(scores["model"]).issubset(allowed)):
        raise ValueError(f"Unexpected or duplicate model in {source}.")
    present_counts = [name for name in COUNT_FIELDS if name in scores.columns]
    if not present_counts:
        raise ValueError(f"Cannot verify held-out count in {source}.")
    if any(not scores[name].eq(count).all() for name in present_counts):
        raise ValueError(f"Wrong fold count in {source}; expected {count}.")
    for name in TRAIN_FIELDS:
        if name in scores and not scores[name].eq(137).all():
            raise ValueError(f"Expected 137 training labels per fold in {source}.")
    present = [col for col in NUMERIC if col in scores.columns]
    if not np.isfinite(scores[present].to_numpy(float)).all():
        raise ValueError(f"Aggregate metrics are not all finite in {source}.")
    if (scores["rmse"] < 0).any() or (
        "mae" in scores and (scores["mae"] < 0).any()
    ):
        raise ValueError(f"Negative error metrics in {source}.")
    # Whitelist columns only: cannot accidentally export an ID/yield.
    result = scores[["model", *present]].copy()
    result.insert(1, "held_out_parcels", int(count))
    result.insert(2, "train_labels_per_fold", 137)
    return result


def export(root: Path = ROOT, destination: Path = OUT) -> list[dict]:
    destination.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    for relative, name, count, allowed in SOURCES:
        source = root / relative
        if not source.is_file():
            print(f"[PUBLIC AGGREGATES] Not available yet: {relative}")
            continue
        clean = audited_scores(source, count=count, allowed=allowed)
        file = destination / name
        clean.to_csv(file, index=False, float_format="%.16g")
        records.append({
            "source": relative,
            "aggregate": str(file.relative_to(destination)),
            "n_models": len(clean),
            "n_heldout_parcels": count,
            "source_sha256": _hash(source),
        })
        print(f"[PUBLIC AGGREGATES] Exported {relative} => {file}")
    if not records:
        raise FileNotFoundError(
            "No completed aggregate summary CSV found. Generate --summary first."
        )
    (destination / "sources.json").write_text(
        json.dumps({"schema": 1, "sources": records}, indent=2) + "\n",
        encoding="utf-8",
    )
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", action="store_true", required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    exports = export(args.root.resolve(), args.output.resolve())
    print(
        f"[PUBLIC AGGREGATES] {len(exports)} summaries archived. "
        "Only results/peer_models is intended for Git. "
        "Do not git add reports/ or data_n/.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
