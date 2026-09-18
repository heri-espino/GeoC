"""Tests for parcel-level feature extraction and the master-table builder."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from geocebada.features.parcel import (
    build_base_features,
    build_satellite_features,
    build_siap_features,
    validate_parcel_feature_table,
    zonal_stats_for_raster,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "data" / "samples" / "integration"
BUILDER = runpy.run_path(ROOT / "tools" / "build_parcel_feature_table.py")


def test_base_features_preserve_one_row_per_fixture_parcel() -> None:
    split = pd.read_csv(FIXTURE / "split.csv")
    parcels = gpd.read_file(FIXTURE / "parcels.geojson")
    admin = pd.read_csv(FIXTURE / "admin.csv", dtype={"cvegeo": "string"})

    features, records = build_base_features(split, parcels, admin=admin)

    assert len(features) == 12
    assert features["ID_POLIGONO"].is_unique
    assert features["base_geom_area_ha"].gt(0).all()
    assert features["base_centroid_lon"].between(-99, -98).all()
    assert features["base_centroid_lat"].between(19, 21).all()
    assert records


def test_satellite_features_keep_clean_and_competition_modes() -> None:
    basic = pd.read_csv(FIXTURE / "basic.csv")

    features, records = build_satellite_features(
        basic,
        source_prefix="sat_basic",
        target_start="2025-04-01",
        target_end="2025-10-31",
        historical_start="2022-01-01",
        historical_end="2024-12-31",
        historical_months=[4, 5, 6, 7, 8, 9, 10],
        target_months=[4, 5, 6, 7, 8, 9, 10],
    )

    modes = {record["mode"] for record in records}
    assert len(features) == 12
    assert features["ID_POLIGONO"].is_unique
    assert modes == {"clean", "competition"}
    assert any(column.startswith("sat_basic_hist") for column in features)
    assert any(column.startswith("sat_basic_2025") for column in features)


def test_siap_features_use_grain_barley_and_keep_2025_separate() -> None:
    admin = pd.read_csv(FIXTURE / "admin.csv", dtype={"cvegeo": "string"})
    siap = pd.read_csv(FIXTURE / "siap.csv", dtype={"cvegeo": "string"})

    features, records = build_siap_features(admin, siap)

    assert len(features) == 12
    assert features["ID_POLIGONO"].is_unique
    assert "siap_hist__yield_mean_recent" in features
    assert "siap_2025__yield" in features
    mode_by_column = {record["column"]: record["mode"] for record in records}
    assert mode_by_column["siap_hist__yield_mean_recent"] == "clean"
    assert mode_by_column["siap_2025__yield"] == "competition"


def test_zonal_stats_return_all_fixture_parcels() -> None:
    parcels = gpd.read_file(FIXTURE / "parcels.geojson")
    raster = FIXTURE / "topography" / "Elevacion_INEGI_CEM4_120m.tif"

    stats = zonal_stats_for_raster(
        parcels,
        raster,
        stats=("mean", "std", "min", "max"),
    )

    assert len(stats) == 12
    assert stats["ID_POLIGONO"].is_unique
    assert np.isfinite(stats["mean"]).all()


def test_fixture_master_table_builds_end_to_end_without_wapor(tmp_path: Path) -> None:
    feature_config = BUILDER["load_yaml"](ROOT / "configs" / "features_v1.yaml")
    contract = BUILDER["load_yaml"](ROOT / "configs" / "data_contract_v2.yaml")

    report = BUILDER["build_tables"](
        root=ROOT,
        feature_config=feature_config,
        contract=contract,
        fixture=True,
        include_wapor=False,
        output_directory=tmp_path,
    )

    all_table = pd.read_csv(tmp_path / "parcel_features_all.csv")
    clean = pd.read_csv(tmp_path / "parcel_features_clean.csv")
    competition = pd.read_csv(tmp_path / "parcel_features_competition.csv")
    manifest = json.loads((tmp_path / "feature_manifest.json").read_text(encoding="utf-8"))

    validate_parcel_feature_table(
        all_table,
        expected_rows=12,
        expected_train=6,
        expected_prediction=6,
    )
    assert report["rows"] == 12
    assert report["features_clean"] > 0
    assert report["features_competition_total"] > report["features_clean"]
    assert len(clean) == len(competition) == 12
    assert len(competition.columns) > len(clean.columns)
    assert {item["mode"] for item in manifest["features"]} == {"clean", "competition"}
