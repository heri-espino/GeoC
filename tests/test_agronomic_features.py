"""Tests for Agronomic Feature Layer v1."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from geocebada.features import (
    build_agronomic_feature_layer,
    join_agronomic_features,
    validate_agronomic_feature_layer,
)

ROOT = Path(__file__).resolve().parents[1]
FEATURE_DIR = ROOT / "data" / "processed" / "features_v1"
BUILDER = runpy.run_path(ROOT / "tools" / "build_agronomic_features_v1.py")


def _config() -> dict:
    return yaml.safe_load(
        (ROOT / "configs" / "agronomic_features_v1.yaml").read_text(encoding="utf-8")
    )


def _base() -> pd.DataFrame:
    return pd.read_csv(FEATURE_DIR / "parcel_features_all.csv")


def test_agronomic_layer_is_additive_target_free_and_one_row_per_parcel() -> None:
    base = _base()
    table, records, diagnostics = build_agronomic_feature_layer(base, _config())

    assert len(table) == 197
    assert table["ID_POLIGONO"].is_unique
    assert "RENDIMIENTO_T_HA" not in table
    assert "CONJUNTO" not in table
    assert diagnostics["features"] == len(table.columns) - 1
    assert diagnostics["features"] >= 200
    assert {record["mode"] for record in records} == {"clean", "competition"}
    assert {record["evidence"] for record in records} <= {
        "literature_backed",
        "mechanistic_proxy",
        "experimental",
    }
    assert all("RENDIMIENTO_T_HA" not in record["inputs"] for record in records)

    validate_agronomic_feature_layer(
        table,
        expected_ids=base["ID_POLIGONO"],
    )
    values = table.drop(columns="ID_POLIGONO").to_numpy(dtype=float)
    assert not np.isinf(values).any()


def test_agronomic_layer_contains_expected_scientific_feature_families() -> None:
    table, records, _ = build_agronomic_feature_layer(_base(), _config())
    columns = set(table.columns)

    assert "agro_pheno__y2025_s2_ndvi__auc" in columns
    assert "agro_anomaly__s2_ndvi__season_mean_delta" in columns
    assert "agro_thermal__2025__gdd_proxy_base5p0c" in columns
    assert "agro_water__2025__npp_per_aeti_season" in columns
    assert "agro_soil__soc_x_cec" in columns
    assert "agro_interaction__siap_recent_yield_x_ndvi_anomaly" in columns

    families = {record["family"] for record in records}
    assert {
        "phenology",
        "phenology_anomaly",
        "sensor_agreement",
        "thermal",
        "water_timing",
        "water_productivity",
        "soil_profile",
        "soil_interaction",
        "cross_domain",
        "nonlinear_basis",
    } <= families


def test_clean_features_do_not_depend_on_2025_or_wapor_inputs() -> None:
    _, records, _ = build_agronomic_feature_layer(_base(), _config())

    for record in records:
        if record["mode"] != "clean":
            continue
        inputs = " ".join(record["inputs"]).casefold()
        assert "wapor_2025" not in inputs
        assert "sat_basic_2025" not in inputs
        assert "sat_pro_2025" not in inputs
        assert "clim_official_2025" not in inputs
        assert "siap_2025" not in inputs


def test_join_agronomic_features_is_one_to_one() -> None:
    base = _base()[["ID_POLIGONO", "CONJUNTO"]]
    agronomic, _, _ = build_agronomic_feature_layer(_base(), _config())

    joined = join_agronomic_features(base, agronomic)

    assert len(joined) == len(base)
    assert joined["ID_POLIGONO"].is_unique
    assert len(joined.columns) == len(base.columns) + len(agronomic.columns) - 1


def test_builder_writes_manifest_and_report(tmp_path: Path) -> None:
    report = BUILDER["build_agronomic_tables"](
        root=ROOT,
        config=_config(),
        output_directory=tmp_path,
    )

    table = pd.read_csv(tmp_path / "parcel_agronomic_features.csv")
    manifest = json.loads((tmp_path / "feature_manifest.json").read_text(encoding="utf-8"))
    written_report = json.loads((tmp_path / "build_report.json").read_text(encoding="utf-8"))

    assert report["rows"] == 197
    assert report["features"] == len(table.columns) - 1
    assert manifest["feature_count"] == report["features"]
    assert len(manifest["features"]) == report["features"]
    assert written_report["target_used_for_feature_construction"] is False
    assert written_report["chirps_used_for_feature_construction"] is False
