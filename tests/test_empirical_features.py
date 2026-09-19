"""Tests for Checkpoint 03B empirical feature discovery."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from geocebada.features import (
    FoldLocalExpressionMiner,
    build_empirical_feature_layer,
    join_empirical_features,
    validate_empirical_feature_layer,
)

ROOT = Path(__file__).resolve().parents[1]
FEATURE_DIR = ROOT / "data" / "processed" / "features_v1"
BUILDER = runpy.run_path(ROOT / "tools" / "build_empirical_features_v1.py")


def _config() -> dict:
    return yaml.safe_load(
        (ROOT / "configs" / "empirical_features_v1.yaml").read_text(
            encoding="utf-8"
        )
    )


def _base() -> pd.DataFrame:
    return pd.read_csv(FEATURE_DIR / "parcel_features_all.csv")


def test_empirical_layer_is_target_free_and_one_row_per_parcel() -> None:
    base = _base()
    table, records, diagnostics = build_empirical_feature_layer(base, _config())

    assert len(table) == 197
    assert table["ID_POLIGONO"].is_unique
    assert "RENDIMIENTO_T_HA" not in table
    assert "CONJUNTO" not in table
    assert diagnostics["features"] == len(table.columns) - 1
    assert diagnostics["features"] >= 200
    assert diagnostics["infinite_values"] == 0
    assert {record["mode"] for record in records} == {"clean", "competition"}
    assert {record["evidence"] for record in records} == {"empirical_x_only"}
    assert all("RENDIMIENTO_T_HA" not in record["inputs"] for record in records)

    validate_empirical_feature_layer(
        table,
        expected_ids=base["ID_POLIGONO"],
    )


def test_empirical_layer_contains_expected_discovery_families() -> None:
    table, records, _ = build_empirical_feature_layer(_base(), _config())
    columns = set(table.columns)

    assert "emp_shape__y2025_s2_ndvi__roughness" in columns
    assert "emp_shape__y2025_s2_ndvi__shape_entropy" in columns
    assert "emp_vci_like_short__s2_ndvi__season_mean" in columns
    assert "emp_symchange__s2_ndwi__season_mean" in columns
    assert (
        "emp_sensor_shape__s2_vs_planet__ndvi__pearson_r"
        in columns
    )
    assert "emp_distribution__hist_s2_ndvi__std_over_range" in columns

    families = {record["family"] for record in records}
    assert {
        "temporal_shape",
        "short_baseline_condition",
        "symmetric_change",
        "sensor_shape_similarity",
        "aggregate_geometry",
    } <= families


def test_short_vci_like_is_explicitly_not_standard_vci() -> None:
    _, records, _ = build_empirical_feature_layer(_base(), _config())
    vci_records = [
        record
        for record in records
        if record["column"].startswith("emp_vci_like_short__")
    ]
    assert vci_records
    for record in vci_records:
        caveats = " ".join(record["caveats"]).casefold()
        assert "not a long-climatology standard vegetation condition index" in caveats
        assert {"Bokusheva2016", "Serban2025"} <= set(record["references"])


def test_clean_empirical_features_do_not_use_2025_sources() -> None:
    _, records, _ = build_empirical_feature_layer(_base(), _config())

    for record in records:
        if record["mode"] != "clean":
            continue
        inputs = " ".join(record["inputs"]).casefold()
        assert "sat_basic_2025" not in inputs
        assert "sat_pro_2025" not in inputs
        assert "clim_official_2025" not in inputs
        assert "wapor_2025" not in inputs
        assert "siap_2025" not in inputs


def test_join_empirical_features_requires_exact_coverage() -> None:
    base = _base()[["ID_POLIGONO", "CONJUNTO"]]
    empirical, _, _ = build_empirical_feature_layer(_base(), _config())

    joined = join_empirical_features(base, empirical)
    assert len(joined) == len(base)
    assert joined["ID_POLIGONO"].is_unique

    with np.testing.assert_raises_regex(ValueError, "coverage mismatch"):
        join_empirical_features(base, empirical.iloc[:-1].copy())


def test_fold_local_expression_miner_selects_only_from_fit_target() -> None:
    rng = np.random.default_rng(42)
    rows = 80
    x1 = rng.normal(size=rows)
    x2 = rng.normal(size=rows)
    x3 = rng.normal(size=rows)
    frame = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})
    target = 2.0 * x1 * x2 + 0.05 * rng.normal(size=rows)

    train = frame.iloc[:60].copy()
    valid = frame.iloc[60:].copy()
    y_train = target[:60]

    miner = FoldLocalExpressionMiner(
        ("x1", "x2", "x3"),
        top_primitives=3,
        top_expressions=5,
    )
    miner.fit(train, y_train)
    transformed = miner.transform(valid)
    report = miner.discovery_report()

    assert transformed.shape == (20, 5)
    assert list(transformed.columns) == list(miner.get_feature_names_out())
    assert len(report) == 5
    assert set(report["left"]) | set(report["right"]) <= {"x1", "x2", "x3"}
    assert np.isfinite(transformed.to_numpy(dtype=float)).all()


def test_fold_local_expression_miner_uses_training_medians_for_missing_values() -> None:
    frame = pd.DataFrame(
        {
            "x1": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0],
            "x2": [2.0, 1.0, 3.0, 5.0, np.nan, 8.0],
            "x3": [5.0, 4.0, 3.0, 2.0, 1.0, 0.0],
        }
    )
    target = np.array([1.0, 1.5, 2.2, 4.0, 4.8, 6.3])
    miner = FoldLocalExpressionMiner(
        ("x1", "x2", "x3"),
        top_primitives=3,
        top_expressions=3,
    )
    miner.fit(frame, target)
    transformed = miner.transform(
        pd.DataFrame({"x1": [np.nan], "x2": [4.0], "x3": [2.0]})
    )

    assert transformed.shape == (1, 3)
    assert np.isfinite(transformed.to_numpy(dtype=float)).all()


def test_builder_writes_manifest_and_report(tmp_path: Path) -> None:
    report = BUILDER["build_empirical_tables"](
        root=ROOT,
        config=_config(),
        output_directory=tmp_path,
    )

    table = pd.read_csv(tmp_path / "parcel_empirical_features.csv")
    manifest = json.loads(
        (tmp_path / "feature_manifest.json").read_text(encoding="utf-8")
    )
    written_report = json.loads(
        (tmp_path / "build_report.json").read_text(encoding="utf-8")
    )

    assert report["rows"] == 197
    assert report["features"] == len(table.columns) - 1
    assert manifest["feature_count"] == report["features"]
    assert len(manifest["features"]) == report["features"]
    assert written_report["target_used_for_materialized_features"] is False
    assert written_report["supervised_discovery_materialized_globally"] is False
    assert written_report["short_vci_like_is_standard_vci"] is False
