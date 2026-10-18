"""Synthetic QA and label-isolation tests for phenology-conditioned spatial GP."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from geocebada.evaluation.checkpoint10_pca import PcaData
from geocebada.evaluation.phenology_spatial_gp import (
    BASELINE,
    CANDIDATE,
    SpatialGPConfig,
    SpatialGPData,
    candidate_grid,
    evaluate_split,
    predict_spatial_gp,
    summarize_matched,
)
from geocebada.features.phenology_spatial import build_phenology_weather_features

ID = "ID_POLIGONO"


def test_physical_features_are_label_free_and_stage_conditional():
    ids = ["p0", "p1", "p2"]
    weather = {ID: ids}
    for var in ("prec", "tmin", "tmax"):
        for month in range(4, 11):
            constant = {"prec": float(month), "tmin": 5.0, "tmax": 15.0}[var]
            weather[f"area_climate__{var}__2025_m{month:02d}"] = [constant] * 3
    hls = pd.DataFrame({
        ID: ["p0", "p0", "p0", "p1", "p1", "p1"],
        "date": ["2025-04-20", "2025-06-20", "2025-08-20"] * 2,
        "sensor": ["S30"] * 6,
        "ndvi_mean": [0.1, 0.25, 0.7, 0.7, 0.25, 0.1],
        "valid_fraction": [0.9] * 6,
    })
    smap = pd.DataFrame({
        ID: ["p0", "p0", "p1", "p1"],
        "date": ["2025-08-01", "2025-08-18"] * 2,
        "soil_moisture_m3_m3": [0.2, 0.3, 0.2, 0.3],
        "recommended_quality": [1] * 4,
    })
    features = build_phenology_weather_features(pd.DataFrame(weather), hls, smap, ids)
    assert features[ID].tolist() == ids
    assert features.loc[0, "peak_month"] == 8
    assert features.loc[1, "peak_month"] == 4
    assert features.loc[0, "prec_before_peak"] > features.loc[1, "prec_before_peak"]
    assert np.isnan(features.loc[2, "peak_month"])
    assert np.isnan(features.loc[2, "prec_before_peak"])
    assert "RENDIMIENTO_T_HA" not in features


@pytest.fixture(scope="module")
def artificial() -> SpatialGPData:
    rng = np.random.default_rng(20261018)
    ids = [f"p{i:03d}" for i in range(197)]
    pc = rng.normal(size=(197, 20))
    locations = np.linspace(0, 25, 197)
    geo = np.abs(locations[:, None] - locations[None, :])
    physical = pd.DataFrame({ID: ids})
    for j in range(20):
        physical[f"physical_{j:02d}"] = rng.normal(size=197)
    physical.loc[0, "physical_00"] = np.nan
    y = pd.Series(
        3 + 0.25 * np.sin(locations[:138] / 2)
        + 0.3 * pc[:138, 0] + rng.normal(0, 0.08, size=138),
        index=ids[:138],
    )
    baseline = PcaData(
        ids=ids, index={pid: j for j, pid in enumerate(ids)},
        y=y, geographic=geo, x={20: pc}, variance={20: 0.5},
        folds={"smoke": (ids[:97], ids[97:138])}, parity_max_diff=0.0,
    )
    data = SpatialGPData(baseline, physical, geo)
    data.validate()
    return data


def test_nested_candidate_contains_spatial_and_ridge_only_arms(artificial):
    grid = candidate_grid(smoke=False)
    assert any(x.spatial_share == 0 for x in grid)
    assert any(x.spatial_share > 0 for x in grid)
    assert any(x.robust_nugget for x in grid)
    assert all(x.length_km > 0 and x.alpha > 0 for x in grid)


def test_gp_predicts_queries_only_from_training_labels(artificial):
    ids = artificial.baseline.ids
    train, held = ids[:97], ids[97:100]
    config = SpatialGPConfig(alpha=10, length_km=3, spatial_share=0.75,
                             robust_nugget=True)
    before = predict_spatial_gp(artificial, train, held, config)
    original = artificial.baseline.y.loc[held].copy()
    artificial.baseline.y.loc[held] = 1e6
    try:
        after = predict_spatial_gp(artificial, train, held, config)
    finally:
        artificial.baseline.y.loc[held] = original
    np.testing.assert_array_equal(before, after)
    assert np.isfinite(after).all()
    with pytest.raises(ValueError, match="disjoint"):
        predict_spatial_gp(artificial, train, train[:1], config)


def test_outer_y_isolation_and_matched_frozen_baseline(artificial):
    before, _, meta = evaluate_split(artificial, "smoke", smoke=True)
    held = artificial.baseline.folds["smoke"][1]
    old = artificial.baseline.y.loc[held].copy()
    artificial.baseline.y.loc[held] = old + 1000
    try:
        after, _, meta2 = evaluate_split(artificial, "smoke", smoke=True)
    finally:
        artificial.baseline.y.loc[held] = old
    np.testing.assert_allclose(before["predicted"], after["predicted"], atol=0, rtol=0)
    assert set(before["model"]) == {BASELINE, CANDIDATE}
    assert meta["selected"] == meta2["selected"]
    assert not np.array_equal(before["observed"], after["observed"])
    summary, pairs = summarize_matched(before)
    assert len(summary) == 2
    assert len(pairs) == 41


def test_rejects_misaligned_physical_rows(artificial):
    invalid = SpatialGPData(
        artificial.baseline, artificial.physical.sample(frac=1, random_state=5),
        artificial.spatial_km,
    )
    with pytest.raises(ValueError, match="aligned"):
        invalid.validate()
