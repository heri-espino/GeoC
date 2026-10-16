"""Weighted climate intersections, temporal feature creation, fair outer splits."""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from geocebada.evaluation.area_weighted_climate_comparison import (
    BASE,
    ENSEMBLE,
    FOUR,
    TEMPORAL,
    TOUCHED,
    WEIGHTED,
    make_features,
    matched_summary,
    predict_local,
    split_predictions,
)
from geocebada.evaluation.checkpoint10_pca import PcaData
from geocebada.features.area_weighted_climate import (
    MONTHS,
    VARIABLES,
    area_weighted_climate_raster,
    temporal_features,
)

ID = "ID_POLIGONO"


def test_exact_cell_overlap_weights_differ_from_all_touched_average(tmp_path):
    import rasterio
    from rasterio.transform import from_origin

    path = tmp_path / "PREC_2025_04.tif"
    with rasterio.open(
        path, "w", driver="GTiff", height=1, width=2,
        count=1, dtype="float32", crs="EPSG:4326",
        transform=from_origin(-98, 20, .01, .01),
    ) as ds:
        ds.write(np.array([[[10., 20.]]], dtype="float32"))
    parcel = gpd.GeoDataFrame({
        ID: ["P1"], "geometry": [box(-98, 19.99, -97.985, 20.)],
    }, crs="EPSG:4326")
    result = area_weighted_climate_raster(parcel, path).iloc[0]
    assert result["unweighted_mean"] == 15.
    assert result["weighted_mean"] == pytest.approx(13.33333, abs=.002)
    assert result["n_cells"] == 2
    assert result["coverage"] == pytest.approx(1., abs=1e-5)
    assert 1. < result["effective_cells"] < 2.


def test_nodata_does_not_receive_weight(tmp_path):
    import rasterio
    from rasterio.transform import from_origin

    path = tmp_path / "PREC_2025_04.tif"
    with rasterio.open(
        path, "w", driver="GTiff", height=1, width=2,
        count=1, dtype="float32", crs="EPSG:4326",
        nodata=-9999., transform=from_origin(-98, 20, .01, .01),
    ) as ds:
        ds.write(np.array([[[10., -9999.]]], dtype="float32"))
    parcel = gpd.GeoDataFrame({
        ID: ["P1"], "geometry": [box(-98, 19.99, -97.985, 20.)],
    }, crs="EPSG:4326")
    score = area_weighted_climate_raster(parcel, path).iloc[0]
    assert score["weighted_mean"] == 10.
    assert score["unweighted_mean"] == 10.
    assert score["coverage"] == pytest.approx(2 / 3, abs=1e-4)
    assert score["n_cells"] == 1


def weather_blocks(ids):
    weighted = pd.DataFrame({ID: ids})
    touched = pd.DataFrame({ID: ids})
    row = np.arange(len(ids), dtype=float)
    for var in VARIABLES:
        for year in (2022, 2023, 2024, 2025):
            for month in MONTHS:
                name = f"{var}__{year}_m{month:02d}"
                base = (100. if var == "prec" else 16. if var == "tmin" else 28.)
                values = base + np.sin(row / 5. + month * .7 + year) * 5. + month
                weighted[f"area_climate__{name}"] = values
                touched[f"touched_climate__{name}"] = values + np.cos(row / 3. + month)
    temporal = temporal_features(weighted, prefix="area_climate")
    return weighted, touched, temporal


def fake_features(*, peer: bool = False):
    ids = [f"P{i:03d}" for i in range(197)]
    rng = np.random.default_rng(31)
    train_ids, held_ids = ids[:97], ids[97:138]
    y = pd.Series(3. + rng.normal(0., .5, 138), index=ids[:138])
    base = PcaData(
        ids=ids, index={pid: i for i, pid in enumerate(ids)}, y=y,
        geographic=np.abs(np.arange(197)[:, None] - np.arange(197)[None, :]).astype(float),
        x={20: rng.normal(size=(197, 20))}, variance={20: .8},
        folds={"confirm_01": (train_ids, held_ids)}, parity_max_diff=0.,
    )
    w, t, d = weather_blocks(ids)
    optional = None
    if peer:
        optional = pd.DataFrame({
            ID: ids,
            "rain": rng.normal(size=197), "heat": rng.normal(size=197),
            "soil": rng.normal(size=197), "terrain": rng.normal(size=197),
        })
    return make_features(base, w, t, d, peer_features=optional)


def test_temporal_features_have_no_y_and_preserve_rows():
    ids = [f"P{i:03d}" for i in range(197)]
    w, t, temporal = weather_blocks(ids)
    assert len(temporal) == len(w) == len(t) == 197
    assert any("rainfall_late_fraction" in c for c in temporal)
    assert any("gdd_monthly_proxy_base5" in c for c in temporal)
    assert sum("anomaly_m" in c for c in temporal) == 21
    assert all("YIELD" not in c for c in temporal)


def test_original_incumbent_exactly_preserved_in_new_arms():
    f = fake_features()
    assert set(f.spaces) == {BASE, TOUCHED, WEIGHTED, TEMPORAL}
    assert f.spaces[BASE].shape == (197, 20)
    assert f.spaces[WEIGHTED].shape == (197, 26)
    assert f.spaces[TEMPORAL].shape == (197, 32)
    assert np.array_equal(f.spaces[BASE], f.spaces[WEIGHTED][:, :20])
    assert np.array_equal(f.spaces[BASE], f.spaces[TEMPORAL][:, :20])


def test_outer_heldout_y_does_not_affect_models():
    f = fake_features(peer=True)
    fit, held = f.baseline.folds["confirm_01"]
    before = {
        name: predict_local(f, fit, held[:5], name) for name in f.spaces
    }
    old = f.baseline.y.loc[held].copy()
    f.baseline.y.loc[held] = 10000.
    for name in f.spaces:
        np.testing.assert_allclose(
            before[name], predict_local(f, fit, held[:5], name),
            atol=0, rtol=0,
        )
    f.baseline.y.loc[held] = old


def test_four_covariate_blend_uses_training_labels_only():
    f = fake_features(peer=True)
    before, info = split_predictions(f, "confirm_01")
    assert len(before) == 41 * 6
    assert {FOUR, ENSEMBLE}.issubset(set(before["model"]))
    assert info["oof_blend_peer_weight"] in (0., .25, .5, .75, 1.)
    _, held = f.baseline.folds["confirm_01"]
    f.baseline.y.loc[held] = 10000.
    after, info2 = split_predictions(f, "confirm_01")
    np.testing.assert_allclose(before["predicted"], after["predicted"])
    assert info2["oof_blend_peer_weight"] == info["oof_blend_peer_weight"]


def test_partial_outer_coverage_refused():
    f = fake_features()
    predictions, _ = split_predictions(f, "confirm_01")
    with pytest.raises(ValueError, match="Paired"):
        matched_summary(predictions)


def test_peer_requires_four_x_only_columns():
    ids = [f"P{i:03d}" for i in range(197)]
    w, t, d = weather_blocks(ids)
    f = fake_features()
    bad = pd.DataFrame({
        ID: ids, "YIELD": np.arange(197), "x": np.arange(197),
        "z": np.arange(197), "w": np.arange(197),
    })
    with pytest.raises(ValueError, match="must not encode"):
        make_features(f.baseline, w, t, d, peer_features=bad)
