"""Synthetic checks for X-only high-dimensional 07B metrics, PCA and small MLP."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from geocebada.evaluation.checkpoint07_pca import (
    dual_pca_fold,
    evaluate_highdim_pca,
    mlp_from_pca,
    retained_components,
    ridge_from_pca,
    summarize_pca_predictions,
)
from geocebada.features.checkpoint07_highdim import (
    build_hls_smap_lag_features,
    build_hls_time_features,
    build_nonlinear_interactions,
    build_smap_time_features,
    irregular_series_features,
    merge_x_only_blocks,
    select_x_only_interaction_seeds,
)


def test_irregular_trajectory_respects_long_gaps_and_repeated_dates() -> None:
    dates = ["2025-04-01", "2025-04-01", "2025-04-11", "2025-07-11"]
    vals = [0.2, 0.4, 0.7, 0.9]
    out = irregular_series_features(dates, vals, max_gap_days=35)
    assert out["n_observations"] == 4
    assert out["n_days"] == 3
    assert out["covered_days"] == 10
    assert out["n_contiguous_intervals"] == 1
    assert np.isclose(out["initial"], 0.3)
    assert np.isclose(out["observed_auc"], 5.0)
    assert out["span_days"] == 101


def test_irregular_trajectory_missing_is_never_fake_zero() -> None:
    out = irregular_series_features(
        ["2025-04-01", "2025-04-02"], [np.nan, np.nan]
    )
    assert out["n_observations"] == 0
    assert "mean" not in out


def test_hls_and_smap_parcel_timelines_no_target_required() -> None:
    ids = ["AGC_001", "AGC_002"]
    hls = pd.DataFrame({
        "ID_POLIGONO": ["AGC_001"] * 4 + ["AGC_002"],
        "date": [
            "2025-04-20", "2025-05-10", "2025-06-01", "2025-07-10",
            "2025-05-01",
        ],
        "sensor": ["L30", "S30", "S30", "L30", "S30"],
        "valid_fraction": [0.9, 0.8, 0.75, 0.7, 0.1],
        "qa_clear_fraction": [0.9, 0.85, 0.8, 0.75, 0.2],
        "n_pixels_polygon": [30, 30, 30, 30, 10],
        "n_pixels_all_bands_valid": [27, 25, 23, 20, 1],
        "ndvi_mean": [0.2, 0.4, 0.6, 0.3, 0.1],
        "red_mean": [0.05, 0.06, 0.08, 0.04, 0.10],
    })
    smap = pd.DataFrame({
        "ID_POLIGONO": ["AGC_001"] * 4,
        "date": ["2025-04-18", "2025-04-19", "2025-05-08", "2025-06-01"],
        "overpass": ["AM", "PM", "AM", "AM"],
        "soil_moisture_m3_m3": [0.14, 0.16, 0.22, np.nan],
        "distance_to_cell_center_km": [2.0] * 4,
    })
    h = build_hls_time_features(hls, ids)
    s = build_smap_time_features(smap, ids)
    lag = build_hls_smap_lag_features(hls, smap, ids)
    assert len(h) == len(s) == len(lag) == 2
    assert "hls_all__ndvi_mean__maximum_growth_velocity" in h
    assert "smap_AM__moisture__mean" in s
    assert lag.loc[0, "lag7__n"] >= 1
    assert pd.isna(s.loc[1, "smap_AM__moisture__mean"])


def test_cross_products_are_bounded_target_free_and_reproducible() -> None:
    n = 35
    frame = pd.DataFrame({
        "ID_POLIGONO": [f"p{i}" for i in range(n)],
        "RENDIMIENTO_T_HA": np.arange(n, dtype=float),
        "CONJUNTO": ["ENTRENAMIENTO"] * n,
        "soilgrids__clay": np.linspace(-1, 2, n),
        "smap_AM__mean": np.linspace(0.01, 0.5, n),
        "hls_all__ndvi": np.linspace(0.1, 0.9, n),
        "clim_prec": np.linspace(20, 120, n),
        "siap_2025__label": np.linspace(3, 5, n),
    })
    seeds = select_x_only_interaction_seeds(frame, max_seeds=4)
    assert "RENDIMIENTO_T_HA" not in seeds
    assert "CONJUNTO" not in seeds
    out, report = build_nonlinear_interactions(
        frame, max_seeds=4, max_products=3
    )
    out2, _ = build_nonlinear_interactions(
        frame.drop(columns=["RENDIMIENTO_T_HA"]), max_seeds=4, max_products=3
    )
    pd.testing.assert_frame_equal(out, out2)
    assert report["n_seeds"] == 4
    assert report["n_products"] <= 3
    assert out.shape[1] <= 11
    with pytest.raises(ValueError, match="Duplicate parcel"):
        merge_x_only_blocks(
            frame[["ID_POLIGONO"]], frame[["ID_POLIGONO", "soilgrids__clay"]]
            .assign(ID_POLIGONO=["p0"] * n)
        )


def test_dual_pca_and_neural_fold_never_see_test_y() -> None:
    rng = np.random.default_rng(207)
    matrix = rng.standard_normal((75, 45))
    train, test = matrix[:50], matrix[50:]
    y = np.sin(train[:, 0]) + 0.3 * train[:, 3]
    pca = dual_pca_fold(train, test)
    assert len(pca.explained_ratio) <= 49
    assert np.isclose(pca.explained_ratio.sum(), 1.0)
    assert retained_components(pca.explained_ratio, 0.95) >= 1
    ridge, n = ridge_from_pca(
        pca, y, components=0.95, alpha=10, pc_products=True
    )
    mlp, count = mlp_from_pca(
        pca, y, components=0.95, hidden_layers=(8,),
        alpha=10, max_iter=70,
    )
    assert n == count
    assert ridge.shape == mlp.shape == (25,)
    assert np.isfinite(ridge).all()
    assert np.isfinite(mlp).all()


def test_pca_benchmark_respects_labeled_pseudo_train_only() -> None:
    rng = np.random.default_rng(42)
    ids = [f"AGC_{i:03d}" for i in range(65)]
    x = rng.normal(size=(65, 30))
    x[1, 4] = np.nan
    features = pd.DataFrame(
        x, columns=[f"f{i}" for i in range(x.shape[1])]
    )
    features.insert(0, "ID_POLIGONO", ids)
    y = 4 + 0.6 * x[:60, 0] - x[:60, 1]
    target = pd.DataFrame({
        "ID_POLIGONO": ids,
        "CONJUNTO": ["ENTRENAMIENTO"] * 60 + ["PREDICCION"] * 5,
        "RENDIMIENTO_T_HA": [*y, *[np.nan] * 5],
    })
    membership = pd.DataFrame({
        "ID_POLIGONO": ids[:60],
        "family": ["target_matched"] * 60,
        "split_id": ["target_matched_01"] * 60,
        "role": ["pseudo_train"] * 42 + ["pseudo_target"] * 18,
    })
    pred, metrics, diagnostics = evaluate_highdim_pca(
        features, target, membership,
        variances=(0.8,), fixed_ranks=(), alphas=(10.0,),
        quadratic=False, neural=True, mlp_hidden=((8,),),
        mlp_alphas=(1.0,), mlp_max_iter=60,
    )
    assert set(pred["ID_POLIGONO"]) == set(ids[42:60])
    assert len(diagnostics) == 1
    assert metrics["model"].str.contains("MLP").any()
    assert metrics["model"].str.contains("linear").any()
    assert len(summarize_pca_predictions(pred)) == 2
    features["RENDIMIENTO_T_HA"] = 123.0
    with pytest.raises(ValueError, match="Target/split metadata"):
        evaluate_highdim_pca(features, target, membership)
