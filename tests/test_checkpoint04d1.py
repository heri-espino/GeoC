"""Tests for Checkpoint 04D.1 local/graph refinement utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd

from geocebada.evaluation.checkpoint04b import build_knn_graph
from geocebada.evaluation.checkpoint04d1 import (
    fit_pls_anchor_with_cross_fitted_residuals,
    leave_one_split_out_method_selection,
    leave_one_split_out_tier_routing,
    predict_residual_graph_correction,
    predict_weighted_local_ridge,
    split_positions_from_membership,
    summarize_nested_predictions,
)


def test_split_positions_from_membership() -> None:
    frame = pd.DataFrame(
        {
            "ID_POLIGONO": ["A", "B", "C", "D", "E"],
            "CONJUNTO": [
                "ENTRENAMIENTO",
                "ENTRENAMIENTO",
                "ENTRENAMIENTO",
                "ENTRENAMIENTO",
                "PREDICCION",
            ],
        }
    )
    membership = pd.DataFrame(
        {
            "split_id": ["s1", "s1", "s1", "s1"],
            "family": ["target_matched"] * 4,
            "repeat": [1] * 4,
            "ID_POLIGONO": ["A", "B", "C", "D"],
            "role": ["pseudo_train", "pseudo_target", "pseudo_train", "pseudo_target"],
        }
    )
    splits = split_positions_from_membership(
        frame,
        membership,
        train_value="ENTRENAMIENTO",
        split_column="CONJUNTO",
    )
    assert len(splits) == 1
    assert splits[0].query_positions == (1, 3)
    assert splits[0].observed_positions == (0, 2)


def test_weighted_local_ridge_returns_finite_predictions() -> None:
    x = np.arange(20, dtype=float).reshape(10, 2)
    y = np.linspace(1.0, 3.0, 10)
    distance = np.abs(np.arange(10)[:, None] - np.arange(10)[None, :]).astype(float)
    prediction = predict_weighted_local_ridge(
        x,
        distance,
        y,
        observed_positions=list(range(8)),
        query_positions=[8, 9],
        k=5,
        alpha=10.0,
        distance_power=1.0,
    )
    assert prediction.shape == (2,)
    assert np.isfinite(prediction).all()


def test_pls_anchor_cross_fitted_residuals_cover_observed_rows() -> None:
    rng = np.random.default_rng(42)
    x = rng.normal(size=(20, 6))
    y = 2.0 + x[:, 0] - 0.5 * x[:, 1]
    observed = list(range(16))
    strata = np.array(["A"] * 8 + ["B"] * 8 + ["C"] * 4)

    predictions, residuals = fit_pls_anchor_with_cross_fitted_residuals(
        x,
        y,
        observed,
        strata=strata,
        n_components=2,
        n_splits=4,
        random_state=42,
    )
    assert predictions.shape == (20,)
    assert np.isfinite(predictions).all()
    assert np.isfinite(residuals[observed]).all()
    assert np.isnan(residuals[16:]).all()


def test_residual_graph_correction_adds_finite_correction() -> None:
    distance = np.abs(np.arange(6)[:, None] - np.arange(6)[None, :]).astype(float)
    adjacency = build_knn_graph(distance, k=2)
    anchor = np.full(6, 2.0)
    residuals = np.array([0.2, 0.1, -0.1, -0.2, np.nan, np.nan])
    prediction = predict_residual_graph_correction(
        anchor,
        residuals,
        adjacency,
        observed_positions=[0, 1, 2, 3],
        query_positions=[4, 5],
        regularization=1.0,
    )
    assert prediction.shape == (2,)
    assert np.isfinite(prediction).all()


def _nested_predictions() -> pd.DataFrame:
    rows = []
    for split_id in ["s1", "s2", "s3"]:
        for parcel_index, tier in enumerate(["low", "mid", "high"]):
            observed = float(parcel_index + 1)
            for method, offset in [("A", 0.05), ("B", 0.30)]:
                rows.append(
                    {
                        "family": "target_matched",
                        "split_id": split_id,
                        "method": method,
                        "ID_POLIGONO": f"{split_id}_{parcel_index}",
                        "observed": observed,
                        "predicted": observed + offset,
                        "x_support_score": 0.2 + 0.3 * parcel_index,
                        "x_support_tier": tier,
                    }
                )
    return pd.DataFrame(rows)


def test_loso_method_selection_scores_only_heldout_rows() -> None:
    selection, predictions = leave_one_split_out_method_selection(
        _nested_predictions(),
        family="target_matched",
    )
    assert len(selection) == 3
    assert set(selection["selected_method"]) == {"A"}
    assert predictions["split_id"].nunique() == 3
    summary = summarize_nested_predictions(predictions, label="test")
    assert summary["n_splits"] == 3
    assert float(summary["rmse_mean"]) < 0.1


def test_loso_tier_routing_returns_one_prediction_per_parcel() -> None:
    selection, predictions = leave_one_split_out_tier_routing(
        _nested_predictions(),
        family="target_matched",
        candidate_methods=["A", "B"],
    )
    assert len(selection) == 9
    assert len(predictions) == 9
    assert set(predictions["selected_method"]) == {"A"}
