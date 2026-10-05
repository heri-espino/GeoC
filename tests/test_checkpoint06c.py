"""Tests for Checkpoint 06C calibration."""

from __future__ import annotations

import numpy as np
import pandas as pd

from geocebada.evaluation.checkpoint06c import (
    fit_predict_calibrators,
    summarize_calibration_predictions,
)


def _config() -> dict:
    return {
        "tails": {
            "primary_low_quantile": 0.15,
            "primary_high_quantile": 0.85,
        },
        "calibration": {
            "stretch_values": [1.10],
            "ridge_alphas": [10.0],
            "piecewise_low_prediction_quantile": 0.15,
            "piecewise_high_prediction_quantile": 0.85,
        },
    }


def test_fit_predict_calibrators_preserves_identity_candidate() -> None:
    train = pd.DataFrame(
        {
            "observed": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "Local04D": [1.4, 2.2, 3.0, 4.0, 4.8, 5.6],
            "CatBoost_C1": [1.5, 2.3, 3.0, 4.0, 4.7, 5.5],
        }
    )
    test = pd.DataFrame(
        {
            "Local04D": [2.5, 4.5],
            "CatBoost_C1": [2.6, 4.4],
        }
    )
    result = fit_predict_calibrators(train, test, config=_config())

    assert np.allclose(result["Local04D"], [2.5, 4.5])
    assert set(result) == {
        "Local04D",
        "LocalStretch_s1p1",
        "LocalResidualRidge_a10p0",
        "LocalPiecewiseRidge_a10p0",
        "LocalCatResidualRidge_a10p0",
    }


def test_stretch_expands_predictions_around_training_y_center() -> None:
    train = pd.DataFrame(
        {
            "observed": [1.0, 2.0, 3.0, 4.0, 5.0],
            "Local04D": [1.5, 2.2, 3.0, 3.8, 4.5],
            "CatBoost_C1": [1.6, 2.3, 3.0, 3.7, 4.4],
        }
    )
    test = pd.DataFrame(
        {
            "Local04D": [2.0, 4.0],
            "CatBoost_C1": [2.1, 3.9],
        }
    )
    result = fit_predict_calibrators(train, test, config=_config())
    stretch = result["LocalStretch_s1p1"]

    assert stretch[0] < 2.0
    assert stretch[1] > 4.0


def test_summary_reports_tail_and_center_metrics() -> None:
    rows = []
    for method, pred in {
        "Local04D": [2.0, 3.0, 4.0],
        "Candidate": [1.8, 3.0, 4.2],
    }.items():
        for parcel_id, y, p, regime in zip(
            ["A", "B", "C"],
            [1.5, 3.0, 4.5],
            pred,
            ["low", "center", "high"],
            strict=True,
        ):
            rows.append(
                {
                    "method": method,
                    "split_id": "s1",
                    "ID_POLIGONO": parcel_id,
                    "observed": y,
                    "predicted": p,
                    "tail_regime": regime,
                }
            )
    split, summary = summarize_calibration_predictions(pd.DataFrame(rows))

    assert set(["tail_rmse", "center_rmse", "low_rmse", "high_rmse"]).issubset(split.columns)
    assert set(["tail_pooled_rmse", "center_pooled_rmse"]).issubset(summary.columns)
