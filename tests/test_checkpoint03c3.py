"""Tests for Checkpoint 03C.3 finalist equal-weight ensembles."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from geocebada.evaluation.checkpoint03c3 import (
    build_finalist_ensemble_predictions,
    summarize_finalist_ensemble_results,
)


def _config() -> dict:
    return {
        "identity": {
            "id_column": "ID_POLIGONO",
            "observed_column": "observed",
            "predicted_column": "predicted",
            "protocol_column": "protocol",
            "fold_column": "fold",
            "representation_column": "representation",
            "model_column": "model",
        },
        "validation": {
            "state_protocol": "fold_state_stratified",
            "grouped_protocol": "fold_municipality_grouped",
            "expected_training_rows": 4,
        },
        "base_candidates": {
            "F1": {"representation": "C0", "model": "PLS"},
            "F2": {"representation": "C3", "model": "Ridge"},
            "F3": {"representation": "C1", "model": "CatBoost"},
        },
        "ensembles": {
            "E13": {"members": ["F1", "F3"], "weights": [0.5, 0.5]},
            "E123": {
                "members": ["F1", "F2", "F3"],
                "weights": [1 / 3, 1 / 3, 1 / 3],
            },
        },
    }


def _oof() -> pd.DataFrame:
    rows: list[dict] = []
    protocols = ["fold_state_stratified", "fold_municipality_grouped"]
    specs = [
        ("C0", "PLS", [1.0, 2.0, 3.0, 4.0]),
        ("C3", "Ridge", [1.2, 1.8, 3.2, 3.8]),
        ("C1", "CatBoost", [0.8, 2.2, 2.8, 4.2]),
    ]
    observed = [1.0, 2.0, 3.0, 4.0]
    for protocol in protocols:
        for representation, model, predicted in specs:
            for index, value in enumerate(predicted):
                rows.append(
                    {
                        "protocol": protocol,
                        "representation": representation,
                        "model": model,
                        "fold": 1 if index < 2 else 2,
                        "ID_POLIGONO": f"P{index}",
                        "observed": observed[index],
                        "predicted": value,
                        "residual": observed[index] - value,
                    }
                )
    return pd.DataFrame(rows)


def test_equal_weight_ensemble_predictions_are_exact() -> None:
    predictions = build_finalist_ensemble_predictions(_oof(), _config())
    row = predictions.loc[
        predictions["protocol"].eq("fold_state_stratified")
        & predictions["candidate"].eq("E13")
        & predictions["ID_POLIGONO"].eq("P1")
    ].iloc[0]
    assert np.isclose(row["predicted"], 2.1)
    assert np.isclose(row["residual"], -0.1)
    assert len(predictions) == 2 * 5 * 4


def test_summary_has_both_protocols_and_all_candidates() -> None:
    predictions = build_finalist_ensemble_predictions(_oof(), _config())
    fold_metrics, protocol_summary, robustness, residual_correlation = (
        summarize_finalist_ensemble_results(predictions, _config())
    )

    assert len(fold_metrics) == 2 * 5 * 2
    assert len(protocol_summary) == 2 * 5
    assert len(robustness) == 5
    assert set(robustness["candidate"]) == {"F1", "F2", "F3", "E13", "E123"}
    assert len(residual_correlation) == 2 * 3
    assert np.isfinite(protocol_summary["oof_rmse"]).all()


def test_missing_finalist_oof_rows_fail() -> None:
    oof = _oof()
    broken = oof.loc[
        ~(
            oof["protocol"].eq("fold_state_stratified")
            & oof["representation"].eq("C1")
            & oof["model"].eq("CatBoost")
            & oof["ID_POLIGONO"].eq("P0")
        )
    ]
    with pytest.raises(ValueError, match="lacks OOF coverage"):
        build_finalist_ensemble_predictions(broken, _config())
