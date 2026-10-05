"""Tests for Checkpoint 06 tail-aware diagnostics."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from geocebada.evaluation.checkpoint06 import (
    aggregate_residuals_by_parcel,
    assign_fold_valid_tails,
    build_residual_diagnostics,
    decide_tail_hypothesis,
    summarize_sse_concentration,
)


def test_checkpoint06_config_is_diagnostic_first_and_competition_only() -> None:
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(
        (root / "configs" / "checkpoint06.yaml").read_text(encoding="utf-8")
    )

    assert config["scope"]["active_track"] == "competition"
    assert config["scope"]["hidden_fira_y_allowed"] is False
    assert config["scope"]["clustering_enabled"] is False
    assert config["scope"]["specialist_models_enabled"] is False
    assert config["scope"]["implementation_scope"] == ["06A", "06B"]


def _synthetic_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = pd.DataFrame(
        {
            "ID_POLIGONO": ["A", "B", "C", "D", "E", "F"],
            "CONJUNTO": ["ENTRENAMIENTO"] * 6,
            "RENDIMIENTO_T_HA": [1.0, 2.0, 3.0, 4.0, 5.0, 100.0],
        }
    )
    membership = pd.DataFrame(
        {
            "family": ["target_matched"] * 6,
            "split_id": ["s1"] * 6,
            "ID_POLIGONO": ["A", "B", "C", "D", "E", "F"],
            "role": [
                "pseudo_train",
                "pseudo_train",
                "pseudo_train",
                "pseudo_train",
                "pseudo_target",
                "pseudo_target",
            ],
        }
    )
    predictions = pd.DataFrame(
        {
            "family": ["target_matched", "target_matched"],
            "split_id": ["s1", "s1"],
            "ID_POLIGONO": ["E", "F"],
            "observed": [5.0, 100.0],
            "Local04D": [4.0, 6.0],
            "CatBoost_C1": [4.2, 5.5],
            "Graph04D": [4.1, 5.8],
        }
    )
    return base, membership, predictions


def test_fold_valid_tail_thresholds_ignore_pseudo_target_y() -> None:
    base, membership, predictions = _synthetic_inputs()
    enriched, thresholds = assign_fold_valid_tails(
        predictions,
        base,
        membership,
        target_column="RENDIMIENTO_T_HA",
        split_column="CONJUNTO",
        train_value="ENTRENAMIENTO",
        low_quantile=0.25,
        high_quantile=0.75,
    )

    assert thresholds.loc[0, "high_threshold"] < 4.0
    assert thresholds.loc[0, "high_threshold"] != pytest.approx(100.0)
    assert set(enriched["tail_regime"]) == {"high"}


def test_residual_builder_uses_observed_minus_predicted() -> None:
    base, membership, predictions = _synthetic_inputs()
    enriched, _ = assign_fold_valid_tails(
        predictions,
        base,
        membership,
        target_column="RENDIMIENTO_T_HA",
        split_column="CONJUNTO",
        train_value="ENTRENAMIENTO",
        low_quantile=0.25,
        high_quantile=0.75,
    )
    residuals = build_residual_diagnostics(
        enriched,
        methods=["Local04D", "CatBoost_C1", "Graph04D"],
    )
    row = residuals.loc[
        residuals["method"].eq("Local04D")
        & residuals["ID_POLIGONO"].eq("F")
    ].iloc[0]

    assert row["residual"] == pytest.approx(94.0)
    assert row["squared_error"] == pytest.approx(94.0**2)
    assert "LocalCatBoostMean" in set(residuals["method"])


def test_sse_concentration_equal_weights_parcels_not_appearances() -> None:
    residuals = pd.DataFrame(
        {
            "method": ["M"] * 4,
            "ID_POLIGONO": ["A", "A", "B", "C"],
            "split_id": ["s1", "s2", "s1", "s1"],
            "observed": [0.0] * 4,
            "predicted": [1.0, 1.0, 2.0, 3.0],
            "residual": [-1.0, -1.0, -2.0, -3.0],
            "absolute_error": [1.0, 1.0, 2.0, 3.0],
            "squared_error": [1.0, 1.0, 4.0, 9.0],
        }
    )
    parcel = aggregate_residuals_by_parcel(residuals)
    concentration = summarize_sse_concentration(
        parcel,
        top_counts=[1],
        top_fractions=[1 / 3],
    )

    row = concentration.loc[
        concentration["selection"].eq("top_1_parcels")
    ].iloc[0]
    assert row["sse_share"] == pytest.approx(9.0 / 14.0)


def test_decision_requires_concentration_shrinkage_and_x_signal() -> None:
    concentration = pd.DataFrame(
        {
            "method": ["Local04D"],
            "selection": ["top_10pct"],
            "sse_share": [0.50],
        }
    )
    tail_bias = pd.DataFrame(
        {
            "method": [
                "Local04D",
                "Local04D",
                "CatBoost_C1",
                "CatBoost_C1",
            ],
            "tail_regime": ["low", "high", "low", "high"],
            "mean_residual": [-0.30, 0.35, -0.25, 0.30],
            "expected_shrinkage_sign_fraction": [0.8, 0.8, 0.7, 0.7],
        }
    )
    classifier = pd.DataFrame(
        {
            "family": ["phenology"],
            "task": ["tail_vs_center"],
            "mean_split_roc_auc": [0.72],
            "mean_split_pr_lift": [1.5],
        }
    )
    risk = pd.DataFrame(
        {
            "method": ["Local04D"],
            "family": ["phenology"],
            "risk_spearman": [0.1],
        }
    )

    decision = decide_tail_hypothesis(
        concentration,
        tail_bias,
        classifier,
        risk,
        thresholds={
            "min_top10pct_sse_share": 0.30,
            "min_abs_tail_bias": 0.10,
            "min_expected_sign_fraction": 0.60,
            "min_tail_auc": 0.65,
            "min_pr_lift": 1.20,
            "min_error_risk_spearman": 0.20,
        },
    )

    assert decision["TAIL_HYPOTHESIS"] == "SUPPORTED"
    assert decision["catboost_shrinkage_corroboration"] is True
