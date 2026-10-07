"""Tests for Checkpoint 06F selective center expert."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from geocebada.evaluation.checkpoint06e import summarize_mixture_predictions
from geocebada.evaluation.checkpoint06f import (
    _apply_selective_center_correction,
    _candidate_names,
    _leave_one_split_out_selective_selection,
    _safe_center_mask,
    _summarize_selective_routing,
)


def test_safe_center_mask_requires_both_tail_scores_below_cutoff() -> None:
    mask = _safe_center_mask(
        np.array([0.2, 0.6, 0.2, 0.4]),
        np.array([0.2, 0.2, 0.7, 0.4]),
        cutoff=0.5,
    )
    assert mask.tolist() == [True, False, False, True]


def test_selective_correction_abstains_to_local04d() -> None:
    local = np.array([2.0, 3.0, 4.0])
    delta = np.array([0.2, -0.3, 0.4])
    predicted, routed = _apply_selective_center_correction(
        local,
        delta,
        np.array([0.1, 0.8, 0.2]),
        np.array([0.1, 0.1, 0.9]),
        cutoff=0.5,
    )
    assert routed.tolist() == [True, False, False]
    assert predicted.tolist() == pytest.approx([2.2, 3.0, 4.0])


def test_candidate_names_only_vary_hard_gate_cutoff() -> None:
    config = {"gate": {"tail_score_cutoffs": [0.4, 0.5, 0.6]}}
    assert _candidate_names(config) == [
        "Local04D",
        "SelectiveCenter_t0p4",
        "SelectiveCenter_t0p5",
        "SelectiveCenter_t0p6",
    ]


def test_routing_summary_reports_tail_intrusion() -> None:
    predictions = pd.DataFrame(
        {
            "method": ["Candidate"] * 4,
            "center_expert_used": [True, True, False, False],
            "tail_regime": ["center", "low", "center", "high"],
        }
    )
    summary = _summarize_selective_routing(predictions).iloc[0]
    assert summary["expert_use_rate"] == pytest.approx(0.5)
    assert summary["true_center_use_rate"] == pytest.approx(0.5)
    assert summary["true_tail_use_rate"] == pytest.approx(0.5)
    assert summary["routed_center_precision"] == pytest.approx(0.5)


def _synthetic_loso_predictions() -> pd.DataFrame:
    rows = []
    for split in ["s1", "s2", "s3", "s4"]:
        for method, values, used in [
            ("Local04D", [1.4, 3.3, 4.6], [False, False, False]),
            ("SelectiveCenter_t0p5", [1.4, 3.0, 4.6], [False, True, False]),
        ]:
            for parcel, observed, predicted, regime, route in zip(
                ["L", "C", "H"],
                [1.0, 3.0, 5.0],
                values,
                ["low", "center", "high"],
                used,
                strict=True,
            ):
                rows.append(
                    {
                        "family": "target_matched",
                        "split_id": split,
                        "ID_POLIGONO": parcel,
                        "method": method,
                        "observed": observed,
                        "predicted": predicted,
                        "tail_regime": regime,
                        "center_expert_used": route,
                    }
                )
    return pd.DataFrame(rows)


def test_loso_selects_center_expert_when_it_improves_center_without_tail_change() -> None:
    predictions = _synthetic_loso_predictions()
    split_metrics, _ = summarize_mixture_predictions(predictions)
    selections, selected_predictions, payload = (
        _leave_one_split_out_selective_selection(
            predictions,
            split_metrics,
            min_center_improvement=0.002,
            tail_tolerance_absolute=0.003,
            tail_tolerance_fraction=0.005,
        )
    )

    assert set(selections["selected_method"]) == {"SelectiveCenter_t0p5"}
    assert len(selected_predictions) == 12
    assert payload["rmse_mean_improvement"] > 0
    assert payload["pooled_rmse_improvement"] > 0
    assert payload["center_rmse_improvement"] > 0
    assert payload["tail_rmse_change"] == pytest.approx(0.0)
