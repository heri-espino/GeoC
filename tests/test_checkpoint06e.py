"""Tests for Checkpoint 06E supervised center/tail mixture."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from geocebada.evaluation.checkpoint06e import (
    normalize_gate_probabilities,
    summarize_mixture_predictions,
)


def test_gate_weights_are_normalized_and_nonnegative() -> None:
    low, center, high = normalize_gate_probabilities(
        np.array([0.8, 0.1, 0.4]),
        np.array([0.1, 0.7, 0.4]),
    )

    assert np.all(low >= 0)
    assert np.all(center >= 0)
    assert np.all(high >= 0)
    assert np.allclose(low + center + high, 1.0)


def test_gate_weights_emphasize_expected_regime() -> None:
    low, center, high = normalize_gate_probabilities(
        np.array([0.9, 0.05, 0.05]),
        np.array([0.05, 0.05, 0.9]),
    )

    assert low[0] > center[0] and low[0] > high[0]
    assert center[1] > low[1] and center[1] > high[1]
    assert high[2] > center[2] and high[2] > low[2]


def test_summary_separates_center_and_tail_rmse() -> None:
    rows = []
    for method, pred in {
        "Local04D": [1.8, 3.0, 4.2],
        "Candidate": [1.6, 3.05, 4.4],
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
    split, summary = summarize_mixture_predictions(pd.DataFrame(rows))

    assert {"center_rmse", "tail_rmse"}.issubset(split.columns)
    candidate = summary.loc[summary["method"].eq("Candidate")].iloc[0]
    assert candidate["center_pooled_rmse"] == pytest.approx(0.05)
