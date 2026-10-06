"""Tests for Checkpoint 06D X-only regime discovery."""

from __future__ import annotations

import numpy as np
import pandas as pd

from geocebada.evaluation.checkpoint06d import (
    _eta_squared,
    decide_regime_hypothesis,
    select_x_only_regime,
)


def test_x_only_selection_ignores_y_columns() -> None:
    candidates = pd.DataFrame(
        {
            "pca_variance": [0.8, 0.9],
            "algorithm": ["kmeans", "gmm"],
            "k": [2, 3],
            "n_components": [4, 6],
            "silhouette": [0.30, 0.40],
            "min_cluster_size": [60, 25],
            "min_cluster_fraction": [0.30, 0.12],
            "seed_ari_mean": [0.95, 0.85],
            "seed_ari_min": [0.90, 0.80],
            "subsample_ari_mean": [0.90, 0.75],
            "subsample_ari_min": [0.80, 0.65],
        }
    )
    selected = select_x_only_regime(
        candidates,
        min_cluster_fraction=0.10,
        min_seed_ari=0.80,
        min_subsample_ari=0.70,
    )

    assert selected["algorithm"] == "gmm"
    assert selected["k"] == 3
    assert selected["x_eligible"] is True


def test_eta_squared_detects_cluster_mean_structure() -> None:
    values = np.array([-1.0, -0.8, -0.9, 0.8, 1.0, 0.9])
    labels = np.array([0, 0, 0, 1, 1, 1])

    assert _eta_squared(values, labels) > 0.9


def test_regime_gate_supported_only_with_x_and_residual_structure() -> None:
    selection = {
        "x_eligible": True,
        "algorithm": "kmeans",
        "k": 2,
        "pca_variance": 0.8,
    }
    payload = {
        "effects": {
            "Local04D": {
                "residual_eta2": 0.12,
                "residual_permutation_p": 0.03,
                "residual_mean_range": 0.22,
            }
        },
        "expert_advantage_range": 0.1,
        "geography": {},
    }
    decision = decide_regime_hypothesis(
        selection,
        payload,
        min_residual_eta2=0.05,
        max_residual_permutation_p=0.10,
        min_residual_mean_range=0.10,
    )

    assert decision["REGIME_HYPOTHESIS"] == "SUPPORTED"
