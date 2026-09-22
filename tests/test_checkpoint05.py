from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from geocebada.evaluation.checkpoint05 import (
    ConvexStackRegressor,
    SupportMixtureOfExperts,
    candidate_prediction_frame,
    support_descriptors,
)


def test_convex_stack_weights_are_valid_and_predict() -> None:
    x = np.asarray(
        [
            [1.0, 1.4, 0.8],
            [2.0, 2.3, 1.7],
            [3.0, 3.4, 2.7],
            [4.0, 4.2, 3.8],
        ]
    )
    y = np.asarray([1.0, 2.0, 3.0, 4.0])
    model = ConvexStackRegressor(l2=0.01).fit(x, y)

    assert np.all(model.coef_ >= 0.0)
    assert np.sum(model.coef_) == pytest.approx(1.0)
    predicted = model.predict(x)
    assert predicted.shape == y.shape
    assert np.isfinite(predicted).all()


def test_support_mixture_produces_sample_specific_convex_weights() -> None:
    expert = np.asarray(
        [
            [1.0, 2.0],
            [1.1, 2.1],
            [3.0, 2.0],
            [3.1, 2.1],
            [5.0, 4.0],
            [5.1, 4.1],
        ]
    )
    support = np.asarray(
        [
            [0.0],
            [0.1],
            [0.4],
            [0.5],
            [0.9],
            [1.0],
        ]
    )
    y = np.asarray([1.0, 1.1, 2.0, 2.1, 4.0, 4.1])

    model = SupportMixtureOfExperts(l2=0.01, max_iter=1000)
    model.fit(expert, support, y)
    weights = model.expert_weights(support)
    predicted = model.predict(expert, support)

    assert weights.shape == expert.shape
    assert np.all(weights >= 0.0)
    assert np.allclose(weights.sum(axis=1), 1.0)
    assert predicted.shape == y.shape
    assert np.isfinite(predicted).all()


def test_support_descriptors_use_only_current_visible_reference_set() -> None:
    frame = pd.DataFrame(
        {
            "ID_POLIGONO": ["A", "B", "C", "D"],
            "meta_estado": ["X", "X", "Y", "X"],
            "meta_municipio": ["M1", "M1", "M2", "M3"],
        }
    )
    geo = np.asarray(
        [
            [0.0, 1.0, 5.0, 2.0],
            [1.0, 0.0, 4.0, 1.5],
            [5.0, 4.0, 0.0, 3.0],
            [2.0, 1.5, 3.0, 0.0],
        ]
    )
    agro = geo * 2.0

    result = support_descriptors(
        frame,
        observed_positions=[0, 1, 2],
        query_positions=[3],
        geographic_distances=geo,
        agronomic_distances=agro,
        local_predictions=np.asarray([2.0]),
        graph_predictions=np.asarray([2.4]),
    )

    assert len(result) == 1
    assert result.loc[0, "local_graph_abs_disagreement"] == pytest.approx(0.4)
    assert 0.0 <= result.loc[0, "geo_support"] <= 1.0
    assert 0.0 <= result.loc[0, "agro_support"] <= 1.0
    assert result.loc[0, "same_state_fraction"] == pytest.approx(2 / 3)
    assert result.loc[0, "same_municipality_fraction"] == pytest.approx(0.0)


def test_candidate_prediction_frame_is_long_and_aligned() -> None:
    result = candidate_prediction_frame(
        ["A", "B"],
        np.asarray([1.0, 2.0]),
        {
            "M1": np.asarray([1.1, 1.9]),
            "M2": np.asarray([0.9, 2.1]),
        },
        family="target_matched",
        split_id="split_01",
    )

    assert len(result) == 4
    assert set(result["method"]) == {"M1", "M2"}
    assert set(result["ID_POLIGONO"]) == {"A", "B"}
