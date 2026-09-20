"""Tests for Checkpoint 04B transductive pseudo-competition utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd

from geocebada.evaluation.checkpoint04b import (
    build_knn_graph,
    build_static_x_profile,
    generate_target_matched_splits,
    hamilton_apportion,
    mixed_distance_matrix,
    predict_graph_laplacian,
    predict_knn,
    split_x_support,
    standardized_profile_coordinates,
    temporal_distance_matrix,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ID_POLIGONO": [f"P{i}" for i in range(1, 13)],
            "CONJUNTO": [
                "ENTRENAMIENTO",
                "ENTRENAMIENTO",
                "ENTRENAMIENTO",
                "ENTRENAMIENTO",
                "ENTRENAMIENTO",
                "ENTRENAMIENTO",
                "ENTRENAMIENTO",
                "ENTRENAMIENTO",
                "PREDICCION",
                "PREDICCION",
                "PREDICCION",
                "PREDICCION",
            ],
            "RENDIMIENTO_T_HA": [
                2.0,
                2.2,
                2.4,
                3.0,
                4.0,
                4.2,
                4.4,
                5.0,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ],
            "meta_estado": [
                "A",
                "A",
                "A",
                "A",
                "B",
                "B",
                "B",
                "B",
                "A",
                "A",
                "B",
                "B",
            ],
            "meta_municipio": [
                "a1",
                "a1",
                "a2",
                "a2",
                "b1",
                "b1",
                "b2",
                "b2",
                "a1",
                "a2",
                "b1",
                "b2",
            ],
        }
    )


def _distance(n: int) -> np.ndarray:
    x = np.arange(n, dtype=float)[:, None]
    return np.abs(x - x.T)


def _temporal_similarity(n: int) -> np.ndarray:
    distance = _distance(n)
    return np.exp(-distance / 3.0)


def _adversarial(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ID_POLIGONO": frame["ID_POLIGONO"],
            "adversarial_full_target_probability": np.linspace(0.2, 0.8, len(frame)),
        }
    )


def test_hamilton_apportion_preserves_total() -> None:
    result = hamilton_apportion({"A": 29, "B": 15, "C": 15}, 41)
    assert sum(result.values()) == 41
    assert all(value >= 0 for value in result.values())


def test_target_matched_split_generator_uses_only_x_profile_inputs() -> None:
    frame = _frame()
    distance = _distance(len(frame))
    profile = build_static_x_profile(
        frame,
        geographic_distances=distance,
        agronomic_distances=distance * 2.0,
        temporal_similarities=_temporal_similarity(len(frame)),
        adversarial_scores=_adversarial(frame),
    )
    coordinates = standardized_profile_coordinates(
        profile,
        columns=[
            "nearest_geo_km",
            "nearest_agro_distance",
            "best_temporal_similarity",
            "adversarial_target_probability",
        ],
    )
    splits = generate_target_matched_splits(
        frame,
        profile_coordinates=coordinates,
        n_pseudo_targets=4,
        repeats=3,
        candidates_per_repeat=10,
        temperature=1.0,
        municipality_smoothing=0.05,
        diversity_penalty=0.05,
        random_state=42,
    )
    assert len(splits) == 3
    assert all(len(split.pseudo_target_positions) == 4 for split in splits)
    train_positions = set(range(8))
    assert all(set(split.pseudo_target_positions) <= train_positions for split in splits)


def test_split_x_support_is_finite_without_target_y() -> None:
    frame = _frame()
    distance = _distance(len(frame))
    support = split_x_support(
        frame,
        query_positions=[1, 6],
        observed_positions=[0, 2, 3, 4, 5, 7],
        geographic_distances=distance,
        agronomic_distances=distance * 1.5,
        temporal_similarities=_temporal_similarity(len(frame)),
        adversarial_scores=_adversarial(frame),
    )
    assert len(support) == 2
    assert support["x_support_score"].between(0.0, 1.0).all()
    assert set(support["x_support_tier"]) <= {"low", "mid", "high"}


def test_knn_prediction_uses_nearest_visible_labels() -> None:
    distance = _distance(5)
    y = np.array([1.0, 2.0, 3.0, np.nan, np.nan])
    prediction = predict_knn(
        distance,
        y,
        observed_positions=[0, 1, 2],
        query_positions=[3],
        k=1,
    )
    assert prediction.shape == (1,)
    assert prediction[0] == 3.0


def test_mixed_distance_and_temporal_distance_are_well_formed() -> None:
    similarity = _temporal_similarity(5)
    temporal = temporal_distance_matrix(similarity)
    mixed = mixed_distance_matrix(
        [
            (_distance(5), 0.75),
            (temporal, 0.25),
        ]
    )
    assert mixed.shape == (5, 5)
    assert np.allclose(np.diag(mixed), 0.0)
    assert np.isfinite(mixed).all()


def test_graph_regression_returns_finite_predictions() -> None:
    distance = _distance(6)
    adjacency = build_knn_graph(distance, k=2)
    y = np.array([1.0, 1.2, 2.0, 4.0, np.nan, np.nan])
    prediction = predict_graph_laplacian(
        adjacency,
        y,
        observed_positions=[0, 1, 2, 3],
        query_positions=[4, 5],
        regularization=1.0,
    )
    assert prediction.shape == (2,)
    assert np.isfinite(prediction).all()
