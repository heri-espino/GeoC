"""Tests for Checkpoint 04A transductive diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from geocebada.evaluation.checkpoint04a import (
    EmbeddingResult,
    build_checkpoint03_ensemble_residuals,
    build_monthly_temporal_arrays,
    build_transductive_pca_embedding,
    feature_neighbor_tables,
    haversine_distance_matrix,
    knn_weight_matrix,
    morans_i,
    temporal_neighbor_tables,
    temporal_pair_metrics,
)


def _synthetic_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ID_POLIGONO": ["P1", "P2", "P3", "P4", "P5"],
            "CONJUNTO": [
                "ENTRENAMIENTO",
                "ENTRENAMIENTO",
                "ENTRENAMIENTO",
                "PREDICCION",
                "PREDICCION",
            ],
            "RENDIMIENTO_T_HA": [2.0, 3.0, 5.0, np.nan, np.nan],
            "meta_estado": ["A", "A", "B", "A", "B"],
            "meta_municipio": ["m1", "m1", "m2", "m1", "m2"],
            "x1": [0.0, 1.0, 4.0, 0.2, 3.8],
            "x2": [0.0, 1.0, 4.0, 0.1, 3.9],
        }
    )


def test_haversine_one_degree_latitude_is_about_111_km() -> None:
    distances = haversine_distance_matrix([0.0, 1.0], [0.0, 0.0])
    assert distances.shape == (2, 2)
    assert distances[0, 0] == 0.0
    assert 110.0 < distances[0, 1] < 112.0


def test_transductive_embedding_and_neighbors_keep_targets_unlabeled() -> None:
    frame = _synthetic_frame()
    embedding = build_transductive_pca_embedding(
        frame,
        name="synthetic",
        features=["x1", "x2"],
        n_components=2,
    )
    assert isinstance(embedding, EmbeddingResult)
    assert len(embedding.frame) == 5
    assert embedding.n_components == 2

    target, train, matrix = feature_neighbor_tables(frame, embedding, k=2)
    assert matrix.shape == (5, 5)
    assert set(target["query_id"]) == {"P4", "P5"}
    assert set(train["query_id"]) == {"P1", "P2", "P3"}
    assert not target["neighbor_y"].isna().any()


def test_temporal_similarity_finds_identical_curve_as_best_neighbor() -> None:
    frame = pd.DataFrame(
        {
            "ID_POLIGONO": ["P1", "P2", "P3", "T1"],
            "CONJUNTO": [
                "ENTRENAMIENTO",
                "ENTRENAMIENTO",
                "ENTRENAMIENTO",
                "PREDICCION",
            ],
        }
    )
    for month, values in zip(
        [4, 5, 6, 7, 8, 9, 10],
        [
            [0.1, 0.8, 0.2, 0.1],
            [0.2, 0.7, 0.3, 0.2],
            [0.4, 0.6, 0.4, 0.4],
            [0.8, 0.5, 0.5, 0.8],
            [0.6, 0.4, 0.6, 0.6],
            [0.3, 0.3, 0.7, 0.3],
            [0.1, 0.2, 0.8, 0.1],
        ],
    ):
        frame[f"series__m{month:02d}"] = values

    arrays = build_monthly_temporal_arrays(
        frame,
        series=[{"name": "s", "prefix": "series__m"}],
        months=[4, 5, 6, 7, 8, 9, 10],
    )
    pairs = temporal_pair_metrics(
        frame,
        series_arrays=arrays,
        max_lag=1,
        min_points=4,
    )
    target, _ = temporal_neighbor_tables(pairs, k=2)
    best = target.loc[target["rank"].eq(1)].iloc[0]
    assert best["query_id"] == "T1"
    assert best["neighbor_id"] == "P1"
    assert best["best_lag_corr_mean"] > 0.99


def test_morans_i_returns_finite_statistic_and_permutation_p() -> None:
    distances = np.array(
        [
            [0.0, 1.0, 4.0, 5.0],
            [1.0, 0.0, 3.0, 4.0],
            [4.0, 3.0, 0.0, 1.0],
            [5.0, 4.0, 1.0, 0.0],
        ]
    )
    weights = knn_weight_matrix(distances, k=1)
    summary, centered, lag = morans_i(
        [1.0, 1.2, 5.0, 5.2],
        weights,
        permutations=99,
        random_state=42,
    )
    assert np.isfinite(summary["morans_i"])
    assert 0.0 < summary["permutation_p_value_two_sided"] <= 1.0
    assert centered.shape == (4,)
    assert lag.shape == (4,)


def test_checkpoint03_ensemble_residuals_reconstruct_expected_means() -> None:
    rows = []
    candidates = [
        ("C0_base", "PLS", [2.0, 4.0]),
        ("C3_base_agro_plus_discovery", "Ridge", [3.0, 5.0]),
        ("C1_agronomic", "CatBoost", [4.0, 6.0]),
    ]
    for representation, model, predictions in candidates:
        for parcel, observed, predicted in zip(
            ["P1", "P2"],
            [3.0, 5.0],
            predictions,
        ):
            rows.append(
                {
                    "protocol": "fold_municipality_grouped",
                    "representation": representation,
                    "model": model,
                    "ID_POLIGONO": parcel,
                    "observed": observed,
                    "predicted": predicted,
                }
            )
    result = build_checkpoint03_ensemble_residuals(
        pd.DataFrame(rows),
        protocol="fold_municipality_grouped",
    ).set_index("ID_POLIGONO")

    assert result.loc["P1", "E13_predicted"] == 3.0
    assert result.loc["P1", "E123_predicted"] == 3.0
    assert result.loc["P2", "E13_predicted"] == 5.0
    assert result.loc["P2", "E123_predicted"] == 5.0
