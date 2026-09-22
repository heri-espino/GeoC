from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from geocebada.evaluation.checkpoint05 import (
    ConvexStackRegressor,
    SupportMixtureOfExperts,
    candidate_prediction_frame,
    constrain_component_param_grid,
    support_descriptors,
)
from geocebada.models.checkpoint05 import (
    fixed_target_predictions,
    load_checkpoint05_bundle,
    verify_fixed_target_bundle,
)



def test_checkpoint05_config_is_competition_only() -> None:
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(
        (root / "configs" / "checkpoint05.yaml").read_text(encoding="utf-8")
    )

    assert config["scope"]["active_track"] == "competition"
    assert (
        Path(config["inputs"]["base_table"]).name
        == "parcel_features_competition.csv"
    )



def test_component_grid_is_clipped_to_smallest_inner_training_fold() -> None:
    grid = [
        {"model__n_components": [2]},
        {"model__n_components": [4]},
        {"model__n_components": [8]},
        {"model__n_components": [12]},
        {"model__n_components": [20]},
    ]
    inner_splits = [
        (np.arange(18), np.arange(18, 24)),
        (np.arange(21), np.arange(21, 24)),
        (np.arange(20), np.arange(20, 24)),
    ]

    constrained, cap = constrain_component_param_grid(
        grid,
        parameter="model__n_components",
        inner_splits=inner_splits,
        n_features=200,
    )

    assert cap == 18
    assert [row["model__n_components"][0] for row in constrained] == [
        2,
        4,
        8,
        12,
    ]


def test_component_grid_also_respects_feature_dimension() -> None:
    grid = [
        {"model__n_components": [2]},
        {"model__n_components": [4]},
        {"model__n_components": [8]},
    ]
    inner_splits = [
        (np.arange(20), np.arange(20, 25)),
        (np.arange(19), np.arange(19, 25)),
    ]

    constrained, cap = constrain_component_param_grid(
        grid,
        parameter="model__n_components",
        inner_splits=inner_splits,
        n_features=4,
    )

    assert cap == 4
    assert [row["model__n_components"][0] for row in constrained] == [2, 4]


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


def test_fixed_target_bundle_helpers_reproduce_selected_candidate() -> None:
    ids = [f"AGC_{index:03d}" for index in range(59)]
    values = np.linspace(2.0, 5.0, 59)
    final = pd.DataFrame(
        {
            "ID_POLIGONO": ids,
            "RENDIMIENTO_T_HA": values,
        }
    )
    candidates = pd.DataFrame(
        {
            "ID_POLIGONO": ids + ids,
            "method": ["M1"] * 59 + ["M2"] * 59,
            "predicted": np.concatenate([values, values + 0.2]),
        }
    )
    diagnostics = pd.DataFrame({"ID_POLIGONO": ids})
    bundle = {
        "manifest": {
            "checkpoint": "05",
            "inference_scope": "fixed_59_competition_targets",
        },
        "final_method": "M1",
        "final_predictions": final,
        "actual_candidates": candidates,
        "final_diagnostics": diagnostics,
        "target_ids": ids,
    }

    recovered = fixed_target_predictions(bundle)
    verification = verify_fixed_target_bundle(bundle)

    assert len(recovered) == 59
    assert verification["verified"] is True
    assert verification["max_abs_difference"] == pytest.approx(0.0)



def test_checkpoint05_bundle_roundtrip_loader(tmp_path) -> None:
    ids = [f"AGC_{index:03d}" for index in range(59)]
    values = np.linspace(2.0, 5.0, 59)
    bundle = {
        "manifest": {
            "checkpoint": "05",
            "inference_scope": "fixed_59_competition_targets",
        },
        "final_method": "Local04D",
        "final_predictions": pd.DataFrame(
            {
                "ID_POLIGONO": ids,
                "RENDIMIENTO_T_HA": values,
            }
        ),
        "actual_candidates": pd.DataFrame(
            {
                "ID_POLIGONO": ids,
                "method": ["Local04D"] * 59,
                "predicted": values,
            }
        ),
        "final_diagnostics": pd.DataFrame({"ID_POLIGONO": ids}),
        "target_ids": ids,
    }

    path = tmp_path / "checkpoint05_model.joblib"
    import joblib

    joblib.dump(bundle, path)
    loaded = load_checkpoint05_bundle(path)

    assert loaded["final_method"] == "Local04D"
    assert verify_fixed_target_bundle(loaded)["verified"] is True
