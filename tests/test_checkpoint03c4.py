"""Tests for Checkpoint 03C.4 nested stacking."""

from __future__ import annotations

import numpy as np
import pandas as pd

from geocebada.evaluation import RepresentationSpec
from geocebada.evaluation.checkpoint03c4 import (
    ConvexStackingRegressor,
    build_stacking_meta_features,
    evaluate_nested_stacking,
    summarize_stacking_results,
)


def test_convex_stacker_learns_simplex_weights() -> None:
    X = np.array(
        [
            [1.0, 1.4, 0.8],
            [2.0, 2.4, 1.7],
            [3.0, 3.5, 2.7],
            [4.0, 4.3, 3.6],
            [5.0, 5.5, 4.6],
        ]
    )
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    model = ConvexStackingRegressor().fit(X, y)

    assert np.isclose(model.coef_.sum(), 1.0)
    assert (model.coef_ >= 0).all()
    assert np.sqrt(np.mean((y - model.predict(X)) ** 2)) < 0.15


def _meta_config() -> dict:
    return {
        "base_candidates": {
            "B1": {},
            "B2": {},
            "B3": {},
        },
        "meta_features": {
            "include_raw_base_predictions": True,
            "include_mean_all": True,
            "include_std_all": True,
            "include_range_all": True,
            "top3": ["B1", "B2", "B3"],
            "include_top3_mean": True,
            "include_top3_std": True,
        },
    }


def test_meta_features_include_disagreement_geometry() -> None:
    base = pd.DataFrame(
        {
            "B1": [1.0, 2.0],
            "B2": [1.2, 1.8],
            "B3": [0.8, 2.2],
        }
    )
    meta = build_stacking_meta_features(base, _meta_config())

    assert {"pred_mean_all", "pred_std_all", "pred_range_all"} <= set(meta)
    assert {"pred_mean_top3", "pred_std_top3"} <= set(meta)
    assert np.isclose(meta.loc[0, "pred_mean_all"], 1.0)
    assert np.isclose(meta.loc[0, "pred_range_all"], 0.4)


def _synthetic_inputs() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, RepresentationSpec],
    dict,
    dict,
    dict,
]:
    rng = np.random.default_rng(42)
    rows = 24
    states = np.array(["A", "B", "C"] * 8)
    municipalities = np.array([f"m{i // 2:02d}" for i in range(rows)])
    x1 = rng.normal(size=rows)
    x2 = rng.normal(size=rows)
    x3 = rng.normal(size=rows)
    x4 = rng.normal(size=rows)
    y = 3.0 + 0.8 * x1 - 0.5 * x3 + 0.1 * rng.normal(size=rows)

    joined = pd.DataFrame(
        {
            "ID_POLIGONO": [f"P{i:02d}" for i in range(rows)],
            "CONJUNTO": ["ENTRENAMIENTO"] * rows,
            "RENDIMIENTO_T_HA": y,
            "meta_estado": states,
            "meta_municipio": municipalities,
            "x1": x1,
            "x2": x2,
            "x3": x3,
            "x4": x4,
        }
    )
    grouped_fold = np.array(
        [1 if int(value[1:]) % 2 == 0 else 2 for value in municipalities]
    )
    folds = pd.DataFrame(
        {
            "ID_POLIGONO": joined["ID_POLIGONO"],
            "fold_state_stratified": [1, 2] * (rows // 2),
            "fold_municipality_grouped": grouped_fold,
        }
    )
    representations = {
        "C0": RepresentationSpec(
            name="C0",
            track="competition",
            features=("x1", "x2"),
            layers=("base",),
            discovery=False,
            primitive_columns=(),
            description="synthetic base",
        ),
        "C1": RepresentationSpec(
            name="C1",
            track="competition",
            features=("x3", "x4"),
            layers=("agronomic",),
            discovery=False,
            primitive_columns=(),
            description="synthetic second view",
        ),
    }
    checkpoint03c2_config = {
        "models": {
            "Ridge": {
                "kind": "ridge",
                "candidates": [{"alpha": 0.1}, {"alpha": 10.0}],
            },
            "PLS": {
                "kind": "pls",
                "candidates": [{"n_components": 1}, {"n_components": 2}],
            },
        }
    }
    config = {
        "identity": {
            "id_column": "ID_POLIGONO",
            "target_column": "RENDIMIENTO_T_HA",
            "split_column": "CONJUNTO",
            "train_value": "ENTRENAMIENTO",
            "state_column": "meta_estado",
            "municipality_column": "meta_municipio",
        },
        "validation": {
            "outer_protocols": [
                "fold_state_stratified",
                "fold_municipality_grouped",
            ],
            "level1_folds": 2,
            "base_tuning_folds": 2,
            "outer_refit_tuning_folds": 2,
            "meta_tuning_folds": 2,
            "random_state": 42,
        },
        "base_candidates": {
            "B1": {
                "representation": "C0",
                "model": "Ridge",
                "source": "checkpoint03c2",
            },
            "B2": {
                "representation": "C1",
                "model": "Ridge",
                "source": "checkpoint03c2",
            },
            "B3": {
                "representation": "C0",
                "model": "PLS",
                "source": "checkpoint03c2",
            },
        },
        "custom_base_models": {},
        "meta_features": {
            "include_raw_base_predictions": True,
            "include_mean_all": True,
            "include_std_all": True,
            "include_range_all": True,
            "top3": ["B1", "B2", "B3"],
            "include_top3_mean": True,
            "include_top3_std": True,
        },
        "meta_models": {
            "EqualTop3": {"kind": "equal_top3"},
            "RidgeStack": {
                "kind": "ridge",
                "candidates": [{"alpha": 0.1}, {"alpha": 10.0}],
            },
        },
        "runtime": {
            "search_n_jobs": 1,
            "catboost_task_type": "CPU",
            "catboost_devices": "0",
            "catboost_allow_cpu_fallback": True,
            "progress": False,
        },
    }
    empirical_config = {"supervised_discovery": {}}
    return (
        joined,
        folds,
        representations,
        checkpoint03c2_config,
        config,
        empirical_config,
    )


def test_nested_stacking_has_complete_outer_oof_coverage() -> None:
    (
        joined,
        folds,
        representations,
        checkpoint03c2_config,
        config,
        empirical_config,
    ) = _synthetic_inputs()

    (
        base_outer,
        meta_training,
        stack_oof,
        base_search,
        meta_search,
        outer_metrics,
        coefficients,
        task_type,
    ) = evaluate_nested_stacking(
        joined=joined,
        folds=folds,
        representations=representations,
        checkpoint03c2_config=checkpoint03c2_config,
        config=config,
        empirical_config=empirical_config,
    )
    protocol_summary, robustness = summarize_stacking_results(
        outer_metrics,
        stack_oof,
        config,
    )

    assert task_type == "CPU"
    assert not base_outer.empty
    assert not meta_training.empty
    assert not base_search.empty
    assert not meta_search.empty
    assert coefficients["meta_model"].eq("RidgeStack").all()
    assert len(outer_metrics) == 2 * 2 * 2
    assert len(protocol_summary) == 2 * 2
    assert len(robustness) == 2
    assert set(robustness["meta_model"]) == {"EqualTop3", "RidgeStack"}

    counts = stack_oof.groupby(
        ["protocol", "meta_model"],
        observed=True,
    )["ID_POLIGONO"].nunique()
    assert counts.eq(24).all()
    assert np.isfinite(protocol_summary["oof_rmse"]).all()
