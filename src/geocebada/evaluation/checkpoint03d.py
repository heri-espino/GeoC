"""Checkpoint 03D large-compute global-model utilities."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from geocebada.evaluation.checkpoint03c import RepresentationSpec
from geocebada.evaluation.checkpoint05 import constrain_component_param_grid


@dataclass(frozen=True)
class _LargeGlobalSearchSpec:
    name: str
    kind: str
    estimator: Pipeline
    param_grid: list[dict[str, list[Any]]]


def _candidate_grid(
    candidates: Sequence[Mapping[str, Any]],
    mapping: Mapping[str, str],
) -> list[dict[str, list[Any]]]:
    grid: list[dict[str, list[Any]]] = []
    for candidate in candidates:
        row: dict[str, list[Any]] = {}
        for key, value in candidate.items():
            if key not in mapping:
                raise KeyError(f"Unsupported 03D tuning key {key!r}.")
            row[mapping[key]] = [value]
        grid.append(row)
    if not grid:
        raise ValueError("Checkpoint 03D requires at least one candidate.")
    return grid


def _base_steps() -> list[tuple[str, Any]]:
    return [
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                add_indicator=False,
                keep_empty_features=True,
            ),
        )
    ]


def _make_large_global_search_spec(
    *,
    model_name: str,
    model_config: Mapping[str, Any],
    representation: RepresentationSpec,
    random_state: int,
    compute: str,
    catboost_devices: str,
    xgboost_device: str,
) -> _LargeGlobalSearchSpec:
    kind = str(model_config["kind"])
    candidates = model_config["candidates"]
    fixed = dict(model_config.get("fixed", {}))
    steps = _base_steps()

    if kind == "ridge":
        steps.extend([("scaler", StandardScaler()), ("model", Ridge())])
        grid = _candidate_grid(candidates, {"alpha": "model__alpha"})

    elif kind == "pls":
        steps.extend(
            [
                ("scaler", StandardScaler()),
                ("model", PLSRegression(scale=False, max_iter=2_000)),
            ]
        )
        grid = _candidate_grid(
            candidates,
            {"n_components": "model__n_components"},
        )

    elif kind == "extra_trees":
        steps.append(
            (
                "model",
                ExtraTreesRegressor(
                    n_estimators=int(fixed.get("n_estimators", 3000)),
                    n_jobs=int(fixed.get("n_jobs", -1)),
                    random_state=int(random_state),
                ),
            )
        )
        grid = _candidate_grid(
            candidates,
            {
                "min_samples_leaf": "model__min_samples_leaf",
                "max_features": "model__max_features",
                "max_depth": "model__max_depth",
            },
        )

    elif kind == "hist_gradient_boosting":
        steps.append(
            (
                "model",
                HistGradientBoostingRegressor(
                    early_stopping=bool(fixed.get("early_stopping", False)),
                    random_state=int(random_state),
                ),
            )
        )
        grid = _candidate_grid(
            candidates,
            {
                "max_iter": "model__max_iter",
                "max_leaf_nodes": "model__max_leaf_nodes",
                "learning_rate": "model__learning_rate",
                "min_samples_leaf": "model__min_samples_leaf",
                "l2_regularization": "model__l2_regularization",
            },
        )

    elif kind == "catboost":
        try:
            from catboost import CatBoostRegressor
        except ImportError as exc:
            raise ImportError(
                "CatBoost is required for Checkpoint 03D. Install .[models]."
            ) from exc

        kwargs: dict[str, Any] = {
            "loss_function": str(fixed.get("loss_function", "RMSE")),
            "verbose": bool(fixed.get("verbose", False)),
            "allow_writing_files": bool(fixed.get("allow_writing_files", False)),
            "random_seed": int(random_state),
            "task_type": "GPU" if compute == "GPU" else "CPU",
            "thread_count": -1,
        }
        if compute == "GPU":
            kwargs["devices"] = str(catboost_devices)
        steps.append(("model", CatBoostRegressor(**kwargs)))
        grid = _candidate_grid(
            candidates,
            {
                "iterations": "model__iterations",
                "depth": "model__depth",
                "learning_rate": "model__learning_rate",
                "l2_leaf_reg": "model__l2_leaf_reg",
                "random_strength": "model__random_strength",
            },
        )

    elif kind == "xgboost":
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise ImportError(
                "XGBoost is required for Checkpoint 03D. Install .[models]."
            ) from exc

        kwargs = {
            "objective": str(fixed.get("objective", "reg:squarederror")),
            "tree_method": str(fixed.get("tree_method", "hist")),
            "verbosity": int(fixed.get("verbosity", 0)),
            "device": str(xgboost_device) if compute == "GPU" else "cpu",
            "random_state": int(random_state),
            "n_jobs": -1,
        }
        steps.append(("model", XGBRegressor(**kwargs)))
        grid = _candidate_grid(
            candidates,
            {
                "n_estimators": "model__n_estimators",
                "max_depth": "model__max_depth",
                "learning_rate": "model__learning_rate",
                "min_child_weight": "model__min_child_weight",
                "subsample": "model__subsample",
                "colsample_bytree": "model__colsample_bytree",
                "reg_alpha": "model__reg_alpha",
                "reg_lambda": "model__reg_lambda",
            },
        )

    elif kind == "lightgbm":
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:
            raise ImportError(
                "LightGBM is required for Checkpoint 03D. Install .[models]."
            ) from exc

        steps.append(
            (
                "model",
                LGBMRegressor(
                    objective=str(fixed.get("objective", "regression")),
                    verbosity=int(fixed.get("verbosity", -1)),
                    random_state=int(random_state),
                    n_jobs=-1,
                    subsample_freq=1,
                ),
            )
        )
        grid = _candidate_grid(
            candidates,
            {
                "n_estimators": "model__n_estimators",
                "num_leaves": "model__num_leaves",
                "max_depth": "model__max_depth",
                "learning_rate": "model__learning_rate",
                "min_child_samples": "model__min_child_samples",
                "subsample": "model__subsample",
                "colsample_bytree": "model__colsample_bytree",
                "reg_alpha": "model__reg_alpha",
                "reg_lambda": "model__reg_lambda",
            },
        )

    else:
        raise ValueError(f"Unsupported Checkpoint 03D model kind: {kind}")

    return _LargeGlobalSearchSpec(
        name=str(model_name),
        kind=kind,
        estimator=Pipeline(steps),
        param_grid=grid,
    )


def _fit_large_search(
    *,
    spec: _LargeGlobalSearchSpec,
    x_train: Any,
    y_train: Any,
    inner_splits: list[tuple[np.ndarray, np.ndarray]],
    search_n_jobs: int,
) -> GridSearchCV:
    param_grid = spec.param_grid
    if spec.kind == "pls":
        param_grid, _ = constrain_component_param_grid(
            param_grid,
            parameter="model__n_components",
            inner_splits=inner_splits,
            n_features=int(x_train.shape[1]),
        )

    search = GridSearchCV(
        estimator=spec.estimator,
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=inner_splits,
        refit=True,
        n_jobs=int(search_n_jobs),
        return_train_score=False,
        error_score="raise",
    )
    search.fit(x_train, y_train)
    return search


def _eligible_pairs(
    *,
    config: Mapping[str, Any],
    representation_names: Sequence[str],
    model_names: Sequence[str],
) -> list[tuple[str, str]]:
    known_representations = set(map(str, representation_names))
    result: list[tuple[str, str]] = []
    for model_name in model_names:
        model_config = config["models"][model_name]
        allowed = list(map(str, model_config.get("representations", representation_names)))
        for representation in allowed:
            if representation in known_representations:
                result.append((representation, str(model_name)))
    return result


def _gpu_smoke_test(*, catboost_devices: str, xgboost_device: str) -> None:
    x = np.asarray(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [0.5, 0.7],
            [0.8, 0.2],
            [0.2, 0.8],
            [0.9, 0.1],
        ],
        dtype=np.float32,
    )
    y = np.asarray([1.0, 2.0, 1.4, 1.8, 1.2, 1.9], dtype=np.float32)

    try:
        from catboost import CatBoostRegressor

        CatBoostRegressor(
            iterations=2,
            depth=2,
            learning_rate=0.1,
            loss_function="RMSE",
            task_type="GPU",
            devices=str(catboost_devices),
            verbose=False,
            allow_writing_files=False,
        ).fit(x, y)
    except Exception as exc:
        raise RuntimeError(f"CatBoost GPU preflight failed: {exc}") from exc

    try:
        from xgboost import XGBRegressor

        XGBRegressor(
            n_estimators=2,
            max_depth=2,
            learning_rate=0.1,
            objective="reg:squarederror",
            tree_method="hist",
            device=str(xgboost_device),
            verbosity=0,
        ).fit(x, y)
    except Exception as exc:
        raise RuntimeError(f"XGBoost GPU preflight failed: {exc}") from exc
