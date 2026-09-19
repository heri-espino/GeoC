"""Checkpoint 03C.4 leakage-safe nested stacking evaluation."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, HuberRegressor, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from geocebada.evaluation.checkpoint03c import RepresentationSpec
from geocebada.evaluation.checkpoint03c2 import (
    ModelSearchSpec,
    make_inner_cv_splits,
    make_model_search_spec,
)
from geocebada.evaluation.parcel_modeling import validate_fixed_fold_assignments
from geocebada.features import FoldLocalExpressionAugmenter


@dataclass(frozen=True)
class MetaSearchSpec:
    """One stacker estimator and its small tuning grid."""

    name: str
    kind: str
    estimator: Pipeline | BaseEstimator
    param_grid: list[dict[str, list[Any]]]


class ConvexStackingRegressor(BaseEstimator, RegressorMixin):
    """Fit nonnegative stacking weights constrained to sum to one."""

    def __init__(self, l2_penalty: float = 0.0) -> None:
        self.l2_penalty = l2_penalty

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
    ) -> ConvexStackingRegressor:
        """Fit simplex-constrained weights by squared-error minimization."""

        matrix = np.asarray(X, dtype=float)
        target = np.asarray(y, dtype=float).reshape(-1)
        if matrix.ndim != 2:
            raise ValueError("ConvexStackingRegressor requires a 2D feature matrix.")
        if len(matrix) != len(target):
            raise ValueError("X and y must contain the same number of rows.")
        if not np.isfinite(matrix).all() or not np.isfinite(target).all():
            raise ValueError("Convex stacking input contains non-finite values.")

        n_features = matrix.shape[1]
        initial = np.full(n_features, 1.0 / n_features, dtype=float)

        def objective(weights: np.ndarray) -> float:
            residual = target - matrix @ weights
            return float(
                np.mean(residual**2)
                + float(self.l2_penalty) * np.sum(weights**2)
            )

        result = minimize(
            objective,
            initial,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * n_features,
            constraints={
                "type": "eq",
                "fun": lambda weights: float(np.sum(weights) - 1.0),
            },
            options={"maxiter": 2_000, "ftol": 1.0e-12},
        )
        if not result.success:
            raise RuntimeError(f"Convex stacking optimization failed: {result.message}")

        weights = np.asarray(result.x, dtype=float)
        weights = np.clip(weights, 0.0, 1.0)
        weights /= weights.sum()
        self.coef_ = weights
        self.n_features_in_ = n_features
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict with the fitted convex combination."""

        check_is_fitted(self, attributes=["coef_", "n_features_in_"])
        matrix = np.asarray(X, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != self.n_features_in_:
            raise ValueError("Prediction matrix has the wrong number of columns.")
        return matrix @ self.coef_


def build_stacking_meta_features(
    base_predictions: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    base_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Build target-free meta features from cross-fitted base predictions."""

    resolved_base_names = (
        list(map(str, base_names))
        if base_names is not None
        else list(map(str, config["base_candidates"].keys()))
    )
    base_names = resolved_base_names
    missing = [name for name in base_names if name not in base_predictions.columns]
    if missing:
        raise KeyError(f"Missing base prediction columns: {missing}")

    raw = base_predictions.loc[:, base_names].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(raw.to_numpy(dtype=float)).all():
        raise ValueError("Base prediction matrix contains non-finite values.")

    feature_config = config["meta_features"]
    output = pd.DataFrame(index=raw.index)
    if bool(feature_config.get("include_raw_base_predictions", True)):
        for column in base_names:
            output[f"pred__{column}"] = raw[column].astype(float)

    if bool(feature_config.get("include_mean_all", True)):
        output["pred_mean_all"] = raw.mean(axis=1)
    if bool(feature_config.get("include_std_all", True)):
        output["pred_std_all"] = raw.std(axis=1, ddof=0)
    if bool(feature_config.get("include_range_all", True)):
        output["pred_range_all"] = raw.max(axis=1) - raw.min(axis=1)

    top3 = list(map(str, feature_config.get("top3", [])))
    if top3:
        unknown = sorted(set(top3) - set(base_names))
        if unknown:
            raise KeyError(f"Unknown top3 base candidates: {unknown}")
        top = raw.loc[:, top3]
        if bool(feature_config.get("include_top3_mean", True)):
            output["pred_mean_top3"] = top.mean(axis=1)
        if bool(feature_config.get("include_top3_std", True)):
            output["pred_std_top3"] = top.std(axis=1, ddof=0)

    if output.empty:
        raise ValueError("Stacking meta-feature configuration produced zero columns.")
    return output


def _singleton_candidate_grid(
    candidates: Sequence[Mapping[str, Any]],
    prefix_by_key: Mapping[str, str],
) -> list[dict[str, list[Any]]]:
    grid: list[dict[str, list[Any]]] = []
    for candidate in candidates:
        row: dict[str, list[Any]] = {}
        for key, value in candidate.items():
            if key not in prefix_by_key:
                raise KeyError(f"Unsupported tuning key: {key}")
            row[prefix_by_key[key]] = [value]
        grid.append(row)
    if not grid:
        raise ValueError("At least one hyperparameter candidate is required.")
    return grid


def _custom_base_search_spec(
    *,
    model_name: str,
    model_config: Mapping[str, Any],
    representation: RepresentationSpec,
    discovery_config: Mapping[str, Any],
    random_state: int,
) -> ModelSearchSpec:
    kind = str(model_config["kind"])
    if kind != "pca_ridge":
        raise ValueError(f"Unsupported custom 03C.4 base model kind: {kind}")

    steps: list[tuple[str, Any]] = []
    if representation.discovery:
        steps.append(
            (
                "discover",
                FoldLocalExpressionAugmenter(
                    representation.primitive_columns,
                    top_primitives=int(discovery_config["top_primitives"]),
                    top_expressions=int(discovery_config["top_expressions"]),
                    operations=tuple(map(str, discovery_config["operations"])),
                    epsilon=float(discovery_config["epsilon"]),
                ),
            )
        )
    steps.extend(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                    keep_empty_features=True,
                ),
            ),
            ("scaler", StandardScaler()),
            (
                "pca",
                PCA(
                    svd_solver="full",
                    random_state=random_state,
                ),
            ),
            ("model", Ridge()),
        ]
    )
    grid = _singleton_candidate_grid(
        model_config["candidates"],
        {
            "variance": "pca__n_components",
            "alpha": "model__alpha",
        },
    )
    return ModelSearchSpec(
        name=model_name,
        kind=kind,
        estimator=Pipeline(steps),
        param_grid=grid,
    )


def _make_base_search_spec(
    *,
    base_name: str,
    base_config: Mapping[str, Any],
    representation: RepresentationSpec,
    checkpoint03c2_config: Mapping[str, Any],
    config: Mapping[str, Any],
    discovery_config: Mapping[str, Any],
    random_state: int,
    catboost_task_type: str,
    catboost_devices: str,
) -> ModelSearchSpec:
    source = str(base_config["source"])
    model_name = str(base_config["model"])
    if source == "checkpoint03c2":
        return make_model_search_spec(
            model_name=base_name,
            model_config=checkpoint03c2_config["models"][model_name],
            representation=representation,
            discovery_config=discovery_config,
            random_state=random_state,
            catboost_task_type=catboost_task_type,
            catboost_devices=catboost_devices,
        )
    if source == "checkpoint03c4":
        return _custom_base_search_spec(
            model_name=base_name,
            model_config=config["custom_base_models"][model_name],
            representation=representation,
            discovery_config=discovery_config,
            random_state=random_state,
        )
    raise ValueError(f"Unknown base-candidate source: {source}")


def _gpu_failure(error: Exception) -> bool:
    message = str(error).casefold()
    return any(token in message for token in ("cuda", "gpu", "driver", "device", "nvidia"))


def _fit_base_search(
    *,
    base_name: str,
    base_config: Mapping[str, Any],
    representation: RepresentationSpec,
    checkpoint03c2_config: Mapping[str, Any],
    config: Mapping[str, Any],
    discovery_config: Mapping[str, Any],
    random_state: int,
    catboost_task_type: str,
    catboost_devices: str,
    allow_cpu_fallback: bool,
    cv_splits: list[tuple[np.ndarray, np.ndarray]],
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[GridSearchCV, str]:
    spec = _make_base_search_spec(
        base_name=base_name,
        base_config=base_config,
        representation=representation,
        checkpoint03c2_config=checkpoint03c2_config,
        config=config,
        discovery_config=discovery_config,
        random_state=random_state,
        catboost_task_type=catboost_task_type,
        catboost_devices=catboost_devices,
    )
    task_type = catboost_task_type
    jobs = (
        1
        if spec.kind == "catboost" and task_type == "GPU"
        else int(config["runtime"].get("search_n_jobs", 1))
    )

    def build_search(current_spec: ModelSearchSpec, n_jobs: int) -> GridSearchCV:
        return GridSearchCV(
            estimator=current_spec.estimator,
            param_grid=current_spec.param_grid,
            scoring="neg_root_mean_squared_error",
            cv=cv_splits,
            refit=True,
            n_jobs=n_jobs,
            return_train_score=False,
            error_score="raise",
        )

    search = build_search(spec, jobs)
    try:
        search.fit(X, y)
    except Exception as exc:
        if (
            spec.kind == "catboost"
            and task_type == "GPU"
            and allow_cpu_fallback
            and _gpu_failure(exc)
        ):
            task_type = "CPU"
            spec = _make_base_search_spec(
                base_name=base_name,
                base_config=base_config,
                representation=representation,
                checkpoint03c2_config=checkpoint03c2_config,
                config=config,
                discovery_config=discovery_config,
                random_state=random_state,
                catboost_task_type="CPU",
                catboost_devices=catboost_devices,
            )
            search = build_search(
                spec,
                int(config["runtime"].get("search_n_jobs", 1)),
            )
            search.fit(X, y)
        else:
            raise
    return search, task_type


def _make_meta_search_spec(
    *,
    meta_name: str,
    meta_config: Mapping[str, Any],
    random_state: int,
) -> MetaSearchSpec:
    kind = str(meta_config["kind"])

    if kind == "ridge":
        estimator: Pipeline | BaseEstimator = Pipeline(
            [("scaler", StandardScaler()), ("model", Ridge())]
        )
        grid = _singleton_candidate_grid(
            meta_config["candidates"],
            {"alpha": "model__alpha"},
        )
    elif kind == "elastic_net":
        estimator = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", ElasticNet(max_iter=50_000, tol=1.0e-5)),
            ]
        )
        grid = _singleton_candidate_grid(
            meta_config["candidates"],
            {
                "alpha": "model__alpha",
                "l1_ratio": "model__l1_ratio",
            },
        )
    elif kind == "huber":
        estimator = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", HuberRegressor(max_iter=2_000)),
            ]
        )
        grid = _singleton_candidate_grid(
            meta_config["candidates"],
            {
                "epsilon": "model__epsilon",
                "alpha": "model__alpha",
            },
        )
    elif kind == "extra_trees":
        fixed = meta_config.get("fixed", {})
        estimator = ExtraTreesRegressor(
            n_estimators=int(fixed.get("n_estimators", 600)),
            max_depth=fixed.get("max_depth"),
            random_state=random_state,
            n_jobs=-1,
        )
        grid = _singleton_candidate_grid(
            meta_config["candidates"],
            {
                "min_samples_leaf": "min_samples_leaf",
                "max_features": "max_features",
            },
        )
    elif kind == "hist_gradient_boosting":
        estimator = HistGradientBoostingRegressor(
            max_iter=300,
            min_samples_leaf=10,
            random_state=random_state,
        )
        grid = _singleton_candidate_grid(
            meta_config["candidates"],
            {
                "learning_rate": "learning_rate",
                "max_leaf_nodes": "max_leaf_nodes",
                "l2_regularization": "l2_regularization",
            },
        )
    else:
        raise ValueError(f"Unsupported tuned stacker kind: {kind}")

    return MetaSearchSpec(
        name=meta_name,
        kind=kind,
        estimator=estimator,
        param_grid=grid,
    )


def _metric_row(observed: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    y_true = observed.to_numpy(dtype=float)
    y_pred = np.asarray(predicted, dtype=float).reshape(-1)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _search_rows(
    search: GridSearchCV,
    *,
    protocol: str,
    outer_fold: int,
    stage: str,
    candidate: str,
    level1_fold: int | None,
) -> list[dict[str, Any]]:
    results = search.cv_results_
    rows: list[dict[str, Any]] = []
    for index, params in enumerate(results["params"]):
        rows.append(
            {
                "protocol": protocol,
                "outer_fold": outer_fold,
                "stage": stage,
                "candidate": candidate,
                "level1_fold": level1_fold,
                "candidate_index": index,
                "mean_test_rmse": float(-results["mean_test_score"][index]),
                "std_test_rmse": float(results["std_test_score"][index]),
                "rank_test_score": int(results["rank_test_score"][index]),
                "params_json": json.dumps(params, sort_keys=True),
            }
        )
    return rows


def _record_meta_importance(
    *,
    estimator: Any,
    feature_names: Sequence[str],
    protocol: str,
    outer_fold: int,
    meta_model: str,
) -> list[dict[str, Any]]:
    fitted = estimator
    if isinstance(fitted, Pipeline):
        fitted = fitted.named_steps["model"]

    values: np.ndarray | None = None
    value_kind = ""
    if hasattr(fitted, "coef_"):
        values = np.asarray(fitted.coef_, dtype=float).reshape(-1)
        value_kind = "coefficient"
    elif hasattr(fitted, "feature_importances_"):
        values = np.asarray(fitted.feature_importances_, dtype=float).reshape(-1)
        value_kind = "importance"

    if values is None or len(values) != len(feature_names):
        return []
    return [
        {
            "protocol": protocol,
            "outer_fold": outer_fold,
            "meta_model": meta_model,
            "feature": str(feature),
            "value_kind": value_kind,
            "value": float(value),
        }
        for feature, value in zip(feature_names, values, strict=True)
    ]


def evaluate_nested_stacking(
    *,
    joined: pd.DataFrame,
    folds: pd.DataFrame,
    representations: Mapping[str, RepresentationSpec],
    checkpoint03c2_config: Mapping[str, Any],
    config: Mapping[str, Any],
    empirical_config: Mapping[str, Any],
    selected_base_candidates: Sequence[str] | None = None,
    selected_meta_models: Sequence[str] | None = None,
    selected_protocols: Sequence[str] | None = None,
    catboost_task_type_override: str | None = None,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    str,
]:
    """Run honest level-1 cross-fitting and untouched outer-fold stack evaluation."""

    identity = config["identity"]
    validation = config["validation"]
    runtime = config["runtime"]
    id_column = str(identity["id_column"])
    target_column = str(identity["target_column"])
    split_column = str(identity["split_column"])
    train_value = str(identity["train_value"])
    state_column = str(identity["state_column"])
    municipality_column = str(identity["municipality_column"])

    configured_protocols = list(map(str, validation["outer_protocols"]))
    protocols = (
        list(map(str, selected_protocols))
        if selected_protocols
        else configured_protocols
    )
    unknown_protocols = sorted(set(protocols) - set(configured_protocols))
    if unknown_protocols:
        raise KeyError(f"Unknown selected protocol(s): {unknown_protocols}")

    validate_fixed_fold_assignments(
        joined,
        folds,
        id_column=id_column,
        split_column=split_column,
        train_value=train_value,
        fold_columns=configured_protocols,
    )

    training = joined.loc[joined[split_column].eq(train_value)].copy()
    training[id_column] = training[id_column].astype(str)
    fold_columns = folds[[id_column, *configured_protocols]].copy()
    fold_columns[id_column] = fold_columns[id_column].astype(str)
    training = training.merge(
        fold_columns,
        on=id_column,
        how="left",
        validate="one_to_one",
    ).reset_index(drop=True)

    y = pd.to_numeric(training[target_column], errors="raise").astype(float)
    if y.isna().any():
        raise ValueError("Training target contains missing values.")

    base_names = (
        list(map(str, selected_base_candidates))
        if selected_base_candidates
        else list(map(str, config["base_candidates"].keys()))
    )
    meta_names = (
        list(map(str, selected_meta_models))
        if selected_meta_models
        else list(map(str, config["meta_models"].keys()))
    )
    unknown_base = sorted(set(base_names) - set(config["base_candidates"]))
    unknown_meta = sorted(set(meta_names) - set(config["meta_models"]))
    if unknown_base:
        raise KeyError(f"Unknown base candidate(s): {unknown_base}")
    if unknown_meta:
        raise KeyError(f"Unknown meta model(s): {unknown_meta}")

    required_top3 = set(map(str, config["meta_features"].get("top3", [])))
    if (
        any(
            str(config["meta_models"][name]["kind"]) == "equal_top3"
            for name in meta_names
        )
        and not required_top3.issubset(base_names)
    ):
        raise ValueError("EqualTop3 requires all configured top3 base candidates.")

    random_state = int(validation["random_state"])
    level1_folds = int(validation["level1_folds"])
    base_tuning_folds = int(validation["base_tuning_folds"])
    outer_refit_tuning_folds = int(validation["outer_refit_tuning_folds"])
    meta_tuning_folds = int(validation["meta_tuning_folds"])
    discovery_config = empirical_config["supervised_discovery"]

    catboost_task_type = (
        str(catboost_task_type_override).upper()
        if catboost_task_type_override
        else str(runtime.get("catboost_task_type", "GPU")).upper()
    )
    catboost_devices = str(runtime.get("catboost_devices", "0"))
    allow_cpu_fallback = bool(
        runtime.get("catboost_allow_cpu_fallback", False)
    )
    progress = bool(runtime.get("progress", True))
    final_catboost_task_type = catboost_task_type

    base_feature_frames: dict[str, pd.DataFrame] = {}
    for base_name in base_names:
        base_config = config["base_candidates"][base_name]
        representation_name = str(base_config["representation"])
        if representation_name not in representations:
            raise KeyError(
                f"Base candidate {base_name!r} references unknown representation "
                f"{representation_name!r}."
            )
        representation = representations[representation_name]
        base_feature_frames[base_name] = training.loc[
            :,
            list(representation.features),
        ].apply(pd.to_numeric, errors="coerce")

    base_outer_rows: list[dict[str, Any]] = []
    meta_training_rows: list[dict[str, Any]] = []
    stack_oof_rows: list[dict[str, Any]] = []
    base_search_rows: list[dict[str, Any]] = []
    meta_search_rows: list[dict[str, Any]] = []
    outer_metric_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []

    for protocol in protocols:
        fold_values = sorted(
            pd.to_numeric(training[protocol], errors="raise")
            .astype(int)
            .unique()
        )
        for outer_fold in fold_values:
            outer_valid_mask = training[protocol].eq(outer_fold).to_numpy()
            outer_train_positions = np.flatnonzero(~outer_valid_mask)
            outer_valid_positions = np.flatnonzero(outer_valid_mask)
            outer_train = training.iloc[outer_train_positions].reset_index(drop=True)
            outer_valid = training.iloc[outer_valid_positions].reset_index(drop=True)
            y_outer_train = y.iloc[outer_train_positions].reset_index(drop=True)
            y_outer_valid = y.iloc[outer_valid_positions].reset_index(drop=True)

            level1_splits = make_inner_cv_splits(
                outer_train,
                protocol=protocol,
                n_splits=level1_folds,
                random_state=random_state + 100 * int(outer_fold),
                state_column=state_column,
                municipality_column=municipality_column,
            )
            meta_splits = make_inner_cv_splits(
                outer_train,
                protocol=protocol,
                n_splits=meta_tuning_folds,
                random_state=random_state + 500 + 100 * int(outer_fold),
                state_column=state_column,
                municipality_column=municipality_column,
            )

            level1_matrix = pd.DataFrame(
                index=np.arange(len(outer_train)),
                columns=base_names,
                dtype=float,
            )
            outer_base_matrix = pd.DataFrame(
                index=np.arange(len(outer_valid)),
                columns=base_names,
                dtype=float,
            )

            for base_index, base_name in enumerate(base_names, start=1):
                base_config = config["base_candidates"][base_name]
                representation_name = str(base_config["representation"])
                representation = representations[representation_name]
                full_x = base_feature_frames[base_name]
                x_outer_train = full_x.iloc[
                    outer_train_positions
                ].reset_index(drop=True)
                x_outer_valid = full_x.iloc[
                    outer_valid_positions
                ].reset_index(drop=True)

                if progress:
                    print(
                        f"[{protocol}] outer {outer_fold} | base "
                        f"{base_index}/{len(base_names)} {base_name}"
                    )

                for level1_fold, (cross_train_idx, cross_valid_idx) in enumerate(
                    level1_splits,
                    start=1,
                ):
                    cross_train_frame = outer_train.iloc[
                        cross_train_idx
                    ].reset_index(drop=True)
                    x_cross_train = x_outer_train.iloc[
                        cross_train_idx
                    ].reset_index(drop=True)
                    y_cross_train = y_outer_train.iloc[
                        cross_train_idx
                    ].reset_index(drop=True)
                    x_cross_valid = x_outer_train.iloc[cross_valid_idx]

                    tune_splits = make_inner_cv_splits(
                        cross_train_frame,
                        protocol=protocol,
                        n_splits=base_tuning_folds,
                        random_state=(
                            random_state
                            + 1_000
                            + 100 * int(outer_fold)
                            + 10 * level1_fold
                            + base_index
                        ),
                        state_column=state_column,
                        municipality_column=municipality_column,
                    )
                    started = perf_counter()
                    search, actual_task = _fit_base_search(
                        base_name=base_name,
                        base_config=base_config,
                        representation=representation,
                        checkpoint03c2_config=checkpoint03c2_config,
                        config=config,
                        discovery_config=discovery_config,
                        random_state=random_state,
                        catboost_task_type=final_catboost_task_type,
                        catboost_devices=catboost_devices,
                        allow_cpu_fallback=allow_cpu_fallback,
                        cv_splits=tune_splits,
                        X=x_cross_train,
                        y=y_cross_train,
                    )
                    if str(base_config["model"]) == "CatBoost":
                        final_catboost_task_type = actual_task
                    predicted = np.asarray(
                        search.predict(x_cross_valid),
                        dtype=float,
                    ).reshape(-1)
                    level1_matrix.loc[cross_valid_idx, base_name] = predicted
                    base_search_rows.extend(
                        _search_rows(
                            search,
                            protocol=protocol,
                            outer_fold=int(outer_fold),
                            stage="level1_crossfit",
                            candidate=base_name,
                            level1_fold=level1_fold,
                        )
                    )
                    if progress:
                        elapsed = perf_counter() - started
                        print(
                            f"  level1 fold {level1_fold}/{level1_folds} "
                            f"done in {elapsed:.1f}s"
                        )

                if level1_matrix[base_name].isna().any():
                    raise ValueError(
                        f"Incomplete level-1 coverage for {base_name!r}, "
                        f"{protocol}, outer fold {outer_fold}."
                    )

                outer_tune_splits = make_inner_cv_splits(
                    outer_train,
                    protocol=protocol,
                    n_splits=outer_refit_tuning_folds,
                    random_state=(
                        random_state + 2_000 + 100 * int(outer_fold) + base_index
                    ),
                    state_column=state_column,
                    municipality_column=municipality_column,
                )
                search, actual_task = _fit_base_search(
                    base_name=base_name,
                    base_config=base_config,
                    representation=representation,
                    checkpoint03c2_config=checkpoint03c2_config,
                    config=config,
                    discovery_config=discovery_config,
                    random_state=random_state,
                    catboost_task_type=final_catboost_task_type,
                    catboost_devices=catboost_devices,
                    allow_cpu_fallback=allow_cpu_fallback,
                    cv_splits=outer_tune_splits,
                    X=x_outer_train,
                    y=y_outer_train,
                )
                if str(base_config["model"]) == "CatBoost":
                    final_catboost_task_type = actual_task
                outer_predicted = np.asarray(
                    search.predict(x_outer_valid),
                    dtype=float,
                ).reshape(-1)
                outer_base_matrix.loc[:, base_name] = outer_predicted
                base_search_rows.extend(
                    _search_rows(
                        search,
                        protocol=protocol,
                        outer_fold=int(outer_fold),
                        stage="outer_refit",
                        candidate=base_name,
                        level1_fold=None,
                    )
                )

                for row_position, prediction in enumerate(
                    level1_matrix[base_name].to_numpy(dtype=float)
                ):
                    meta_training_rows.append(
                        {
                            "protocol": protocol,
                            "outer_fold": int(outer_fold),
                            "ID_POLIGONO": str(
                                outer_train.iloc[row_position][id_column]
                            ),
                            "observed": float(y_outer_train.iloc[row_position]),
                            "base_candidate": base_name,
                            "predicted": float(prediction),
                        }
                    )
                for row_position, prediction in enumerate(outer_predicted):
                    base_outer_rows.append(
                        {
                            "protocol": protocol,
                            "outer_fold": int(outer_fold),
                            "ID_POLIGONO": str(
                                outer_valid.iloc[row_position][id_column]
                            ),
                            "observed": float(y_outer_valid.iloc[row_position]),
                            "base_candidate": base_name,
                            "predicted": float(prediction),
                        }
                    )

            z_train = build_stacking_meta_features(
                level1_matrix,
                config,
                base_names=base_names,
            )
            z_valid = build_stacking_meta_features(
                outer_base_matrix,
                config,
                base_names=base_names,
            )
            raw_top3 = list(map(str, config["meta_features"]["top3"]))

            for meta_index, meta_name in enumerate(meta_names, start=1):
                meta_config = config["meta_models"][meta_name]
                kind = str(meta_config["kind"])
                if progress:
                    print(
                        f"[{protocol}] outer {outer_fold} | meta "
                        f"{meta_index}/{len(meta_names)} {meta_name}"
                    )
                started = perf_counter()

                if kind == "equal_top3":
                    predicted = outer_base_matrix.loc[:, raw_top3].mean(
                        axis=1
                    ).to_numpy(dtype=float)
                    fitted_estimator: Any = None
                    best_params_json = "{}"
                    inner_best_rmse = float("nan")
                elif kind == "mean_all":
                    predicted = outer_base_matrix.loc[:, base_names].mean(
                        axis=1
                    ).to_numpy(dtype=float)
                    fitted_estimator = None
                    best_params_json = "{}"
                    inner_best_rmse = float("nan")
                elif kind == "convex_simplex":
                    fitted_estimator = ConvexStackingRegressor().fit(
                        level1_matrix.loc[:, base_names],
                        y_outer_train,
                    )
                    predicted = fitted_estimator.predict(
                        outer_base_matrix.loc[:, base_names]
                    )
                    best_params_json = "{}"
                    inner_best_rmse = float("nan")
                    coefficient_rows.extend(
                        [
                            {
                                "protocol": protocol,
                                "outer_fold": int(outer_fold),
                                "meta_model": meta_name,
                                "feature": base_name,
                                "value_kind": "convex_weight",
                                "value": float(weight),
                            }
                            for base_name, weight in zip(
                                base_names,
                                fitted_estimator.coef_,
                                strict=True,
                            )
                        ]
                    )
                else:
                    spec = _make_meta_search_spec(
                        meta_name=meta_name,
                        meta_config=meta_config,
                        random_state=random_state,
                    )
                    search = GridSearchCV(
                        estimator=spec.estimator,
                        param_grid=spec.param_grid,
                        scoring="neg_root_mean_squared_error",
                        cv=meta_splits,
                        refit=True,
                        n_jobs=int(runtime.get("search_n_jobs", 1)),
                        return_train_score=False,
                        error_score="raise",
                    )
                    search.fit(z_train, y_outer_train)
                    fitted_estimator = search.best_estimator_
                    predicted = np.asarray(
                        search.predict(z_valid),
                        dtype=float,
                    ).reshape(-1)
                    best_params_json = json.dumps(
                        search.best_params_,
                        sort_keys=True,
                    )
                    inner_best_rmse = float(-search.best_score_)
                    meta_search_rows.extend(
                        _search_rows(
                            search,
                            protocol=protocol,
                            outer_fold=int(outer_fold),
                            stage="meta_tuning",
                            candidate=meta_name,
                            level1_fold=None,
                        )
                    )
                    coefficient_rows.extend(
                        _record_meta_importance(
                            estimator=fitted_estimator,
                            feature_names=list(z_train.columns),
                            protocol=protocol,
                            outer_fold=int(outer_fold),
                            meta_model=meta_name,
                        )
                    )

                metrics = _metric_row(y_outer_valid, predicted)
                elapsed = perf_counter() - started
                outer_metric_rows.append(
                    {
                        "protocol": protocol,
                        "outer_fold": int(outer_fold),
                        "meta_model": meta_name,
                        "meta_kind": kind,
                        "n_train": int(len(outer_train)),
                        "n_validation": int(len(outer_valid)),
                        "n_base_candidates": len(base_names),
                        "n_meta_features": int(z_train.shape[1]),
                        "inner_best_rmse": inner_best_rmse,
                        "best_params_json": best_params_json,
                        "elapsed_seconds": elapsed,
                        **metrics,
                    }
                )
                for row_position, prediction in enumerate(predicted):
                    stack_oof_rows.append(
                        {
                            "protocol": protocol,
                            "outer_fold": int(outer_fold),
                            "meta_model": meta_name,
                            "ID_POLIGONO": str(
                                outer_valid.iloc[row_position][id_column]
                            ),
                            "observed": float(y_outer_valid.iloc[row_position]),
                            "predicted": float(prediction),
                            "residual": float(
                                y_outer_valid.iloc[row_position] - prediction
                            ),
                        }
                    )

    return (
        pd.DataFrame(base_outer_rows),
        pd.DataFrame(meta_training_rows),
        pd.DataFrame(stack_oof_rows),
        pd.DataFrame(base_search_rows),
        pd.DataFrame(meta_search_rows),
        pd.DataFrame(outer_metric_rows),
        pd.DataFrame(coefficient_rows),
        final_catboost_task_type,
    )


def summarize_stacking_results(
    outer_metrics: pd.DataFrame,
    stack_oof: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build pooled protocol metrics and cross-protocol robustness for stackers."""

    protocol_rows: list[dict[str, Any]] = []
    for (protocol, meta_model), subset in stack_oof.groupby(
        ["protocol", "meta_model"],
        observed=True,
        sort=True,
    ):
        metrics = _metric_row(
            subset["observed"],
            subset["predicted"].to_numpy(dtype=float),
        )
        fold_subset = outer_metrics.loc[
            outer_metrics["protocol"].eq(protocol)
            & outer_metrics["meta_model"].eq(meta_model)
        ]
        yield_mean = float(pd.to_numeric(subset["observed"], errors="raise").mean())
        protocol_rows.append(
            {
                "protocol": str(protocol),
                "meta_model": str(meta_model),
                "oof_rmse": metrics["rmse"],
                "oof_rmse_kg_ha": 1_000.0 * metrics["rmse"],
                "oof_nrmse_pct": 100.0 * metrics["rmse"] / yield_mean,
                "oof_mae": metrics["mae"],
                "oof_mae_kg_ha": 1_000.0 * metrics["mae"],
                "oof_r2": metrics["r2"],
                "fold_rmse_mean": float(fold_subset["rmse"].mean()),
                "fold_rmse_std": float(fold_subset["rmse"].std(ddof=1)),
                "inner_best_rmse_mean": float(
                    fold_subset["inner_best_rmse"].mean(skipna=True)
                ),
            }
        )
    protocol_summary = pd.DataFrame(protocol_rows).sort_values(
        ["protocol", "oof_rmse", "meta_model"],
        kind="stable",
    )

    protocols = list(map(str, config["validation"]["outer_protocols"]))
    if len(protocols) != 2:
        raise ValueError("03C.4 robustness summary expects exactly two protocols.")
    state_protocol, grouped_protocol = protocols

    state = protocol_summary.loc[
        protocol_summary["protocol"].eq(state_protocol),
        ["meta_model", "oof_rmse"],
    ].rename(columns={"oof_rmse": "state_oof_rmse"})
    grouped = protocol_summary.loc[
        protocol_summary["protocol"].eq(grouped_protocol),
        ["meta_model", "oof_rmse"],
    ].rename(columns={"oof_rmse": "grouped_oof_rmse"})
    robustness = state.merge(
        grouped,
        on="meta_model",
        how="inner",
        validate="one_to_one",
    )
    robustness["mean_oof_rmse"] = (
        robustness["state_oof_rmse"] + robustness["grouped_oof_rmse"]
    ) / 2.0
    robustness["worst_protocol_rmse"] = robustness[
        ["state_oof_rmse", "grouped_oof_rmse"]
    ].max(axis=1)
    robustness["protocol_gap"] = (
        robustness["state_oof_rmse"] - robustness["grouped_oof_rmse"]
    ).abs()
    robustness["worst_protocol_rmse_kg_ha"] = (
        1_000.0 * robustness["worst_protocol_rmse"]
    )
    robustness = robustness.sort_values(
        ["worst_protocol_rmse", "mean_oof_rmse", "meta_model"],
        kind="stable",
    ).reset_index(drop=True)

    return protocol_summary, robustness
