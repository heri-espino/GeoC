"""Checkpoint 03C.2 competition-only nested model-family evaluation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from geocebada.evaluation.checkpoint03c import (
    RepresentationSpec,
    build_layer_feature_catalog,
)
from geocebada.evaluation.parcel_modeling import validate_fixed_fold_assignments
from geocebada.features import FoldLocalExpressionAugmenter


@dataclass(frozen=True)
class ModelSearchSpec:
    """Pipeline plus a small deterministic candidate grid."""

    name: str
    kind: str
    estimator: Pipeline
    param_grid: list[dict[str, list[Any]]]


def build_competition_representation_specs(
    *,
    joined: pd.DataFrame,
    manifests: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
    empirical_config: Mapping[str, Any],
) -> dict[str, RepresentationSpec]:
    """Resolve competition-only 03C.2 representations."""

    if str(config["scope"]["active_track"]) != "competition":
        raise ValueError("Checkpoint 03C.2 must run on the competition track only.")
    if bool(config["scope"].get("clean_track_enabled", False)):
        raise ValueError("Clean track must remain disabled in Checkpoint 03C.2.")

    policy = config["feature_policy"]
    catalog = build_layer_feature_catalog(
        joined=joined,
        manifests=manifests,
        numeric_only=bool(policy.get("numeric_only", True)),
        exclude_sources=tuple(map(str, policy.get("exclude_sources", []))),
        exclude_columns=tuple(map(str, policy.get("exclude_columns", []))),
    )
    track_catalog = catalog.loc[catalog["mode"].isin(["clean", "competition"])].copy()
    layer_by_feature = dict(zip(catalog["feature"], catalog["layer"], strict=True))

    primitive_pool = tuple(
        map(str, empirical_config["supervised_discovery"]["primitive_columns"])
    )

    result: dict[str, RepresentationSpec] = {}
    for name, raw in config["representations"].items():
        layers = tuple(map(str, raw["layers"]))
        selected = track_catalog.loc[
            track_catalog["layer"].isin(layers),
            "feature",
        ].astype(str).tolist()

        discovery = bool(raw.get("discovery", False))
        support_layers = tuple(
            map(str, raw.get("discovery_support_layers", []))
        )
        if discovery and support_layers:
            support = [
                column
                for column in primitive_pool
                if layer_by_feature.get(column) in support_layers
                and column in joined.columns
            ]
            selected.extend(support)

        features = tuple(dict.fromkeys(selected))
        if not features:
            raise ValueError(f"Representation {name!r} resolved to zero features.")

        primitives: tuple[str, ...] = ()
        if discovery:
            feature_set = set(features)
            primitives = tuple(
                column for column in primitive_pool if column in feature_set
            )
            if len(primitives) < 2:
                raise ValueError(
                    f"Representation {name!r} has fewer than two discovery primitives."
                )

        result[str(name)] = RepresentationSpec(
            name=str(name),
            track="competition",
            features=features,
            layers=layers,
            discovery=discovery,
            primitive_columns=primitives,
            description=str(raw.get("description", "")),
        )

    return result


def make_inner_cv_splits(
    outer_training: pd.DataFrame,
    *,
    protocol: str,
    n_splits: int,
    random_state: int,
    state_column: str,
    municipality_column: str,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create leakage-safe inner splits aligned with the outer protocol."""

    n_rows = len(outer_training)
    dummy = np.zeros(n_rows)

    if protocol == "fold_state_stratified":
        labels = outer_training[state_column].astype(str).to_numpy()
        counts = pd.Series(labels).value_counts()
        if counts.min() < n_splits:
            raise ValueError(
                "Not enough rows per state for inner stratification: "
                f"{counts.to_dict()}"
            )
        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        )
        return list(splitter.split(dummy, labels))

    if protocol == "fold_municipality_grouped":
        groups = (
            outer_training[state_column].astype(str)
            + "|"
            + outer_training[municipality_column].astype(str)
        ).to_numpy()
        n_groups = len(np.unique(groups))
        if n_groups < n_splits:
            raise ValueError(
                f"Need at least {n_splits} municipality groups, found {n_groups}."
            )
        splitter = GroupKFold(n_splits=n_splits)
        return list(splitter.split(dummy, groups=groups))

    raise ValueError(f"Unknown outer protocol: {protocol}")


def _candidate_grid(
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


def _base_steps(
    *,
    representation: RepresentationSpec,
    discovery_config: Mapping[str, Any],
) -> list[tuple[str, Any]]:
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
    steps.append(
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                add_indicator=True,
                keep_empty_features=True,
            ),
        )
    )
    return steps


def make_model_search_spec(
    *,
    model_name: str,
    model_config: Mapping[str, Any],
    representation: RepresentationSpec,
    discovery_config: Mapping[str, Any],
    random_state: int,
    catboost_task_type: str,
    catboost_devices: str,
) -> ModelSearchSpec:
    """Construct one 03C.2 estimator and its disciplined candidate grid."""

    kind = str(model_config["kind"])
    candidates = model_config["candidates"]
    steps = _base_steps(
        representation=representation,
        discovery_config=discovery_config,
    )

    if kind == "ridge":
        steps.extend([("scaler", StandardScaler()), ("model", Ridge())])
        grid = _candidate_grid(candidates, {"alpha": "model__alpha"})

    elif kind == "elastic_net":
        steps.extend(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    ElasticNet(
                        max_iter=50_000,
                        tol=1.0e-4,
                    ),
                ),
            ]
        )
        grid = _candidate_grid(
            candidates,
            {
                "alpha": "model__alpha",
                "l1_ratio": "model__l1_ratio",
            },
        )

    elif kind == "extra_trees":
        fixed = model_config.get("fixed", {})
        steps.append(
            (
                "model",
                ExtraTreesRegressor(
                    n_estimators=int(fixed.get("n_estimators", 400)),
                    max_depth=fixed.get("max_depth"),
                    random_state=random_state,
                    n_jobs=-1,
                ),
            )
        )
        grid = _candidate_grid(
            candidates,
            {
                "min_samples_leaf": "model__min_samples_leaf",
                "max_features": "model__max_features",
            },
        )

    elif kind == "catboost":
        try:
            from catboost import CatBoostRegressor
        except ImportError as exc:
            raise ImportError(
                "CatBoost is required for 03C.2. Install with "
                "python -m pip install -e '.[models]'."
            ) from exc

        fixed = dict(model_config.get("fixed", {}))
        kwargs: dict[str, Any] = {
            "iterations": int(fixed.get("iterations", 400)),
            "loss_function": str(fixed.get("loss_function", "RMSE")),
            "verbose": bool(fixed.get("verbose", False)),
            "allow_writing_files": bool(
                fixed.get("allow_writing_files", False)
            ),
            "random_seed": random_state,
            "task_type": catboost_task_type,
            "thread_count": -1,
        }
        if catboost_task_type == "GPU":
            kwargs["devices"] = catboost_devices

        steps.append(("model", CatBoostRegressor(**kwargs)))
        grid = _candidate_grid(
            candidates,
            {
                "depth": "model__depth",
                "learning_rate": "model__learning_rate",
                "l2_leaf_reg": "model__l2_leaf_reg",
            },
        )

    elif kind == "pls":
        steps.extend(
            [
                ("scaler", StandardScaler()),
                ("model", PLSRegression(scale=False, max_iter=1_000)),
            ]
        )
        grid = _candidate_grid(
            candidates,
            {"n_components": "model__n_components"},
        )

    elif kind in {"pca_kernel_rbf", "pca_kernel_poly2"}:
        kernel = "rbf" if kind == "pca_kernel_rbf" else "polynomial"
        steps.extend(
            [
                ("scaler", StandardScaler()),
                (
                    "pca",
                    PCA(
                        svd_solver="full",
                        random_state=random_state,
                    ),
                ),
                (
                    "model",
                    KernelRidge(
                        kernel=kernel,
                        degree=2,
                        coef0=1.0,
                    ),
                ),
            ]
        )
        grid = _candidate_grid(
            candidates,
            {
                "n_components": "pca__n_components",
                "alpha": "model__alpha",
                "gamma": "model__gamma",
            },
        )

    else:
        raise ValueError(f"Unsupported Checkpoint 03C.2 model kind: {kind}")

    return ModelSearchSpec(
        name=model_name,
        kind=kind,
        estimator=Pipeline(steps),
        param_grid=grid,
    )


def _metric_row(
    observed: pd.Series,
    predicted: np.ndarray,
) -> dict[str, float]:
    y_true = observed.to_numpy(dtype=float)
    y_pred = np.asarray(predicted, dtype=float).reshape(-1)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _strip_param_prefixes(params: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key.split("__", 1)[-1]: value
        for key, value in params.items()
    }


def _gpu_failure(error: Exception) -> bool:
    message = str(error).casefold()
    tokens = ("cuda", "gpu", "driver", "device", "nvidia")
    return any(token in message for token in tokens)


def evaluate_nested_model_families(
    *,
    joined: pd.DataFrame,
    folds: pd.DataFrame,
    representations: Mapping[str, RepresentationSpec],
    config: Mapping[str, Any],
    empirical_config: Mapping[str, Any],
    selected_models: Sequence[str] | None = None,
    selected_representations: Sequence[str] | None = None,
    catboost_task_type_override: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """Run nested tuning under both frozen outer validation protocols."""

    identity = config["identity"]
    validation = config["validation"]
    runtime = config["runtime"]

    id_column = str(identity["id_column"])
    target_column = str(identity["target_column"])
    split_column = str(identity["split_column"])
    train_value = str(identity["train_value"])
    state_column = str(identity["state_column"])
    municipality_column = str(identity["municipality_column"])
    protocols = tuple(map(str, validation["outer_protocols"]))

    validate_fixed_fold_assignments(
        joined,
        folds,
        id_column=id_column,
        split_column=split_column,
        train_value=train_value,
        fold_columns=protocols,
    )

    training = joined.loc[joined[split_column].eq(train_value)].copy()
    training[id_column] = training[id_column].astype(str)
    fold_columns = folds[[id_column, *protocols]].copy()
    fold_columns[id_column] = fold_columns[id_column].astype(str)
    training = training.merge(
        fold_columns,
        on=id_column,
        how="left",
        validate="one_to_one",
    )

    y = pd.to_numeric(training[target_column], errors="raise").astype(float)
    if y.isna().any():
        raise ValueError("Training target contains missing values.")

    model_names = (
        list(map(str, selected_models))
        if selected_models
        else list(map(str, config["models"].keys()))
    )
    representation_names = (
        list(map(str, selected_representations))
        if selected_representations
        else list(representations.keys())
    )
    missing_models = sorted(set(model_names) - set(config["models"]))
    missing_representations = sorted(
        set(representation_names) - set(representations)
    )
    if missing_models:
        raise KeyError(f"Unknown selected model(s): {missing_models}")
    if missing_representations:
        raise KeyError(
            f"Unknown selected representation(s): {missing_representations}"
        )

    discovery_config = empirical_config["supervised_discovery"]
    random_state = int(validation["random_state"])
    inner_folds = int(validation["inner_folds"])
    search_n_jobs = int(runtime.get("search_n_jobs", 1))
    catboost_task_type = (
        str(catboost_task_type_override).upper()
        if catboost_task_type_override
        else str(runtime.get("catboost_task_type", "GPU")).upper()
    )
    catboost_devices = str(runtime.get("catboost_devices", "0"))
    allow_cpu_fallback = bool(
        runtime.get("catboost_allow_cpu_fallback", True)
    )
    progress = bool(runtime.get("progress", True))

    outer_rows: list[dict[str, Any]] = []
    oof_rows: list[dict[str, Any]] = []
    inner_rows: list[dict[str, Any]] = []
    discovery_rows: list[dict[str, Any]] = []

    for representation_name in representation_names:
        representation = representations[representation_name]
        x = training.loc[:, list(representation.features)].apply(
            pd.to_numeric,
            errors="coerce",
        )

        for protocol in protocols:
            fold_values = sorted(
                pd.to_numeric(training[protocol], errors="raise")
                .astype(int)
                .unique()
            )
            for fold in fold_values:
                validation_mask = training[protocol].eq(fold)
                train_mask = ~validation_mask
                outer_train = training.loc[train_mask].reset_index(drop=True)
                x_train = x.loc[train_mask].reset_index(drop=True)
                y_train = y.loc[train_mask].reset_index(drop=True)
                x_valid = x.loc[validation_mask]
                y_valid = y.loc[validation_mask]

                inner_splits = make_inner_cv_splits(
                    outer_train,
                    protocol=protocol,
                    n_splits=inner_folds,
                    random_state=random_state + int(fold),
                    state_column=state_column,
                    municipality_column=municipality_column,
                )

                for model_name in model_names:
                    model_config = config["models"][model_name]
                    actual_task_type = catboost_task_type

                    def build_search(task_type: str) -> GridSearchCV:
                        spec = make_model_search_spec(
                            model_name=model_name,
                            model_config=model_config,
                            representation=representation,
                            discovery_config=discovery_config,
                            random_state=random_state,
                            catboost_task_type=task_type,
                            catboost_devices=catboost_devices,
                        )
                        jobs = (
                            1
                            if spec.kind == "catboost" and task_type == "GPU"
                            else search_n_jobs
                        )
                        return GridSearchCV(
                            estimator=spec.estimator,
                            param_grid=spec.param_grid,
                            scoring="neg_root_mean_squared_error",
                            cv=inner_splits,
                            refit=True,
                            n_jobs=jobs,
                            return_train_score=False,
                            error_score="raise",
                        )

                    if progress:
                        print(
                            f"[{protocol}] {representation_name} / "
                            f"{model_name} / fold {fold}"
                        )

                    started = perf_counter()
                    search = build_search(actual_task_type)
                    try:
                        search.fit(x_train, y_train)
                    except Exception as exc:
                        is_catboost = str(model_config["kind"]) == "catboost"
                        if (
                            is_catboost
                            and actual_task_type == "GPU"
                            and allow_cpu_fallback
                            and _gpu_failure(exc)
                        ):
                            actual_task_type = "CPU"
                            if progress:
                                print(
                                    "  CatBoost GPU unavailable; retrying on CPU."
                                )
                            search = build_search(actual_task_type)
                            search.fit(x_train, y_train)
                            catboost_task_type = "CPU"
                        else:
                            raise

                    predicted = np.asarray(
                        search.predict(x_valid),
                        dtype=float,
                    ).reshape(-1)
                    elapsed = perf_counter() - started
                    metrics = _metric_row(y_valid, predicted)
                    best_params = _strip_param_prefixes(search.best_params_)

                    best_estimator = search.best_estimator_
                    n_discovered = 0
                    if representation.discovery:
                        augmenter = best_estimator.named_steps["discover"]
                        discovery_report = augmenter.discovery_report()
                        n_discovered = len(discovery_report)
                        for row in discovery_report.itertuples(index=False):
                            discovery_rows.append(
                                {
                                    "protocol": protocol,
                                    "representation": representation_name,
                                    "model": model_name,
                                    "fold": int(fold),
                                    "rank": int(row.rank),
                                    "feature": str(row.feature),
                                    "left": str(row.left),
                                    "right": str(row.right),
                                    "operation": str(row.operation),
                                    "formula": str(row.formula),
                                    "abs_spearman_train": float(
                                        row.abs_spearman_train
                                    ),
                                }
                            )

                    outer_rows.append(
                        {
                            "protocol": protocol,
                            "representation": representation_name,
                            "model": model_name,
                            "fold": int(fold),
                            "n_features": len(representation.features),
                            "n_primitives": len(
                                representation.primitive_columns
                            ),
                            "n_discovered": n_discovered,
                            "n_train": int(train_mask.sum()),
                            "n_validation": int(validation_mask.sum()),
                            "inner_best_rmse": float(-search.best_score_),
                            "best_params_json": json.dumps(
                                best_params,
                                sort_keys=True,
                            ),
                            "catboost_task_type": (
                                actual_task_type
                                if str(model_config["kind"]) == "catboost"
                                else ""
                            ),
                            "elapsed_seconds": float(elapsed),
                            **metrics,
                        }
                    )

                    for candidate_index, params in enumerate(
                        search.cv_results_["params"]
                    ):
                        inner_rows.append(
                            {
                                "protocol": protocol,
                                "representation": representation_name,
                                "model": model_name,
                                "fold": int(fold),
                                "candidate": int(candidate_index),
                                "params_json": json.dumps(
                                    _strip_param_prefixes(params),
                                    sort_keys=True,
                                ),
                                "inner_rmse_mean": float(
                                    -search.cv_results_[
                                        "mean_test_score"
                                    ][candidate_index]
                                ),
                                "inner_rmse_std": float(
                                    search.cv_results_[
                                        "std_test_score"
                                    ][candidate_index]
                                ),
                                "rank": int(
                                    search.cv_results_[
                                        "rank_test_score"
                                    ][candidate_index]
                                ),
                            }
                        )

                    ids = training.loc[validation_mask, id_column].astype(str)
                    for parcel_id, truth, pred in zip(
                        ids,
                        y_valid.to_numpy(dtype=float),
                        predicted,
                        strict=True,
                    ):
                        oof_rows.append(
                            {
                                "protocol": protocol,
                                "representation": representation_name,
                                "model": model_name,
                                "fold": int(fold),
                                id_column: parcel_id,
                                "observed": float(truth),
                                "predicted": float(pred),
                                "residual": float(truth - pred),
                            }
                        )

    outer = pd.DataFrame(outer_rows)
    oof = pd.DataFrame(oof_rows)
    inner = pd.DataFrame(inner_rows)
    discovery = pd.DataFrame(discovery_rows)
    validate_nested_oof_coverage(
        oof,
        expected_ids=training[id_column],
        id_column=id_column,
    )
    return outer, oof, inner, discovery, catboost_task_type


def validate_nested_oof_coverage(
    oof: pd.DataFrame,
    *,
    expected_ids: pd.Series,
    id_column: str,
) -> None:
    """Require one OOF prediction per parcel for every protocol/representation/model."""

    expected = set(expected_ids.astype(str))
    grouping = ["protocol", "representation", "model"]
    for key, group in oof.groupby(grouping, sort=False):
        observed = set(group[id_column].astype(str))
        if observed != expected:
            raise ValueError(
                f"OOF coverage mismatch for {key}; "
                f"missing={sorted(expected-observed)[:5]}, "
                f"extra={sorted(observed-expected)[:5]}"
            )
        if group[id_column].duplicated().any():
            raise ValueError(f"Duplicate OOF parcel prediction for {key}.")


def summarize_nested_results(
    outer: pd.DataFrame,
    oof: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build protocol-specific and cross-protocol robustness summaries."""

    grouping = ["protocol", "representation", "model"]
    fold_summary = (
        outer.groupby(grouping, as_index=False)
        .agg(
            n_features=("n_features", "first"),
            n_primitives=("n_primitives", "first"),
            n_discovered=("n_discovered", "max"),
            fold_rmse_mean=("rmse", "mean"),
            fold_rmse_std=("rmse", "std"),
            fold_mae_mean=("mae", "mean"),
            fold_r2_mean=("r2", "mean"),
            inner_best_rmse_mean=("inner_best_rmse", "mean"),
            elapsed_seconds=("elapsed_seconds", "sum"),
        )
    )

    oof_rows: list[dict[str, Any]] = []
    for key, group in oof.groupby(grouping, sort=False):
        protocol, representation, model = key
        observed = group["observed"].to_numpy(dtype=float)
        predicted = group["predicted"].to_numpy(dtype=float)
        oof_rows.append(
            {
                "protocol": protocol,
                "representation": representation,
                "model": model,
                "oof_rows": len(group),
                "oof_rmse": float(
                    np.sqrt(mean_squared_error(observed, predicted))
                ),
                "oof_mae": float(
                    mean_absolute_error(observed, predicted)
                ),
                "oof_r2": float(r2_score(observed, predicted)),
            }
        )

    protocol_summary = fold_summary.merge(
        pd.DataFrame(oof_rows),
        on=grouping,
        how="left",
        validate="one_to_one",
    ).sort_values(
        ["protocol", "oof_rmse", "representation", "model"],
        kind="stable",
    )

    pivot = protocol_summary.pivot_table(
        index=["representation", "model"],
        columns="protocol",
        values="oof_rmse",
        aggfunc="first",
    )
    required = {
        "fold_state_stratified",
        "fold_municipality_grouped",
    }
    if not required <= set(pivot.columns):
        raise ValueError(
            f"Robustness summary requires both protocols, got {list(pivot.columns)}"
        )

    robustness = pivot.reset_index().rename(
        columns={
            "fold_state_stratified": "state_oof_rmse",
            "fold_municipality_grouped": "grouped_oof_rmse",
        }
    )
    robustness["mean_oof_rmse"] = (
        robustness["state_oof_rmse"]
        + robustness["grouped_oof_rmse"]
    ) / 2.0
    robustness["worst_protocol_rmse"] = robustness[
        ["state_oof_rmse", "grouped_oof_rmse"]
    ].max(axis=1)
    robustness["protocol_gap"] = (
        robustness["grouped_oof_rmse"]
        - robustness["state_oof_rmse"]
    )
    robustness = robustness.sort_values(
        [
            "worst_protocol_rmse",
            "mean_oof_rmse",
            "grouped_oof_rmse",
            "state_oof_rmse",
        ],
        kind="stable",
    ).reset_index(drop=True)

    return protocol_summary.reset_index(drop=True), robustness
