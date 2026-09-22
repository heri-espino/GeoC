#!/usr/bin/env python
"""Run Checkpoint 05 final global-local stacked mixture experiment."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNet, HuberRegressor, Ridge
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from geocebada.evaluation.checkpoint03c2 import (
    build_competition_representation_specs,
    make_inner_cv_splits,
    make_model_search_spec,
)
from geocebada.evaluation.checkpoint04a import (
    ID_COLUMN,
    haversine_distance_matrix,
)
from geocebada.evaluation.checkpoint04b import (
    build_knn_graph,
    build_static_x_profile,
    generate_target_matched_splits,
    mixed_distance_matrix,
    normalized_distance_matrix,
    predict_graph_laplacian,
    regression_metrics,
    standardized_profile_coordinates,
    temporal_similarity_matrix,
)
from geocebada.evaluation.checkpoint04c import residual_correlation_table
from geocebada.evaluation.checkpoint04d1 import (
    SplitPositions,
    leave_one_split_out_method_selection,
    predict_weighted_local_ridge,
    split_positions_from_membership,
    summarize_nested_predictions,
    summarize_predictions_by_method,
)
from geocebada.evaluation.checkpoint05 import (
    ConvexStackRegressor,
    FittedMetaCandidate,
    SupportMixtureOfExperts,
    candidate_prediction_frame,
    support_descriptors,
)
from geocebada.features import join_agronomic_features, join_empirical_features
from geocebada.paths import find_project_root

ROOT = Path(__file__).resolve().parents[1]


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping in {path}.")
    return payload


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}.")
    return payload


def _git_commit(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _validate_contract(frame: pd.DataFrame, config: dict[str, Any]) -> None:
    identity = config["identity"]
    required = {
        identity["id_column"],
        identity["target_column"],
        identity["split_column"],
        identity["state_column"],
        identity["municipality_column"],
        identity["latitude_column"],
        identity["longitude_column"],
    }
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Checkpoint 05 missing columns: {sorted(missing)}")

    expected = (
        int(config["validation"]["expected_total_rows"]),
        int(config["validation"]["expected_training_rows"]),
        int(config["validation"]["expected_prediction_rows"]),
    )
    actual = (
        len(frame),
        int(frame[identity["split_column"]].eq(identity["train_value"]).sum()),
        int(frame[identity["split_column"]].eq(identity["prediction_value"]).sum()),
    )
    if actual != expected:
        raise ValueError(f"Unexpected parcel counts: {actual}; expected {expected}.")
    if not frame[identity["id_column"]].is_unique:
        raise ValueError("Checkpoint 05 requires unique parcel IDs.")

    train = frame[identity["split_column"]].eq(identity["train_value"])
    target = frame[identity["split_column"]].eq(identity["prediction_value"])
    observed = pd.to_numeric(
        frame.loc[train, identity["target_column"]],
        errors="coerce",
    )
    hidden = pd.to_numeric(
        frame.loc[target, identity["target_column"]],
        errors="coerce",
    )
    if observed.isna().any():
        raise ValueError("All 138 training yields must be observed.")
    if hidden.notna().any():
        raise ValueError("The 59 hidden FIRA yields must remain missing.")


def _embedding_matrix(
    embeddings: pd.DataFrame,
    frame: pd.DataFrame,
    representation: str,
) -> np.ndarray:
    block = embeddings.loc[
        embeddings["representation"].eq(str(representation))
    ].copy()
    pc_columns = [column for column in block.columns if column.startswith("pc")]
    if not pc_columns:
        raise ValueError(f"No PCA columns found for {representation}.")
    block = block.set_index(ID_COLUMN).reindex(frame[ID_COLUMN].astype(str))
    if block[pc_columns].isna().any().any():
        raise ValueError(f"Embedding {representation} does not cover all parcels.")
    return block.loc[:, pc_columns].to_numpy(float)


def _distance_system(
    frame: pd.DataFrame,
    embeddings: pd.DataFrame,
    config: dict[str, Any],
    checkpoint04d1_config: dict[str, Any],
) -> dict[str, Any]:
    identity = config["identity"]
    c1 = _embedding_matrix(embeddings, frame, "C1_agronomic")
    c4 = _embedding_matrix(embeddings, frame, "C4_all_deterministic")
    c4 = StandardScaler().fit_transform(c4)

    latitude = pd.to_numeric(frame[identity["latitude_column"]], errors="raise")
    longitude = pd.to_numeric(frame[identity["longitude_column"]], errors="raise")
    geographic = haversine_distance_matrix(latitude, longitude)
    agronomic = pairwise_distances(c1, metric="euclidean")

    train_positions = np.flatnonzero(
        frame[identity["split_column"]].eq(identity["train_value"]).to_numpy()
    )
    geo_norm, _ = normalized_distance_matrix(
        geographic,
        reference_positions=train_positions,
    )
    agro_norm, _ = normalized_distance_matrix(
        agronomic,
        reference_positions=train_positions,
    )

    distance_topologies: dict[str, np.ndarray] = {}
    for name, weights in checkpoint04d1_config["distance_topologies"].items():
        distance_topologies[str(name)] = mixed_distance_matrix(
            [
                (geo_norm, float(weights["geographic"])),
                (agro_norm, float(weights["agronomic"])),
            ]
        )

    graph_cfg = config["transductive_experts"]["Graph04D"]
    graph_distance = distance_topologies[str(graph_cfg["distance"])]
    graph = build_knn_graph(graph_distance, k=int(graph_cfg["k"]))
    return {
        "c1": c1,
        "c4": c4,
        "geographic": geographic,
        "agronomic": agronomic,
        "distances": distance_topologies,
        "graph": graph,
        "train_positions": train_positions,
    }


def _fresh_confirmation_splits(
    *,
    frame: pd.DataFrame,
    matrices: dict[str, Any],
    temporal_pairs: pd.DataFrame,
    adversarial_scores: pd.DataFrame,
    checkpoint04b_config: dict[str, Any],
    config: dict[str, Any],
    development_splits: list[SplitPositions],
) -> tuple[list[SplitPositions], pd.DataFrame]:
    """Generate a fresh X-only target-matched confirmation bank."""

    confirmation = config["validation"]["confirmation"]
    if not bool(confirmation.get("enabled", True)):
        return [], pd.DataFrame()

    temporal = temporal_similarity_matrix(
        frame,
        temporal_pairs,
        value_column=str(
            checkpoint04b_config["distance"]["temporal_component"]
        ),
    )
    profile = build_static_x_profile(
        frame,
        geographic_distances=matrices["geographic"],
        agronomic_distances=matrices["agronomic"],
        temporal_similarities=temporal,
        adversarial_scores=adversarial_scores,
    )
    coordinates = standardized_profile_coordinates(
        profile,
        columns=checkpoint04b_config["profile_matching"]["columns"],
    )

    desired = int(confirmation["repeats"])
    generated = generate_target_matched_splits(
        frame,
        profile_coordinates=coordinates,
        n_pseudo_targets=int(confirmation["n_pseudo_targets"]),
        repeats=max(desired * 2, desired + 8),
        candidates_per_repeat=int(confirmation["candidates_per_repeat"]),
        temperature=float(confirmation["temperature"]),
        municipality_smoothing=float(
            confirmation["municipality_smoothing"]
        ),
        diversity_penalty=float(confirmation["diversity_penalty"]),
        random_state=int(confirmation["random_state"]),
        verbose=False,
    )

    development_sets = [
        set(map(int, split.query_positions))
        for split in development_splits
    ]
    chosen: list[Any] = []
    chosen_sets: list[set[int]] = []
    for split in generated:
        candidate = set(map(int, split.pseudo_target_positions))
        if any(candidate == previous for previous in development_sets):
            continue
        if any(candidate == previous for previous in chosen_sets):
            continue
        chosen.append(split)
        chosen_sets.append(candidate)
        if len(chosen) == desired:
            break
    if len(chosen) != desired:
        raise RuntimeError(
            "Could not construct the requested number of fresh confirmation splits."
        )

    train_positions = set(map(int, matrices["train_positions"]))
    resolved: list[SplitPositions] = []
    membership_rows: list[dict[str, Any]] = []
    ids = frame[ID_COLUMN].astype(str).to_numpy()
    for repeat, split in enumerate(chosen, start=1):
        query = tuple(sorted(map(int, split.pseudo_target_positions)))
        observed = tuple(sorted(train_positions.difference(query)))
        split_id = f"target_matched_confirm_{repeat:02d}"
        resolved.append(
            SplitPositions(
                split_id=split_id,
                family="target_matched_confirm",
                repeat=repeat,
                observed_positions=observed,
                query_positions=query,
            )
        )
        max_dev_jaccard = 0.0
        query_set = set(query)
        for previous in development_sets:
            union = query_set | previous
            score = len(query_set & previous) / len(union) if union else 0.0
            max_dev_jaccard = max(max_dev_jaccard, float(score))
        for position in sorted(train_positions):
            membership_rows.append(
                {
                    "split_id": split_id,
                    "family": "target_matched_confirm",
                    "repeat": repeat,
                    ID_COLUMN: ids[position],
                    "role": (
                        "pseudo_target"
                        if position in query_set
                        else "pseudo_train"
                    ),
                    "match_objective": float(split.match_objective),
                    "municipality_tv": float(split.municipality_tv),
                    "profile_mean_distance": float(
                        split.profile_mean_distance
                    ),
                    "max_jaccard_vs_development": max_dev_jaccard,
                }
            )

    return resolved, pd.DataFrame(membership_rows)


def _inner_protocol(family: str) -> str:
    if str(family) == "fold_municipality_grouped":
        return "fold_municipality_grouped"
    return "fold_state_stratified"


def _fit_global_expert(
    *,
    joined: pd.DataFrame,
    y: np.ndarray,
    train_positions: np.ndarray,
    query_positions: np.ndarray,
    representation: Any,
    model_name: str,
    model_config: dict[str, Any],
    empirical_config: dict[str, Any],
    family: str,
    tuning_folds: int,
    random_state: int,
    state_column: str,
    municipality_column: str,
    task_type: str,
    devices: str,
    seed_ensemble: list[int],
    search_n_jobs: int,
    return_estimators: bool = False,
) -> tuple[np.ndarray, dict[str, Any], Any]:
    x_all = joined.loc[:, list(representation.features)].apply(
        pd.to_numeric,
        errors="coerce",
    )
    train_positions = np.asarray(train_positions, dtype=int)
    query_positions = np.asarray(query_positions, dtype=int)
    x_train = x_all.iloc[train_positions].reset_index(drop=True)
    y_train = np.asarray(y, dtype=float)[train_positions]
    x_query = x_all.iloc[query_positions]

    train_frame = joined.iloc[train_positions].reset_index(drop=True)
    inner_splits = make_inner_cv_splits(
        train_frame,
        protocol=_inner_protocol(family),
        n_splits=int(tuning_folds),
        random_state=int(random_state),
        state_column=state_column,
        municipality_column=municipality_column,
    )

    search_spec = make_model_search_spec(
        model_name=model_name,
        model_config=model_config,
        representation=representation,
        discovery_config=empirical_config["supervised_discovery"],
        random_state=int(random_state),
        catboost_task_type=str(task_type),
        catboost_devices=str(devices),
    )
    jobs = (
        1
        if search_spec.kind in {"catboost", "extra_trees"}
        else int(search_n_jobs)
    )
    search = GridSearchCV(
        estimator=search_spec.estimator,
        param_grid=search_spec.param_grid,
        scoring="neg_root_mean_squared_error",
        cv=inner_splits,
        refit=True,
        n_jobs=jobs,
        return_train_score=False,
        error_score="raise",
    )
    search.fit(x_train, y_train)
    best_params = dict(search.best_params_)

    if search_spec.kind != "catboost":
        predicted = np.asarray(
            search.best_estimator_.predict(x_query),
            dtype=float,
        ).reshape(-1)
        fitted = search.best_estimator_ if return_estimators else None
        return predicted, best_params, fitted

    predictions: list[np.ndarray] = []
    fitted_models: list[Any] = []
    for seed in seed_ensemble:
        seeded_spec = make_model_search_spec(
            model_name=model_name,
            model_config=model_config,
            representation=representation,
            discovery_config=empirical_config["supervised_discovery"],
            random_state=int(seed),
            catboost_task_type=str(task_type),
            catboost_devices=str(devices),
        )
        estimator = seeded_spec.estimator
        estimator.set_params(**best_params)
        estimator.fit(x_train, y_train)
        predictions.append(
            np.asarray(estimator.predict(x_query), dtype=float).reshape(-1)
        )
        if return_estimators:
            fitted_models.append(estimator)

    return (
        np.mean(np.vstack(predictions), axis=0),
        best_params,
        fitted_models if return_estimators else None,
    )


def _predict_local_graph(
    *,
    matrices: dict[str, Any],
    y: np.ndarray,
    observed_positions: np.ndarray,
    query_positions: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    local_cfg = config["transductive_experts"]["Local04D"]
    graph_cfg = config["transductive_experts"]["Graph04D"]

    local = predict_weighted_local_ridge(
        matrices["c4"],
        matrices["distances"][str(local_cfg["distance"])],
        y,
        observed_positions,
        query_positions,
        k=int(local_cfg["k"]),
        alpha=float(local_cfg["alpha"]),
        distance_power=float(local_cfg["distance_power"]),
    )
    graph = predict_graph_laplacian(
        matrices["graph"],
        y,
        observed_positions,
        query_positions,
        regularization=float(graph_cfg["regularization"]),
    )
    return local, graph


def _fit_base_experts(
    *,
    joined: pd.DataFrame,
    y: np.ndarray,
    train_positions: np.ndarray,
    query_positions: np.ndarray,
    family: str,
    representations: dict[str, Any],
    config: dict[str, Any],
    checkpoint03c2_config: dict[str, Any],
    empirical_config: dict[str, Any],
    matrices: dict[str, Any],
    random_state: int,
    task_type: str,
    return_estimators: bool = False,
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    identity = config["identity"]
    expert_predictions: dict[str, np.ndarray] = {}
    tuning_rows: list[dict[str, Any]] = []
    fitted_experts: dict[str, Any] = {}

    for expert_name, expert_cfg in config["global_experts"].items():
        representation_name = str(expert_cfg["representation"])
        model_name = str(expert_cfg["model"])
        if bool(config.get("runtime", {}).get("progress", True)):
            print(
                f"        base expert {expert_name}: {representation_name} / {model_name} "
                f"(train={len(train_positions)}, query={len(query_positions)})",
                flush=True,
            )
        predicted, best_params, fitted = _fit_global_expert(
            joined=joined,
            y=y,
            train_positions=train_positions,
            query_positions=query_positions,
            representation=representations[representation_name],
            model_name=model_name,
            model_config=checkpoint03c2_config["models"][model_name],
            empirical_config=empirical_config,
            family=family,
            tuning_folds=int(config["validation"]["tuning_inner_folds"]),
            random_state=int(random_state),
            state_column=str(identity["state_column"]),
            municipality_column=str(identity["municipality_column"]),
            task_type=task_type,
            devices=str(config["catboost"]["devices"]),
            seed_ensemble=list(map(int, config["catboost"]["seed_ensemble"])),
            search_n_jobs=int(config["runtime"]["search_n_jobs"]),
            return_estimators=return_estimators,
        )
        expert_predictions[str(expert_name)] = predicted
        tuning_rows.append(
            {
                "expert": str(expert_name),
                "representation": representation_name,
                "model": model_name,
                "best_params": json.dumps(best_params, sort_keys=True),
            }
        )
        if return_estimators:
            fitted_experts[str(expert_name)] = fitted

    local, graph = _predict_local_graph(
        matrices=matrices,
        y=y,
        observed_positions=train_positions,
        query_positions=query_positions,
        config=config,
    )
    expert_predictions["Local04D"] = local
    expert_predictions["Graph04D"] = graph

    support = support_descriptors(
        joined,
        observed_positions=train_positions,
        query_positions=query_positions,
        geographic_distances=matrices["geographic"],
        agronomic_distances=matrices["agronomic"],
        local_predictions=local,
        graph_predictions=graph,
        state_column=str(identity["state_column"]),
        municipality_column=str(identity["municipality_column"]),
    )

    result = pd.DataFrame(expert_predictions)
    result = pd.concat(
        [result.reset_index(drop=True), support.reset_index(drop=True)],
        axis=1,
    )
    positions = np.asarray(query_positions, dtype=int)
    result[ID_COLUMN] = joined.iloc[positions][ID_COLUMN].astype(str).to_numpy()
    result["observed"] = np.asarray(y, dtype=float)[positions]
    result["meta_estado"] = (
        joined.iloc[positions][identity["state_column"]].astype(str).to_numpy()
    )
    result["meta_municipio"] = (
        joined.iloc[positions][identity["municipality_column"]]
        .astype(str)
        .to_numpy()
    )
    return result, tuning_rows, fitted_experts


def _crossfit_base_experts(
    *,
    joined: pd.DataFrame,
    y: np.ndarray,
    visible_positions: np.ndarray,
    family: str,
    representations: dict[str, Any],
    config: dict[str, Any],
    checkpoint03c2_config: dict[str, Any],
    empirical_config: dict[str, Any],
    matrices: dict[str, Any],
    random_state: int,
    task_type: str,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    identity = config["identity"]
    visible_positions = np.asarray(visible_positions, dtype=int)
    expert_columns = list(map(str, config["meta_model"]["expert_columns"]))
    support_columns = list(map(str, config["meta_model"]["support_columns"]))
    accum = {
        column: np.zeros(len(visible_positions), dtype=float)
        for column in [*expert_columns, *support_columns]
    }
    counts = np.zeros(len(visible_positions), dtype=float)
    tuning_rows: list[dict[str, Any]] = []
    local_index = {
        int(position): idx
        for idx, position in enumerate(visible_positions)
    }

    repeats = int(config["validation"]["meta_crossfit_repeats"])
    for repeat in range(repeats):
        visible_frame = joined.iloc[visible_positions].reset_index(drop=True)
        folds = make_inner_cv_splits(
            visible_frame,
            protocol=_inner_protocol(family),
            n_splits=int(config["validation"]["meta_crossfit_folds"]),
            random_state=int(random_state) + 1009 * repeat,
            state_column=str(identity["state_column"]),
            municipality_column=str(identity["municipality_column"]),
        )

        for fold_index, (train_idx, valid_idx) in enumerate(folds, start=1):
            if bool(config.get("runtime", {}).get("progress", True)):
                print(
                    f"      meta crossfit repeat {repeat + 1}/{repeats}, "
                    f"fold {fold_index}/{len(folds)}",
                    flush=True,
                )
            train_positions = visible_positions[np.asarray(train_idx, dtype=int)]
            valid_positions = visible_positions[np.asarray(valid_idx, dtype=int)]
            block, fold_tuning, _ = _fit_base_experts(
                joined=joined,
                y=y,
                train_positions=train_positions,
                query_positions=valid_positions,
                family=family,
                representations=representations,
                config=config,
                checkpoint03c2_config=checkpoint03c2_config,
                empirical_config=empirical_config,
                matrices=matrices,
                random_state=int(random_state) + repeat * 100 + fold_index,
                task_type=task_type,
                return_estimators=False,
            )
            for row in fold_tuning:
                tuning_rows.append(
                    {
                        **row,
                        "crossfit_repeat": repeat,
                        "crossfit_fold": fold_index,
                    }
                )
            for row_index, position in enumerate(valid_positions):
                destination = local_index[int(position)]
                for column in accum:
                    accum[column][destination] += float(
                        block.iloc[row_index][column]
                    )
                counts[destination] += 1.0

    if np.any(counts <= 0):
        raise RuntimeError("Cross-fitting did not cover every visible label.")
    frame = pd.DataFrame(
        {
            column: values / counts
            for column, values in accum.items()
        }
    )
    frame[ID_COLUMN] = (
        joined.iloc[visible_positions][ID_COLUMN].astype(str).to_numpy()
    )
    frame["observed"] = np.asarray(y, dtype=float)[visible_positions]
    frame["meta_estado"] = (
        joined.iloc[visible_positions][identity["state_column"]]
        .astype(str)
        .to_numpy()
    )
    frame["meta_municipio"] = (
        joined.iloc[visible_positions][identity["municipality_column"]]
        .astype(str)
        .to_numpy()
    )
    return frame, tuning_rows


def _add_derived_global_ensembles(
    frame: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    result = frame.copy()
    for name, spec in config["derived_global_ensembles"].items():
        members = list(map(str, spec["members"]))
        weights = np.asarray(list(map(float, spec["weights"])), dtype=float)
        weights = weights / np.sum(weights)
        result[str(name)] = result.loc[:, members].to_numpy(float) @ weights
    return result


def _meta_splits(
    frame: pd.DataFrame,
    family: str,
    config: dict[str, Any],
) -> list[tuple[np.ndarray, np.ndarray]]:
    identity = config["identity"]
    return make_inner_cv_splits(
        frame,
        protocol=_inner_protocol(family),
        n_splits=int(config["validation"]["meta_tuning_folds"]),
        random_state=int(config["validation"]["random_state"]) + 313,
        state_column=str(identity["state_column"]),
        municipality_column=str(identity["municipality_column"]),
    )


def _sklearn_meta_estimator(
    kind: str,
    params: dict[str, Any],
    config: dict[str, Any],
) -> Any:
    if kind == "ridge":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", Ridge(alpha=float(params["alpha"]))),
            ]
        )
    if kind == "elastic_net":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    ElasticNet(
                        alpha=float(params["alpha"]),
                        l1_ratio=float(params["l1_ratio"]),
                        max_iter=100_000,
                    ),
                ),
            ]
        )
    if kind == "huber":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    HuberRegressor(
                        epsilon=float(params["epsilon"]),
                        alpha=float(params["alpha"]),
                        max_iter=5000,
                    ),
                ),
            ]
        )
    if kind == "extra_trees":
        settings = config["meta_model"]["candidates"]["ExtraTreesStack"]
        return ExtraTreesRegressor(
            n_estimators=int(settings["n_estimators"]),
            max_depth=int(params["max_depth"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            max_features=float(params["max_features"]),
            random_state=int(config["validation"]["random_state"]),
            n_jobs=-1,
        )
    if kind == "hist_gradient_boosting":
        settings = config["meta_model"]["candidates"]["HistGBStack"]
        return HistGradientBoostingRegressor(
            max_iter=int(settings["max_iter"]),
            max_leaf_nodes=int(params["max_leaf_nodes"]),
            l2_regularization=float(params["l2_regularization"]),
            learning_rate=float(params["learning_rate"]),
            random_state=int(config["validation"]["random_state"]),
        )
    raise ValueError(f"Unsupported sklearn meta kind: {kind}")


def _meta_param_candidates(
    spec: dict[str, Any],
) -> list[dict[str, Any]]:
    kind = str(spec["kind"])
    if kind == "ridge":
        return [{"alpha": value} for value in spec["alpha_values"]]
    if kind in {"convex", "mixture_of_experts"}:
        return [{"l2": value} for value in spec["l2_values"]]
    return [dict(row) for row in spec["candidates"]]


def _fit_meta_candidate(
    *,
    name: str,
    spec: dict[str, Any],
    train: pd.DataFrame,
    query: pd.DataFrame,
    family: str,
    config: dict[str, Any],
) -> tuple[np.ndarray, FittedMetaCandidate, dict[str, Any]]:
    expert_columns = list(map(str, config["meta_model"]["expert_columns"]))
    support_columns = list(map(str, config["meta_model"]["support_columns"]))
    x_expert = train.loc[:, expert_columns].to_numpy(float)
    q_expert = query.loc[:, expert_columns].to_numpy(float)
    x_support = train.loc[:, support_columns].to_numpy(float)
    q_support = query.loc[:, support_columns].to_numpy(float)
    target = train["observed"].to_numpy(float)
    folds = _meta_splits(train, family, config)
    kind = str(spec["kind"])
    candidates = _meta_param_candidates(spec)

    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        fold_rmses: list[float] = []
        for train_idx, valid_idx in folds:
            train_idx = np.asarray(train_idx, dtype=int)
            valid_idx = np.asarray(valid_idx, dtype=int)
            if kind == "convex":
                estimator = ConvexStackRegressor(l2=float(candidate["l2"]))
                estimator.fit(x_expert[train_idx], target[train_idx])
                predicted = estimator.predict(x_expert[valid_idx])
            elif kind == "mixture_of_experts":
                estimator = SupportMixtureOfExperts(
                    l2=float(candidate["l2"]),
                    max_iter=int(spec["max_iter"]),
                )
                estimator.fit(
                    x_expert[train_idx],
                    x_support[train_idx],
                    target[train_idx],
                )
                predicted = estimator.predict(
                    x_expert[valid_idx],
                    x_support[valid_idx],
                )
            else:
                uses_support = kind in {
                    "extra_trees",
                    "hist_gradient_boosting",
                }
                x_train = (
                    np.column_stack([x_expert, x_support])
                    if uses_support
                    else x_expert
                )
                estimator = _sklearn_meta_estimator(kind, candidate, config)
                estimator.fit(x_train[train_idx], target[train_idx])
                predicted = estimator.predict(x_train[valid_idx])
            fold_rmses.append(
                regression_metrics(target[valid_idx], predicted)["rmse"]
            )
        rows.append(
            {
                "candidate": dict(candidate),
                "cv_rmse": float(np.mean(fold_rmses)),
            }
        )

    rows.sort(
        key=lambda row: (
            row["cv_rmse"],
            json.dumps(row["candidate"], sort_keys=True),
        )
    )
    selected = rows[0]
    params = dict(selected["candidate"])
    uses_support = kind in {
        "extra_trees",
        "hist_gradient_boosting",
        "mixture_of_experts",
    }

    if kind == "convex":
        estimator = ConvexStackRegressor(l2=float(params["l2"]))
        estimator.fit(x_expert, target)
        predicted = estimator.predict(q_expert)
    elif kind == "mixture_of_experts":
        estimator = SupportMixtureOfExperts(
            l2=float(params["l2"]),
            max_iter=int(spec["max_iter"]),
        )
        estimator.fit(x_expert, x_support, target)
        predicted = estimator.predict(q_expert, q_support)
    else:
        train_matrix = (
            np.column_stack([x_expert, x_support])
            if uses_support
            else x_expert
        )
        query_matrix = (
            np.column_stack([q_expert, q_support])
            if uses_support
            else q_expert
        )
        estimator = _sklearn_meta_estimator(kind, params, config)
        estimator.fit(train_matrix, target)
        predicted = estimator.predict(query_matrix)

    fitted = FittedMetaCandidate(
        name=str(name),
        kind=kind,
        estimator=estimator,
        parameters=params,
        uses_support=uses_support,
    )
    detail = {
        "method": str(name),
        "kind": kind,
        "selected_params": json.dumps(params, sort_keys=True),
        "meta_cv_rmse": float(selected["cv_rmse"]),
    }
    return np.asarray(predicted, dtype=float).reshape(-1), fitted, detail


def _predict_all_candidates(
    *,
    meta_train: pd.DataFrame,
    base_query: pd.DataFrame,
    family: str,
    config: dict[str, Any],
    requested_methods: set[str] | None = None,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, FittedMetaCandidate],
    list[dict[str, Any]],
]:
    train = _add_derived_global_ensembles(meta_train, config)
    query = _add_derived_global_ensembles(base_query, config)
    predictions: dict[str, np.ndarray] = {}
    for expert in config["meta_model"]["expert_columns"]:
        predictions[str(expert)] = query[str(expert)].to_numpy(float)

    for name in config["derived_global_ensembles"]:
        predictions[str(name)] = query[str(name)].to_numpy(float)

    for name, spec in config["fixed_blends"].items():
        predictions[str(name)] = (
            float(spec["local_weight"])
            * query["Local04D"].to_numpy(float)
            + float(spec["global_weight"])
            * query[str(spec["global_method"])].to_numpy(float)
        )

    fitted: dict[str, FittedMetaCandidate] = {}
    details: list[dict[str, Any]] = []
    for name, spec in config["meta_model"]["candidates"].items():
        if (
            requested_methods is not None
            and str(name) not in requested_methods
        ):
            continue
        predicted, fitted_model, detail = _fit_meta_candidate(
            name=str(name),
            spec=spec,
            train=train,
            query=query,
            family=family,
            config=config,
        )
        predictions[str(name)] = predicted
        fitted[str(name)] = fitted_model
        details.append(detail)
    return predictions, fitted, details


def _split_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (family, split_id, method), group in predictions.groupby(
        ["family", "split_id", "method"],
        sort=False,
    ):
        rows.append(
            {
                "family": str(family),
                "split_id": str(split_id),
                "method": str(method),
                **regression_metrics(group["observed"], group["predicted"]),
            }
        )
    return pd.DataFrame(rows)


def _promotion_candidates(config: dict[str, Any]) -> list[str]:
    return list(
        dict.fromkeys(
            [
                *map(str, config["meta_model"]["expert_columns"]),
                *map(str, config["derived_global_ensembles"].keys()),
                *map(str, config["fixed_blends"].keys()),
                *map(str, config["meta_model"]["candidates"].keys()),
            ]
        )
    )


def _actual_candidate_table(
    base_query: pd.DataFrame,
    predictions: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for method, values in predictions.items():
        rows.append(
            pd.DataFrame(
                {
                    ID_COLUMN: base_query[ID_COLUMN].astype(str).to_numpy(),
                    "method": str(method),
                    "predicted": np.asarray(values, dtype=float),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def _render_report(report: dict[str, Any], primary: pd.DataFrame) -> str:
    lines = [
        "# Checkpoint 05 — Final global-local mixture",
        "",
        f"Generated: {report['generated_at']}",
        f"Git commit: {report.get('git_commit') or 'unknown'}",
        "",
        "## Decision",
        "",
        f"- incumbent: **{report['incumbent_method']}**",
        f"- best development candidate: **{report['best_development_method']}**",
        f"- promoted final method: **{report['final_method']}**",
        f"- promotion passed: **{report['promotion_passed']}**",
        "- hidden FIRA y used/scored: **no**",
        "",
        "## Target-matched ranking",
        "",
        "| method | rmse_mean | pooled_rmse | pooled_mae | rmse_worst |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in primary.sort_values("rmse_mean").itertuples(index=False):
        lines.append(
            f"| {row.method} | {row.rmse_mean:.6f} | "
            f"{row.pooled_rmse:.6f} | {row.pooled_mae:.6f} | "
            f"{row.rmse_worst:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Split-excluded promotion gate",
            "",
            f"- LOSO selector mean RMSE: **{report['loso_selection']['rmse_mean']:.6f}**",
            f"- LOSO selector pooled RMSE: **{report['loso_selection']['pooled_rmse']:.6f}**",
            f"- incumbent mean RMSE: **{report['incumbent']['rmse_mean']:.6f}**",
            f"- incumbent pooled RMSE: **{report['incumbent']['pooled_rmse']:.6f}**",
            "",
            "The full target-matched table may identify the lowest development score,",
            "but Checkpoint 05 first requires the leave-one-development-split-out",
            "selector to improve both mean and pooled RMSE relative to Local04D.",
            "",
            "## Fresh target-matched confirmation",
            "",
        ]
    )
    confirmation = report["confirmation"]
    if confirmation["executed"]:
        lines.extend(
            [
                f"- candidate: **{confirmation['candidate']}**",
                f"- confirmation passed: **{confirmation['passed']}**",
                (
                    "- candidate mean RMSE: "
                    f"**{confirmation['candidate_metrics']['rmse_mean']:.6f}**"
                ),
                (
                    "- incumbent mean RMSE: "
                    f"**{confirmation['incumbent']['rmse_mean']:.6f}**"
                ),
                (
                    "- candidate pooled RMSE: "
                    f"**{confirmation['candidate_metrics']['pooled_rmse']:.6f}**"
                ),
                (
                    "- incumbent pooled RMSE: "
                    f"**{confirmation['incumbent']['pooled_rmse']:.6f}**"
                ),
            ]
        )
    else:
        lines.append(
            "Fresh confirmation was not executed because the development/LOSO "
            "gate did not produce a justified challenger."
        )
    lines.extend(
        [
            "",
            "A candidate replaces Local04D only if the preselected fixed method",
            "also improves both mean and pooled RMSE on this fresh X-only split bank.",
            "",
            "## Export",
            "",
            "- canonical predictions: reports/checkpoint_05/final_predictions.csv",
            "- model manifest: reports/checkpoint_05/model_manifest.json",
            "- local model bundle: models/final/checkpoint05_model.joblib",
            "",
            "The joblib bundle is intended for reproducibility and the later Python app.",
            "It is ignored by Git and is not automatically committed.",
            "",
        ]
    )
    return "\\n".join(lines)


def run_checkpoint_05(
    root: Path,
    config: dict[str, Any],
    *,
    task_type_override: str | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    inputs = config["inputs"]
    outputs = config["outputs"]
    identity = config["identity"]
    output_dir = root / str(outputs["directory"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[05 1/9] Loading frozen Checkpoint 03/04 artifacts and representations...")
    base = pd.read_csv(root / str(inputs["base_table"]))
    agronomic = pd.read_csv(root / str(inputs["agronomic_table"]))
    empirical = pd.read_csv(root / str(inputs["empirical_table"]))
    joined = join_empirical_features(
        join_agronomic_features(base, agronomic),
        empirical,
    )
    _validate_contract(joined, config)

    manifests = {
        "base": _load_json(root / str(inputs["base_manifest"])),
        "agronomic": _load_json(root / str(inputs["agronomic_manifest"])),
        "empirical": _load_json(root / str(inputs["empirical_manifest"])),
    }
    empirical_config = _load_yaml(root / str(inputs["empirical_config"]))
    checkpoint03c2_config = _load_yaml(
        root / str(inputs["checkpoint03c2_config"])
    )
    checkpoint04d1_config = _load_yaml(
        root / str(inputs["checkpoint04d1_config"])
    )
    checkpoint04b_config = _load_yaml(
        root / str(inputs["checkpoint04b_config"])
    )
    adversarial_scores = pd.read_csv(
        root / str(inputs["adversarial_scores"])
    )
    temporal_pairs = pd.read_csv(
        root / str(inputs["temporal_pairs"])
    )
    representations = build_competition_representation_specs(
        joined=joined,
        manifests=manifests,
        config=checkpoint03c2_config,
        empirical_config=empirical_config,
    )
    membership = pd.read_csv(root / str(inputs["pseudo_membership"]))
    embeddings = pd.read_csv(root / str(inputs["transductive_embeddings"]))
    matrices = _distance_system(
        joined,
        embeddings,
        config,
        checkpoint04d1_config,
    )

    y = pd.to_numeric(
        joined[identity["target_column"]],
        errors="coerce",
    ).to_numpy(float)
    splits = split_positions_from_membership(
        joined,
        membership,
        train_value=str(identity["train_value"]),
        split_column=str(identity["split_column"]),
    )
    families = set(map(str, config["validation"]["families"]))
    splits = [split for split in splits if split.family in families]
    primary_family = str(config["validation"]["primary_family"])
    development_splits = [
        split
        for split in splits
        if split.family == primary_family
    ]
    primary_count = len(development_splits)
    if primary_count != int(config["validation"]["expected_primary_splits"]):
        raise ValueError(
            "Checkpoint 05 primary split count does not match the frozen contract."
        )

    task_type = (
        str(task_type_override).upper()
        if task_type_override
        else str(config["catboost"]["task_type"]).upper()
    )

    partial_predictions = output_dir / "_partial_pseudo_predictions.csv"
    partial_base = output_dir / "_partial_base_oof_predictions.csv"
    partial_details = output_dir / "_partial_meta_selection_details.csv"
    partial_confirmation = output_dir / "_partial_confirmation_predictions.csv"
    prediction_blocks: list[pd.DataFrame] = []
    base_blocks: list[pd.DataFrame] = []
    detail_rows: list[dict[str, Any]] = []
    completed: set[str] = set()
    if resume and partial_predictions.is_file():
        existing = pd.read_csv(partial_predictions)
        prediction_blocks.append(existing)
        completed = set(existing["split_id"].astype(str).unique())
        if partial_base.is_file():
            base_blocks.append(pd.read_csv(partial_base))
        if partial_details.is_file():
            detail_rows.extend(
                pd.read_csv(partial_details).to_dict("records")
            )
        print(f"  resume: {len(completed)} outer splits already complete")

    print(
        f"[05 2/9] Running nested cross-fitted experts on "
        f"{len(splits)} outer splits..."
    )
    start = perf_counter()
    for split_index, split in enumerate(splits, start=1):
        if split.split_id in completed:
            continue
        observed_positions = np.asarray(
            split.observed_positions,
            dtype=int,
        )
        query_positions = np.asarray(split.query_positions, dtype=int)

        meta_train, base_tuning = _crossfit_base_experts(
            joined=joined,
            y=y,
            visible_positions=observed_positions,
            family=split.family,
            representations=representations,
            config=config,
            checkpoint03c2_config=checkpoint03c2_config,
            empirical_config=empirical_config,
            matrices=matrices,
            random_state=(
                int(config["validation"]["random_state"])
                + split_index * 1000
            ),
            task_type=task_type,
        )
        base_query, outer_tuning, _ = _fit_base_experts(
            joined=joined,
            y=y,
            train_positions=observed_positions,
            query_positions=query_positions,
            family=split.family,
            representations=representations,
            config=config,
            checkpoint03c2_config=checkpoint03c2_config,
            empirical_config=empirical_config,
            matrices=matrices,
            random_state=(
                int(config["validation"]["random_state"])
                + split_index * 1000
                + 777
            ),
            task_type=task_type,
            return_estimators=False,
        )
        candidate_predictions, _, meta_details = _predict_all_candidates(
            meta_train=meta_train,
            base_query=base_query,
            family=split.family,
            config=config,
        )
        block = candidate_prediction_frame(
            base_query[ID_COLUMN].astype(str).tolist(),
            base_query["observed"].to_numpy(float),
            candidate_predictions,
            family=split.family,
            split_id=split.split_id,
        )
        prediction_blocks.append(block)

        base_record = meta_train.copy()
        base_record["family"] = split.family
        base_record["split_id"] = split.split_id
        base_blocks.append(base_record)

        for row in [*base_tuning, *outer_tuning]:
            detail_rows.append(
                {
                    "family": split.family,
                    "split_id": split.split_id,
                    "stage": "base_expert",
                    **row,
                }
            )
        for row in meta_details:
            detail_rows.append(
                {
                    "family": split.family,
                    "split_id": split.split_id,
                    "stage": "meta_model",
                    **row,
                }
            )

        if bool(
            config["runtime"].get(
                "checkpoint_every_outer_split",
                True,
            )
        ):
            pd.concat(prediction_blocks, ignore_index=True).to_csv(
                partial_predictions,
                index=False,
            )
            pd.concat(base_blocks, ignore_index=True).to_csv(
                partial_base,
                index=False,
            )
            pd.DataFrame(detail_rows).to_csv(partial_details, index=False)

        elapsed = (perf_counter() - start) / 60.0
        print(
            f"  [{split_index:02d}/{len(splits):02d}] "
            f"{split.family}/{split.split_id} complete "
            f"({elapsed:.1f} min elapsed)",
            flush=True,
        )

    predictions = pd.concat(prediction_blocks, ignore_index=True)
    base_oof = pd.concat(base_blocks, ignore_index=True)
    meta_details_frame = pd.DataFrame(detail_rows)

    print("[05 3/9] Summarizing candidate performance and residual diversity...")
    split_metrics = _split_metrics(predictions)
    protocol_summary = summarize_predictions_by_method(predictions)
    promotion_candidates = _promotion_candidates(config)
    primary = protocol_summary.loc[
        protocol_summary["family"].eq(primary_family)
        & protocol_summary["method"].isin(promotion_candidates)
    ].copy()
    if primary.empty:
        raise RuntimeError("No target-matched promotion candidates were produced.")

    residual_methods = [
        method
        for method in [
            "Local04D",
            "Graph04D",
            "E13_PLS_CatBoost",
            "E123_PLS_Ridge_CatBoost",
            "ConvexStack",
            "MoE_Support",
        ]
        if method in set(predictions["method"].astype(str))
    ]
    correlations = residual_correlation_table(
        predictions,
        family=primary_family,
        methods=residual_methods,
    )

    print("[05 4/9] Running leave-one-development-split-out promotion gate...")
    loso_selection, loso_predictions = leave_one_split_out_method_selection(
        predictions,
        family=primary_family,
        candidate_methods=promotion_candidates,
    )
    loso_summary_dict = summarize_nested_predictions(
        loso_predictions,
        label="Checkpoint05_LOSO_method_selection",
    )
    loso_summary = pd.DataFrame([loso_summary_dict])

    incumbent_method = str(config["scope"]["incumbent_method"])
    incumbent = primary.loc[primary["method"].eq(incumbent_method)]
    if len(incumbent) != 1:
        raise RuntimeError(
            "Could not resolve exactly one Local04D incumbent summary."
        )
    incumbent_row = incumbent.iloc[0]
    best_row = primary.sort_values(
        ["rmse_mean", "pooled_rmse", "method"]
    ).iloc[0]
    best_method = str(best_row["method"])
    preliminary_passed = bool(
        best_method != incumbent_method
        and float(best_row["rmse_mean"])
        < float(incumbent_row["rmse_mean"])
        and float(best_row["pooled_rmse"])
        < float(incumbent_row["pooled_rmse"])
        and float(loso_summary_dict["rmse_mean"])
        < float(incumbent_row["rmse_mean"])
        and float(loso_summary_dict["pooled_rmse"])
        < float(incumbent_row["pooled_rmse"])
    )

    confirmation_membership = pd.DataFrame(
        columns=[
            "split_id",
            "family",
            "repeat",
            ID_COLUMN,
            "role",
            "match_objective",
            "municipality_tv",
            "profile_mean_distance",
            "max_jaccard_vs_development",
        ]
    )
    confirmation_predictions = pd.DataFrame(
        columns=[
            "family",
            "split_id",
            "method",
            ID_COLUMN,
            "observed",
            "predicted",
        ]
    )
    confirmation_summary = pd.DataFrame(
        columns=[
            "family",
            "method",
            "n_splits",
            "n_predictions",
            "rmse_mean",
            "rmse_median",
            "rmse_worst",
            "pooled_rmse",
            "pooled_mae",
            "pooled_r2",
        ]
    )
    confirmation_passed = False

    if preliminary_passed:
        print(
            "[05 5/9] Evaluating the preselected candidate on a fresh "
            "target-matched confirmation bank..."
        )
        confirmation_splits, confirmation_membership = (
            _fresh_confirmation_splits(
                frame=joined,
                matrices=matrices,
                temporal_pairs=temporal_pairs,
                adversarial_scores=adversarial_scores,
                checkpoint04b_config=checkpoint04b_config,
                config=config,
                development_splits=development_splits,
            )
        )

        confirmation_blocks: list[pd.DataFrame] = []
        completed_confirmation: set[str] = set()
        if resume and partial_confirmation.is_file():
            existing_confirmation = pd.read_csv(partial_confirmation)
            confirmation_blocks.append(existing_confirmation)
            completed_confirmation = set(
                existing_confirmation["split_id"].astype(str).unique()
            )
            print(
                "  resume confirmation: "
                f"{len(completed_confirmation)} splits already complete"
            )

        confirm_start = perf_counter()
        for confirm_index, split in enumerate(
            confirmation_splits,
            start=1,
        ):
            if split.split_id in completed_confirmation:
                continue
            observed_positions = np.asarray(
                split.observed_positions,
                dtype=int,
            )
            query_positions = np.asarray(
                split.query_positions,
                dtype=int,
            )
            confirm_meta_train, _ = _crossfit_base_experts(
                joined=joined,
                y=y,
                visible_positions=observed_positions,
                family=split.family,
                representations=representations,
                config=config,
                checkpoint03c2_config=checkpoint03c2_config,
                empirical_config=empirical_config,
                matrices=matrices,
                random_state=(
                    int(config["validation"]["confirmation"]["random_state"])
                    + confirm_index * 1000
                ),
                task_type=task_type,
            )
            confirm_base_query, _, _ = _fit_base_experts(
                joined=joined,
                y=y,
                train_positions=observed_positions,
                query_positions=query_positions,
                family=split.family,
                representations=representations,
                config=config,
                checkpoint03c2_config=checkpoint03c2_config,
                empirical_config=empirical_config,
                matrices=matrices,
                random_state=(
                    int(config["validation"]["confirmation"]["random_state"])
                    + confirm_index * 1000
                    + 777
                ),
                task_type=task_type,
                return_estimators=False,
            )
            confirm_candidates, _, _ = _predict_all_candidates(
                meta_train=confirm_meta_train,
                base_query=confirm_base_query,
                family=split.family,
                config=config,
                requested_methods={best_method},
            )
            selected_predictions = {
                method: values
                for method, values in confirm_candidates.items()
                if method in {incumbent_method, best_method}
            }
            if set(selected_predictions) != {
                incumbent_method,
                best_method,
            }:
                raise RuntimeError(
                    "Fresh confirmation did not produce both the incumbent "
                    "and the preselected candidate."
                )
            confirmation_blocks.append(
                candidate_prediction_frame(
                    confirm_base_query[ID_COLUMN].astype(str).tolist(),
                    confirm_base_query["observed"].to_numpy(float),
                    selected_predictions,
                    family=split.family,
                    split_id=split.split_id,
                )
            )
            pd.concat(
                confirmation_blocks,
                ignore_index=True,
            ).to_csv(partial_confirmation, index=False)
            elapsed = (perf_counter() - confirm_start) / 60.0
            print(
                f"  confirmation [{confirm_index:02d}/"
                f"{len(confirmation_splits):02d}] {split.split_id} "
                f"complete ({elapsed:.1f} min elapsed)",
                flush=True,
            )

        confirmation_predictions = pd.concat(
            confirmation_blocks,
            ignore_index=True,
        )
        confirmation_summary = summarize_predictions_by_method(
            confirmation_predictions
        )
        confirmation_incumbent = confirmation_summary.loc[
            confirmation_summary["method"].eq(incumbent_method)
        ]
        confirmation_candidate = confirmation_summary.loc[
            confirmation_summary["method"].eq(best_method)
        ]
        if len(confirmation_incumbent) != 1 or len(confirmation_candidate) != 1:
            raise RuntimeError(
                "Fresh confirmation summary does not contain exactly one "
                "incumbent and one candidate row."
            )
        confirm_local = confirmation_incumbent.iloc[0]
        confirm_best = confirmation_candidate.iloc[0]
        confirmation_passed = bool(
            float(confirm_best["rmse_mean"])
            < float(confirm_local["rmse_mean"])
            and float(confirm_best["pooled_rmse"])
            < float(confirm_local["pooled_rmse"])
        )
    else:
        print(
            "[05 5/9] Fresh confirmation skipped: the development/LOSO "
            "gate did not justify a challenger."
        )

    promotion_passed = bool(
        preliminary_passed and confirmation_passed
    )
    final_method = best_method if promotion_passed else incumbent_method

    confirmation_payload: dict[str, Any] = {
        "executed": bool(preliminary_passed),
        "passed": bool(confirmation_passed),
        "candidate": best_method,
        "repeats": int(
            config["validation"]["confirmation"]["repeats"]
        ),
    }
    if preliminary_passed:
        confirmation_payload["incumbent"] = {
            "rmse_mean": float(confirm_local["rmse_mean"]),
            "pooled_rmse": float(confirm_local["pooled_rmse"]),
            "pooled_mae": float(confirm_local["pooled_mae"]),
        }
        confirmation_payload["candidate_metrics"] = {
            "rmse_mean": float(confirm_best["rmse_mean"]),
            "pooled_rmse": float(confirm_best["pooled_rmse"]),
            "pooled_mae": float(confirm_best["pooled_mae"]),
        }

    print("[05 6/9] Fitting full-label experts and meta-models for the actual 59...")
    train_positions = np.flatnonzero(
        joined[identity["split_column"]].eq(identity["train_value"]).to_numpy()
    )
    target_positions = np.flatnonzero(
        joined[identity["split_column"]]
        .eq(identity["prediction_value"])
        .to_numpy()
    )
    actual_meta_train, actual_base_tuning = _crossfit_base_experts(
        joined=joined,
        y=y,
        visible_positions=train_positions,
        family=primary_family,
        representations=representations,
        config=config,
        checkpoint03c2_config=checkpoint03c2_config,
        empirical_config=empirical_config,
        matrices=matrices,
        random_state=int(config["validation"]["random_state"]) + 900000,
        task_type=task_type,
    )
    actual_base_query, actual_outer_tuning, fitted_global_experts = (
        _fit_base_experts(
            joined=joined,
            y=y,
            train_positions=train_positions,
            query_positions=target_positions,
            family=primary_family,
            representations=representations,
            config=config,
            checkpoint03c2_config=checkpoint03c2_config,
            empirical_config=empirical_config,
            matrices=matrices,
            random_state=int(config["validation"]["random_state"]) + 999999,
            task_type=task_type,
            return_estimators=True,
        )
    )
    (
        actual_predictions,
        actual_meta_models,
        actual_meta_details,
    ) = _predict_all_candidates(
        meta_train=actual_meta_train,
        base_query=actual_base_query,
        family=primary_family,
        config=config,
    )
    actual_candidates = _actual_candidate_table(
        actual_base_query,
        actual_predictions,
    )

    print("[05 7/9] Verifying exact Local04D/Graph04D provenance against 04F...")
    frozen_final = pd.read_csv(
        root / str(inputs["frozen_final_predictions"])
    )
    frozen_diag = pd.read_csv(
        root / str(inputs["frozen_final_diagnostics"])
    )
    actual_lookup = actual_candidates.pivot(
        index=ID_COLUMN,
        columns="method",
        values="predicted",
    )
    frozen_final = frozen_final.set_index(ID_COLUMN)
    frozen_diag = frozen_diag.set_index(ID_COLUMN)
    local_diff = float(
        np.max(
            np.abs(
                actual_lookup.loc[
                    frozen_final.index,
                    "Local04D",
                ].to_numpy(float)
                - frozen_final[identity["target_column"]].to_numpy(float)
            )
        )
    )
    graph_diff = float(
        np.max(
            np.abs(
                actual_lookup.loc[
                    frozen_diag.index,
                    "Graph04D",
                ].to_numpy(float)
                - frozen_diag["predicted_graph"].to_numpy(float)
            )
        )
    )
    if local_diff > 1.0e-10 or graph_diff > 1.0e-10:
        raise RuntimeError(
            "Checkpoint 05 Local/Graph reconstruction does not reproduce 04F "
            f"(Local diff={local_diff:g}, Graph diff={graph_diff:g})."
        )

    selected_actual = actual_candidates.loc[
        actual_candidates["method"].eq(final_method),
        [ID_COLUMN, "predicted"],
    ].copy()
    selected_actual = selected_actual.rename(
        columns={"predicted": identity["target_column"]}
    ).sort_values(ID_COLUMN).reset_index(drop=True)
    if len(selected_actual) != int(
        config["validation"]["expected_prediction_rows"]
    ):
        raise RuntimeError(
            "Final Checkpoint 05 table does not contain exactly 59 rows."
        )

    print("[05 8/9] Exporting model bundle, manifest and diagnostics...")
    final_diagnostics = actual_base_query.copy()
    for method, values in actual_predictions.items():
        if method not in final_diagnostics.columns:
            final_diagnostics[method] = np.asarray(values, dtype=float)
    final_diagnostics["selected_method"] = final_method
    final_diagnostics["selected_prediction"] = actual_predictions[final_method]

    moe_model = actual_meta_models.get("MoE_Support")
    if moe_model is not None:
        expert_columns = list(map(str, config["meta_model"]["expert_columns"]))
        support_columns = list(map(str, config["meta_model"]["support_columns"]))
        moe_weights = moe_model.estimator.expert_weights(
            actual_base_query.loc[:, support_columns].to_numpy(float)
        )
        for index, expert in enumerate(expert_columns):
            final_diagnostics[f"moe_weight__{expert}"] = moe_weights[:, index]

    selected_meta_model = actual_meta_models.get(final_method)
    model_manifest = {
        "schema_version": int(config["schema_version"]),
        "generated_at": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(root),
        "checkpoint": "05",
        "inference_scope": "fixed_59_competition_targets",
        "target": str(identity["target_column"]),
        "training_rows": int(len(train_positions)),
        "prediction_rows": int(len(target_positions)),
        "final_method": final_method,
        "incumbent_method": incumbent_method,
        "promotion_passed": promotion_passed,
        "development_gate_passed": preliminary_passed,
        "confirmation_gate_passed": confirmation_passed,
        "best_development_method": best_method,
        "confirmation": confirmation_payload,
        "selected_meta_model": (
            None
            if selected_meta_model is None
            else {
                "name": selected_meta_model.name,
                "kind": selected_meta_model.kind,
                "parameters": selected_meta_model.parameters,
                "uses_support": selected_meta_model.uses_support,
            }
        ),
        "expert_columns": list(
            map(str, config["meta_model"]["expert_columns"])
        ),
        "support_columns": list(
            map(str, config["meta_model"]["support_columns"])
        ),
        "catboost_task_type": task_type,
        "hidden_prediction_targets_used": False,
        "actual_hidden_y_scored": False,
        "local04d_reproduction_max_abs_diff": local_diff,
        "graph04d_reproduction_max_abs_diff": graph_diff,
        "bundle_note": (
            "The joblib bundle reproduces the fixed 59-target competition "
            "inference. A generic arbitrary-new-parcel API will be implemented "
            "with the web app."
        ),
    }

    feature_schema = {
        "schema_version": 1,
        "checkpoint": "05",
        "id_column": str(identity["id_column"]),
        "target_column": str(identity["target_column"]),
        "split_column": str(identity["split_column"]),
        "meta_expert_columns": list(
            map(str, config["meta_model"]["expert_columns"])
        ),
        "support_columns": list(
            map(str, config["meta_model"]["support_columns"])
        ),
        "global_experts": {
            str(name): {
                "representation": str(spec["representation"]),
                "model": str(spec["model"]),
                "n_input_features": int(
                    len(
                        representations[
                            str(spec["representation"])
                        ].features
                    )
                ),
                "input_features": list(
                    representations[
                        str(spec["representation"])
                    ].features
                ),
                "uses_fold_local_discovery": bool(
                    representations[
                        str(spec["representation"])
                    ].discovery
                ),
                "discovery_primitive_columns": list(
                    representations[
                        str(spec["representation"])
                    ].primitive_columns
                ),
            }
            for name, spec in config["global_experts"].items()
        },
        "transductive_experts": config["transductive_experts"],
        "derived_global_ensembles": config[
            "derived_global_ensembles"
        ],
        "fixed_blends": config["fixed_blends"],
    }

    bundle = {
        "manifest": model_manifest,
        "feature_schema": feature_schema,
        "final_method": final_method,
        "final_predictions": selected_actual,
        "actual_candidates": actual_candidates,
        "final_diagnostics": final_diagnostics,
        "actual_base_query": actual_base_query,
        "fitted_global_experts": fitted_global_experts,
        "selected_meta_model": selected_meta_model,
        "all_meta_models": actual_meta_models,
        "transductive_expert_config": config["transductive_experts"],
        "training_ids": (
            joined.iloc[train_positions][ID_COLUMN].astype(str).tolist()
        ),
        "target_ids": (
            joined.iloc[target_positions][ID_COLUMN].astype(str).tolist()
        ),
    }
    model_bundle_path = root / str(outputs["model_bundle"])
    model_bundle_path.parent.mkdir(parents=True, exist_ok=True)
    if bool(config["runtime"].get("persist_model_bundle", True)):
        joblib.dump(bundle, model_bundle_path, compress=3)

    print("[05 9/9] Writing final Checkpoint 05 artifacts...")
    base_oof.to_csv(
        output_dir / str(outputs["base_oof_predictions"]),
        index=False,
    )
    predictions.to_csv(
        output_dir / str(outputs["pseudo_predictions"]),
        index=False,
    )
    split_metrics.to_csv(
        output_dir / str(outputs["split_metrics"]),
        index=False,
    )
    protocol_summary.to_csv(
        output_dir / str(outputs["protocol_summary"]),
        index=False,
    )
    correlations.to_csv(
        output_dir / str(outputs["residual_correlation"]),
        index=False,
    )
    meta_details_frame.to_csv(
        output_dir / str(outputs["meta_selection_details"]),
        index=False,
    )
    loso_selection.to_csv(
        output_dir / str(outputs["loso_selection"]),
        index=False,
    )
    loso_predictions.to_csv(
        output_dir / str(outputs["loso_predictions"]),
        index=False,
    )
    loso_summary.to_csv(
        output_dir / str(outputs["loso_summary"]),
        index=False,
    )
    confirmation_membership.to_csv(
        output_dir / str(outputs["confirmation_split_membership"]),
        index=False,
    )
    confirmation_predictions.to_csv(
        output_dir / str(outputs["confirmation_predictions"]),
        index=False,
    )
    confirmation_summary.to_csv(
        output_dir / str(outputs["confirmation_summary"]),
        index=False,
    )
    actual_candidates.to_csv(
        output_dir / str(outputs["actual_candidates"]),
        index=False,
    )
    selected_actual.to_csv(
        output_dir / str(outputs["final_predictions"]),
        index=False,
    )
    final_diagnostics.to_csv(
        output_dir / str(outputs["final_diagnostics"]),
        index=False,
    )
    (output_dir / str(outputs["model_manifest"])).write_text(
        json.dumps(
            model_manifest,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\\n",
        encoding="utf-8",
    )
    (output_dir / str(outputs["feature_schema"])).write_text(
        json.dumps(
            feature_schema,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\\n",
        encoding="utf-8",
    )

    report = {
        **model_manifest,
        "stage": str(config["stage"]["name"]),
        "families": sorted(
            predictions["family"].astype(str).unique().tolist()
        ),
        "n_outer_splits": int(predictions["split_id"].nunique()),
        "n_candidate_methods": int(predictions["method"].nunique()),
        "incumbent": {
            "rmse_mean": float(incumbent_row["rmse_mean"]),
            "pooled_rmse": float(incumbent_row["pooled_rmse"]),
            "pooled_mae": float(incumbent_row["pooled_mae"]),
        },
        "best_development": {
            "method": best_method,
            "rmse_mean": float(best_row["rmse_mean"]),
            "pooled_rmse": float(best_row["pooled_rmse"]),
            "pooled_mae": float(best_row["pooled_mae"]),
        },
        "loso_selection": {
            key: (
                int(value)
                if key in {"n_splits", "n_predictions"}
                else float(value)
            )
            for key, value in loso_summary_dict.items()
            if key != "label"
        },
        "actual_base_tuning_records": (
            len(actual_base_tuning) + len(actual_outer_tuning)
        ),
        "actual_meta_models": actual_meta_details,
    }
    (output_dir / str(outputs["report_json"])).write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
        + "\\n",
        encoding="utf-8",
    )
    (output_dir / str(outputs["report_markdown"])).write_text(
        _render_report(report, primary),
        encoding="utf-8",
    )

    for path in [
        partial_predictions,
        partial_base,
        partial_details,
        partial_confirmation,
    ]:
        if path.is_file():
            path.unlink()

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/checkpoint05.yaml"),
    )
    parser.add_argument(
        "--catboost-task-type",
        choices=("GPU", "CPU"),
        help="Override configured CatBoost task type. GPU is the default.",
    )
    parser.add_argument(
        "--confirm-cpu",
        default="n",
        help="Must be exactly 'y' when explicitly requesting CPU execution.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from split-level partial artifacts after interruption.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    if not (root / "pyproject.toml").is_file():
        root = find_project_root(root)

    config_path = (
        args.config
        if args.config.is_absolute()
        else root / args.config
    )
    config = _load_yaml(config_path)
    task_type = (
        str(args.catboost_task_type).upper()
        if args.catboost_task_type
        else str(config["catboost"]["task_type"]).upper()
    )
    if (
        task_type == "CPU"
        and str(args.confirm_cpu).casefold() != "y"
    ):
        raise SystemExit(
            "CPU execution was requested but not confirmed. Rerun with "
            "--catboost-task-type CPU --confirm-cpu y."
        )

    report = run_checkpoint_05(
        root,
        config,
        task_type_override=args.catboost_task_type,
        resume=bool(args.resume),
    )
    print("Checkpoint 05 final global-local mixture: PASS")
    print(f"Incumbent: {report['incumbent_method']}")
    print(
        "Best development method: "
        f"{report['best_development_method']}"
    )
    print(f"Promotion passed: {report['promotion_passed']}")
    print(f"Final method: {report['final_method']}")
    print("Final table: reports/checkpoint_05/final_predictions.csv")
    print("Model bundle: models/final/checkpoint05_model.joblib")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
