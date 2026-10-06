"""Checkpoint 06D X-only regime discovery.

Cluster selection is target-free: all 197 parcel covariates may define the
transductive geometry, but no yield or residual may influence the choice of PCA
variance, clustering algorithm, K, or random seed.

Only after one X-only regime solution is frozen do we characterize yield and
honest residual behavior among the 138 labeled parcels.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from geocebada.evaluation.checkpoint06 import ID_COLUMN, _numeric_feature_families


def _x_pipeline(variance: float) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", RobustScaler()),
            ("pca", PCA(n_components=float(variance), svd_solver="full")),
        ]
    )


def _fit_clusterer(
    x: np.ndarray,
    *,
    algorithm: str,
    k: int,
    random_state: int,
) -> Any:
    if algorithm == "kmeans":
        model = KMeans(
            n_clusters=int(k),
            n_init=25,
            random_state=int(random_state),
        )
    elif algorithm == "gmm":
        model = GaussianMixture(
            n_components=int(k),
            covariance_type="full",
            n_init=10,
            reg_covar=1.0e-6,
            random_state=int(random_state),
        )
    else:
        raise ValueError(f"Unsupported 06D clustering algorithm: {algorithm}")
    model.fit(x)
    return model


def _predict_clusterer(model: Any, x: np.ndarray) -> np.ndarray:
    return np.asarray(model.predict(x), dtype=int)


def _cluster_balance(labels: np.ndarray) -> tuple[int, float]:
    counts = pd.Series(labels).value_counts()
    minimum = int(counts.min())
    return minimum, float(minimum / len(labels))


def discover_x_regimes(
    agronomic: pd.DataFrame,
    manifest: Mapping[str, Any],
    *,
    pca_variances: Sequence[float],
    algorithms: Sequence[str],
    k_values: Sequence[int],
    seeds: Sequence[int],
    subsample_repeats: int,
    subsample_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Evaluate and select X-only regime candidates without reading y."""

    families = _numeric_feature_families(agronomic, manifest)
    columns = families["all_agronomic"]
    x_raw = agronomic[columns]
    ids = agronomic[ID_COLUMN].astype(str).to_numpy()
    primary_seed = int(seeds[0])
    candidate_rows: list[dict[str, Any]] = []
    assignments: list[dict[str, Any]] = []
    fitted_payload: dict[tuple[float, str, int], tuple[np.ndarray, np.ndarray, Pipeline]] = {}

    for variance in pca_variances:
        pipeline = _x_pipeline(float(variance))
        x = np.asarray(pipeline.fit_transform(x_raw), dtype=float)
        for algorithm in algorithms:
            for k in k_values:
                key = (float(variance), str(algorithm), int(k))
                primary = _fit_clusterer(
                    x,
                    algorithm=str(algorithm),
                    k=int(k),
                    random_state=primary_seed,
                )
                labels = _predict_clusterer(primary, x)
                minimum, minimum_fraction = _cluster_balance(labels)
                silhouette = float(silhouette_score(x, labels))

                seed_aris: list[float] = []
                for seed in seeds[1:]:
                    model = _fit_clusterer(
                        x,
                        algorithm=str(algorithm),
                        k=int(k),
                        random_state=int(seed),
                    )
                    seed_aris.append(
                        float(adjusted_rand_score(labels, _predict_clusterer(model, x)))
                    )

                rng = np.random.default_rng(primary_seed + int(k) * 100 + int(variance * 100))
                subsample_aris: list[float] = []
                sample_size = max(int(k) * 8, int(round(float(subsample_fraction) * len(x))))
                sample_size = min(sample_size, len(x) - 1)
                for repeat in range(int(subsample_repeats)):
                    index = np.sort(rng.choice(len(x), size=sample_size, replace=False))
                    model = _fit_clusterer(
                        x[index],
                        algorithm=str(algorithm),
                        k=int(k),
                        random_state=primary_seed + 1000 + repeat,
                    )
                    subsample_aris.append(
                        float(adjusted_rand_score(labels, _predict_clusterer(model, x)))
                    )

                seed_stability = float(np.mean(seed_aris)) if seed_aris else 1.0
                subsample_stability = (
                    float(np.mean(subsample_aris)) if subsample_aris else np.nan
                )
                candidate_rows.append(
                    {
                        "pca_variance": float(variance),
                        "algorithm": str(algorithm),
                        "k": int(k),
                        "n_components": int(x.shape[1]),
                        "silhouette": silhouette,
                        "min_cluster_size": minimum,
                        "min_cluster_fraction": minimum_fraction,
                        "seed_ari_mean": seed_stability,
                        "seed_ari_min": float(np.min(seed_aris)) if seed_aris else 1.0,
                        "subsample_ari_mean": subsample_stability,
                        "subsample_ari_min": (
                            float(np.min(subsample_aris)) if subsample_aris else np.nan
                        ),
                    }
                )
                fitted_payload[key] = (x, labels, pipeline)

    candidates = pd.DataFrame(candidate_rows)
    return candidates, pd.DataFrame(assignments), {
        "feature_columns": columns,
        "ids": ids,
        "fitted_payload": fitted_payload,
    }


def select_x_only_regime(
    candidates: pd.DataFrame,
    *,
    min_cluster_fraction: float,
    min_seed_ari: float,
    min_subsample_ari: float,
) -> dict[str, Any]:
    """Freeze one clustering candidate using only X-derived diagnostics."""

    table = candidates.copy()
    table["x_eligible"] = (
        table["min_cluster_fraction"].ge(float(min_cluster_fraction))
        & table["seed_ari_mean"].ge(float(min_seed_ari))
        & table["subsample_ari_mean"].ge(float(min_subsample_ari))
    )
    eligible = table.loc[table["x_eligible"]].copy()
    if eligible.empty:
        ranked = table.sort_values(
            ["silhouette", "seed_ari_mean", "subsample_ari_mean"],
            ascending=False,
        )
        selected = ranked.iloc[0]
        status = "NO_ELIGIBLE_X_CLUSTERING"
    else:
        ranked = eligible.sort_values(
            ["silhouette", "seed_ari_mean", "subsample_ari_mean"],
            ascending=False,
        )
        selected = ranked.iloc[0]
        status = "X_CLUSTERING_SELECTED"

    return {
        "selection_status": status,
        "pca_variance": float(selected["pca_variance"]),
        "algorithm": str(selected["algorithm"]),
        "k": int(selected["k"]),
        "n_components": int(selected["n_components"]),
        "silhouette": float(selected["silhouette"]),
        "min_cluster_size": int(selected["min_cluster_size"]),
        "min_cluster_fraction": float(selected["min_cluster_fraction"]),
        "seed_ari_mean": float(selected["seed_ari_mean"]),
        "subsample_ari_mean": float(selected["subsample_ari_mean"]),
        "x_eligible": bool(selected["x_eligible"]),
    }


def selected_regime_assignments(
    agronomic: pd.DataFrame,
    manifest: Mapping[str, Any],
    selection: Mapping[str, Any],
    *,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit the frozen X-only clustering to all 197 X and return assignments/PCA coordinates."""

    columns = _numeric_feature_families(agronomic, manifest)["all_agronomic"]
    pipeline = _x_pipeline(float(selection["pca_variance"]))
    x = np.asarray(pipeline.fit_transform(agronomic[columns]), dtype=float)
    model = _fit_clusterer(
        x,
        algorithm=str(selection["algorithm"]),
        k=int(selection["k"]),
        random_state=int(random_state),
    )
    labels = _predict_clusterer(model, x)

    assignments = pd.DataFrame(
        {
            ID_COLUMN: agronomic[ID_COLUMN].astype(str),
            "cluster": labels.astype(int),
        }
    )
    pca = pd.DataFrame({ID_COLUMN: agronomic[ID_COLUMN].astype(str)})
    for index in range(min(8, x.shape[1])):
        pca[f"PC{index + 1}"] = x[:, index]
    return assignments, pca


def _eta_squared(values: np.ndarray, labels: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    labels = np.asarray(labels)
    overall = float(np.mean(values))
    total = float(np.sum((values - overall) ** 2))
    if total <= 1.0e-15:
        return 0.0
    between = 0.0
    for label in np.unique(labels):
        group = values[labels == label]
        between += len(group) * (float(np.mean(group)) - overall) ** 2
    return float(between / total)


def _permutation_eta_pvalue(
    values: np.ndarray,
    labels: np.ndarray,
    *,
    repeats: int,
    random_state: int,
) -> tuple[float, float]:
    observed = _eta_squared(values, labels)
    rng = np.random.default_rng(int(random_state))
    null = np.empty(int(repeats), dtype=float)
    for index in range(int(repeats)):
        null[index] = _eta_squared(rng.permutation(values), labels)
    pvalue = float((1.0 + np.sum(null >= observed)) / (1.0 + len(null)))
    return observed, pvalue


def characterize_selected_regimes(
    assignments: pd.DataFrame,
    base_frame: pd.DataFrame,
    parcel_residual_summary: pd.DataFrame,
    *,
    split_column: str,
    train_value: str,
    prediction_value: str,
    target_column: str,
    state_column: str | None,
    municipality_column: str | None,
    permutation_repeats: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Characterize frozen X-only clusters using labeled y/residuals only after selection."""

    base = base_frame.copy()
    base[ID_COLUMN] = base[ID_COLUMN].astype(str)
    merged = assignments.merge(base, on=ID_COLUMN, how="left", validate="one_to_one")

    composition_rows: list[dict[str, Any]] = []
    for cluster, group in merged.groupby("cluster", sort=True):
        train = group[split_column].eq(train_value)
        target = group[split_column].eq(prediction_value)
        row: dict[str, Any] = {
            "cluster": int(cluster),
            "n_all": int(len(group)),
            "n_labeled": int(train.sum()),
            "n_target": int(target.sum()),
            "target_fraction": float(target.mean()),
        }
        y = pd.to_numeric(group.loc[train, target_column], errors="coerce").dropna()
        row["yield_mean"] = float(y.mean()) if len(y) else np.nan
        row["yield_std"] = float(y.std(ddof=0)) if len(y) else np.nan
        composition_rows.append(row)
    composition = pd.DataFrame(composition_rows)

    residual = parcel_residual_summary.copy()
    residual[ID_COLUMN] = residual[ID_COLUMN].astype(str)
    residual = residual.merge(assignments, on=ID_COLUMN, how="inner", validate="many_to_one")
    residual_rows: list[dict[str, Any]] = []
    for (method, cluster), group in residual.groupby(["method", "cluster"], sort=True):
        residual_rows.append(
            {
                "method": method,
                "cluster": int(cluster),
                "n_parcels": int(len(group)),
                "mean_residual": float(group["mean_residual"].mean()),
                "median_residual": float(group["median_residual"].median()),
                "mean_squared_error": float(group["mean_squared_error"].mean()),
                "rmse_across_parcels": float(np.sqrt(group["mean_squared_error"].mean())),
                "mean_absolute_error": float(group["mean_absolute_error"].mean()),
            }
        )
    residual_by_cluster = pd.DataFrame(residual_rows)

    effect_rows: list[dict[str, Any]] = []
    effect_payload: dict[str, Any] = {}
    for method in ["Local04D", "CatBoost_C1"]:
        group = residual.loc[residual["method"].eq(method)].copy()
        values = group["mean_residual"].to_numpy(float)
        labels = group["cluster"].to_numpy(int)
        eta2, pvalue = _permutation_eta_pvalue(
            values,
            labels,
            repeats=int(permutation_repeats),
            random_state=int(random_state) + sum(ord(char) for char in method),
        )
        mse_eta2, mse_pvalue = _permutation_eta_pvalue(
            group["mean_squared_error"].to_numpy(float),
            labels,
            repeats=int(permutation_repeats),
            random_state=int(random_state) + 5000 + sum(ord(char) for char in method),
        )
        means = group.groupby("cluster")["mean_residual"].mean()
        row = {
            "method": method,
            "residual_eta2": eta2,
            "residual_permutation_p": pvalue,
            "mse_eta2": mse_eta2,
            "mse_permutation_p": mse_pvalue,
            "residual_mean_range": float(means.max() - means.min()),
        }
        effect_rows.append(row)
        effect_payload[method] = row

    effects = pd.DataFrame(effect_rows)

    expert = (
        residual.loc[residual["method"].isin(["Local04D", "CatBoost_C1"])]
        .pivot(index=[ID_COLUMN, "cluster"], columns="method", values="mean_squared_error")
        .dropna()
        .reset_index()
    )
    expert["cat_minus_local_mse"] = expert["CatBoost_C1"] - expert["Local04D"]
    advantage_rows = []
    for cluster, group in expert.groupby("cluster", sort=True):
        advantage_rows.append(
            {
                "cluster": int(cluster),
                "n_parcels": int(len(group)),
                "mean_cat_minus_local_mse": float(group["cat_minus_local_mse"].mean()),
                "median_cat_minus_local_mse": float(group["cat_minus_local_mse"].median()),
                "catboost_better_fraction": float(
                    np.mean(group["cat_minus_local_mse"].to_numpy(float) < 0.0)
                ),
            }
        )
    expert_advantage = pd.DataFrame(advantage_rows)

    geography: dict[str, Any] = {}
    labeled = merged.loc[merged[split_column].eq(train_value)].copy()
    if state_column and state_column in labeled.columns:
        geography["cluster_state_nmi"] = float(
            normalized_mutual_info_score(
                labeled["cluster"].astype(str),
                labeled[state_column].fillna("missing").astype(str),
            )
        )
    if municipality_column and municipality_column in labeled.columns:
        geography["cluster_municipality_nmi"] = float(
            normalized_mutual_info_score(
                labeled["cluster"].astype(str),
                labeled[municipality_column].fillna("missing").astype(str),
            )
        )

    payload = {
        "effects": effect_payload,
        "geography": geography,
        "expert_advantage_range": (
            float(
                expert_advantage["mean_cat_minus_local_mse"].max()
                - expert_advantage["mean_cat_minus_local_mse"].min()
            )
            if not expert_advantage.empty
            else np.nan
        ),
    }
    return composition, residual_by_cluster, effects, expert_advantage, payload


def decide_regime_hypothesis(
    selection: Mapping[str, Any],
    effect_payload: Mapping[str, Any],
    *,
    min_residual_eta2: float,
    max_residual_permutation_p: float,
    min_residual_mean_range: float,
) -> dict[str, Any]:
    """Decide whether X-only regimes warrant a partially pooled 06E correction."""

    local = effect_payload["effects"]["Local04D"]
    x_ok = bool(selection["x_eligible"])
    residual_ok = bool(
        float(local["residual_eta2"]) >= float(min_residual_eta2)
        and float(local["residual_permutation_p"]) <= float(max_residual_permutation_p)
        and float(local["residual_mean_range"]) >= float(min_residual_mean_range)
    )
    if x_ok and residual_ok:
        status = "SUPPORTED"
    elif x_ok and (
        float(local["residual_eta2"]) >= float(min_residual_eta2)
        or float(local["residual_mean_range"]) >= float(min_residual_mean_range)
    ):
        status = "AMBIGUOUS"
    else:
        status = "NOT_SUPPORTED"
    return {
        "REGIME_HYPOTHESIS": status,
        "x_clustering_supported": x_ok,
        "local04d_residual_structure_supported": residual_ok,
        "selected_x_regime": dict(selection),
        "local04d_residual_effect": dict(local),
        "expert_advantage_range": float(effect_payload["expert_advantage_range"]),
        "geography": dict(effect_payload["geography"]),
    }
