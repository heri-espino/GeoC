"""Checkpoint 06 tail-aware residual and regime diagnostics.

The first Checkpoint 06 implementation is intentionally diagnostic-first. It
reuses honest target-matched base predictions from Checkpoint 05, reconstructs
fold-valid yield tails from the visible pseudo-train labels, and asks whether
remaining Local04D/CatBoost errors are concentrated, systematically biased, and
predictable from the target-free G1 agronomic representation.

No function in this module reads hidden FIRA target values.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import json
import math

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ID_COLUMN = "ID_POLIGONO"


def _rmse(observed: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(observed, predicted)))


def _safe_spearman(left: pd.Series, right: pd.Series) -> float:
    pair = pd.concat(
        [
            pd.to_numeric(left, errors="coerce"),
            pd.to_numeric(right, errors="coerce"),
        ],
        axis=1,
    ).dropna()
    if len(pair) < 4 or pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
        return np.nan
    return float(pair.iloc[:, 0].corr(pair.iloc[:, 1], method="spearman"))


def _standardized_difference(values: pd.Series, mask_a: pd.Series, mask_b: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    a = numeric[mask_a].dropna().to_numpy(float)
    b = numeric[mask_b].dropna().to_numpy(float)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pooled = math.sqrt(
        ((len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1))
        / max(len(a) + len(b) - 2, 1)
    )
    if not np.isfinite(pooled) or pooled <= 1.0e-12:
        return np.nan
    return float((np.mean(a) - np.mean(b)) / pooled)


def _bootstrap_mean_ci(
    values: Sequence[float],
    *,
    repeats: int,
    random_state: int,
) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if len(array) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(int(random_state))
    means = np.empty(int(repeats), dtype=float)
    for index in range(int(repeats)):
        sample = rng.choice(array, size=len(array), replace=True)
        means[index] = float(np.mean(sample))
    return (
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def _numeric_feature_families(
    agronomic: pd.DataFrame,
    manifest: Mapping[str, Any],
) -> dict[str, list[str]]:
    records = manifest.get("features", [])
    families: dict[str, list[str]] = {}
    for record in records:
        column = str(record.get("column", ""))
        family = str(record.get("family", "unknown"))
        mode = str(record.get("mode", ""))
        if (
            column
            and column in agronomic.columns
            and mode in {"clean", "competition"}
            and pd.api.types.is_numeric_dtype(agronomic[column])
        ):
            families.setdefault(family, []).append(column)
    families = {key: sorted(set(value)) for key, value in families.items() if value}
    all_columns = sorted({column for value in families.values() for column in value})
    if all_columns:
        families["all_agronomic"] = all_columns
    return dict(sorted(families.items()))


def assign_fold_valid_tails(
    prediction_rows: pd.DataFrame,
    base_frame: pd.DataFrame,
    split_membership: pd.DataFrame,
    *,
    target_column: str,
    split_column: str,
    train_value: str,
    low_quantile: float = 0.15,
    high_quantile: float = 0.85,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign pseudo-target tails using thresholds computed from visible pseudo-train y only."""

    required_predictions = {"family", "split_id", ID_COLUMN, "observed"}
    missing = required_predictions.difference(prediction_rows.columns)
    if missing:
        raise KeyError(f"Prediction rows missing columns: {sorted(missing)}")
    required_membership = {"family", "split_id", ID_COLUMN, "role"}
    missing = required_membership.difference(split_membership.columns)
    if missing:
        raise KeyError(f"Split membership missing columns: {sorted(missing)}")
    required_base = {ID_COLUMN, target_column, split_column}
    missing = required_base.difference(base_frame.columns)
    if missing:
        raise KeyError(f"Base frame missing columns: {sorted(missing)}")

    target_matched = prediction_rows.loc[prediction_rows["family"].eq("target_matched")].copy()
    membership = split_membership.loc[
        split_membership["family"].eq("target_matched")
    ].copy()
    y_lookup = base_frame.set_index(ID_COLUMN)[target_column]
    train_ids = set(
        base_frame.loc[base_frame[split_column].eq(train_value), ID_COLUMN].astype(str)
    )

    threshold_rows: list[dict[str, Any]] = []
    pieces: list[pd.DataFrame] = []
    for split_id, group in target_matched.groupby("split_id", sort=True):
        split_members = membership.loc[membership["split_id"].eq(split_id)]
        pseudo_train_ids = split_members.loc[
            split_members["role"].eq("pseudo_train"), ID_COLUMN
        ].astype(str)
        pseudo_target_ids = split_members.loc[
            split_members["role"].eq("pseudo_target"), ID_COLUMN
        ].astype(str)

        if not set(pseudo_train_ids).issubset(train_ids):
            raise ValueError(f"{split_id}: pseudo-train contains non-training parcel IDs.")
        predicted_ids = set(group[ID_COLUMN].astype(str))
        if predicted_ids != set(pseudo_target_ids):
            raise ValueError(
                f"{split_id}: base prediction IDs do not match frozen pseudo-target membership."
            )

        visible_y = pd.to_numeric(y_lookup.reindex(pseudo_train_ids), errors="coerce").dropna()
        if len(visible_y) != len(pseudo_train_ids):
            raise ValueError(f"{split_id}: pseudo-train yield is incomplete.")
        low = float(visible_y.quantile(float(low_quantile)))
        high = float(visible_y.quantile(float(high_quantile)))

        enriched = group.copy()
        observed = pd.to_numeric(enriched["observed"], errors="coerce")
        enriched["tail_low_threshold"] = low
        enriched["tail_high_threshold"] = high
        enriched["tail_regime"] = np.where(
            observed <= low,
            "low",
            np.where(observed >= high, "high", "center"),
        )
        pieces.append(enriched)
        threshold_rows.append(
            {
                "split_id": split_id,
                "n_pseudo_train": int(len(visible_y)),
                "n_pseudo_target": int(len(group)),
                "low_quantile": float(low_quantile),
                "high_quantile": float(high_quantile),
                "low_threshold": low,
                "high_threshold": high,
                "pseudo_train_y_mean": float(visible_y.mean()),
                "pseudo_train_y_std": float(visible_y.std(ddof=0)),
            }
        )

    return pd.concat(pieces, ignore_index=True), pd.DataFrame(threshold_rows)


def build_residual_diagnostics(
    prediction_rows: pd.DataFrame,
    *,
    methods: Sequence[str],
    include_local_catboost_mean: bool = True,
) -> pd.DataFrame:
    """Convert wide honest base predictions into a long residual diagnostic table."""

    required = {
        ID_COLUMN,
        "split_id",
        "family",
        "observed",
        "tail_regime",
        "tail_low_threshold",
        "tail_high_threshold",
        *methods,
    }
    missing = required.difference(prediction_rows.columns)
    if missing:
        raise KeyError(f"Prediction rows missing columns: {sorted(missing)}")

    support_columns = [
        column
        for column in [
            "meta_estado",
            "meta_municipio",
            "log_nearest_geo_km",
            "log_nearest_agro_distance",
            "geo_support",
            "agro_support",
            "same_state_fraction",
            "same_municipality_fraction",
            "local_graph_abs_disagreement",
        ]
        if column in prediction_rows.columns
    ]
    rows: list[pd.DataFrame] = []
    method_payload = {
        method: pd.to_numeric(prediction_rows[method], errors="coerce")
        for method in methods
    }
    if (
        include_local_catboost_mean
        and "Local04D" in method_payload
        and "CatBoost_C1" in method_payload
    ):
        method_payload["LocalCatBoostMean"] = (
            method_payload["Local04D"] + method_payload["CatBoost_C1"]
        ) / 2.0

    observed = pd.to_numeric(prediction_rows["observed"], errors="coerce")
    for method, predicted in method_payload.items():
        frame = prediction_rows[
            [
                ID_COLUMN,
                "split_id",
                "family",
                "observed",
                "tail_regime",
                "tail_low_threshold",
                "tail_high_threshold",
                *support_columns,
            ]
        ].copy()
        frame["method"] = method
        frame["predicted"] = predicted.to_numpy(float)
        frame["residual"] = observed.to_numpy(float) - frame["predicted"].to_numpy(float)
        frame["absolute_error"] = np.abs(frame["residual"])
        frame["squared_error"] = frame["residual"] ** 2
        if "Local04D" in method_payload and "CatBoost_C1" in method_payload:
            frame["local_catboost_abs_disagreement"] = np.abs(
                method_payload["Local04D"].to_numpy(float)
                - method_payload["CatBoost_C1"].to_numpy(float)
            )
        rows.append(frame)

    result = pd.concat(rows, ignore_index=True)
    if result[["observed", "predicted", "residual"]].isna().any().any():
        raise ValueError("Residual diagnostic table contains missing prediction/target values.")
    return result


def aggregate_residuals_by_parcel(residual_rows: pd.DataFrame) -> pd.DataFrame:
    """Aggregate repeated honest pseudo-target appearances to equal-weight parcel summaries."""

    required = {
        ID_COLUMN,
        "method",
        "observed",
        "predicted",
        "residual",
        "absolute_error",
        "squared_error",
    }
    missing = required.difference(residual_rows.columns)
    if missing:
        raise KeyError(f"Residual rows missing columns: {sorted(missing)}")

    summary = (
        residual_rows.groupby(["method", ID_COLUMN], sort=True)
        .agg(
            n_honest_appearances=("split_id", "nunique"),
            observed=("observed", "first"),
            mean_prediction=("predicted", "mean"),
            median_prediction=("predicted", "median"),
            prediction_std=("predicted", "std"),
            mean_residual=("residual", "mean"),
            median_residual=("residual", "median"),
            mean_absolute_error=("absolute_error", "mean"),
            mean_squared_error=("squared_error", "mean"),
            worst_absolute_error=("absolute_error", "max"),
        )
        .reset_index()
    )
    summary["prediction_std"] = summary["prediction_std"].fillna(0.0)
    summary["rmse_across_appearances"] = np.sqrt(summary["mean_squared_error"])
    return summary


def summarize_sse_concentration(
    parcel_summary: pd.DataFrame,
    *,
    top_counts: Sequence[int] = (5, 10),
    top_fractions: Sequence[float] = (0.10, 0.15, 0.20),
) -> pd.DataFrame:
    """Summarize how much equal-weight parcel SSE is concentrated in the largest errors."""

    rows: list[dict[str, Any]] = []
    for method, group in parcel_summary.groupby("method", sort=True):
        ordered = group.sort_values("mean_squared_error", ascending=False).reset_index(drop=True)
        total = float(ordered["mean_squared_error"].sum())
        n = len(ordered)
        for count in top_counts:
            k = min(int(count), n)
            share = (
                float(ordered.head(k)["mean_squared_error"].sum() / total)
                if total > 0
                else np.nan
            )
            rows.append(
                {
                    "method": method,
                    "aggregation": "equal_weight_parcel_mse",
                    "selection": f"top_{k}_parcels",
                    "k": k,
                    "fraction": float(k / n) if n else np.nan,
                    "sse_share": share,
                }
            )
        for fraction in top_fractions:
            k = max(1, int(math.ceil(float(fraction) * n)))
            share = (
                float(ordered.head(k)["mean_squared_error"].sum() / total)
                if total > 0
                else np.nan
            )
            rows.append(
                {
                    "method": method,
                    "aggregation": "equal_weight_parcel_mse",
                    "selection": f"top_{int(round(float(fraction) * 100))}pct",
                    "k": k,
                    "fraction": float(fraction),
                    "sse_share": share,
                }
            )
    return pd.DataFrame(rows)


def summarize_tail_bias(
    residual_rows: pd.DataFrame,
    *,
    bootstrap_repeats: int = 2000,
    random_state: int = 20261005,
) -> pd.DataFrame:
    """Summarize fold-valid low/center/high residual bias with parcel-level bootstrap CIs."""

    per_parcel = (
        residual_rows.groupby(["method", ID_COLUMN, "tail_regime"], sort=True)
        .agg(
            observed=("observed", "mean"),
            residual=("residual", "mean"),
            absolute_error=("absolute_error", "mean"),
            squared_error=("squared_error", "mean"),
        )
        .reset_index()
    )
    rows: list[dict[str, Any]] = []
    for (method, regime), group in per_parcel.groupby(
        ["method", "tail_regime"], sort=True
    ):
        residual = group["residual"].to_numpy(float)
        expected_mask = (
            residual < 0.0
            if regime == "low"
            else residual > 0.0
            if regime == "high"
            else np.full(len(residual), False)
        )
        lower, upper = _bootstrap_mean_ci(
            residual,
            repeats=int(bootstrap_repeats),
            random_state=int(random_state)
            + sum(ord(char) for char in f"{method}:{regime}"),
        )
        rows.append(
            {
                "method": method,
                "tail_regime": regime,
                "n_parcels": int(len(group)),
                "mean_observed": float(group["observed"].mean()),
                "mean_residual": float(np.mean(residual)),
                "median_residual": float(np.median(residual)),
                "rmse": float(np.sqrt(group["squared_error"].mean())),
                "mae": float(group["absolute_error"].mean()),
                "mean_residual_ci_low": lower,
                "mean_residual_ci_high": upper,
                "expected_shrinkage_sign_fraction": (
                    float(np.mean(expected_mask))
                    if regime in {"low", "high"}
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def expert_tail_overlap(
    parcel_summary: pd.DataFrame,
    *,
    method_a: str = "Local04D",
    method_b: str = "CatBoost_C1",
) -> pd.DataFrame:
    """Compare the parcel-level error sets and residual direction of two honest experts."""

    left = parcel_summary.loc[parcel_summary["method"].eq(method_a)].set_index(ID_COLUMN)
    right = parcel_summary.loc[parcel_summary["method"].eq(method_b)].set_index(ID_COLUMN)
    common = left.index.intersection(right.index)
    left = left.loc[common]
    right = right.loc[common]

    residual_corr = pd.Series(left["mean_residual"].to_numpy()).corr(
        pd.Series(right["mean_residual"].to_numpy()), method="pearson"
    )
    residual_spearman = pd.Series(left["mean_residual"].to_numpy()).corr(
        pd.Series(right["mean_residual"].to_numpy()), method="spearman"
    )
    squared_corr = pd.Series(left["mean_squared_error"].to_numpy()).corr(
        pd.Series(right["mean_squared_error"].to_numpy()), method="spearman"
    )
    rows: list[dict[str, Any]] = [
        {"metric": "residual_pearson", "value": float(residual_corr), "k": np.nan},
        {
            "metric": "residual_spearman",
            "value": float(residual_spearman),
            "k": np.nan,
        },
        {
            "metric": "squared_error_spearman",
            "value": float(squared_corr),
            "k": np.nan,
        },
        {
            "metric": "residual_sign_agreement",
            "value": float(
                np.mean(
                    np.sign(left["mean_residual"].to_numpy())
                    == np.sign(right["mean_residual"].to_numpy())
                )
            ),
            "k": np.nan,
        },
    ]
    for k in (5, 10):
        left_ids = set(left.nlargest(min(k, len(left)), "mean_squared_error").index)
        right_ids = set(right.nlargest(min(k, len(right)), "mean_squared_error").index)
        union = left_ids | right_ids
        rows.append(
            {
                "metric": f"top_{k}_error_jaccard",
                "value": float(len(left_ids & right_ids) / len(union)) if union else np.nan,
                "k": int(k),
            }
        )
    return pd.DataFrame(rows)


def feature_association_tables(
    residual_rows: pd.DataFrame,
    agronomic: pd.DataFrame,
    manifest: Mapping[str, Any],
    *,
    methods: Sequence[str] = ("Local04D", "CatBoost_C1"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Measure fold-wise feature association with yield and honest residual behavior."""

    families = _numeric_feature_families(agronomic, manifest)
    all_columns = families.pop("all_agronomic", [])
    family_lookup = {
        column: family for family, columns in families.items() for column in columns
    }
    feature_frame = agronomic.set_index(ID_COLUMN)
    rows: list[dict[str, Any]] = []

    subset = residual_rows.loc[residual_rows["method"].isin(list(methods))].copy()
    for (method, split_id), group in subset.groupby(["method", "split_id"], sort=True):
        ids = group[ID_COLUMN].astype(str).tolist()
        joined = group.set_index(ID_COLUMN).join(feature_frame.reindex(ids), how="left")
        low_mask = joined["tail_regime"].eq("low")
        high_mask = joined["tail_regime"].eq("high")
        center_mask = joined["tail_regime"].eq("center")
        for feature in all_columns:
            values = pd.to_numeric(joined[feature], errors="coerce")
            rows.append(
                {
                    "method": method,
                    "split_id": split_id,
                    "feature": feature,
                    "feature_family": family_lookup.get(feature, "unknown"),
                    "n_finite": int(values.notna().sum()),
                    "rho_y": _safe_spearman(values, joined["observed"]),
                    "rho_residual": _safe_spearman(values, joined["residual"]),
                    "rho_absolute_error": _safe_spearman(
                        values, joined["absolute_error"]
                    ),
                    "rho_squared_error": _safe_spearman(
                        values, joined["squared_error"]
                    ),
                    "std_diff_low_vs_center": _standardized_difference(
                        values, low_mask, center_mask
                    ),
                    "std_diff_high_vs_center": _standardized_difference(
                        values, high_mask, center_mask
                    ),
                }
            )

    by_split = pd.DataFrame(rows)
    summaries: list[dict[str, Any]] = []
    metrics = [
        "rho_y",
        "rho_residual",
        "rho_absolute_error",
        "rho_squared_error",
        "std_diff_low_vs_center",
        "std_diff_high_vs_center",
    ]
    for (method, feature, family), group in by_split.groupby(
        ["method", "feature", "feature_family"], sort=True
    ):
        row: dict[str, Any] = {
            "method": method,
            "feature": feature,
            "feature_family": family,
            "n_splits": int(group["split_id"].nunique()),
        }
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"median_{metric}"] = float(values.median()) if len(values) else np.nan
            row[f"median_abs_{metric}"] = (
                float(values.abs().median()) if len(values) else np.nan
            )
            if metric.startswith("rho_") and len(values):
                median = float(values.median())
                row[f"sign_stability_{metric}"] = (
                    float(
                        np.mean(
                            np.sign(values.to_numpy())
                            == np.sign(median)
                        )
                    )
                    if abs(median) > 1.0e-12
                    else np.nan
                )
        summaries.append(row)
    return by_split, pd.DataFrame(summaries)


def _binary_metrics(observed: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    observed = np.asarray(observed, dtype=int)
    probability = np.asarray(probability, dtype=float)
    prevalence = float(np.mean(observed)) if len(observed) else np.nan
    brier = float(np.mean((probability - observed) ** 2)) if len(observed) else np.nan
    if len(np.unique(observed)) < 2:
        auc = np.nan
        ap = np.nan
    else:
        auc = float(roc_auc_score(observed, probability))
        ap = float(average_precision_score(observed, probability))
    return {
        "roc_auc": auc,
        "average_precision": ap,
        "prevalence": prevalence,
        "pr_lift": float(ap / prevalence)
        if np.isfinite(ap) and prevalence > 0
        else np.nan,
        "brier": brier,
    }


def evaluate_tail_classifier_families(
    base_frame: pd.DataFrame,
    split_membership: pd.DataFrame,
    agronomic: pd.DataFrame,
    manifest: Mapping[str, Any],
    *,
    target_column: str,
    low_quantile: float = 0.15,
    high_quantile: float = 0.85,
    c_value: float = 1.0,
    random_state: int = 20261005,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cross-fit simple family-level tail classifiers on pseudo-train/pseudo-target rows."""

    families = _numeric_feature_families(agronomic, manifest)
    feature_frame = agronomic.set_index(ID_COLUMN)
    base = base_frame.set_index(ID_COLUMN)
    membership = split_membership.loc[
        split_membership["family"].eq("target_matched")
    ].copy()
    prediction_rows: list[dict[str, Any]] = []
    tasks = ("low_vs_rest", "high_vs_rest", "tail_vs_center")

    for split_id, split_group in membership.groupby("split_id", sort=True):
        train_ids = split_group.loc[
            split_group["role"].eq("pseudo_train"), ID_COLUMN
        ].astype(str).tolist()
        test_ids = split_group.loc[
            split_group["role"].eq("pseudo_target"), ID_COLUMN
        ].astype(str).tolist()
        train_y = pd.to_numeric(base.loc[train_ids, target_column], errors="coerce")
        test_y = pd.to_numeric(base.loc[test_ids, target_column], errors="coerce")
        if train_y.isna().any() or test_y.isna().any():
            raise ValueError(f"{split_id}: missing labeled yield in diagnostic split.")
        low = float(train_y.quantile(float(low_quantile)))
        high = float(train_y.quantile(float(high_quantile)))
        train_values = train_y.to_numpy(float)
        test_values = test_y.to_numpy(float)
        task_labels = {
            "low_vs_rest": (
                (train_values <= low).astype(int),
                (test_values <= low).astype(int),
            ),
            "high_vs_rest": (
                (train_values >= high).astype(int),
                (test_values >= high).astype(int),
            ),
            "tail_vs_center": (
                ((train_values <= low) | (train_values >= high)).astype(int),
                ((test_values <= low) | (test_values >= high)).astype(int),
            ),
        }

        for family, columns in families.items():
            x_train = feature_frame.loc[train_ids, columns]
            x_test = feature_frame.loc[test_ids, columns]
            for task in tasks:
                y_train_binary, y_test_binary = task_labels[task]
                if len(np.unique(y_train_binary)) < 2:
                    continue
                pipeline = Pipeline(
                    [
                        (
                            "impute",
                            SimpleImputer(
                                strategy="median",
                                keep_empty_features=True,
                            ),
                        ),
                        ("scale", StandardScaler()),
                        (
                            "model",
                            LogisticRegression(
                                C=float(c_value),
                                class_weight="balanced",
                                max_iter=2500,
                                solver="liblinear",
                                random_state=int(random_state),
                            ),
                        ),
                    ]
                )
                pipeline.fit(x_train, y_train_binary)
                probability = pipeline.predict_proba(x_test)[:, 1]
                for parcel_id, label, prob, y_value in zip(
                    test_ids,
                    y_test_binary,
                    probability,
                    test_values,
                    strict=True,
                ):
                    prediction_rows.append(
                        {
                            "split_id": split_id,
                            "family": family,
                            "task": task,
                            ID_COLUMN: parcel_id,
                            "observed_y": float(y_value),
                            "tail_label": int(label),
                            "tail_probability": float(prob),
                            "low_threshold": low,
                            "high_threshold": high,
                            "n_features": int(len(columns)),
                        }
                    )

    predictions = pd.DataFrame(prediction_rows)
    split_rows: list[dict[str, Any]] = []
    for (family, task, split_id), group in predictions.groupby(
        ["family", "task", "split_id"], sort=True
    ):
        split_rows.append(
            {
                "family": family,
                "task": task,
                "split_id": split_id,
                "n_test": int(len(group)),
                "n_features": int(group["n_features"].iloc[0]),
                **_binary_metrics(
                    group["tail_label"].to_numpy(int),
                    group["tail_probability"].to_numpy(float),
                ),
            }
        )
    split_metrics = pd.DataFrame(split_rows)

    summaries: list[dict[str, Any]] = []
    for (family, task), group in split_metrics.groupby(["family", "task"], sort=True):
        pooled = predictions.loc[
            predictions["family"].eq(family) & predictions["task"].eq(task)
        ]
        pooled_metrics = _binary_metrics(
            pooled["tail_label"].to_numpy(int),
            pooled["tail_probability"].to_numpy(float),
        )
        summaries.append(
            {
                "family": family,
                "task": task,
                "n_splits": int(group["split_id"].nunique()),
                "n_features": int(group["n_features"].iloc[0]),
                "mean_split_roc_auc": float(group["roc_auc"].mean()),
                "median_split_roc_auc": float(group["roc_auc"].median()),
                "mean_split_average_precision": float(
                    group["average_precision"].mean()
                ),
                "mean_split_prevalence": float(group["prevalence"].mean()),
                "mean_split_pr_lift": float(group["pr_lift"].mean()),
                "mean_split_brier": float(group["brier"].mean()),
                "pooled_row_roc_auc": pooled_metrics["roc_auc"],
                "pooled_row_average_precision": pooled_metrics[
                    "average_precision"
                ],
                "pooled_row_pr_lift": pooled_metrics["pr_lift"],
            }
        )
    return predictions, pd.DataFrame(summaries)


def evaluate_error_risk_families(
    parcel_summary: pd.DataFrame,
    agronomic: pd.DataFrame,
    manifest: Mapping[str, Any],
    *,
    methods: Sequence[str] = ("Local04D", "CatBoost_C1"),
    n_splits: int = 5,
    ridge_alpha: float = 10.0,
    random_state: int = 20261005,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cross-validate agronomic family models for honest parcel-level error risk."""

    families = _numeric_feature_families(agronomic, manifest)
    feature_frame = agronomic.set_index(ID_COLUMN)
    oof_rows: list[dict[str, Any]] = []

    for method in methods:
        target_rows = (
            parcel_summary.loc[parcel_summary["method"].eq(method)]
            .sort_values(ID_COLUMN)
            .reset_index(drop=True)
        )
        ids = target_rows[ID_COLUMN].astype(str).tolist()
        log_risk = np.log1p(target_rows["mean_squared_error"].to_numpy(float))
        for family, columns in families.items():
            x = feature_frame.loc[ids, columns]
            predicted = np.full(len(target_rows), np.nan, dtype=float)
            fold_index = np.full(len(target_rows), -1, dtype=int)
            folds = KFold(
                n_splits=min(int(n_splits), len(target_rows)),
                shuffle=True,
                random_state=int(random_state),
            )
            for fold_number, (train_idx, test_idx) in enumerate(
                folds.split(x), start=1
            ):
                pipeline = Pipeline(
                    [
                        (
                            "impute",
                            SimpleImputer(
                                strategy="median",
                                keep_empty_features=True,
                            ),
                        ),
                        ("scale", StandardScaler()),
                        ("model", Ridge(alpha=float(ridge_alpha))),
                    ]
                )
                pipeline.fit(x.iloc[train_idx], log_risk[train_idx])
                predicted[test_idx] = pipeline.predict(x.iloc[test_idx])
                fold_index[test_idx] = fold_number
            for index, parcel_id in enumerate(ids):
                oof_rows.append(
                    {
                        "method": method,
                        "family": family,
                        ID_COLUMN: parcel_id,
                        "fold": int(fold_index[index]),
                        "observed_mse": float(
                            target_rows.loc[index, "mean_squared_error"]
                        ),
                        "observed_log1p_mse": float(log_risk[index]),
                        "predicted_log1p_mse": float(predicted[index]),
                        "predicted_mse": float(max(np.expm1(predicted[index]), 0.0)),
                        "n_features": int(len(columns)),
                    }
                )

    oof = pd.DataFrame(oof_rows)
    summaries: list[dict[str, Any]] = []
    for (method, family), group in oof.groupby(["method", "family"], sort=True):
        summaries.append(
            {
                "method": method,
                "family": family,
                "n_parcels": int(len(group)),
                "n_features": int(group["n_features"].iloc[0]),
                "risk_spearman": _safe_spearman(
                    group["observed_mse"], group["predicted_mse"]
                ),
                "risk_r2_log": float(
                    r2_score(
                        group["observed_log1p_mse"],
                        group["predicted_log1p_mse"],
                    )
                ),
                "risk_rmse_log": _rmse(
                    group["observed_log1p_mse"].to_numpy(float),
                    group["predicted_log1p_mse"].to_numpy(float),
                ),
            }
        )
    return oof, pd.DataFrame(summaries)


def decide_tail_hypothesis(
    concentration: pd.DataFrame,
    tail_bias: pd.DataFrame,
    tail_classifier_summary: pd.DataFrame,
    error_risk_summary: pd.DataFrame,
    *,
    thresholds: Mapping[str, float],
) -> dict[str, Any]:
    """Apply predeclared continuation thresholds to the diagnostic evidence."""

    local_concentration = concentration.loc[
        concentration["method"].eq("Local04D")
        & concentration["selection"].eq("top_10pct")
    ]
    top10_share = (
        float(local_concentration["sse_share"].iloc[0])
        if len(local_concentration)
        else np.nan
    )
    concentration_supported = bool(
        np.isfinite(top10_share)
        and top10_share >= float(thresholds["min_top10pct_sse_share"])
    )

    def tail_flag(
        method: str,
        regime: str,
        expected_direction: int,
    ) -> tuple[bool, dict[str, float]]:
        row = tail_bias.loc[
            tail_bias["method"].eq(method)
            & tail_bias["tail_regime"].eq(regime)
        ]
        if row.empty:
            return False, {}
        mean_residual = float(row["mean_residual"].iloc[0])
        sign_fraction = float(
            row["expected_shrinkage_sign_fraction"].iloc[0]
        )
        supported = (
            abs(mean_residual) >= float(thresholds["min_abs_tail_bias"])
            and np.sign(mean_residual) == int(expected_direction)
            and sign_fraction >= float(thresholds["min_expected_sign_fraction"])
        )
        return bool(supported), {
            "mean_residual": mean_residual,
            "expected_sign_fraction": sign_fraction,
        }

    local_low, local_low_values = tail_flag("Local04D", "low", -1)
    local_high, local_high_values = tail_flag("Local04D", "high", 1)
    cat_low, cat_low_values = tail_flag("CatBoost_C1", "low", -1)
    cat_high, cat_high_values = tail_flag("CatBoost_C1", "high", 1)
    shrinkage_supported = bool(local_low and local_high)
    catboost_corroborates = bool(cat_low and cat_high)

    tail_rows = tail_classifier_summary.loc[
        tail_classifier_summary["task"].eq("tail_vs_center")
    ].copy()
    classifier_supported = bool(
        (
            tail_rows["mean_split_roc_auc"].ge(
                float(thresholds["min_tail_auc"])
            )
            & tail_rows["mean_split_pr_lift"].ge(
                float(thresholds["min_pr_lift"])
            )
        ).any()
    ) if not tail_rows.empty else False
    best_classifier: dict[str, Any] = {}
    if not tail_rows.empty:
        best = tail_rows.sort_values(
            ["mean_split_roc_auc", "mean_split_pr_lift"],
            ascending=False,
        ).iloc[0]
        best_classifier = {
            "family": str(best["family"]),
            "mean_split_roc_auc": float(best["mean_split_roc_auc"]),
            "mean_split_pr_lift": float(best["mean_split_pr_lift"]),
        }

    local_risk = error_risk_summary.loc[
        error_risk_summary["method"].eq("Local04D")
    ]
    risk_supported = bool(
        local_risk["risk_spearman"]
        .ge(float(thresholds["min_error_risk_spearman"]))
        .any()
    ) if not local_risk.empty else False
    best_risk: dict[str, Any] = {}
    if not local_risk.empty:
        best = local_risk.sort_values("risk_spearman", ascending=False).iloc[0]
        best_risk = {
            "family": str(best["family"]),
            "risk_spearman": float(best["risk_spearman"]),
        }

    x_predictable = bool(classifier_supported or risk_supported)
    if concentration_supported and shrinkage_supported and x_predictable:
        status = "SUPPORTED"
    elif concentration_supported and (shrinkage_supported or x_predictable):
        status = "AMBIGUOUS"
    else:
        status = "NOT_SUPPORTED"

    return {
        "TAIL_HYPOTHESIS": status,
        "concentration_supported": concentration_supported,
        "shrinkage_supported_local04d": shrinkage_supported,
        "catboost_shrinkage_corroboration": catboost_corroborates,
        "x_predictability_supported": x_predictable,
        "tail_classifier_supported": classifier_supported,
        "error_risk_supported": risk_supported,
        "local04d_top10pct_sse_share": top10_share,
        "local04d_low_tail": local_low_values,
        "local04d_high_tail": local_high_values,
        "catboost_low_tail": cat_low_values,
        "catboost_high_tail": cat_high_values,
        "best_tail_classifier": best_classifier,
        "best_error_risk_model": best_risk,
        "thresholds": {
            key: float(value)
            for key, value in thresholds.items()
            if isinstance(value, (int, float))
        },
    }


def checkpoint06_preflight(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate Checkpoint 06 input provenance without fitting diagnostic models."""

    inputs = config["inputs"]
    identity = config["identity"]
    validation = config["validation"]
    paths = {key: root / str(value) for key, value in inputs.items()}

    missing_paths = [str(path) for path in paths.values() if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(f"Missing Checkpoint 06 inputs: {missing_paths}")

    lfs_pointers = []
    for path in paths.values():
        with path.open("rb") as handle:
            prefix = handle.read(80)
        if prefix.startswith(b"version https://git-lfs.github.com/spec/v1"):
            lfs_pointers.append(str(path.relative_to(root)))
    if lfs_pointers:
        raise RuntimeError(
            "Git LFS inputs are not materialized: "
            + ", ".join(lfs_pointers)
            + ". Run 'git lfs pull' before Checkpoint 06."
        )

    base = pd.read_csv(paths["base_table"])
    predictions = pd.read_csv(paths["base_oof_predictions"])
    membership = pd.read_csv(paths["split_membership"])
    agronomic = pd.read_csv(paths["agronomic_table"])
    manifest = json.loads(
        paths["agronomic_manifest"].read_text(encoding="utf-8")
    )

    id_column = str(identity["id_column"])
    target_column = str(identity["target_column"])
    split_column = str(identity["split_column"])
    train_value = str(identity["train_value"])
    prediction_value = str(identity["prediction_value"])

    if len(base) != int(validation["expected_total_rows"]):
        raise ValueError("Unexpected base-table row count.")
    if base[id_column].duplicated().any() or agronomic[id_column].duplicated().any():
        raise ValueError("Duplicate parcel IDs in Checkpoint 06 inputs.")

    train = base[split_column].eq(train_value)
    target = base[split_column].eq(prediction_value)
    if int(train.sum()) != int(validation["expected_training_rows"]):
        raise ValueError("Unexpected number of labeled rows.")
    if int(target.sum()) != int(validation["expected_prediction_rows"]):
        raise ValueError("Unexpected number of hidden-target rows.")
    if pd.to_numeric(base.loc[target, target_column], errors="coerce").notna().any():
        raise ValueError("Hidden FIRA target y is present; Checkpoint 06 refuses to run.")
    if set(base[id_column].astype(str)) != set(agronomic[id_column].astype(str)):
        raise ValueError("Agronomic table does not cover the same 197 parcel IDs.")

    methods = list(config["diagnostics"]["methods"])
    required_prediction_columns = {
        id_column,
        "family",
        "split_id",
        "observed",
        *methods,
    }
    missing = required_prediction_columns.difference(predictions.columns)
    if missing:
        raise KeyError(f"Base OOF predictions missing columns: {sorted(missing)}")

    target_predictions = predictions.loc[predictions["family"].eq("target_matched")]
    target_membership = membership.loc[membership["family"].eq("target_matched")]
    n_splits = int(target_predictions["split_id"].nunique())
    if n_splits != int(validation["expected_primary_splits"]):
        raise ValueError(
            f"Expected {validation['expected_primary_splits']} target-matched splits."
        )
    counts = target_predictions.groupby("split_id")[id_column].nunique()
    if not counts.eq(int(validation["expected_pseudo_targets"])).all():
        raise ValueError("Unexpected pseudo-target count in base OOF predictions.")

    enriched, _ = assign_fold_valid_tails(
        target_predictions,
        base,
        target_membership,
        target_column=target_column,
        split_column=split_column,
        train_value=train_value,
        low_quantile=float(config["tails"]["primary_low_quantile"]),
        high_quantile=float(config["tails"]["primary_high_quantile"]),
    )
    local_rmse = _rmse(
        enriched["observed"].to_numpy(float),
        enriched["Local04D"].to_numpy(float),
    )
    expected_rmse = float(validation["expected_local04d_pooled_rmse"])
    tolerance = float(validation["local04d_rmse_tolerance"])
    if abs(local_rmse - expected_rmse) > tolerance:
        raise ValueError(
            f"Local04D OOF provenance mismatch: pooled RMSE {local_rmse:.6f}, "
            f"expected approximately {expected_rmse:.6f}."
        )

    families = _numeric_feature_families(agronomic, manifest)
    if "all_agronomic" not in families or len(families["all_agronomic"]) < 100:
        raise ValueError("Agronomic manifest did not resolve the expected G1 feature space.")

    return {
        "rows": int(len(base)),
        "training_rows": int(train.sum()),
        "prediction_rows": int(target.sum()),
        "target_matched_splits": n_splits,
        "pseudo_targets_per_split": int(counts.iloc[0]),
        "base_oof_rows": int(len(target_predictions)),
        "local04d_pooled_rmse": local_rmse,
        "agronomic_features": int(len(families["all_agronomic"])),
        "agronomic_families": int(len(families) - 1),
        "hidden_target_y_present": False,
        "compute": "CPU diagnostics only; no CatBoost refit in 06A/06B",
    }


def run_checkpoint06_diagnostics(
    root: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Run the Checkpoint 06A/06B analysis from frozen honest base predictions."""

    preflight = checkpoint06_preflight(root, config)
    inputs = config["inputs"]
    identity = config["identity"]

    base = pd.read_csv(root / str(inputs["base_table"]))
    predictions = pd.read_csv(root / str(inputs["base_oof_predictions"]))
    membership = pd.read_csv(root / str(inputs["split_membership"]))
    agronomic = pd.read_csv(root / str(inputs["agronomic_table"]))
    manifest = json.loads(
        (root / str(inputs["agronomic_manifest"])).read_text(encoding="utf-8")
    )

    target_predictions = predictions.loc[
        predictions["family"].eq("target_matched")
    ].copy()
    target_membership = membership.loc[
        membership["family"].eq("target_matched")
    ].copy()

    enriched, tail_thresholds = assign_fold_valid_tails(
        target_predictions,
        base,
        target_membership,
        target_column=str(identity["target_column"]),
        split_column=str(identity["split_column"]),
        train_value=str(identity["train_value"]),
        low_quantile=float(config["tails"]["primary_low_quantile"]),
        high_quantile=float(config["tails"]["primary_high_quantile"]),
    )
    residuals = build_residual_diagnostics(
        enriched,
        methods=list(config["diagnostics"]["methods"]),
        include_local_catboost_mean=bool(
            config["diagnostics"].get("include_local_catboost_mean", True)
        ),
    )
    parcel_summary = aggregate_residuals_by_parcel(residuals)
    concentration = summarize_sse_concentration(
        parcel_summary,
        top_counts=list(config["diagnostics"]["top_counts"]),
        top_fractions=list(config["diagnostics"]["top_fractions"]),
    )
    tail_bias = summarize_tail_bias(
        residuals,
        bootstrap_repeats=int(config["diagnostics"]["bootstrap_repeats"]),
        random_state=int(config["runtime"]["random_state"]),
    )
    overlap = expert_tail_overlap(parcel_summary)
    assoc_by_split, assoc_summary = feature_association_tables(
        residuals,
        agronomic,
        manifest,
    )
    classifier_oof, classifier_summary = evaluate_tail_classifier_families(
        base,
        target_membership,
        agronomic,
        manifest,
        target_column=str(identity["target_column"]),
        low_quantile=float(config["tails"]["primary_low_quantile"]),
        high_quantile=float(config["tails"]["primary_high_quantile"]),
        c_value=float(config["feature_analysis"]["tail_classifier_c"]),
        random_state=int(config["runtime"]["random_state"]),
    )
    risk_oof, risk_summary = evaluate_error_risk_families(
        parcel_summary,
        agronomic,
        manifest,
        n_splits=int(config["feature_analysis"]["error_risk_folds"]),
        ridge_alpha=float(config["feature_analysis"]["error_risk_ridge_alpha"]),
        random_state=int(config["runtime"]["random_state"]),
    )
    decision = decide_tail_hypothesis(
        concentration,
        tail_bias,
        classifier_summary,
        risk_summary,
        thresholds=config["continuation_thresholds"],
    )

    labeled_y = pd.to_numeric(
        base.loc[
            base[str(identity["split_column"])].eq(str(identity["train_value"])),
            str(identity["target_column"]),
        ],
        errors="coerce",
    ).dropna()
    descriptive_quantiles = pd.DataFrame(
        [
            {
                "quantile": float(quantile),
                "yield_t_ha": float(labeled_y.quantile(quantile)),
            }
            for quantile in [0.10, 0.15, 0.20, 0.80, 0.85, 0.90]
        ]
    )

    family_importance = classifier_summary.copy()
    if not family_importance.empty:
        family_importance["evidence_type"] = "tail_classifier"
    risk_importance = risk_summary.copy()
    if not risk_importance.empty:
        risk_importance["evidence_type"] = "error_risk_regression"

    return {
        "preflight": preflight,
        "tail_thresholds": tail_thresholds,
        "descriptive_quantiles": descriptive_quantiles,
        "residuals": residuals,
        "parcel_summary": parcel_summary,
        "concentration": concentration,
        "tail_bias": tail_bias,
        "overlap": overlap,
        "feature_associations_by_split": assoc_by_split,
        "feature_associations": assoc_summary,
        "tail_classifier_oof": classifier_oof,
        "tail_classifier_summary": classifier_summary,
        "error_risk_oof": risk_oof,
        "error_risk_summary": risk_summary,
        "feature_family_tail_importance": family_importance,
        "feature_family_error_risk": risk_importance,
        "decision": decision,
    }
