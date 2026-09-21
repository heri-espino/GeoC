"""Checkpoint 04D.1 local/graph refinement and nested routing utilities.

04D.1 reuses the frozen 04B pseudo-competition memberships. It refines only
methods that survived 04B: local Ridge, graph-Laplacian regression and a global
PLS anchor. Hyperparameter selection and support-tier routing can then be
evaluated leave-one-pseudo-split-out.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, StratifiedKFold

from geocebada.evaluation.checkpoint04a import ID_COLUMN
from geocebada.evaluation.checkpoint04b import (
    predict_graph_laplacian,
    regression_metrics,
)


@dataclass(frozen=True)
class SplitPositions:
    """Resolved pseudo-train and pseudo-target row positions for one frozen split."""

    split_id: str
    family: str
    repeat: int
    observed_positions: tuple[int, ...]
    query_positions: tuple[int, ...]


def split_positions_from_membership(
    frame: pd.DataFrame,
    membership: pd.DataFrame,
    *,
    train_value: str,
    split_column: str,
) -> list[SplitPositions]:
    """Resolve committed 04B split membership into row-position tuples."""

    id_to_position = {
        parcel_id: position
        for position, parcel_id in enumerate(frame[ID_COLUMN].astype(str))
    }
    train_positions = set(
        np.flatnonzero(frame[split_column].eq(train_value).to_numpy()).tolist()
    )
    required = {"split_id", "family", "repeat", ID_COLUMN, "role"}
    missing = required.difference(membership.columns)
    if missing:
        raise KeyError(f"Pseudo-split membership missing columns: {sorted(missing)}")

    result: list[SplitPositions] = []
    for split_id, group in membership.groupby("split_id", sort=False):
        query_ids = group.loc[group["role"].eq("pseudo_target"), ID_COLUMN].astype(str)
        query_positions = sorted(id_to_position[parcel_id] for parcel_id in query_ids)
        observed_positions = sorted(train_positions.difference(query_positions))
        first = group.iloc[0]
        result.append(
            SplitPositions(
                split_id=str(split_id),
                family=str(first["family"]),
                repeat=int(first["repeat"]),
                observed_positions=tuple(observed_positions),
                query_positions=tuple(query_positions),
            )
        )
    return result


def predict_weighted_local_ridge(
    x: np.ndarray,
    distance_matrix: np.ndarray,
    y: np.ndarray,
    observed_positions: Sequence[int],
    query_positions: Sequence[int],
    *,
    k: int,
    alpha: float,
    distance_power: float,
) -> np.ndarray:
    """Fit one inverse-distance weighted Ridge model per query parcel."""

    observed = np.asarray(list(map(int, observed_positions)), dtype=int)
    query = list(map(int, query_positions))
    if len(observed) < 3:
        raise ValueError("Local Ridge requires at least three observed parcels.")

    result: list[float] = []
    for position in query:
        distances = np.asarray(distance_matrix[position, observed], dtype=float)
        finite = np.isfinite(distances)
        candidates = observed[finite]
        candidate_distances = distances[finite]
        if len(candidates) < 3:
            result.append(float(np.mean(y[observed])))
            continue

        order = np.argsort(candidate_distances)
        take = order[: max(3, min(int(k), len(order)))]
        local_positions = candidates[take]
        local_distances = candidate_distances[take]
        power = max(float(distance_power), 0.0)
        if power == 0:
            weights = np.ones(len(local_positions), dtype=float)
        else:
            weights = 1.0 / np.power(local_distances + 1.0e-6, power)

        model = Ridge(alpha=float(alpha))
        model.fit(
            np.asarray(x, dtype=float)[local_positions],
            np.asarray(y, dtype=float)[local_positions],
            sample_weight=weights,
        )
        result.append(float(model.predict(np.asarray(x, dtype=float)[[position]])[0]))
    return np.asarray(result, dtype=float)


def fit_pls_anchor_with_cross_fitted_residuals(
    x: np.ndarray,
    y: np.ndarray,
    observed_positions: Sequence[int],
    *,
    strata: Sequence[str] | None,
    n_components: int,
    n_splits: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a full PLS anchor and cross-fitted residuals on visible labels only."""

    features = np.asarray(x, dtype=float)
    targets = np.asarray(y, dtype=float)
    observed = np.asarray(list(map(int, observed_positions)), dtype=int)
    if len(observed) < 4:
        raise ValueError("PLS anchor requires at least four observed parcels.")

    components = max(1, min(int(n_components), len(observed) - 1, features.shape[1]))
    full_model = PLSRegression(n_components=components, scale=False, max_iter=1_000)
    full_model.fit(features[observed], targets[observed])
    full_predictions = np.asarray(full_model.predict(features), dtype=float).reshape(-1)

    folds = max(2, min(int(n_splits), len(observed)))
    if strata is not None:
        labels = np.asarray(list(map(str, strata)), dtype=object)[observed]
        counts = pd.Series(labels).value_counts()
    else:
        labels = None
        counts = pd.Series(dtype=int)

    if labels is not None and not counts.empty and int(counts.min()) >= folds:
        splitter = StratifiedKFold(
            n_splits=folds,
            shuffle=True,
            random_state=int(random_state),
        )
        split_iterator = splitter.split(np.zeros(len(observed)), labels)
    else:
        splitter = KFold(
            n_splits=folds,
            shuffle=True,
            random_state=int(random_state),
        )
        split_iterator = splitter.split(np.zeros(len(observed)))

    oof = np.full(len(observed), np.nan, dtype=float)
    for train_index, validation_index in split_iterator:
        train_positions = observed[np.asarray(train_index, dtype=int)]
        validation_positions = observed[np.asarray(validation_index, dtype=int)]
        fold_components = max(
            1,
            min(int(n_components), len(train_positions) - 1, features.shape[1]),
        )
        model = PLSRegression(
            n_components=fold_components,
            scale=False,
            max_iter=1_000,
        )
        model.fit(features[train_positions], targets[train_positions])
        oof[np.asarray(validation_index, dtype=int)] = np.asarray(
            model.predict(features[validation_positions]),
            dtype=float,
        ).reshape(-1)

    if not np.isfinite(oof).all():
        raise RuntimeError("Cross-fitted PLS anchor did not cover every observed parcel.")

    residuals = np.full(len(targets), np.nan, dtype=float)
    residuals[observed] = targets[observed] - oof
    return full_predictions, residuals


def predict_residual_graph_correction(
    anchor_predictions: np.ndarray,
    cross_fitted_residuals: np.ndarray,
    adjacency: np.ndarray,
    observed_positions: Sequence[int],
    query_positions: Sequence[int],
    *,
    regularization: float,
) -> np.ndarray:
    """Add graph-propagated cross-fitted residuals to a global anchor."""

    query = np.asarray(list(map(int, query_positions)), dtype=int)
    correction = predict_graph_laplacian(
        adjacency,
        np.asarray(cross_fitted_residuals, dtype=float),
        observed_positions,
        query,
        regularization=float(regularization),
    )
    return np.asarray(anchor_predictions, dtype=float)[query] + correction


def summarize_predictions_by_method(predictions: pd.DataFrame) -> pd.DataFrame:
    """Summarize equal-weight split RMSE and pooled row metrics by method."""

    frame = predictions.copy()
    frame["squared_error"] = np.square(frame["observed"] - frame["predicted"])
    frame["absolute_error"] = np.abs(frame["observed"] - frame["predicted"])

    split_rows: list[dict[str, Any]] = []
    for (family, split_id, method), group in frame.groupby(
        ["family", "split_id", "method"]
    ):
        metrics = regression_metrics(group["observed"], group["predicted"])
        split_rows.append(
            {
                "family": family,
                "split_id": split_id,
                "method": method,
                **metrics,
            }
        )
    split_metrics = pd.DataFrame(split_rows)

    rows: list[dict[str, Any]] = []
    for (family, method), group in frame.groupby(["family", "method"]):
        matching = split_metrics.loc[
            split_metrics["family"].eq(family)
            & split_metrics["method"].eq(method)
        ]
        pooled = regression_metrics(group["observed"], group["predicted"])
        rows.append(
            {
                "family": family,
                "method": method,
                "n_splits": int(matching["split_id"].nunique()),
                "n_predictions": int(len(group)),
                "rmse_mean": float(matching["rmse"].mean()),
                "rmse_median": float(matching["rmse"].median()),
                "rmse_std": float(matching["rmse"].std(ddof=1)),
                "rmse_worst": float(matching["rmse"].max()),
                "pooled_rmse": pooled["rmse"],
                "pooled_mae": pooled["mae"],
                "pooled_r2": pooled["r2"],
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["family", "rmse_mean", "method"]
    ).reset_index(drop=True)


def leave_one_split_out_method_selection(
    predictions: pd.DataFrame,
    *,
    family: str,
    candidate_methods: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate split-level hyperparameter selection without scoring on its training splits."""

    frame = predictions.loc[predictions["family"].eq(family)].copy()
    if candidate_methods is not None:
        frame = frame.loc[frame["method"].isin(list(map(str, candidate_methods)))].copy()
    splits = list(dict.fromkeys(frame["split_id"].astype(str)))
    rows: list[dict[str, Any]] = []
    selected_predictions: list[pd.DataFrame] = []

    for holdout in splits:
        training = frame.loc[~frame["split_id"].eq(holdout)].copy()
        validation = frame.loc[frame["split_id"].eq(holdout)].copy()
        training["squared_error"] = np.square(
            training["observed"] - training["predicted"]
        )
        split_method = (
            training.groupby(["split_id", "method"], as_index=False)["squared_error"]
            .mean()
            .assign(rmse=lambda table: np.sqrt(table["squared_error"]))
        )
        ranking = (
            split_method.groupby("method", as_index=False)["rmse"]
            .mean()
            .sort_values(["rmse", "method"])
        )
        if ranking.empty:
            raise ValueError("No candidate methods available for LOSO selection.")
        selected = str(ranking.iloc[0]["method"])
        selected_block = validation.loc[validation["method"].eq(selected)].copy()
        metrics = regression_metrics(
            selected_block["observed"],
            selected_block["predicted"],
        )
        rows.append(
            {
                "holdout_split": holdout,
                "selected_method": selected,
                "selection_rmse": float(ranking.iloc[0]["rmse"]),
                **{f"holdout_{key}": value for key, value in metrics.items()},
            }
        )
        selected_predictions.append(selected_block)

    return pd.DataFrame(rows), pd.concat(selected_predictions, ignore_index=True)


def leave_one_split_out_tier_routing(
    predictions: pd.DataFrame,
    *,
    family: str,
    candidate_methods: Sequence[str],
    tiers: Sequence[str] = ("low", "mid", "high"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate support-tier method routing on pseudo-splits excluded from route fitting."""

    frame = predictions.loc[
        predictions["family"].eq(family)
        & predictions["method"].isin(list(map(str, candidate_methods)))
    ].copy()
    split_ids = list(dict.fromkeys(frame["split_id"].astype(str)))
    tier_values = tuple(map(str, tiers))
    selection_rows: list[dict[str, Any]] = []
    routed_rows: list[dict[str, Any]] = []

    for holdout in split_ids:
        training = frame.loc[~frame["split_id"].eq(holdout)].copy()
        validation = frame.loc[frame["split_id"].eq(holdout)].copy()
        route: dict[str, str] = {}

        training["squared_error"] = np.square(
            training["observed"] - training["predicted"]
        )
        split_tier_method = (
            training.groupby(
                ["split_id", "x_support_tier", "method"],
                as_index=False,
            )["squared_error"]
            .mean()
            .assign(rmse=lambda table: np.sqrt(table["squared_error"]))
        )

        for tier in tier_values:
            ranking = (
                split_tier_method.loc[
                    split_tier_method["x_support_tier"].eq(tier)
                ]
                .groupby("method", as_index=False)["rmse"]
                .mean()
                .sort_values(["rmse", "method"])
            )
            if ranking.empty:
                continue
            selected = str(ranking.iloc[0]["method"])
            route[tier] = selected
            selection_rows.append(
                {
                    "holdout_split": holdout,
                    "x_support_tier": tier,
                    "selected_method": selected,
                    "selection_rmse": float(ranking.iloc[0]["rmse"]),
                }
            )

        identity_columns = [
            "split_id",
            ID_COLUMN,
            "observed",
            "x_support_score",
            "x_support_tier",
        ]
        parcels = validation.loc[:, identity_columns].drop_duplicates()
        lookup = validation.set_index([ID_COLUMN, "method"])

        for parcel in parcels.itertuples(index=False):
            tier = str(parcel.x_support_tier)
            selected = route.get(tier)
            if selected is None:
                continue
            row = lookup.loc[(str(parcel.ID_POLIGONO), selected)]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            routed_rows.append(
                {
                    "split_id": str(parcel.split_id),
                    ID_COLUMN: str(parcel.ID_POLIGONO),
                    "observed": float(parcel.observed),
                    "predicted": float(row["predicted"]),
                    "x_support_score": float(parcel.x_support_score),
                    "x_support_tier": tier,
                    "selected_method": selected,
                }
            )

    return pd.DataFrame(selection_rows), pd.DataFrame(routed_rows)


def summarize_nested_predictions(
    predictions: pd.DataFrame,
    *,
    label: str,
) -> dict[str, float | int | str]:
    """Summarize nested selected/routed predictions with equal-weight split RMSE."""

    split_metrics: list[float] = []
    for _, group in predictions.groupby("split_id"):
        split_metrics.append(regression_metrics(group["observed"], group["predicted"])["rmse"])
    pooled = regression_metrics(predictions["observed"], predictions["predicted"])
    return {
        "label": str(label),
        "n_splits": int(predictions["split_id"].nunique()),
        "n_predictions": int(len(predictions)),
        "rmse_mean": float(np.mean(split_metrics)),
        "rmse_median": float(np.median(split_metrics)),
        "rmse_worst": float(np.max(split_metrics)),
        "pooled_rmse": pooled["rmse"],
        "pooled_mae": pooled["mae"],
        "pooled_r2": pooled["r2"],
    }
