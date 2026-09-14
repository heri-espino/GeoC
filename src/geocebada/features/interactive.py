"""Reusable feature-engineering recipes for notebooks and the dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

Aggregation = Literal["mean", "median", "min", "max", "std", "sum", "slope", "auc"]


@dataclass(frozen=True)
class FeatureRecipe:
    """Declarative recipe for one grouped feature derived from a source column."""

    name: str
    value_column: str
    group_column: str = "ID_POLIGONO"
    aggregation: Aggregation = "mean"
    date_column: str | None = None
    start: str | None = None
    end: str | None = None


def apply_feature_recipe(frame: pd.DataFrame, recipe: FeatureRecipe) -> pd.DataFrame:
    """Apply a feature recipe and return one row per group with the new feature."""

    required = {recipe.group_column, recipe.value_column}
    if recipe.date_column is not None:
        required.add(recipe.date_column)
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Missing required column(s): {sorted(missing)}")

    work = frame[list(required)].copy()
    if recipe.date_column is not None:
        work[recipe.date_column] = pd.to_datetime(work[recipe.date_column], errors="coerce")
        work = work.dropna(subset=[recipe.date_column])
        if recipe.start is not None:
            work = work.loc[work[recipe.date_column] >= pd.Timestamp(recipe.start)]
        if recipe.end is not None:
            work = work.loc[work[recipe.date_column] <= pd.Timestamp(recipe.end)]

    work[recipe.value_column] = pd.to_numeric(work[recipe.value_column], errors="coerce")
    work = work.dropna(subset=[recipe.value_column])

    if recipe.aggregation in {"mean", "median", "min", "max", "std", "sum"}:
        series = work.groupby(recipe.group_column)[recipe.value_column].agg(recipe.aggregation)
    elif recipe.aggregation == "slope":
        if recipe.date_column is None:
            raise ValueError("slope aggregation requires date_column.")
        series = work.groupby(recipe.group_column).apply(
            lambda group: _temporal_slope(group, recipe.value_column, recipe.date_column),
            include_groups=False,
        )
    elif recipe.aggregation == "auc":
        if recipe.date_column is None:
            raise ValueError("auc aggregation requires date_column.")
        series = work.groupby(recipe.group_column).apply(
            lambda group: _temporal_auc(group, recipe.value_column, recipe.date_column),
            include_groups=False,
        )
    else:  # pragma: no cover - protected by Literal/type usage
        raise ValueError(f"Unsupported aggregation: {recipe.aggregation}")

    return series.rename(recipe.name).reset_index()


def merge_features(
    base: pd.DataFrame,
    *feature_frames: pd.DataFrame,
    on: str = "ID_POLIGONO",
    validate: str = "one_to_one",
) -> pd.DataFrame:
    """Left-join one-row-per-parcel feature tables onto a base table."""

    result = base.copy()
    for features in feature_frames:
        if on not in features.columns:
            raise KeyError(f"Feature frame is missing join key {on!r}.")
        result = result.merge(features, on=on, how="left", validate=validate)
    return result


def feature_recipe_to_dict(recipe: FeatureRecipe) -> dict[str, str | None]:
    """Serialize a feature recipe into a plain dictionary for YAML/JSON storage."""

    return {
        "name": recipe.name,
        "value_column": recipe.value_column,
        "group_column": recipe.group_column,
        "aggregation": recipe.aggregation,
        "date_column": recipe.date_column,
        "start": recipe.start,
        "end": recipe.end,
    }


def _temporal_slope(group: pd.DataFrame, value: str, date: str) -> float:
    ordered = group.sort_values(date)
    if len(ordered) < 2:
        return float("nan")
    x = (ordered[date] - ordered[date].min()).dt.total_seconds().to_numpy() / 86400.0
    y = ordered[value].to_numpy(dtype=float)
    if np.ptp(x) == 0:
        return float("nan")
    return float(np.polyfit(x, y, 1)[0])


def _temporal_auc(group: pd.DataFrame, value: str, date: str) -> float:
    ordered = group.sort_values(date)
    if len(ordered) < 2:
        return float("nan")
    x = (ordered[date] - ordered[date].min()).dt.total_seconds().to_numpy() / 86400.0
    y = ordered[value].to_numpy(dtype=float)
    if np.ptp(x) == 0:
        return float("nan")
    return float(np.trapezoid(y, x))
