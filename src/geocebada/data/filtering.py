"""Reusable filtering helpers for exploratory dashboards and notebooks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Iterable

import pandas as pd


def filter_frame(
    frame: pd.DataFrame,
    *,
    categorical: Mapping[str, Iterable[object]] | None = None,
    numeric_ranges: Mapping[str, tuple[float, float]] | None = None,
    text_contains: Mapping[str, str] | None = None,
    drop_missing: bool = False,
) -> pd.DataFrame:
    """Return a filtered copy of ``frame`` using declarative filter specifications.

    Parameters
    ----------
    frame:
        Source table.
    categorical:
        Mapping of column name to accepted values. Empty selections are ignored.
    numeric_ranges:
        Inclusive lower/upper bounds for numeric columns.
    text_contains:
        Case-insensitive substring filters for text-like columns.
    drop_missing:
        If ``True``, remove rows containing any missing value after the other filters.
    """

    result = frame.copy()

    for column, values in (categorical or {}).items():
        if column not in result.columns:
            raise KeyError(column)
        accepted = list(values)
        if accepted:
            result = result.loc[result[column].isin(accepted)]

    for column, bounds in (numeric_ranges or {}).items():
        if column not in result.columns:
            raise KeyError(column)
        lower, upper = bounds
        numeric = pd.to_numeric(result[column], errors="coerce")
        result = result.loc[numeric.between(lower, upper, inclusive="both")]

    for column, query in (text_contains or {}).items():
        if column not in result.columns:
            raise KeyError(column)
        if query:
            mask = result[column].astype("string").str.contains(query, case=False, na=False)
            result = result.loc[mask]

    if drop_missing:
        result = result.dropna()

    return result.copy()
