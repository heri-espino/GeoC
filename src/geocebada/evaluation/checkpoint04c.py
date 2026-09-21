"""Focused global CatBoost anchor utilities for Checkpoint 04C.1.

This module deliberately avoids another broad model search. It evaluates a small
set of CatBoost configurations inherited from Checkpoint 03C.2 on the frozen
Checkpoint 04B pseudo-competition splits.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

ID_COLUMN = "ID_POLIGONO"


def resolve_agronomic_competition_features(
    joined: pd.DataFrame,
    manifest: Mapping[str, Any],
) -> tuple[str, ...]:
    """Resolve numeric clean+competition agronomic features from the manifest."""

    records = manifest.get("features")
    if not isinstance(records, list):
        raise ValueError("Agronomic manifest must contain a 'features' list.")

    selected: list[str] = []
    for record in records:
        if not isinstance(record, Mapping):
            continue
        column = str(record.get("column", ""))
        mode = str(record.get("mode", ""))
        if (
            column
            and mode in {"clean", "competition"}
            and column in joined.columns
            and pd.api.types.is_numeric_dtype(joined[column])
        ):
            selected.append(column)

    features = tuple(dict.fromkeys(selected))
    if not features:
        raise ValueError("No numeric competition-track agronomic features resolved.")
    return features


def predict_catboost_seed_ensemble(
    x: pd.DataFrame,
    y: np.ndarray,
    observed_positions: Sequence[int],
    query_positions: Sequence[int],
    *,
    params: Mapping[str, Any],
    seeds: Sequence[int],
    task_type: str,
    devices: str = "0",
) -> np.ndarray:
    """Fit one CatBoost model per seed on visible labels and average predictions."""

    try:
        from catboost import CatBoostRegressor
    except ImportError as exc:
        raise ImportError(
            "Checkpoint 04C.1 requires CatBoost. Install the project with "
            "python -m pip install -e '.[models]'."
        ) from exc

    observed = np.asarray(list(map(int, observed_positions)), dtype=int)
    query = np.asarray(list(map(int, query_positions)), dtype=int)
    if len(observed) < 10:
        raise ValueError("CatBoost anchor requires at least 10 visible labels.")
    if len(query) == 0:
        return np.asarray([], dtype=float)

    features = x.apply(pd.to_numeric, errors="coerce").replace(
        [np.inf, -np.inf],
        np.nan,
    )
    targets = np.asarray(y, dtype=float)
    if not np.isfinite(targets[observed]).all():
        raise ValueError("Visible CatBoost targets must all be finite.")

    resolved_task = str(task_type).upper()
    if resolved_task not in {"GPU", "CPU"}:
        raise ValueError("task_type must be GPU or CPU.")

    seed_values = tuple(map(int, seeds))
    if not seed_values:
        raise ValueError("At least one CatBoost seed is required.")

    predictions: list[np.ndarray] = []
    for seed in seed_values:
        kwargs: dict[str, Any] = {
            "iterations": int(params.get("iterations", 400)),
            "depth": int(params["depth"]),
            "learning_rate": float(params["learning_rate"]),
            "l2_leaf_reg": float(params["l2_leaf_reg"]),
            "loss_function": "RMSE",
            "eval_metric": "RMSE",
            "verbose": False,
            "allow_writing_files": False,
            "random_seed": seed,
            "task_type": resolved_task,
            "thread_count": -1,
        }
        if resolved_task == "GPU":
            kwargs["devices"] = str(devices)

        model = CatBoostRegressor(**kwargs)
        model.fit(features.iloc[observed], targets[observed])
        predictions.append(
            np.asarray(model.predict(features.iloc[query]), dtype=float).reshape(-1)
        )

    return np.mean(np.vstack(predictions), axis=0)


def blend_anchor_predictions(
    baseline: Sequence[float],
    anchor: Sequence[float],
    *,
    anchor_weight: float,
) -> np.ndarray:
    """Blend a frozen baseline with a global anchor using a fixed convex weight."""

    weight = float(anchor_weight)
    if not 0.0 <= weight <= 1.0:
        raise ValueError("anchor_weight must lie in [0, 1].")
    baseline_array = np.asarray(baseline, dtype=float)
    anchor_array = np.asarray(anchor, dtype=float)
    if baseline_array.shape != anchor_array.shape:
        raise ValueError("Baseline and anchor predictions must have identical shapes.")
    return (1.0 - weight) * baseline_array + weight * anchor_array


def residual_correlation_table(
    predictions: pd.DataFrame,
    *,
    family: str,
    methods: Sequence[str],
) -> pd.DataFrame:
    """Return pairwise residual correlations for selected methods within one family."""

    selected = tuple(map(str, methods))
    frame = predictions.loc[
        predictions["family"].eq(str(family))
        & predictions["method"].isin(selected)
    ].copy()
    frame["residual"] = frame["observed"] - frame["predicted"]

    pivot = frame.pivot_table(
        index=["split_id", ID_COLUMN],
        columns="method",
        values="residual",
        aggfunc="first",
    )
    rows: list[dict[str, Any]] = []
    for left_index, left in enumerate(selected):
        for right in selected[left_index + 1 :]:
            if left not in pivot or right not in pivot:
                continue
            pair = pivot[[left, right]].dropna()
            rows.append(
                {
                    "family": str(family),
                    "method_a": left,
                    "method_b": right,
                    "n_pairs": int(len(pair)),
                    "residual_correlation": (
                        float(pair[left].corr(pair[right]))
                        if len(pair) >= 2
                        else float("nan")
                    ),
                }
            )
    return pd.DataFrame(rows)
