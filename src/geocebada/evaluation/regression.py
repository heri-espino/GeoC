"""Small-sample regression benchmarking utilities."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin, clone
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def default_regressors(*, random_state: int = 42) -> dict[str, RegressorMixin]:
    """Return lightweight baseline regressors suitable for early comparison."""

    return {
        "LinearRegression": LinearRegression(),
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        ),
    }


def benchmark_regressors(
    frame: pd.DataFrame,
    target: str,
    features: Iterable[str],
    *,
    regressors: Mapping[str, RegressorMixin] | None = None,
    n_splits: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compare regression baselines using identical shuffled K-fold splits.

    This is an exploratory random-CV benchmark only. It must not be treated as
    the final GeoCebada validation protocol until spatial/grouped validation is
    established.
    """

    feature_list = list(features)
    if not feature_list:
        raise ValueError("At least one feature is required.")

    clean = frame[[target, *feature_list]].dropna().astype(float)
    if len(clean) < n_splits * 2:
        raise ValueError("Not enough complete observations for the requested CV folds.")

    x = clean[feature_list]
    y = clean[target]
    models = dict(regressors or default_regressors(random_state=random_state))
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    rows: list[dict[str, float | int | str]] = []
    for model_name, estimator in models.items():
        for fold, (train_index, test_index) in enumerate(splitter.split(x), start=1):
            fitted = clone(estimator)
            fitted.fit(x.iloc[train_index], y.iloc[train_index])
            prediction = fitted.predict(x.iloc[test_index])
            observed = y.iloc[test_index]
            rows.append(
                {
                    "model": model_name,
                    "fold": fold,
                    "n_test": int(len(test_index)),
                    "rmse": float(np.sqrt(mean_squared_error(observed, prediction))),
                    "mae": float(mean_absolute_error(observed, prediction)),
                    "r2": float(r2_score(observed, prediction)),
                }
            )

    return pd.DataFrame(rows)


def summarize_benchmark(scores: pd.DataFrame) -> pd.DataFrame:
    """Summarize fold-level benchmark metrics as mean and standard deviation."""

    required = {"model", "rmse", "mae", "r2"}
    missing = required.difference(scores.columns)
    if missing:
        raise KeyError(f"Missing score column(s): {sorted(missing)}")

    summary = scores.groupby("model")[["rmse", "mae", "r2"]].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.reset_index().sort_values("rmse_mean").reset_index(drop=True)
