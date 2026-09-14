"""Statistical tests and diagnostics used by notebooks and the interactive lab."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def correlation_test(
    x: pd.Series,
    y: pd.Series,
    *,
    method: str = "pearson",
) -> dict[str, float | int | str]:
    """Test association between two numeric series after pairwise NA removal."""

    pair = pd.concat([x, y], axis=1).dropna()
    if len(pair) < 3:
        raise ValueError("At least three complete paired observations are required.")

    x_clean = pair.iloc[:, 0].astype(float)
    y_clean = pair.iloc[:, 1].astype(float)
    method = method.lower()

    if method == "pearson":
        statistic, p_value = stats.pearsonr(x_clean, y_clean)
    elif method == "spearman":
        statistic, p_value = stats.spearmanr(x_clean, y_clean)
    else:
        raise ValueError("method must be 'pearson' or 'spearman'.")

    return {
        "method": method,
        "statistic": float(statistic),
        "p_value": float(p_value),
        "n": int(len(pair)),
    }


def correlation_screen(
    frame: pd.DataFrame,
    target: str,
    *,
    features: Iterable[str] | None = None,
    method: str = "spearman",
    correction: str = "fdr_bh",
) -> pd.DataFrame:
    """Screen numeric features against a target and adjust for multiple testing."""

    if target not in frame.columns:
        raise KeyError(target)

    if features is None:
        features = [
            column
            for column in frame.select_dtypes(include="number").columns
            if column != target
        ]

    rows: list[dict[str, float | int | str]] = []
    for feature in features:
        if feature not in frame.columns:
            raise KeyError(feature)
        result = correlation_test(frame[feature], frame[target], method=method)
        rows.append({"feature": feature, **result})

    result_frame = pd.DataFrame(rows)
    if result_frame.empty:
        return result_frame

    result_frame["p_adjusted"] = adjust_pvalues(
        result_frame["p_value"].to_numpy(),
        method=correction,
    )
    result_frame["abs_statistic"] = result_frame["statistic"].abs()
    return result_frame.sort_values(
        ["p_adjusted", "abs_statistic"],
        ascending=[True, False],
    ).reset_index(drop=True)


def adjust_pvalues(
    pvalues: Iterable[float],
    *,
    method: str = "fdr_bh",
) -> np.ndarray:
    """Adjust a collection of p-values for multiple hypothesis testing."""

    values = np.asarray(list(pvalues), dtype=float)
    if values.size == 0:
        return values
    if np.isnan(values).any() or ((values < 0) | (values > 1)).any():
        raise ValueError("p-values must be finite numbers between 0 and 1.")

    method = method.lower()
    if method == "bonferroni":
        return np.minimum(values * len(values), 1.0)
    if method == "holm":
        order = np.argsort(values)
        adjusted = np.empty_like(values)
        running_max = 0.0
        m = len(values)
        for rank, index in enumerate(order):
            candidate = min((m - rank) * values[index], 1.0)
            running_max = max(running_max, candidate)
            adjusted[index] = running_max
        return adjusted
    if method == "fdr_bh":
        order = np.argsort(values)
        ordered = values[order]
        m = len(values)
        adjusted_ordered = ordered * m / np.arange(1, m + 1)
        adjusted_ordered = np.minimum.accumulate(adjusted_ordered[::-1])[::-1]
        adjusted_ordered = np.minimum(adjusted_ordered, 1.0)
        adjusted = np.empty_like(values)
        adjusted[order] = adjusted_ordered
        return adjusted

    raise ValueError("method must be one of: 'bonferroni', 'holm', 'fdr_bh'.")


def vif_table(frame: pd.DataFrame, features: Iterable[str]) -> pd.DataFrame:
    """Compute variance inflation factors using auxiliary linear regressions."""

    feature_list = list(features)
    if not feature_list:
        raise ValueError("At least one feature is required.")

    matrix = frame[feature_list].dropna().astype(float)
    if len(matrix) < 3:
        raise ValueError("At least three complete observations are required.")

    rows = []
    for feature in feature_list:
        others = [column for column in feature_list if column != feature]
        if not others:
            vif = 1.0
        else:
            model = LinearRegression().fit(matrix[others], matrix[feature])
            r_squared = model.score(matrix[others], matrix[feature])
            vif = float("inf") if r_squared >= 1.0 else 1.0 / (1.0 - r_squared)
        rows.append({"feature": feature, "vif": float(vif)})

    return pd.DataFrame(rows).sort_values("vif", ascending=False).reset_index(drop=True)


def linear_regression_diagnostics(
    frame: pd.DataFrame,
    target: str,
    features: Iterable[str],
) -> dict[str, object]:
    """Fit OLS-style linear regression and return core assumption diagnostics."""

    feature_list = list(features)
    if not feature_list:
        raise ValueError("At least one predictor is required.")

    columns = [target, *feature_list]
    clean = frame[columns].dropna().astype(float)
    if len(clean) <= len(feature_list) + 2:
        raise ValueError("Not enough complete rows for the requested linear model.")

    x = clean[feature_list]
    y = clean[target]
    model = LinearRegression().fit(x, y)
    predictions = model.predict(x)
    residuals = y.to_numpy() - predictions

    shapiro_stat, shapiro_p = stats.shapiro(residuals)
    bp_stat, bp_p = _breusch_pagan(residuals, x.to_numpy())

    coefficients = pd.DataFrame(
        {
            "feature": feature_list,
            "coefficient": model.coef_.astype(float),
        }
    )

    return {
        "n": int(len(clean)),
        "intercept": float(model.intercept_),
        "coefficients": coefficients,
        "predictions": pd.Series(predictions, index=clean.index, name="prediction"),
        "residuals": pd.Series(residuals, index=clean.index, name="residual"),
        "rmse": float(np.sqrt(mean_squared_error(y, predictions))),
        "mae": float(mean_absolute_error(y, predictions)),
        "r2": float(r2_score(y, predictions)),
        "shapiro_w": float(shapiro_stat),
        "shapiro_p": float(shapiro_p),
        "breusch_pagan_lm": float(bp_stat),
        "breusch_pagan_p": float(bp_p),
        "vif": vif_table(clean, feature_list),
    }


def _breusch_pagan(residuals: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    squared = np.square(residuals)
    auxiliary = LinearRegression().fit(x, squared)
    r_squared = auxiliary.score(x, squared)
    statistic = len(squared) * max(float(r_squared), 0.0)
    p_value = stats.chi2.sf(statistic, df=x.shape[1])
    return float(statistic), float(p_value)
