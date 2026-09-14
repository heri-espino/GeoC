"""Exploratory plots shared by notebooks and the Streamlit laboratory."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def missingness_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize missing values by column."""

    missing = frame.isna().sum()
    result = pd.DataFrame(
        {
            "column": frame.columns,
            "missing": missing.to_numpy(),
            "missing_pct": (missing.to_numpy() / max(len(frame), 1)) * 100,
            "dtype": [str(dtype) for dtype in frame.dtypes],
        }
    )
    return result.sort_values(["missing_pct", "column"], ascending=[False, True]).reset_index(
        drop=True
    )


def distribution_figure(frame: pd.DataFrame, column: str, *, color: str | None = None) -> go.Figure:
    """Build an interactive histogram for one numeric variable."""

    if column not in frame.columns:
        raise KeyError(column)
    if color is not None and color not in frame.columns:
        raise KeyError(color)
    return px.histogram(frame, x=column, color=color, marginal="box")


def correlation_scatter(
    frame: pd.DataFrame,
    x: str,
    y: str,
    *,
    color: str | None = None,
    trendline: bool = True,
) -> go.Figure:
    """Build an interactive scatter plot for exploring bivariate relationships."""

    for column in [x, y, color]:
        if column is not None and column not in frame.columns:
            raise KeyError(column)

    kwargs: dict[str, object] = {"data_frame": frame, "x": x, "y": y, "color": color}
    if trendline:
        kwargs["trendline"] = "ols"
    return px.scatter(**kwargs)


def residual_diagnostic_figure(
    predictions: pd.Series,
    residuals: pd.Series,
) -> go.Figure:
    """Plot residuals against fitted values with a horizontal zero reference line."""

    frame = pd.DataFrame({"prediction": predictions, "residual": residuals}).dropna()
    figure = px.scatter(frame, x="prediction", y="residual")
    figure.add_hline(y=0.0, line_dash="dash")
    return figure
