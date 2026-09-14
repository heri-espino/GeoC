"""Exploratory plots shared by notebooks and the Streamlit laboratory."""

from __future__ import annotations

from collections.abc import Iterable

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


def pairplot_figure(
    frame: pd.DataFrame,
    columns: Iterable[str],
    *,
    color: str | None = None,
    max_points: int = 3000,
) -> go.Figure:
    """Build an interactive scatter-matrix for a selected set of numeric variables."""

    dimensions = list(dict.fromkeys(columns))
    if len(dimensions) < 2:
        raise ValueError("At least two variables are required for a pairplot.")
    missing = [column for column in dimensions if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing column(s): {missing}")
    if color is not None and color not in frame.columns:
        raise KeyError(color)

    view = frame
    if len(view) > max_points:
        view = view.sample(max_points, random_state=42)

    figure = px.scatter_matrix(view, dimensions=dimensions, color=color)
    figure.update_traces(diagonal_visible=False, showupperhalf=False)
    figure.update_layout(height=max(650, 130 * len(dimensions)))
    return figure


def correlation_heatmap(
    frame: pd.DataFrame,
    columns: Iterable[str],
    *,
    method: str = "spearman",
) -> go.Figure:
    """Build an interactive correlation heatmap for selected numeric variables."""

    selected = list(dict.fromkeys(columns))
    if len(selected) < 2:
        raise ValueError("At least two variables are required for a correlation heatmap.")
    missing = [column for column in selected if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing column(s): {missing}")
    if method not in {"pearson", "spearman", "kendall"}:
        raise ValueError("method must be 'pearson', 'spearman', or 'kendall'.")

    correlation = frame[selected].corr(method=method)
    return px.imshow(
        correlation,
        text_auto=".2f",
        zmin=-1,
        zmax=1,
        aspect="auto",
        title=f"{method.title()} correlation",
    )


def outlier_summary(
    frame: pd.DataFrame,
    columns: Iterable[str],
    *,
    whisker_width: float = 1.5,
) -> pd.DataFrame:
    """Summarize univariate IQR outliers for selected numeric variables."""

    rows: list[dict[str, float | int | str]] = []
    for column in columns:
        if column not in frame.columns:
            raise KeyError(column)
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        if values.empty:
            rows.append(
                {
                    "feature": column,
                    "n": 0,
                    "lower_bound": float("nan"),
                    "upper_bound": float("nan"),
                    "outliers": 0,
                    "outlier_pct": float("nan"),
                }
            )
            continue
        q1 = float(values.quantile(0.25))
        q3 = float(values.quantile(0.75))
        iqr = q3 - q1
        lower = q1 - whisker_width * iqr
        upper = q3 + whisker_width * iqr
        count = int(((values < lower) | (values > upper)).sum())
        rows.append(
            {
                "feature": column,
                "n": int(len(values)),
                "lower_bound": lower,
                "upper_bound": upper,
                "outliers": count,
                "outlier_pct": 100 * count / len(values),
            }
        )
    return pd.DataFrame(rows).sort_values("outlier_pct", ascending=False).reset_index(drop=True)


def residual_diagnostic_figure(
    predictions: pd.Series,
    residuals: pd.Series,
) -> go.Figure:
    """Plot residuals against fitted values with a horizontal zero reference line."""

    frame = pd.DataFrame({"prediction": predictions, "residual": residuals}).dropna()
    figure = px.scatter(frame, x="prediction", y="residual")
    figure.add_hline(y=0.0, line_dash="dash")
    return figure
