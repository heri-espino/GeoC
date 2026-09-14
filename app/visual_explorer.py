"""High-density visual exploration dashboard for GeoCebada Lab."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from geocebada.data import filter_frame
from geocebada.geo import load_parcels
from geocebada.visualization import (
    correlation_heatmap,
    correlation_scatter,
    missingness_table,
    outlier_summary,
    pairplot_figure,
    parcel_map_figure,
)

ID_COLUMN = "ID_POLIGONO"
TARGET_COLUMN = "RENDIMIENTO_T_HA"
SPLIT_COLUMN = "CONJUNTO"


@st.cache_data(show_spinner=False)
def _load_parcels_cached():
    return load_parcels()


def _numeric_columns(frame: pd.DataFrame) -> list[str]:
    return frame.select_dtypes(include="number").columns.tolist()


def _categorical_candidates(frame: pd.DataFrame, *, max_levels: int = 40) -> list[str]:
    candidates: list[str] = []
    for column in frame.columns:
        unique = frame[column].nunique(dropna=True)
        if 1 < unique <= max_levels and not pd.api.types.is_numeric_dtype(frame[column]):
            candidates.append(column)
    return candidates


def _filter_panel(frame: pd.DataFrame) -> pd.DataFrame:
    categorical: dict[str, list[object]] = {}
    numeric_ranges: dict[str, tuple[float, float]] = {}
    text_contains: dict[str, str] = {}

    with st.expander("Filters", expanded=True):
        top_left, top_right = st.columns(2)
        if SPLIT_COLUMN in frame.columns:
            values = frame[SPLIT_COLUMN].dropna().unique().tolist()
            selected = top_left.multiselect(
                "Train / prediction",
                values,
                default=values,
                key="visual_split_filter",
            )
            categorical[SPLIT_COLUMN] = selected

        text_column = ID_COLUMN if ID_COLUMN in frame.columns else None
        if text_column is not None:
            query = top_right.text_input(
                f"Search {text_column}",
                key="visual_id_search",
                placeholder="e.g. AGC_047",
            )
            if query:
                text_contains[text_column] = query

        category_options = [
            column
            for column in _categorical_candidates(frame)
            if column not in {SPLIT_COLUMN, ID_COLUMN}
        ]
        category_columns = st.multiselect(
            "Additional categorical filters",
            category_options,
            key="visual_category_columns",
        )
        for column in category_columns:
            values = frame[column].dropna().unique().tolist()
            selected = st.multiselect(
                column,
                values,
                default=values,
                key=f"visual_category_{column}",
            )
            categorical[column] = selected

        numeric_options = _numeric_columns(frame)
        range_columns = st.multiselect(
            "Numeric range filters",
            numeric_options,
            key="visual_numeric_filter_columns",
        )
        for column in range_columns:
            values = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
            values = values.dropna()
            if values.empty:
                continue
            lower = float(values.min())
            upper = float(values.max())
            if lower == upper:
                st.caption(f"{column}: constant at {lower:g}")
                continue
            selected_range = st.slider(
                column,
                min_value=lower,
                max_value=upper,
                value=(lower, upper),
                key=f"visual_range_{column}",
            )
            numeric_ranges[column] = selected_range

        drop_missing = st.checkbox(
            "Drop rows containing any missing value",
            value=False,
            key="visual_drop_missing",
        )

    return filter_frame(
        frame,
        categorical=categorical,
        numeric_ranges=numeric_ranges,
        text_contains=text_contains,
        drop_missing=drop_missing,
    )


def _render_map(frame: pd.DataFrame) -> None:
    st.markdown("#### Parcel map")
    st.caption(
        "The map reprojects only when the parcel source already carries CRS metadata. "
        "GeoCebada never guesses a missing CRS."
    )
    if frame.empty:
        st.info("No rows remain after filtering.")
        return

    try:
        parcels = _load_parcels_cached()
    except (ImportError, FileNotFoundError, ValueError, OSError) as exc:
        st.warning(f"Parcel geometry could not be loaded: {exc}")
        return

    geometry_columns = [column for column in parcels.columns if column != parcels.geometry.name]
    if not geometry_columns:
        st.warning("The parcel geometry source has no attribute column that can be used as an ID.")
        return

    data_id_default = ID_COLUMN if ID_COLUMN in frame.columns else frame.columns[0]
    parcel_id_default = ID_COLUMN if ID_COLUMN in geometry_columns else geometry_columns[0]

    controls = st.columns(3)
    data_id = controls[0].selectbox(
        "Data parcel ID",
        frame.columns.tolist(),
        index=frame.columns.tolist().index(data_id_default),
        key="map_data_id",
    )
    parcel_id = controls[1].selectbox(
        "Geometry parcel ID",
        geometry_columns,
        index=geometry_columns.index(parcel_id_default),
        key="map_geometry_id",
    )

    color_candidates = [
        column
        for column in frame.columns
        if column != data_id and frame[column].notna().any()
    ]
    if not color_candidates:
        st.info("No non-empty variable is available to color the map.")
        return

    priority = [SPLIT_COLUMN, TARGET_COLUMN, "AREA_HA"]
    default_color = next(
        (column for column in priority if column in color_candidates),
        color_candidates[0],
    )
    color = controls[2].selectbox(
        "Color parcels by",
        color_candidates,
        index=color_candidates.index(default_color),
        key="map_color",
    )

    hover_candidates = [column for column in frame.columns if column not in {data_id, color}]
    default_hover = [
        column
        for column in [SPLIT_COLUMN, "AREA_HA", TARGET_COLUMN]
        if column in hover_candidates
    ]
    hover = st.multiselect(
        "Hover fields",
        hover_candidates,
        default=default_hover,
        key="map_hover",
    )

    st.caption(f"Source parcel CRS: {parcels.crs}")
    try:
        figure = parcel_map_figure(
            parcels,
            frame,
            parcel_id=parcel_id,
            data_id=data_id,
            color=color,
            hover_columns=hover,
        )
    except (KeyError, ValueError, TypeError) as exc:
        st.warning(str(exc))
        return
    st.plotly_chart(figure, use_container_width=True)


def _render_pairplot(frame: pd.DataFrame) -> None:
    st.markdown("#### Pairplot / scatter matrix")
    numeric = _numeric_columns(frame)
    if len(numeric) < 2:
        st.info("At least two numeric variables are required.")
        return

    preferred = [column for column in ["AREA_HA", TARGET_COLUMN] if column in numeric]
    defaults = list(dict.fromkeys([*preferred, *numeric]))[: min(4, len(numeric))]
    dimensions = st.multiselect(
        "Variables (2–8 recommended)",
        numeric,
        default=defaults,
        max_selections=8,
        key="pairplot_dimensions",
    )
    color_options = ["(none)", *frame.columns.tolist()]
    default_color_index = (
        color_options.index(SPLIT_COLUMN) if SPLIT_COLUMN in frame.columns else 0
    )
    color = st.selectbox(
        "Color / group",
        color_options,
        index=default_color_index,
        key="pairplot_color",
    )
    if len(dimensions) < 2:
        st.info("Select at least two variables.")
        return
    try:
        figure = pairplot_figure(
            frame,
            dimensions,
            color=None if color == "(none)" else color,
        )
    except (KeyError, ValueError) as exc:
        st.warning(str(exc))
        return
    st.plotly_chart(figure, use_container_width=True)
    st.caption(
        "Pairplots are exploratory. Strong-looking relationships can be driven by outliers, "
        "group structure, repeated observations, spatial dependence, or temporal leakage."
    )


def _render_correlations(frame: pd.DataFrame) -> None:
    st.markdown("#### Correlation structure")
    numeric = _numeric_columns(frame)
    if len(numeric) < 2:
        st.info("At least two numeric variables are required.")
        return

    controls = st.columns([3, 1])
    selected = controls[0].multiselect(
        "Variables",
        numeric,
        default=numeric[: min(10, len(numeric))],
        max_selections=20,
        key="heatmap_columns",
    )
    method = controls[1].selectbox(
        "Method",
        ["spearman", "pearson", "kendall"],
        key="heatmap_method",
    )
    if len(selected) < 2:
        st.info("Select at least two variables.")
        return

    st.plotly_chart(
        correlation_heatmap(frame, selected, method=method),
        use_container_width=True,
    )
    correlation = frame[selected].corr(method=method)
    with st.expander("Correlation matrix as table"):
        st.dataframe(correlation, use_container_width=True)


def _render_bivariate(frame: pd.DataFrame) -> None:
    st.markdown("#### Bivariate explorer")
    numeric = _numeric_columns(frame)
    if len(numeric) < 2:
        st.info("At least two numeric variables are required.")
        return

    controls = st.columns(4)
    x = controls[0].selectbox("X", numeric, key="visual_bivariate_x")
    y_candidates = [column for column in numeric if column != x]
    y = controls[1].selectbox("Y", y_candidates, key="visual_bivariate_y")
    color_options = ["(none)", *frame.columns.tolist()]
    default_color_index = (
        color_options.index(SPLIT_COLUMN) if SPLIT_COLUMN in frame.columns else 0
    )
    color = controls[2].selectbox(
        "Color",
        color_options,
        index=default_color_index,
        key="visual_bivariate_color",
    )
    trendline = controls[3].checkbox("OLS trendline", value=True, key="visual_bivariate_ols")

    st.plotly_chart(
        correlation_scatter(
            frame,
            x,
            y,
            color=None if color == "(none)" else color,
            trendline=trendline,
        ),
        use_container_width=True,
    )


def _render_outliers(frame: pd.DataFrame) -> None:
    st.markdown("#### Univariate outlier screening")
    numeric = _numeric_columns(frame)
    if not numeric:
        st.info("No numeric variables are available.")
        return

    selected = st.multiselect(
        "Variables",
        numeric,
        default=numeric[: min(8, len(numeric))],
        key="outlier_columns",
    )
    if not selected:
        return
    summary = outlier_summary(frame, selected)
    st.dataframe(summary, hide_index=True, use_container_width=True)

    variable = st.selectbox("Inspect variable", selected, key="outlier_variable")
    st.plotly_chart(
        px.box(frame, y=variable, points="all", title=f"IQR view: {variable}"),
        use_container_width=True,
    )
    st.caption(
        "IQR flags are diagnostic, not automatic deletion rules. Agronomic extremes may be "
        "real and potentially important for prediction."
    )


def _render_groups(frame: pd.DataFrame) -> None:
    st.markdown("#### Group summaries")
    group_candidates = _categorical_candidates(frame, max_levels=60)
    numeric = _numeric_columns(frame)
    if not group_candidates or not numeric:
        st.info("This view requires a low-cardinality grouping variable and numeric data.")
        return

    controls = st.columns([1, 2])
    default_group = SPLIT_COLUMN if SPLIT_COLUMN in group_candidates else group_candidates[0]
    group = controls[0].selectbox(
        "Group",
        group_candidates,
        index=group_candidates.index(default_group),
        key="group_summary_group",
    )
    values = controls[1].multiselect(
        "Numeric variables",
        numeric,
        default=numeric[: min(4, len(numeric))],
        key="group_summary_values",
    )
    aggregations = st.multiselect(
        "Statistics",
        ["mean", "median", "std", "min", "max", "count"],
        default=["mean", "median", "std"],
        key="group_summary_aggs",
    )
    if not values or not aggregations:
        return

    summary = frame.groupby(group, dropna=False)[values].agg(aggregations)
    summary.columns = [f"{column}_{stat}" for column, stat in summary.columns]
    st.dataframe(summary.reset_index(), use_container_width=True, hide_index=True)


def _render_missingness(frame: pd.DataFrame) -> None:
    st.markdown("#### Missingness and schema after filtering")
    table = missingness_table(frame)
    st.dataframe(table, use_container_width=True, hide_index=True)
    top = table.head(20).sort_values("missing_pct")
    st.plotly_chart(
        px.bar(top, x="missing_pct", y="column", orientation="h", title="Top missingness"),
        use_container_width=True,
    )
    with st.expander("Descriptive statistics"):
        st.dataframe(frame.describe(include="all").T, use_container_width=True)


def render_visual_explorer(frame: pd.DataFrame) -> None:
    """Render the filtered spatial and multivariate exploration dashboard."""

    st.subheader("Visual Explorer")
    st.caption(
        "Filter once, then inspect the same subset spatially and statistically. "
        "All operations are read-only and happen in memory."
    )

    filtered = _filter_panel(frame)
    metric_columns = st.columns(4)
    metric_columns[0].metric("Filtered rows", f"{len(filtered):,}")
    share = len(filtered) / len(frame) if len(frame) else 0.0
    metric_columns[1].metric("Rows retained", f"{share:.1%}")
    if ID_COLUMN in filtered.columns:
        metric_columns[2].metric("Unique parcels", f"{filtered[ID_COLUMN].nunique():,}")
    else:
        metric_columns[2].metric("Unique parcels", "n/a")
    metric_columns[3].metric("Missing cells", f"{int(filtered.isna().sum().sum()):,}")

    if filtered.empty:
        st.warning("No observations match the current filters.")
        return

    map_tab, pair_tab, corr_tab, biv_tab, outlier_tab, group_tab, missing_tab = st.tabs(
        [
            "Map",
            "Pairplot",
            "Correlations",
            "Bivariate",
            "Outliers",
            "Groups",
            "Missingness",
        ]
    )
    with map_tab:
        _render_map(filtered)
    with pair_tab:
        _render_pairplot(filtered)
    with corr_tab:
        _render_correlations(filtered)
    with biv_tab:
        _render_bivariate(filtered)
    with outlier_tab:
        _render_outliers(filtered)
    with group_tab:
        _render_groups(filtered)
    with missing_tab:
        _render_missingness(filtered)
