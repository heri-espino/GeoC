"""GeoCebada interactive data, statistics and modeling laboratory."""

from __future__ import annotations

import io
import json

import pandas as pd
import plotly.express as px
import streamlit as st

from geocebada.data import load_yield_split, partition_yield_split
from geocebada.evaluation import benchmark_regressors, summarize_benchmark
from geocebada.features import FeatureRecipe, apply_feature_recipe, feature_recipe_to_dict
from geocebada.statistics import (
    correlation_screen,
    correlation_test,
    linear_regression_diagnostics,
)
from geocebada.visualization import (
    correlation_scatter,
    distribution_figure,
    missingness_table,
    residual_diagnostic_figure,
)

st.set_page_config(page_title="GeoCebada Lab", page_icon="🌾", layout="wide")

TARGET_COLUMN = "RENDIMIENTO_T_HA"
SPLIT_COLUMN = "CONJUNTO"
TRAIN_VALUE = "ENTRENAMIENTO"


@st.cache_data
def _load_official_split() -> pd.DataFrame:
    return load_yield_split()


def _analysis_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Use labeled rows only when the official split contract is present."""

    if SPLIT_COLUMN in frame.columns and TARGET_COLUMN in frame.columns:
        labeled = frame.loc[frame[SPLIT_COLUMN].eq(TRAIN_VALUE)].copy()
        if not labeled.empty:
            return labeled
    return frame.copy()


def _numeric_columns(frame: pd.DataFrame) -> list[str]:
    return frame.select_dtypes(include="number").columns.tolist()


def _render_overview(frame: pd.DataFrame) -> None:
    st.subheader("Overview")
    columns = st.columns(4)
    columns[0].metric("Rows", f"{len(frame):,}")
    columns[1].metric("Columns", len(frame.columns))
    columns[2].metric("Numeric columns", len(_numeric_columns(frame)))
    columns[3].metric("Missing cells", f"{int(frame.isna().sum().sum()):,}")

    if SPLIT_COLUMN in frame.columns:
        counts = frame[SPLIT_COLUMN].value_counts(dropna=False)
        st.plotly_chart(
            px.bar(
                counts.rename_axis(SPLIT_COLUMN).reset_index(name="count"),
                x=SPLIT_COLUMN,
                y="count",
                title="Official train/prediction split",
            ),
            use_container_width=True,
        )

    if TARGET_COLUMN in frame.columns:
        labeled = frame[TARGET_COLUMN].dropna()
        if not labeled.empty:
            st.plotly_chart(
                px.histogram(
                    labeled,
                    x=TARGET_COLUMN,
                    marginal="box",
                    title="Observed yield distribution",
                ),
                use_container_width=True,
            )

    st.markdown("#### Missingness / schema")
    st.dataframe(missingness_table(frame), use_container_width=True, hide_index=True)



def _render_explorer(frame: pd.DataFrame) -> None:
    st.subheader("Data Explorer")
    st.caption("Interactive inspection only; this does not modify source files.")

    columns = frame.columns.tolist()
    selected = st.multiselect(
        "Columns to display",
        columns,
        default=columns[: min(10, len(columns))],
        key="explorer_columns",
    )
    view = frame[selected] if selected else frame
    st.dataframe(view, use_container_width=True, height=380)

    numeric = _numeric_columns(frame)
    if numeric:
        left, right = st.columns([1, 2])
        with left:
            variable = st.selectbox("Distribution variable", numeric, key="distribution_variable")
            color_candidates = ["(none)", *[column for column in frame.columns if column != variable]]
            color = st.selectbox("Color/group", color_candidates, key="distribution_color")
        with right:
            st.plotly_chart(
                distribution_figure(
                    frame,
                    variable,
                    color=None if color == "(none)" else color,
                ),
                use_container_width=True,
            )

    buffer = io.StringIO()
    view.to_csv(buffer, index=False)
    st.download_button(
        "Download current table as CSV",
        data=buffer.getvalue(),
        file_name="geocebada_explorer.csv",
        mime="text/csv",
    )



def _render_statistics(frame: pd.DataFrame) -> None:
    st.subheader("Statistical Lab")
    st.caption(
        "Inference is performed only on complete observed rows. For the official split, hidden "
        "prediction targets are excluded automatically. Statistical significance is not causality."
    )

    analysis = _analysis_frame(frame)
    numeric = _numeric_columns(analysis)
    if len(numeric) < 2:
        st.warning("At least two numeric columns are required.")
        return

    association_tab, screen_tab, assumptions_tab = st.tabs(
        ["Bivariate association", "Multiple-testing screen", "Linear assumptions"]
    )

    with association_tab:
        x = st.selectbox("X", numeric, key="stats_x")
        y_options = [column for column in numeric if column != x]
        y = st.selectbox("Y", y_options, key="stats_y")
        method = st.radio("Association", ["spearman", "pearson"], horizontal=True)
        result = correlation_test(analysis[x], analysis[y], method=method)

        m1, m2, m3 = st.columns(3)
        m1.metric("Association", f"{result['statistic']:.4f}")
        m2.metric("p-value", f"{result['p_value']:.4g}")
        m3.metric("Complete pairs", int(result["n"]))

        group_options = ["(none)", *[column for column in analysis.columns if column not in {x, y}]]
        group = st.selectbox("Optional color/group", group_options, key="stats_group")
        st.plotly_chart(
            correlation_scatter(
                analysis,
                x,
                y,
                color=None if group == "(none)" else group,
                trendline=True,
            ),
            use_container_width=True,
        )
        if method == "pearson":
            st.info(
                "Pearson targets linear association and is sensitive to outliers. Compare with "
                "Spearman when the relationship is monotonic but non-linear or ranks are more stable."
            )

    with screen_tab:
        target = st.selectbox(
            "Target for feature screening",
            numeric,
            index=numeric.index(TARGET_COLUMN) if TARGET_COLUMN in numeric else 0,
            key="screen_target",
        )
        candidates = [column for column in numeric if column != target]
        selected = st.multiselect(
            "Candidate features",
            candidates,
            default=candidates,
            key="screen_features",
        )
        correction = st.selectbox(
            "Multiple-testing correction",
            ["fdr_bh", "holm", "bonferroni"],
            key="screen_correction",
        )
        if selected:
            screen = correlation_screen(
                analysis,
                target,
                features=selected,
                method="spearman",
                correction=correction,
            )
            st.dataframe(screen, use_container_width=True, hide_index=True)
            st.caption(
                "Adjusted p-values help control false discoveries when many variables are explored. "
                "They do not solve confounding, spatial dependence, or post-selection bias."
            )

    with assumptions_tab:
        target = st.selectbox(
            "Regression target",
            numeric,
            index=numeric.index(TARGET_COLUMN) if TARGET_COLUMN in numeric else 0,
            key="diag_target",
        )
        predictors = st.multiselect(
            "Predictors",
            [column for column in numeric if column != target],
            key="diag_predictors",
        )
        if predictors:
            try:
                diagnostics = linear_regression_diagnostics(analysis, target, predictors)
            except ValueError as exc:
                st.warning(str(exc))
            else:
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("R² (in-sample)", f"{diagnostics['r2']:.3f}")
                m2.metric("RMSE (in-sample)", f"{diagnostics['rmse']:.3f}")
                m3.metric("Shapiro p", f"{diagnostics['shapiro_p']:.4g}")
                m4.metric("Breusch-Pagan p", f"{diagnostics['breusch_pagan_p']:.4g}")

                st.plotly_chart(
                    residual_diagnostic_figure(
                        diagnostics["predictions"],
                        diagnostics["residuals"],
                    ),
                    use_container_width=True,
                )
                left, right = st.columns(2)
                with left:
                    st.markdown("**Coefficients**")
                    st.dataframe(diagnostics["coefficients"], hide_index=True, use_container_width=True)
                with right:
                    st.markdown("**Variance inflation factors**")
                    st.dataframe(diagnostics["vif"], hide_index=True, use_container_width=True)

                st.warning(
                    "These are diagnostics, not automatic pass/fail rules. Spatial independence and "
                    "valid temporal ordering must be checked separately for GeoCebada."
                )



def _render_feature_lab(frame: pd.DataFrame) -> None:
    st.subheader("Feature Engineering Lab")
    st.caption(
        "Create reproducible feature recipes interactively. Useful recipes should later be promoted "
        "into src/geocebada/features/ and covered by tests."
    )

    columns = frame.columns.tolist()
    numeric = _numeric_columns(frame)
    if not numeric:
        st.warning("No numeric source columns are available.")
        return

    default_group = "ID_POLIGONO" if "ID_POLIGONO" in columns else columns[0]
    group = st.selectbox("Group / parcel identifier", columns, index=columns.index(default_group))
    value = st.selectbox("Source numeric variable", numeric)
    aggregation = st.selectbox(
        "Aggregation",
        ["mean", "median", "min", "max", "std", "sum", "slope", "auc"],
    )

    date_candidates = ["(none)", *columns]
    date_column = st.selectbox("Date column", date_candidates)
    requires_date = aggregation in {"slope", "auc"}
    if requires_date and date_column == "(none)":
        st.info("Slope and AUC require a date column.")
        return

    start = end = None
    if date_column != "(none)":
        parsed_dates = pd.to_datetime(frame[date_column], errors="coerce").dropna()
        if not parsed_dates.empty:
            min_date = parsed_dates.min().date()
            max_date = parsed_dates.max().date()
            left, right = st.columns(2)
            start_date = left.date_input("Start", value=min_date, min_value=min_date, max_value=max_date)
            end_date = right.date_input("End", value=max_date, min_value=min_date, max_value=max_date)
            start = start_date.isoformat()
            end = end_date.isoformat()

    default_name = f"{value}_{aggregation}"
    feature_name = st.text_input("Feature name", value=default_name)
    recipe = FeatureRecipe(
        name=feature_name,
        value_column=value,
        group_column=group,
        aggregation=aggregation,
        date_column=None if date_column == "(none)" else date_column,
        start=start,
        end=end,
    )

    try:
        generated = apply_feature_recipe(frame, recipe)
    except (ValueError, KeyError, TypeError) as exc:
        st.error(str(exc))
        return

    left, right = st.columns([2, 1])
    with left:
        st.dataframe(generated, use_container_width=True, height=350)
    with right:
        st.plotly_chart(
            px.histogram(generated, x=feature_name, marginal="box", title="Derived feature"),
            use_container_width=True,
        )

    serialized = feature_recipe_to_dict(recipe)
    st.code(json.dumps(serialized, indent=2), language="json")
    st.download_button(
        "Download feature recipe",
        data=json.dumps(serialized, indent=2),
        file_name=f"{feature_name}.json",
        mime="application/json",
    )
    st.download_button(
        "Download generated feature table",
        data=generated.to_csv(index=False),
        file_name=f"{feature_name}.csv",
        mime="text/csv",
    )



def _render_model_lab(frame: pd.DataFrame) -> None:
    st.subheader("Model Lab")
    st.caption(
        "Quick random-CV baselines for exploration. They are not the final competition estimate: "
        "GeoCebada still needs spatial/grouped validation after the parcel audit."
    )

    analysis = _analysis_frame(frame)
    numeric = _numeric_columns(analysis)
    if len(numeric) < 2:
        st.warning("At least two numeric columns are required.")
        return

    target = st.selectbox(
        "Target",
        numeric,
        index=numeric.index(TARGET_COLUMN) if TARGET_COLUMN in numeric else 0,
        key="model_target",
    )
    features = st.multiselect(
        "Features",
        [column for column in numeric if column != target],
        key="model_features",
    )
    folds = st.slider("K-folds", min_value=3, max_value=10, value=5)

    if features and st.button("Run baseline benchmark", type="primary"):
        with st.spinner("Running cross-validation..."):
            try:
                scores = benchmark_regressors(
                    analysis,
                    target,
                    features,
                    n_splits=folds,
                    random_state=42,
                )
            except ValueError as exc:
                st.error(str(exc))
                return

        summary = summarize_benchmark(scores)
        st.dataframe(summary, hide_index=True, use_container_width=True)
        st.plotly_chart(
            px.box(scores, x="model", y="rmse", points="all", title="RMSE across folds"),
            use_container_width=True,
        )
        with st.expander("Fold-level scores"):
            st.dataframe(scores, hide_index=True, use_container_width=True)
        st.warning(
            "Do not choose the final model from this random K-fold result alone. Nearby parcels may "
            "share environmental information and make random CV optimistic."
        )



def _render_methodology() -> None:
    st.subheader("Interpretation guardrails")
    st.markdown(
        """
This laboratory deliberately separates four questions:

1. **Description:** what patterns are present in the observed data?
2. **Statistical inference:** how uncertain are estimated relationships under explicit assumptions?
3. **Prediction:** how well does a model generalize to unseen parcels under a defensible validation design?
4. **Causality:** what would happen under an intervention? The current observational challenge data do
   **not** establish causal effects by themselves.

For GeoCebada, the main unresolved methodological risks remain temporal leakage, spatial dependence,
the agricultural cycle represented by the target, and the final validation mechanism. Results shown
here should be treated as exploratory until those questions are resolved.
"""
    )


st.title("GeoCebada Lab")
st.caption("Exploración · Inferencia estadística · Feature engineering · Modelado")

with st.sidebar:
    st.header("Data source")
    source = st.radio("Choose data", ["Official yield split", "Upload CSV"])
    uploaded = None
    if source == "Upload CSV":
        uploaded = st.file_uploader("CSV file", type=["csv"])
    st.divider()
    st.caption(
        "Official source files remain immutable. Interactive transformations only create in-memory "
        "or downloadable derived tables."
    )

if source == "Official yield split":
    data = _load_official_split()
elif uploaded is not None:
    data = pd.read_csv(uploaded)
else:
    st.info("Upload a CSV to start the laboratory.")
    st.stop()

st.success(f"Loaded {len(data):,} rows × {len(data.columns):,} columns")

overview, explorer, statistics_tab, feature_tab, model_tab, methodology = st.tabs(
    [
        "Overview",
        "Data Explorer",
        "Statistical Lab",
        "Feature Engineering",
        "Model Lab",
        "Methodology",
    ]
)

with overview:
    _render_overview(data)
with explorer:
    _render_explorer(data)
with statistics_tab:
    _render_statistics(data)
with feature_tab:
    _render_feature_lab(data)
with model_tab:
    _render_model_lab(data)
with methodology:
    _render_methodology()
