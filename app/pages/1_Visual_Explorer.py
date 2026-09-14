"""Dedicated spatial and multivariate exploration page for GeoCebada Lab."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.visual_explorer import render_visual_explorer
from geocebada.data import attach_yield_split_metadata, load_yield_split
from geocebada.paths import source_path

st.set_page_config(page_title="GeoCebada Visual Explorer", page_icon="🗺️", layout="wide")

BASIC_FILE = "Conjunto_datos_BASICO_AgroCebada2026.csv"
PRO_FILE = "Conjunto_datos_PRO_AgroCebada.csv"
ID_COLUMN = "ID_POLIGONO"


@st.cache_data(show_spinner=False)
def _load_official_csv(filename: str) -> pd.DataFrame:
    return pd.read_csv(source_path("tabular", filename, must_exist=True))


@st.cache_data(show_spinner=False)
def _load_split() -> pd.DataFrame:
    return load_yield_split()


st.title("GeoCebada — Visual Explorer")
st.caption("Mapa · filtros · pairplot · correlaciones · outliers · grupos · missingness")

with st.sidebar:
    st.header("Dataset")
    source = st.radio(
        "Choose data",
        [
            "Official yield split",
            "Official BASIC",
            "Official PRO",
            "Upload CSV",
        ],
    )
    uploaded = None
    if source == "Upload CSV":
        uploaded = st.file_uploader("CSV file", type=["csv"])

if source == "Official yield split":
    data = _load_split()
elif source == "Official BASIC":
    data = _load_official_csv(BASIC_FILE)
elif source == "Official PRO":
    data = _load_official_csv(PRO_FILE)
elif uploaded is not None:
    data = pd.read_csv(uploaded)
else:
    st.info("Upload a CSV to start the explorer.")
    st.stop()

if source in {"Official BASIC", "Official PRO"}:
    st.warning(
        "BASIC/PRO are loaded exactly as delivered. Their temporal schema and relationship to the "
        "target agricultural cycle are still unresolved, so time-dependent interpretation must "
        "remain exploratory until the source documentation is audited."
    )

if ID_COLUMN in data.columns:
    attach_metadata = st.sidebar.checkbox(
        "Attach official split/target metadata",
        value=source in {"Official BASIC", "Official PRO"},
        help=(
            "Adds AREA_HA, CONJUNTO and observed training yield when those columns are absent. "
            "Prediction-set yield remains missing."
        ),
    )
    if attach_metadata:
        data = attach_yield_split_metadata(data, id_column=ID_COLUMN)

st.success(f"Loaded {len(data):,} rows × {len(data.columns):,} columns")
render_visual_explorer(data)
