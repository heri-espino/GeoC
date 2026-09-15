"""Dedicated spatial and multivariate exploration page for GeoCebada Lab."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.visual_explorer import render_visual_explorer
from geocebada.data import (
    attach_yield_split_metadata,
    load_basic_data,
    load_pro_data,
    load_yield_split,
)

st.set_page_config(page_title="GeoCebada Visual Explorer", page_icon="🗺️", layout="wide")

ID_COLUMN = "ID_POLIGONO"
SPLIT_SOURCE = "Yield split (4 columns)"
BASIC_SOURCE = "BASIC remote sensing (96 columns)"
PRO_SOURCE = "PRO remote sensing (20 columns)"
UPLOAD_SOURCE = "Upload CSV"


@st.cache_data(show_spinner=False)
def _load_split() -> pd.DataFrame:
    return load_yield_split()


@st.cache_data(show_spinner=False)
def _load_basic() -> pd.DataFrame:
    return load_basic_data()


@st.cache_data(show_spinner=False)
def _load_pro() -> pd.DataFrame:
    return load_pro_data()


st.title("GeoCebada — Visual Explorer")
st.caption("Mapa · filtros · pairplot · correlaciones · outliers · grupos · missingness")

with st.sidebar:
    st.header("Dataset")
    source = st.radio(
        "Choose data",
        [SPLIT_SOURCE, BASIC_SOURCE, PRO_SOURCE, UPLOAD_SOURCE],
        index=1,
        help=(
            "The 4-column table only defines parcel ID, area, target and train/prediction split. "
            "BASIC and PRO contain the remote-sensing observations used for richer exploration."
        ),
    )
    uploaded = None
    if source == UPLOAD_SOURCE:
        uploaded = st.file_uploader("CSV file", type=["csv"])

if source == SPLIT_SOURCE:
    data = _load_split()
elif source == BASIC_SOURCE:
    data = _load_basic()
elif source == PRO_SOURCE:
    data = _load_pro()
elif uploaded is not None:
    data = pd.read_csv(uploaded)
else:
    st.info("Upload a CSV to start the explorer.")
    st.stop()

if source in {BASIC_SOURCE, PRO_SOURCE}:
    st.info(
        "This is longitudinal remote-sensing data: each parcel appears repeatedly across capture "
        "dates/sensors. Use it for exploration and feature engineering; do not treat rows as "
        "independent parcels in supervised validation."
    )

if ID_COLUMN in data.columns:
    attach_metadata = st.sidebar.checkbox(
        "Attach official split/target metadata",
        value=source in {BASIC_SOURCE, PRO_SOURCE},
        help=(
            "Adds AREA_HA, CONJUNTO and observed training yield when those columns are absent. "
            "Prediction-set yield remains missing."
        ),
    )
    if attach_metadata:
        data = attach_yield_split_metadata(data, id_column=ID_COLUMN)

st.success(f"Loaded {len(data):,} rows × {len(data.columns):,} columns")
render_visual_explorer(data)
