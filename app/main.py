"""GeoCebada web application.

Initial Streamlit scaffold. The inference layer will be connected once the final
feature pipeline and model interface are defined.
"""

import streamlit as st

st.set_page_config(page_title="GeoCebada", page_icon="🌾", layout="wide")

st.title("GeoCebada")
st.caption("Predicción agroclimática del rendimiento de cebada")

st.info(
    "Aplicación en construcción. Aquí se integrarán las predicciones, mapas, "
    "explicabilidad del modelo y consulta por parcela."
)

left, right = st.columns(2)

with left:
    st.subheader("Predicción")
    st.write("Consulta del rendimiento estimado por parcela.")

with right:
    st.subheader("Interpretación")
    st.write("Importancia de variables, factores agroclimáticos y diagnóstico del modelo.")
