# Mini muestras reales de datos

Esta carpeta se genera con `python tools/build_data_catalog.py`. Contiene recortes
deterministas y pequenos de los datos reales para desarrollar, depurar y probar loaders y
feature engineering sin leer los archivos completos.

**No son un dataset de entrenamiento**, no preservan distribuciones estadisticas y no deben
usarse para estimar metricas. El contrato completo esta en `data/data_catalog.json`.

Subcarpetas posibles: `tabular/`, `vector/`, `raster/` y `netcdf/`.
