# Mini muestras reales de datos

Esta carpeta se genera y actualiza con:

```bash
python tools/build_data_catalog.py
```

Contiene recortes deterministas y pequeños de los datos reales para desarrollar, depurar y probar loaders y feature engineering sin leer las fuentes completas.

**No son un dataset de entrenamiento**, no preservan distribuciones estadísticas y no deben usarse para estimar métricas. El contrato estructural completo vive en `data/data_catalog.json`.

Subcarpetas generadas posibles: `tabular/`, `vector/`, `raster/` y `netcdf/`.
