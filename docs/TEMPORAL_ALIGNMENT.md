# Alineación temporal de BASIC/PRO

Las tablas satelitales oficiales son longitudinales y contienen muchas observaciones por parcela. Para comparar parcelas y construir una tabla model-ready, GeoCebada puede alinear las capturas irregulares a una rejilla temporal común mediante vecinos cercanos en fecha.

## Objetivo

Para una parcela y sensor dados, sea una serie de observaciones

\[
(t_1, x_1),\ldots,(t_n,x_n).
\]

Para cada fecha objetivo \(\tau\) de una rejilla regular se seleccionan las `k` capturas más cercanas en tiempo. El peso temporal es

\[
w_t=\exp\left(-\frac{|t-\tau|}{h}\right),
\]

con `h = bandwidth_days`.

Si se activa el ajuste por nubosidad, el peso total es

\[
w=w_t\left(1-\frac{\text{nubosidad}}{100}\right).
\]

Los valores faltantes se manejan de forma independiente por variable. Una captura puede contribuir a NDVI aunque otro índice de esa fila esté ausente.

## Separación por sensor

Por defecto la alineación se realiza por `ID_POLIGONO` y `sensor`. Esto evita mezclar silenciosamente índices con el mismo nombre provenientes de Sentinel-2, Landsat o Planet.

`aligned_to_wide()` conserva el sensor en los nombres de las columnas, por ejemplo:

```text
SENTINEL-2__ndvi_promedio__2025_06_10
LANDSAT__ndvi_promedio__2025_06_10
```

Una armonización entre sensores debe ser una decisión metodológica posterior y explícita.

## Funciones

```python
from geocebada.features import (
    align_temporal_knn,
    aligned_to_wide,
    make_temporal_grid,
    temporal_backend_available,
    temporal_coverage_summary,
)
```

`temporal_coverage_summary()` permite comparar disponibilidad, gaps y nubosidad entre parcelas de entrenamiento y predicción sin utilizar el target oculto.

`make_temporal_grid()` crea una rejilla regular, por ejemplo cada 14 días.

`align_temporal_knn()` alinea muchas variables simultáneamente a esa rejilla.

`aligned_to_wide()` convierte la representación alineada a una fila por parcela para experimentos tabulares posteriores.

## CPU y GPU

El backend puede ser `cpu`, `gpu` o `auto`.

```python
aligned = align_temporal_knn(
    basic,
    value_columns,
    start="2025-04-01",
    end="2025-10-31",
    freq="14D",
    k=3,
    bandwidth_days=14,
    max_distance_days=30,
    backend="auto",
)
```

El backend GPU usa CuPy para distancias y reducciones ponderadas. Pandas sigue realizando el parsing y agrupamiento en CPU. Por tanto, la GPU no está garantizada como más rápida: con ~100 mil filas y muchos grupos pequeños, un CPU moderno puede ser competitivo. La GPU gana relevancia al alinear muchas columnas simultáneamente y en experimentos repetidos.

En una máquina NVIDIA/CUDA 12 compatible:

```bash
python -m pip install -e ".[dev,geo,gpu]"
```

Luego:

```python
from geocebada.features import temporal_backend_available

temporal_backend_available("auto")
```

retornará `gpu` si CuPy detecta una GPU CUDA y `cpu` en caso contrario.

## Guardrails

El target corresponde al ciclo abril–octubre de 2025, pero el cutoff operacional exacto de una predicción pre-cosecha todavía debe fijarse. Alinear hasta octubre es válido para exploración descriptiva de la cobertura, pero no autoriza automáticamente esas fechas como predictores del modelo final.

La selección de `k`, `bandwidth_days`, frecuencia de rejilla y `max_distance_days` debe hacerse dentro de una estrategia de validación adecuada. No deben elegirse mirando los rendimientos ocultos de las 59 parcelas de predicción.

La notebook `notebooks/02_cobertura_alineacion_temporal.ipynb` está diseñada para ejecutar este flujo paso a paso.
