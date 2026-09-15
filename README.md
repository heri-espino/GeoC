# GeoCebada

**Predicción agroclimática y geoespacial del rendimiento de cebada** para el Reto AgroCebada FIRA 2026.

GeoCebada desarrolla un pipeline reproducible para estimar el rendimiento agrícola de parcelas de cebada a partir de percepción remota, clima, topografía y geometría espacial. El proyecto incluye una librería Python compartida (`geocebada`) y un laboratorio interactivo en Streamlit para exploración, inferencia estadística, feature engineering y comparación de modelos.

> Para trabajo asistido por agentes/Codex, leer primero `AGENTS.md`, `.ai_handoff`, `data/.ai_handoff`, `docs/VARIABLES.md` y `src/geocebada/README.md`.

## Problema

La unidad final de predicción es una **parcela georreferenciada**:

- 197 parcelas totales;
- 138 `ENTRENAMIENTO` con `RENDIMIENTO_T_HA` observado;
- 59 `PREDICCION` con rendimiento oculto por FIRA.

El identificador canónico es `ID_POLIGONO`. La documentación oficial confirma que el target corresponde al **ciclo de producción abril–octubre de 2025**.

Formalmente,

\[
\hat y_i=f(X_i),
\]

donde \(X_i\) puede integrar señal satelital, clima, topografía, geometría y features espaciales/temporales derivados.

## Datos confirmados

### Target y split

Archivo: `data/source/tabular/ID_area_rendimiento_70_30_Reto_AgroCebada.csv`.

Columnas: `ID_POLIGONO`, `AREA_HA`, `RENDIMIENTO_T_HA`, `CONJUNTO`.

### BASIC

`Conjunto_datos_BASICO_AgroCebada2026.csv`:

- 107,666 filas × 96 columnas;
- 197 parcelas;
- Sentinel-2 + Landsat;
- cobertura 2022–2025;
- 23 familias de índices con `promedio`, `std`, `max`, `min`;
- VI6T corresponde a la señal Landsat; la mayor parte de los otros índices proviene de Sentinel-2;
- incluye `fecha_captura`, `sensor` y `porcentaje_nubosidad`.

### PRO

`Conjunto_datos_PRO_AgroCebada.csv`:

- 47,804 filas × 20 columnas;
- 197 parcelas;
- Planet;
- cobertura 2025;
- NDVI, EVI, LAI y MSAVI con `promedio`, `std`, `max`, `min`;
- incluye `fecha_captura`, `sensor` y `porcentaje_nubosidad`.

BASIC y PRO son longitudinales: las filas son observaciones repetidas de parcela/fecha/sensor, no muestras independientes de rendimiento. Variables con el mismo nombre provenientes de sensores diferentes no deben fusionarse automáticamente como si fueran equivalentes.

El diccionario completo y source-backed está en **`docs/VARIABLES.md`**.

### Clima y topografía

| Fuente | Variable | Cobertura | CRS | Resolución aproximada |
|---|---|---|---|---:|
| CHIRPS | precipitación mensual acumulada | 2022–2025 | EPSG:4326 | 0.05° (~5 km) |
| CHIRTS-ERA5 | Tmin/Tmax mensuales | 2022–2025 | EPSG:4326 | 0.05° (~5 km) |
| INEGI CEM | elevación y pendiente | estática | EPSG:6372 | producto entregado a 120 m |

El CRS de las geometrías de parcela sigue pendiente de verificación directa.

## Preguntas todavía abiertas

No asumir respuestas sin evidencia:

1. fecha exacta de corte/horizonte en la que debe entenderse una predicción operativa dentro del ciclo abril–octubre de 2025;
2. estrategia de armonización entre sensores cuando se combinen BASIC y PRO;
3. mecanismo de estratificación del split 70/30, si existe;
4. estrategia de validación que mejor aproxime la evaluación oculta de FIRA;
5. CRS exacto de las geometrías de parcela.

## Arquitectura

```text
notebooks / app / experimentos
             ↓
        import geocebada
             ↓
      src/geocebada/
```

```text
GeoCebada/
├── app/                         # interfaces Streamlit
├── configs/
├── data/
│   ├── source/                  # fuentes oficiales, inmutables
│   ├── raw/                     # ingestión derivada
│   ├── interim/                 # transformaciones intermedias
│   └── processed/               # datos model-ready
├── docs/
│   ├── VARIABLES.md             # diccionario de variables
│   ├── FUNCTION_INDEX.md        # índice autogenerado de funciones
│   ├── VISUAL_EXPLORER.md
│   ├── api/                     # Sphinx
│   ├── official/
│   ├── reference/
│   └── AI_USAGE.md
├── models/
├── notebooks/
├── reports/
├── src/geocebada/               # lógica reusable / fuente de verdad
└── tests/
```

## GeoCebada Lab

`app/main.py` incluye Overview, Data Explorer, Prediction Set, Statistical Lab, Feature Engineering, Model Lab y Methodology.

`app/pages/1_Visual_Explorer.py` permite cargar BASIC, PRO, el split oficial o un CSV, adjuntar metadata oficial por `ID_POLIGONO` y explorar:

- mapa de parcelas;
- filtros compartidos;
- pairplot;
- correlaciones Pearson/Spearman/Kendall;
- relaciones bivariadas;
- outliers;
- resúmenes por grupo;
- missingness.

Los targets ocultos de `PREDICCION` nunca se completan ni se usan para scoring.

## Feature engineering

BASIC/PRO deben convertirse en información a nivel parcela respetando fecha y sensor. Ejemplos de features potenciales:

```text
ndvi_mean_apr_jun
ndvi_peak
ndvi_auc
ndvi_slope
precip_sum_apr_jun
tmin_mean_apr_jun
tmax_mean_apr_jun
elevation_mean
slope_mean
```

Son features derivados de GeoCebada, no columnas originales. Las recetas exploratorias que se vuelvan estables deben promoverse a `src/geocebada/features/` y cubrirse con tests.

## Protección contra leakage

- **Target:** nunca usar o reconstruir los 59 rendimientos ocultos.
- **Temporal:** el target es abril–octubre de 2025, pero el cutoff operativo aún debe fijarse. Ningún feature final puede utilizar información posterior al horizonte de predicción elegido.
- **Agrupación:** nunca hacer CV fila-a-fila sobre BASIC/PRO; como mínimo, las observaciones de una misma parcela deben quedar en el mismo fold.
- **Espacial:** parcelas cercanas pueden compartir señal ambiental; random parcel CV puede ser optimista.
- **Preprocesamiento:** imputación, escalado, selección, PCA y transformaciones aprendidas deben ajustarse dentro de cada fold.

## Librería y documentación

Instalación editable:

```bash
pip install -e ".[dev,geo]"
```

Ejemplo:

```python
from geocebada.data import load_basic_data, load_yield_split
from geocebada.statistics import correlation_screen
```

Antes de crear una función nueva, buscar en `docs/FUNCTION_INDEX.md`. Tras modificar API pública:

```bash
python tools/generate_function_index.py
```

Sphinx:

```bash
pip install -e ".[docs]"
sphinx-build -b html docs docs/_build/html
```

## Ejecutar la app

```bash
pip install -e ".[dev,geo]"
streamlit run app/main.py
```

## Estrategia de modelado

Sólo existen 138 targets observados. La prioridad es controlar capacidad, construir features defendibles y validar correctamente. Baselines/candidatos incluyen media/mediana, Linear/Ridge/Elastic Net, Random Forest/Extra Trees y boosting si aporta evidencia CV estable.

No se seleccionará un modelo final a partir del K-fold aleatorio exploratorio del dashboard. Antes deben definirse el cutoff temporal, agrupación espacial y protocolo de validación.

## Pipeline previsto

```text
official sources
      ↓
data + variable audit
      ↓
EDA / hipótesis / estadística
      ↓
definir horizonte de predicción
      ↓
spatial + temporal alignment
      ↓
feature engineering por parcela
      ↓
leakage-safe grouped/spatial CV
      ↓
model comparison + ablations
      ↓
explainability + diagnostics
      ↓
final frozen pipeline
      ↓
59 predictions
```

## Próximos pasos

1. auditar las fechas/sensores y missingness de BASIC/PRO con foco en el ciclo 2025;
2. inspeccionar geometrías, CRS, validez y cobertura de las 197 parcelas;
3. fijar un cutoff/horizonte de predicción pre-cosecha y ventanas temporales legales;
4. construir el explorador temporal BASIC/PRO;
5. extraer clima/topografía por parcela y crear features reproducibles;
6. añadir Moran's I y diagnósticos espaciales cuando la geometría esté validada;
7. añadir grouped/spatial CV al Model Lab;
8. construir la tabla model-ready y realizar ablations por dominio;
9. congelar preprocessing/modelo antes de generar las 59 predicciones finales.

## Calidad y trazabilidad

CI valida Ruff, pytest, sincronización de `FUNCTION_INDEX` y build de Sphinx con warnings como errores.

El uso material de IA se registra en `docs/AI_USAGE.md`. `.ai_handoff` y `AGENTS.md` son handoffs operativos y no sustituyen el registro solicitado por FIRA.
