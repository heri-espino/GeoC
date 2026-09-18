# GeoCebada

**Predicción agroclimática y geoespacial del rendimiento de cebada** para el Reto AgroCebada FIRA 2026.

GeoCebada desarrolla un pipeline reproducible para estimar el rendimiento agrícola de parcelas de cebada a partir de percepción remota, clima, topografía y geometría espacial. El proyecto incluye una librería Python compartida (`geocebada`) y un laboratorio interactivo en Streamlit para exploración, inferencia estadística, feature engineering y comparación de modelos.

> Para trabajo asistido por agentes/Codex, empezar por **`docs/AGENT_GUIDE.md`**, luego `.ai_handoff` y **`docs/PROJECT_HISTORY.md`**. Ahí está el estado actual, qué ya se cerró, qué artifacts son canónicos y qué no debe repetirse.

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

El diccionario completo y source-backed está en **`docs/VARIABLES.md`**. El inventario actualizado de fuentes oficiales/externas está en **`docs/DATA_SOURCES.md`**. El contrato ejecutable previo a feature engineering está en **`docs/DATA_CONTRACT_V2.md`** y **`configs/data_contract_v2.yaml`**.

### Clima y topografía

| Fuente | Variable | Cobertura | CRS | Resolución aproximada |
|---|---|---|---|---:|
| CHIRPS | precipitación mensual acumulada | 2022–2025 | EPSG:4326 | 0.05° (~5 km) |
| CHIRTS-ERA5 | Tmin/Tmax mensuales | 2022–2025 | EPSG:4326 | 0.05° (~5 km) |
| INEGI CEM | elevación y pendiente | estática | EPSG:6372 | producto entregado a 120 m |

El CRS de las geometrías de parcela fue verificado directamente como `EPSG:4326`.

## Preguntas todavía abiertas

No asumir respuestas sin evidencia:

1. fecha exacta de corte/horizonte en la que debe entenderse una predicción operativa dentro del ciclo abril–octubre de 2025;
2. estrategia de armonización entre sensores cuando se combinen BASIC y PRO;
3. mecanismo de estratificación del split 70/30, si existe;
4. estrategia de validación que mejor aproxime la evaluación oculta de FIRA;

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
├── checkpoints/                  # hitos reproducibles del proyecto
│   └── 01_data/                  # adquisición, audit y fixtures de datos
├── app/                         # interfaces Streamlit
├── configs/
├── data/
│   ├── source/                  # fuentes oficiales, inmutables
│   ├── raw/                     # ingestión derivada
│   ├── interim/                 # transformaciones intermedias
│   └── processed/               # derivados; features_v1 canónico sí se versiona
├── docs/
│   ├── AGENT_GUIDE.md           # entrada operativa para agentes
│   ├── PROJECT_HISTORY.md       # historia cronológica del proyecto
│   ├── DATA_SOURCES.md          # inventario e integración de datos
│   ├── DATA_CONTRACT_V2.md      # audit gate antes de features
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

## Checkpoints

El cierre de la fase de datos está documentado en **`checkpoints/01_data/README.md`**. La tabla maestra determinista ya fue construida y validada en la workstation: 197 parcelas = 138 `ENTRENAMIENTO` + 59 `PREDICCION`, con 694 features `clean` y 1,403 features totales en el track `competition`. **`checkpoints/02_features/README.md`** registra históricamente este estado y congela el pipeline de inventario, covariate shift, folds fijos, ablations y baselines iniciales.

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

1. mantener cerrado el **Data Contract v2** reejecutando el audit cuando cambien fuentes/raw;
2. mantener versionada la **Feature Table v1** canónica bajo `data/processed/features_v1/`;
3. reutilizar siempre `reports/checkpoint_02/cv_folds.csv` para comparaciones de modelos;
4. abrir **Checkpoint 03 — Modeling** con CatBoost, modelos lineales regularizados y boosting bajo los mismos folds;
5. tratar la gran brecha entre CV state-stratified y municipality-grouped como riesgo espacial explícito;
6. probar reducción/selección de dimensionalidad sólo dentro de folds y sólo si mejora evidencia fuera de muestra;
7. congelar preprocessing/modelo antes de generar las 59 predicciones finales.


## Calidad y trazabilidad

CI valida Ruff, pytest, sincronización de `FUNCTION_INDEX` y build de Sphinx con warnings como errores.

El uso material de IA se registra en `docs/AI_USAGE.md`. `.ai_handoff` y `AGENTS.md` son handoffs operativos y no sustituyen el registro solicitado por FIRA.
