# GeoCebada

## Objetivo activo — reconstrucción transductiva de 59 rendimientos

Desde el 19 de septiembre de 2026, GeoCebada se modela explícitamente como un problema de
**Transductive Competition Modeling**. Conocemos las covariables de las 197 parcelas, pero sólo
138 tienen `RENDIMIENTO_T_HA`; las otras 59 son consultas fijas cuyos valores de referencia
permanecen ocultos por FIRA.

El objetivo activo no es aprender el mejor modelo universal para parcelas futuras. Es reconstruir
los 59 rendimientos faltantes con el menor error posible usando toda la evidencia observable y
permitida: series temporales BASIC/PRO, `competition` full-season, geometría, similitud entre
parcelas, clima, suelo, topografía, SIAP 2025 y demás fuentes públicas justificadas.

Las X de las 59 parcelas **sí forman parte del problema** y pueden utilizarse en aprendizaje
transductivo X-only. Los 59 y ocultos no pueden utilizarse ni obtenerse directamente.

Checkpoint 03C queda congelado como **First Modeling Delivery**, Checkpoint 04 produjo el incumbent local/transductivo y Checkpoint 05 no logró promover un ensamble sobre `Local04D`. Por decisión explícita final, la fase activa es **Checkpoint 03D**, un benchmark global de cómputo grande; sólo si produce señal global nueva se abrirá un 05B restringido. Leer `docs/TRANSDUCTIVE_OBJECTIVE.md`, `docs/CHECKPOINT_03D_PLAN.md` y `docs/CHECKPOINT_05B_PLAN.md`.


**Predicción agroclimática y geoespacial del rendimiento de cebada** para el Reto AgroCebada FIRA 2026.

GeoCebada desarrolla un pipeline reproducible para estimar el rendimiento agrícola de parcelas de cebada a partir de percepción remota, clima, topografía y geometría espacial. El proyecto incluye una librería Python compartida (`geocebada`) y un laboratorio interactivo en Streamlit para exploración, inferencia estadística, feature engineering y comparación de modelos.

> Para trabajo asistido por agentes/Codex, empezar por **`docs/AGENT_GUIDE.md`**, luego `.ai_handoff` y **`docs/PROJECT_HISTORY.md`**. Ahí está el estado actual, qué ya se cerró, qué artifacts son canónicos y qué no debe repetirse.

## Problema

La unidad final de predicción es una **parcela georreferenciada**:

- 197 parcelas totales;
- 138 `ENTRENAMIENTO` con `RENDIMIENTO_T_HA` observado;
- 59 `PREDICCION` con rendimiento oculto por FIRA.

El identificador canónico es `ID_POLIGONO`. La documentación oficial confirma que el target corresponde al **ciclo de producción abril–octubre de 2025**.

Formalmente, el objeto activo es el vector de 59 valores ocultos \(y_U\): conocemos \(X_L\), \(X_U\) y \(y_L\), y buscamos reconstruir \(y_U\). Una única función global \(f(X)\) es sólo una de varias estrategias posibles.



## Checkpoint 03D — activo: large-compute global benchmark

03D cierra una pregunta que los experimentos anteriores dejaron deliberadamente
abierta: los modelos globales de 03C usaron grids pequeños y conservadores. El
nuevo benchmark usa un presupuesto serio con CatBoost y XGBoost en GPU,
LightGBM, ExtraTrees, HistGradientBoosting y controles Ridge/PLS sobre cuatro
representaciones deterministas competition-only.

La validación sigue siendo nested y reutiliza los folds congelados de 03:
state-stratified y municipality-grouped. Los 59 targets FIRA nunca se puntúan.

El runner es resumible por outer fit y, al terminar, refitea los tres finalistas
globales sobre los 138 labels. El mejor global se guarda para deployment como:

```text
models/final/checkpoint03d_global.onnx
models/final/checkpoint03d_global.onnx.json
models/final/checkpoint03d_global.joblib
```

El `.onnx` se valida con ONNX Runtime contra las predicciones Python antes de
declarar PASS. El JSON conserva el orden exacto de features y las medianas de
imputación.

Ejecutar primero:

```powershell
git pull
conda activate geocebada
python -m pip install -e ".[dev,models,deployment]"
python tools\run_checkpoint_03d.py --preflight
```

y sólo si pasa:

```powershell
python tools\run_checkpoint_03d.py
```

Si 03D encuentra globales realmente más fuertes, `docs/CHECKPOINT_05B_PLAN.md`
define una remezcla final muy restringida con Local04D. Hasta entonces,
Local04D sigue siendo el método final de competencia.

## Checkpoint 05 — completado: global + local mixture

Checkpoint 05 fue el último experimento de modelado previsto. Combinó PLS,
Ridge, CatBoost y ExtraTrees de la línea global de 03 con Local04D y Graph04D
mediante stacking completamente cross-fitted y un mixture-of-experts condicionado
en soporte.

El mejor resultado de desarrollo fue un blend simple:

```text
LocalE13_w0p10 = 90% Local04D + 10% E13
mean RMSE      = 0.487243
pooled RMSE    = 0.495261
```

frente a Local04D:

```text
mean RMSE      = 0.487845
pooled RMSE    = 0.495741
```

La mejora nominal no sobrevivió el gate leave-one-split-out. El selector LOSO
obtuvo 0.487934 de mean RMSE y 0.495933 pooled, ligeramente peor que Local04D
en ambas métricas. Por ello no se ejecutó la confirmación fresca y no hubo
promoción.

El método final permanece:

```text
Local04D = LocalRidge_C4_all_deterministic_Geo_k24_a30_p1
```

La salida canónica post-05 es:

```text
reports/checkpoint_05/final_predictions.csv
```

Interpretación canónica: `docs/CHECKPOINT_05_FINDINGS.md`.

## Checkpoint 04F — completado

La reconstrucción transductiva final ya está congelada. La regla final es
`Baseline_Local04D`, equivalente a
`LocalRidge_C4_all_deterministic_Geo_k24_a30_p1`.

Resultados finales de validación:

```text
target-matched mean RMSE     0.487845
pooled RMSE                  0.495741
LOSO Local/Graph check       0.487941
```

Un blend fijo 75% Local / 25% Graph alcanza 0.486768 en la tabla completa de
desarrollo, pero el check leave-one-split-out no mejora el baseline congelado.
Por eso se conserva Local04D sin tuning adicional post-hoc.

La tabla canónica de entrega está en:

```text
reports/checkpoint_04f/final_predictions.csv
```

y contiene exactamente 59 filas con
`ID_POLIGONO,RENDIMIENTO_T_HA`. Los diagnósticos de soporte y discrepancia
Local–Graph están separados en
`reports/checkpoint_04f/final_prediction_diagnostics.csv`.

Interpretación canónica: `docs/CHECKPOINT_04F_FINDINGS.md`.

## Checkpoint 04D.1 — completado

La búsqueda de 626 configuraciones confirmó una región local estable. El mejor
target-matched fue `LocalRidge_C4_all_deterministic_Geo_k24_a30_p1` con RMSE
medio **0.4878**; la selección leave-one-split-out quedó en **0.4884** y el
routing LOSO en **0.4858**.

El grafo `GraphDirect_GeoAgro25_k6_lam8` quedó en **0.4949** target-matched,
pero fue mucho más robusto en municipality-grouped (**0.5279**). La interpretación
canónica está en `docs/CHECKPOINT_04D1_FINDINGS.md`.

## Checkpoint 04B — completado

04B ya fue ejecutado y sus resultados están versionados. Para reproducirlo:

```powershell
git pull
conda activate geocebada
python -m pip install -e ".[dev,geo]"
python tools\run_checkpoint_04b.py
```

El runner simula repetidamente el concurso real: oculta únicamente `y` de pseudo-targets,
mantiene sus `X` visibles, y calcula RMSE para modelos globales, kNN geográficos/agronómicos,
modelos locales, grafos y blends. También conserva los folds state/municipality de Checkpoint 03
como stress tests.

Los resultados se generan en `reports/checkpoint_04b/`. La interpretación canónica está en `docs/CHECKPOINT_04B_FINDINGS.md`.

El resultado principal favorece modelos locales y de grafo: LocalRidge k20 alcanza RMSE target-matched 0.4946 y Graph k8/lambda2 0.4974, con el grafo como opción más robusta bajo stress tests. La siguiente fase activa es refinamiento local/graph; el modelado global de alta capacidad queda como anchor secundario.

## Checkpoint 04A — completado

La primera etapa transductiva ya fue ejecutada y sus resultados están versionados en `reports/checkpoint_04a/`. Para reproducirla:

```powershell
git pull
conda activate geocebada
python -m pip install -e ".[dev,geo]"
python tools\run_checkpoint_04a.py
```

04A no entrena todavía el modelo final ni genera los 59 rendimientos. Estudia la topología de
las 59 parcelas objetivo: vecinos geográficos y multivariados, similitud temporal con lag/DTW,
covariate shift, relación similitud–diferencia de rendimiento, autocorrelación espacial y
soporte/extrapolación por parcela.

Los resultados se generan en `reports/checkpoint_04a/`, incluyendo CSV/JSON, un reporte
Markdown, figuras resumen y paneles temporales para cada una de las 59 parcelas.

La interpretación canónica del resultado está en `docs/CHECKPOINT_04A_FINDINGS.md`. El siguiente paso activo es Checkpoint 04B: validación pseudo-competition transductiva con RMSE.

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

## Preguntas que guiaron Checkpoint 04

No asumir respuestas sin evidencia:

1. ¿qué tan cerca está cada una de las 59 parcelas de alguna de las 138 etiquetadas, espacialmente y en espacio de features?;
2. ¿la similitud de curvas temporales BASIC/PRO implica realmente similitud de rendimiento?;
3. ¿existe autocorrelación espacial del rendimiento o de los residuos del baseline?;
4. ¿qué pseudo-competition splits reproducen mejor la geometría X de las 59 parcelas objetivo?;
5. ¿qué gana covariate shift, kNN/local regression, grafos y mixtures of experts frente al baseline de Checkpoint 03?;
6. ¿puede SIAP 2025 alinearse específicamente a `Cebada grano` + ciclo + `Temporal` + municipio?;
7. ¿qué fuentes externas agregan señal no redundante para estas 59 parcelas?;

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
│   ├── 01_data/                  # adquisición, audit y fixtures de datos
│   ├── 03_modeling/              # First Modeling Delivery, congelado
│   ├── 04_transductive_competition/ # línea transductiva/local congelada
│   └── 05_global_local_mixture/     # fase activa: combinación 03 + 04
├── app/                         # interfaces Streamlit
├── configs/
├── data/
│   ├── source/                  # fuentes oficiales, inmutables
│   ├── raw/                     # ingestión derivada
│   ├── interim/                 # transformaciones intermedias
│   └── processed/               # features_v1 + agronomic_features_v1 + empirical_features_v1 canónicos
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

El cierre de la fase de datos está documentado en **`checkpoints/01_data/README.md`**. La tabla maestra determinista ya fue construida y validada en la workstation: 197 parcelas = 138 `ENTRENAMIENTO` + 59 `PREDICCION`, con 694 features `clean` y 1,403 features totales en el track `competition`. **`checkpoints/02_features/README.md`** está cerrado y registra el inventario, covariate shift, folds fijos, ablations y baselines ya ejecutados. **Checkpoint 03A** también está cerrado: `data/processed/agronomic_features_v1/` contiene 350 variables agronómicas/no lineales target-free (123 clean, 227 competition) para las 197 parcelas. **Checkpoint 03B** está cerrado: `data/processed/empirical_features_v1/` contiene 335 variables empíricas X-only (91 clean, 244 competition), y el audit fold-local encontró 88 expresiones únicas, con 7 expresiones recurrentes que se reducen a cuatro motivos conceptuales. **Checkpoint 03C.1** está cerrado. El benchmark mostró que las variables agronómicas conservan señal con mucha menor dimensionalidad, `empirical-only` es débil como representación aislada y el discovery fold-local puede mejorar ExtraTrees, especialmente en el track clean. Checkpoints 03C.2/03C.3 y 04A--04F están cerrados como evidencia reproducible. Checkpoint 05 también está cerrado. No hubo promoción y `Local04D` permanece como método final. La salida canónica de competencia es `reports/checkpoint_05/final_predictions.csv`; el siguiente trabajo es integración de submission/app/export, no otra búsqueda amplia de modelos.

## Agronomic Features v1

Checkpoint 03A creó una capa aditiva separada de Feature Table v1:

```text
data/processed/agronomic_features_v1/
  parcel_agronomic_features.csv   197 × 351
  feature_manifest.json           350 variables derivadas
  build_report.json
```

La capa no contiene target ni split y no fue seleccionada mirando rendimiento. Incluye forma
fenológica, anomalías 2025 vs histórico, acuerdo entre sensores, tiempo térmico, timing de
lluvia, productividad hídrica WaPOR, perfiles/interacciones de suelo y cruces agronómicos.
CHIRPS está excluido hasta resolver su QC. Ver `docs/AGRONOMIC_FEATURES_V1.md`.

## Empirical Features v1

Checkpoint 03B construye una tercera representación separada:

```text
data/processed/empirical_features_v1/
  parcel_empirical_features.csv
  feature_manifest.json
  build_report.json
```

La capa materializada es **X-only**: describe geometría temporal, condición relativa contra el
histórico corto, cambios simétricos 2025-vs-histórico, similitud Sentinel-2/Planet y geometría
normalizada de distribuciones. No contiene target ni selección supervisada.

La parte target-aware se implementa con `FoldLocalExpressionMiner`: las fórmulas candidatas se
descubren únicamente dentro del training fold y luego se aplican al validation fold. Ver
`docs/EMPIRICAL_FEATURES_V1.md`.

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

La primera capa formal de hipótesis agronómicas está documentada en
**`docs/AGRONOMIC_FEATURES_V1.md`** y versionada separadamente en
**`data/processed/agronomic_features_v1/`**. Incluye fenología, anomalías 2025 vs histórico,
tiempo térmico, agua/productividad, perfiles de suelo e interacciones cross-domain. Feature
Table v1 no se modificó; ambas capas se unen uno-a-uno por `ID_POLIGONO`.

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

1. ejecutar **Checkpoint 05** una sola vez en la workstation GPU con `python tools\run_checkpoint_05.py`;
2. si el proceso se interrumpe después de completar splits, continuar con `--resume` en lugar de reiniciar desde cero;
3. revisar el gate de promoción LOSO antes de sustituir el incumbent 04F;
4. versionar los reportes pequeños de `reports/checkpoint_05/`; los bundles `checkpoint05_app_bundle.joblib` y `checkpoint05_model.joblib` permanecen locales/ignorados por Git;
5. una vez congelado 05, pasar a la app/interfaz y hacer que consuma el manifest y la predicción final sin reabrir búsqueda de modelos.


## Calidad y trazabilidad

CI ligero valida Ruff, pytest, sincronización de `FUNCTION_INDEX` y build de Sphinx con warnings como errores. Los builds pesados de features, figuras y el reporte técnico/PDF son manuales mediante `workflow_dispatch`; un `push` normal no los ejecuta ni hace commits automáticos de outputs. Ver `docs/GITHUB_ACTIONS.md`.

El uso material de IA se registra en `docs/AI_USAGE.md`. `.ai_handoff` y `AGENTS.md` son handoffs operativos y no sustituyen el registro solicitado por FIRA.
