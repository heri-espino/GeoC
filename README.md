# GeoCebada

**Predicción agroclimática y geoespacial del rendimiento de cebada** para el Reto AgroCebada FIRA 2026.

GeoCebada desarrolla un pipeline reproducible para estimar el rendimiento agrícola de parcelas de cebada a partir de percepción remota, clima, topografía y geometría espacial. El proyecto incluye una librería Python compartida (`geocebada`) y un laboratorio interactivo en Streamlit para exploración, inferencia estadística, feature engineering y comparación de modelos.

> Para trabajo asistido por agentes/Codex, leer primero `AGENTS.md`, `.ai_handoff`, `data/.ai_handoff` y `src/geocebada/README.md`.

## Problema

La unidad de predicción es una **parcela georreferenciada**. El split oficial contiene:

- 197 parcelas totales;
- 138 parcelas `ENTRENAMIENTO` con `RENDIMIENTO_T_HA` observado;
- 59 parcelas `PREDICCION` cuyo rendimiento permanece oculto por FIRA.

El split es por **parcela completa**, no por observaciones internas de cada parcela.

Formalmente:

\[
\hat y_i = f(X_i),
\]

con `ID_POLIGONO` como identificador canónico, `RENDIMIENTO_T_HA` como target en ton/ha y \(X_i\) compuesto por covariables satelitales, climáticas, topográficas, geométricas y espaciales.

## Hechos confirmados

A fecha de 2026-09-14:

- `CONJUNTO` toma los valores `ENTRENAMIENTO` y `PREDICCION`;
- ámbito geográfico: Hidalgo, Puebla y Tlaxcala;
- CHIRPS: precipitación mensual enero 2022–diciembre 2025, EPSG:4326, ~5 km;
- CHIRTS-ERA5: `Tmin`/`Tmax` mensuales enero 2022–diciembre 2025, EPSG:4326, ~5 km;
- INEGI CEM: elevación y pendiente a 120 m, EPSG:6372;
- existen tablas satelitales oficiales `BASICO` y `PRO`;
- el esquema/semántica temporal exacta de BASIC/PRO todavía debe auditarse antes de fijar features canónicos.

La documentación detallada del dataset vive en `data/README.md`.

## Preguntas todavía abiertas

No asumir respuestas hasta verificarlas en los archivos oficiales o con FIRA:

1. ciclo/año agrícola exacto representado por `RENDIMIENTO_T_HA`;
2. granularidad temporal y semántica exacta de BASIC y PRO;
3. diferencia operacional entre BASIC y PRO;
4. mecanismo de estratificación del split 70/30, si existe;
5. ventanas temporales legítimas para predicción sin post-harvest leakage;
6. estrategia de validación que mejor aproxime la evaluación oculta de FIRA;
7. CRS exacto de las geometrías de parcela hasta inspección directa.

## Arquitectura

```text
GeoCebada/
├── AGENTS.md
├── .ai_handoff
├── app/
│   └── main.py                    # GeoCebada Lab (Streamlit)
├── configs/
├── data/
│   ├── source/                    # fuentes oficiales, inmutables
│   ├── raw/                       # ingestión derivada
│   ├── interim/                   # transformaciones intermedias
│   └── processed/                 # datasets model-ready
├── docs/
│   ├── FUNCTION_INDEX.md          # índice autogenerado de API pública
│   ├── api/                       # documentación Sphinx
│   ├── official/
│   ├── reference/
│   └── AI_USAGE.md
├── models/
├── notebooks/
├── reports/
├── src/geocebada/
│   ├── data/                      # carga, inventario y contratos de datos
│   ├── geo/                       # CRS y geoprocesamiento
│   ├── features/                  # feature engineering reproducible
│   ├── statistics/                # inferencia y diagnósticos estadísticos
│   ├── visualization/             # visualización reusable
│   ├── evaluation/                # CV, métricas y benchmarks
│   └── models/                    # entrenamiento/inferencia final
├── tests/
└── tools/
    └── generate_function_index.py
```

La regla central es:

```text
notebooks / app / experimentos
             ↓
        import geocebada
             ↓
      src/geocebada/
```

La aplicación es una interfaz. La lógica reusable y reproducible vive en la librería.

## GeoCebada Lab

`app/main.py` ya implementa un laboratorio interactivo con seis áreas.

### Overview

- tamaño y esquema del dataset;
- missingness;
- distribución del split oficial;
- distribución del rendimiento observado.

### Data Explorer

- selección interactiva de columnas;
- tablas filtrables;
- histogramas y boxplots;
- agrupación/color por variables;
- descarga de vistas a CSV.

### Statistical Lab

Incluye:

- correlación Pearson y Spearman;
- visualización bivariada con tendencia;
- screening de múltiples variables contra un target;
- corrección de multiplicidad Benjamini-Hochberg/FDR, Holm y Bonferroni;
- diagnóstico de regresión lineal;
- residual-vs-fitted;
- Shapiro-Wilk para residuos;
- diagnóstico de heterocedasticidad tipo Breusch-Pagan;
- VIF para multicolinealidad.

Estos resultados son diagnósticos, no reglas automáticas de aceptación/rechazo. Un `p < 0.05` no implica causalidad y la independencia espacial sigue siendo una hipótesis especialmente delicada en este proyecto.

### Feature Engineering Lab

Permite declarar recetas reproducibles, por ejemplo:

```python
from geocebada.features import FeatureRecipe, apply_feature_recipe

recipe = FeatureRecipe(
    name="ndvi_mean_may_aug",
    value_column="NDVI",
    group_column="ID_POLIGONO",
    aggregation="mean",
    date_column="date",
    start="2025-05-01",
    end="2025-08-31",
)

feature = apply_feature_recipe(long_table, recipe)
```

Agregaciones disponibles actualmente:

- `mean`, `median`, `min`, `max`, `std`, `sum`;
- `slope` temporal;
- `auc` temporal.

El laboratorio permite descargar tanto la tabla derivada como la receta JSON. Una receta exploratoria que pase a formar parte del modelo debe promoverse después a un módulo estable bajo `src/geocebada/features/` y cubrirse con tests.

### Model Lab

Implementa un benchmark inicial con splits K-fold idénticos para:

- Linear Regression;
- Ridge;
- Random Forest;
- Extra Trees.

Reporta RMSE, MAE y \(R^2\) por fold y resumen de media/desviación. **Este K-fold aleatorio es sólo exploratorio**: no sustituye la futura validación espacial/agrupada.

### Methodology

La interfaz distingue explícitamente:

1. descripción;
2. inferencia estadística;
3. inferencia predictiva;
4. inferencia causal.

El dataset observacional del reto no permite convertir asociaciones en efectos causales automáticamente.

## Protección contra leakage

### Hidden-target leakage

Cuando la aplicación detecta el esquema oficial (`CONJUNTO`, `RENDIMIENTO_T_HA`), los análisis supervisados y benchmarks utilizan únicamente filas `ENTRENAMIENTO`. Las 59 filas `PREDICCION` no se usan como etiquetas ni para scoring.

### Leakage temporal

Tener datos 2022–2025 no hace válidos todos esos meses para un target. Hasta resolver el ciclo agrícola, una feature temporal puede explorarse, pero no debe promoverse al pipeline final si usa información posterior a la cosecha.

### Leakage espacial

Parcelas cercanas pueden compartir clima, elevación y otras condiciones. El random CV puede sobreestimar desempeño. La validación final deberá incorporar un diagnóstico espacial/agrupado después de auditar las geometrías.

### Leakage de preprocesamiento

Imputación, escalado, selección de variables, PCA y transformaciones aprendidas deberán entrenarse dentro de cada fold cuando entren al pipeline model-ready.

## CRS

| Fuente | CRS conocido | Resolución aproximada |
|---|---|---:|
| CHIRPS precipitación | EPSG:4326 | 0.05° (~5 km) |
| CHIRTS-ERA5 temperatura | EPSG:4326 | 0.05° (~5 km) |
| INEGI CEM topografía | EPSG:6372 | 120 m |
| Parcelas | por verificar | por verificar |

Nunca realizar overlay, zonal statistics, áreas o distancias sin inspeccionar/reproyectar explícitamente el CRS.

## Librería y documentación de funciones

Instalar el proyecto en modo editable:

```bash
pip install -e ".[dev,geo]"
```

Después los notebooks pueden importar normalmente:

```python
from geocebada.data import load_yield_split
from geocebada.statistics import correlation_screen
from geocebada.features import FeatureRecipe
```

Antes de crear una función nueva, buscar en:

```text
docs/FUNCTION_INDEX.md
```

Después de cambiar la API pública:

```bash
python tools/generate_function_index.py
```

Sphinx genera la documentación profunda de API desde los docstrings:

```bash
pip install -e ".[docs]"
sphinx-build -b html docs docs/_build/html
```

## Ejecutar GeoCebada Lab

```bash
pip install -e ".[dev,geo]"
streamlit run app/main.py
```

La app puede cargar el split oficial directamente o recibir un CSV para exploración. Los archivos de `data/source/` nunca se modifican desde la interfaz.

## Estrategia de modelado

Con sólo 138 targets observados, la prioridad es bajo-\(n\), validación rigurosa y feature engineering trazable. Familias candidatas:

- baseline media/mediana;
- Linear/Ridge/Elastic Net;
- Random Forest / Extra Trees;
- CatBoost / LightGBM / XGBoost;
- otros modelos sólo con evidencia CV clara.

El objetivo no es encontrar el modelo más complejo, sino el que generalice de manera estable al mecanismo de evaluación de FIRA.

## Pipeline previsto

```text
official sources
      ↓
data audit + schema validation
      ↓
GeoCebada Lab: EDA + hipótesis + estadística
      ↓
spatial/temporal alignment
      ↓
feature recipes / feature engineering
      ↓
parcel-level model-ready table
      ↓
leakage-safe spatial/grouped CV
      ↓
baselines + model comparison + ablations
      ↓
explainability + residual diagnostics
      ↓
final frozen pipeline
      ↓
59 hidden-target predictions
      ↓
GeoCebada decision dashboard + report/video
```

## Próximos pasos

1. auditar BASIC y PRO: columnas, fechas, cardinalidades, missingness y relación con `ID_POLIGONO`;
2. conectar BASIC/PRO al explorador temporal mediante loaders reproducibles;
3. inspeccionar geometrías, CRS, validez y cobertura de las 197 parcelas;
4. añadir mapa interactivo de parcelas y auditoría espacial train/test;
5. incorporar Moran's I y diagnósticos espaciales una vez validada la geometría;
6. resolver el ciclo/año exacto del target;
7. definir ventanas temporales leakage-safe;
8. añadir spatial/grouped CV al Model Lab;
9. construir y congelar la tabla/modelo final;
10. exponer inferencia final y explicabilidad en la misma aplicación.

## Calidad y CI

El workflow de CI valida en cada push/PR:

```text
Ruff
  ↓
pytest
  ↓
FUNCTION_INDEX sincronizado
  ↓
Sphinx build con warnings como errores
```

No considerar una nueva utilidad estable hasta que pase estas verificaciones.

## Uso de IA

El uso material de IA debe registrarse en `docs/AI_USAGE.md`. Los archivos `.ai_handoff` y `AGENTS.md` son handoffs operativos y no sustituyen el registro requerido por FIRA.
