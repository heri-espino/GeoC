# GeoCebada

**Predicción agroclimática del rendimiento de cebada** para el Reto AgroCebada FIRA 2026.

GeoCebada desarrolla un pipeline reproducible de ciencia de datos para estimar el rendimiento agrícola de parcelas de cebada a partir de información agroclimática, geoespacial y de percepción remota, y exponer los resultados mediante una aplicación web interactiva.

## Objetivos

1. Preparar y validar los datos proporcionados por FIRA.
2. Explorar índices de vegetación, variables climáticas, topográficas y geoespaciales.
3. Construir baselines estadísticos y modelos de machine learning.
4. Evaluar el desempeño con validación reproducible y métricas como RMSE, MAE y R².
5. Analizar importancia de variables e incertidumbre de las predicciones.
6. Generar las predicciones finales por `ID_parcela`.
7. Desplegar una aplicación web para consultar e interpretar resultados.

## Estructura

```text
GeoCebada/
├── app/                         # aplicación / dashboard
├── configs/                     # configuración de experimentos
├── data/
│   ├── source/                  # archivos oficiales proporcionados por FIRA
│   │   ├── tabular/
│   │   ├── geospatial/
│   │   ├── climate/
│   │   │   ├── precipitation/
│   │   │   └── temperature/
│   │   └── topography/
│   ├── raw/                     # ingestión local, no versionada
│   ├── interim/                 # datos intermedios, no versionados
│   └── processed/               # datos listos para modelado, no versionados
├── docs/
│   ├── official/                # bases y lineamientos FIRA
│   ├── reference/               # diccionarios y descripción del dataset
│   └── AI_USAGE.md              # registro de IA y prompts
├── models/                      # artefactos entrenados
├── notebooks/                   # exploración y prototipos
├── reports/                     # resultados, figuras y entregables
├── src/geocebada/
│   ├── data/                    # carga, validación y limpieza
│   ├── features/                # ingeniería de variables
│   ├── models/                  # entrenamiento y predicción
│   └── evaluation/              # métricas, CV y diagnóstico
├── tests/                       # pruebas automáticas
├── .env.example
├── .gitignore
└── pyproject.toml
```

## Datos fuente

Los archivos descargados de la plataforma de FIRA se conservan sin modificar en `data/source/`, separados por dominio:

- tablas BÁSICO y PRO y archivo de rendimiento 70/30;
- parcelas georreferenciadas;
- precipitación CHIRPS 2022–2025;
- temperatura 2022–2025;
- topografía INEGI CEM 4.

Los documentos oficiales de la convocatoria están en `docs/official/` y los diccionarios/descripciones técnicas del dataset en `docs/reference/`.

No se deben editar manualmente los archivos de `data/source/`. Cualquier transformación debe ser reproducible y escribirse en `data/raw/`, `data/interim/` o `data/processed/`.

## Flujo de trabajo

```text
data/source
   ↓
ingestion + validation
   ↓
data/raw / data/interim
   ↓
feature engineering
   ↓
data/processed
   ↓
EDA + baselines
   ↓
cross-validation
   ↓
model comparison / tuning
   ↓
final model
   ↓
predictions + explainability
   ↓
web application + technical report
```

## Instalación

Requiere Python 3.11+.

```bash
git clone https://github.com/heri-espino/GeoCebada.git
cd GeoCebada
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
# .venv\Scripts\activate  # Windows
pip install -e ".[dev]"
```

## Convenciones de experimentos

- No modificar los archivos de `data/source/`.
- Fijar semillas aleatorias cuando aplique.
- Mantener separado el conjunto usado para evaluación final.
- Registrar configuración, métricas y fecha de cada experimento.
- Evitar selección de variables o tuning usando información del conjunto final.
- Guardar predicciones con `ID_parcela` como llave explícita.

## Aplicación

La interfaz se desarrollará en `app/`. El prototipo inicial está preparado para Streamlit y posteriormente se conectará al pipeline de inferencia final.

```bash
streamlit run app/main.py
```

## Entregables

Los documentos, tablas y figuras del reto se organizan en `reports/`. El proyecto contempla:

- reporte técnico;
- tabla final `ID_parcela -> rendimiento_predicho_ton_ha`;
- anexos y figuras;
- bibliografía;
- aplicación o dashboard funcional;
- registro del uso de herramientas de IA y prompts.

## Estado

Proyecto en desarrollo — Reto AgroCebada FIRA 2026.
