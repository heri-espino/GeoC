# GeoCebada

**Predicción agroclimática del rendimiento de cebada** para el Reto AgroCebada FIRA 2026.

GeoCebada busca construir un pipeline reproducible de ciencia de datos para estimar el rendimiento agrícola de parcelas de cebada a partir de información agroclimática, geoespacial y de percepción remota, y exponer los resultados mediante una aplicación web interactiva.

## Objetivos

1. Preparar y validar los datos proporcionados por FIRA.
2. Explorar índices de vegetación, variables climáticas y variables geoespaciales.
3. Construir baselines estadísticos y modelos de machine learning.
4. Evaluar el desempeño con validación reproducible y métricas como RMSE, MAE y R².
5. Analizar importancia de variables e incertidumbre de las predicciones.
6. Generar las predicciones finales por `ID_parcela`.
7. Desplegar una aplicación web para consultar e interpretar resultados.

## Estructura

```text
GeoCebada/
├── app/                    # aplicación / dashboard
├── configs/                # configuración de experimentos
├── data/
│   ├── raw/                # datos originales, sin modificar
│   ├── interim/            # datos intermedios
│   └── processed/          # tablas listas para modelado
├── docs/                   # documentación y registro de IA/prompts
├── models/                 # artefactos entrenados (no versionados)
├── notebooks/              # exploración y prototipos
├── reports/                # resultados, tablas, figuras y entregables
├── src/geocebada/
│   ├── data/               # carga, validación y limpieza
│   ├── features/           # ingeniería de variables
│   ├── models/             # entrenamiento y predicción
│   └── evaluation/         # métricas, CV y diagnóstico
├── tests/                  # pruebas automáticas
├── .env.example
├── .gitignore
└── pyproject.toml
```

## Flujo de trabajo

```text
raw data
   ↓
data validation / cleaning
   ↓
feature engineering
   ↓
EDA + baseline
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

## Datos

Los archivos originales deben colocarse en `data/raw/` y **no deben modificarse**. Todo archivo derivado debe escribirse en `data/interim/` o `data/processed/`.

Los datos del reto y los artefactos pesados no se versionan en Git. Esto evita publicar accidentalmente información proporcionada por la competencia y mantiene el repositorio liviano.

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

- No modificar los datos en `data/raw/`.
- Fijar semillas aleatorias cuando aplique.
- Mantener separado el conjunto usado para evaluación final.
- Registrar configuración, métricas y fecha de cada experimento.
- Evitar selección de variables o tuning usando información del conjunto final.
- Guardar predicciones con `ID_parcela` como llave explícita.

## Aplicación

La interfaz se desarrollará en `app/`. El prototipo inicial está preparado para Streamlit y posteriormente podrá conectarse al pipeline de inferencia final.

Ejecución local:

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
