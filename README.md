# GeoCebada

**Predicción agroclimática y geoespacial del rendimiento de cebada** para el Reto AgroCebada FIRA 2026.

GeoCebada desarrolla un pipeline reproducible de ciencia de datos para estimar el rendimiento agrícola de parcelas de cebada a partir de información de percepción remota, clima, topografía y geometría de parcela, y exponer los resultados mediante una aplicación web interactiva.

> Para trabajo asistido por agentes/Codex, leer primero `AGENTS.md`, `.ai_handoff` y `data/.ai_handoff`.

## Problema

La unidad de predicción es una **parcela georreferenciada**. El archivo oficial de split contiene 197 parcelas:

- 138 parcelas de `ENTRENAMIENTO` con rendimiento observado;
- 59 parcelas de `PREDICCION` con rendimiento oculto por FIRA.

El split es por **parcela completa**. No se entrega 70 % de observaciones dentro de cada parcela.

Formalmente, para la parcela \(i\):

\[
\hat y_i = f(X_i),
\]

con:

- \(y_i\): `RENDIMIENTO_T_HA`, rendimiento real en toneladas por hectárea;
- \(X_i\): covariables satelitales, climáticas, topográficas, geométricas y espaciales;
- \(\hat y_i\): rendimiento estimado.

El objetivo técnico es construir un modelo de **regresión supervisada** que generalice al conjunto de 59 parcelas cuyo target permanece oculto.

## Hechos confirmados

A fecha de 2026-09-14:

- el identificador oficial del archivo de split es `ID_POLIGONO`;
- el target oficial es `RENDIMIENTO_T_HA`;
- `CONJUNTO` toma los valores `ENTRENAMIENTO` y `PREDICCION`;
- hay 197 parcelas: 138 train y 59 test/predicción;
- las parcelas pertenecen al ámbito de Hidalgo, Puebla y Tlaxcala;
- hay precipitación CHIRPS mensual de enero de 2022 a diciembre de 2025;
- hay temperatura CHIRTS-ERA5 mensual (`Tmin`, `Tmax`) de enero de 2022 a diciembre de 2025;
- precipitación y temperatura se entregan a 0.05° (~5 km), EPSG:4326;
- topografía INEGI CEM incluye elevación y pendiente a 120 m, EPSG:6372;
- existen tablas satelitales `BASICO` y `PRO`, pero su esquema y semántica temporal deben auditarse directamente antes de diseñar features.

La documentación detallada de datos vive en `data/README.md`.

## Preguntas todavía abiertas

No asumir respuestas hasta verificarlas en los archivos oficiales o con FIRA:

1. ¿A qué ciclo/año agrícola corresponde exactamente `RENDIMIENTO_T_HA`?
2. ¿Cuál es la granularidad temporal y semántica exacta de `Conjunto_datos_BASICO_AgroCebada2026.csv` y `Conjunto_datos_PRO_AgroCebada.csv`?
3. ¿Qué diferencia operacional existe entre BASIC y PRO?
4. ¿El split 70/30 fue estratificado por estado, zona, año, tamaño de parcela u otra variable?
5. ¿Qué observaciones temporales son legítimas para predecir cada rendimiento sin usar información posterior a la cosecha?
6. ¿Cuál estrategia de validación reproduce mejor el mecanismo de evaluación oculto de FIRA?

Estas preguntas son bloqueantes para una evaluación honesta. No deben resolverse por intuición.

## Riesgos metodológicos principales

### Leakage del conjunto oculto

Las 59 filas de `PREDICCION` no pueden influir en selección de variables, tuning, selección de modelo ni evaluación supervisada. Sus covariables pueden procesarse con transformaciones previamente definidas, pero nunca debe inventarse o inferirse un target para tratarlas como observaciones etiquetadas.

### Leakage temporal

Tener clima 2022–2025 no implica que todos esos meses sean válidos para todos los targets. Hasta confirmar el ciclo del rendimiento, no usar variables posteriores a la cosecha.

### Leakage espacial

Parcelas cercanas pueden compartir clima, elevación y condiciones productivas. Un K-fold aleatorio puede sobreestimar generalización. Debe compararse con una estrategia espacial o agrupada cuando la geometría y distribución de las parcelas estén auditadas.

### Preprocesamiento fuera de CV

Imputación, escalado, selección de variables, reducción dimensional y cualquier transformación aprendida deben ajustarse dentro de cada fold. No ajustar transformaciones supervisadas sobre todo el conjunto antes de CV.

## Sistemas de referencia espacial (CRS)

No asumir que todos los insumos comparten CRS.

| Fuente | CRS conocido | Resolución aproximada |
|---|---|---:|
| CHIRPS precipitación | EPSG:4326 | 0.05° (~5 km) |
| CHIRTS-ERA5 temperatura | EPSG:4326 | 0.05° (~5 km) |
| INEGI CEM topografía | EPSG:6372 | 120 m |
| Parcelas | verificar al extraer | verificar |

Reglas:

- inspeccionar CRS antes de cualquier overlay/zonal statistic;
- reproyectar explícitamente cuando sea necesario;
- evitar cálculos métricos de área/distancia en coordenadas geográficas;
- documentar el CRS de cada artefacto derivado;
- no sobrescribir geometrías fuente.

## Arquitectura del repositorio

```text
GeoCebada/
├── AGENTS.md                     # instrucciones cortas para agentes/Codex
├── .ai_handoff                   # contexto operativo global del proyecto
├── app/                          # aplicación / dashboard
│   └── main.py
├── configs/                      # configuración reproducible
│   └── base.yaml
├── data/
│   ├── README.md                 # documentación humana del dataset
│   ├── .ai_handoff               # contexto operativo específico de datos
│   ├── source/                   # fuentes oficiales FIRA, inmutables
│   ├── raw/                      # ingestión local, no versionada
│   ├── interim/                  # artefactos intermedios, no versionados
│   └── processed/                # datasets model-ready, no versionados
├── docs/
│   ├── official/                 # bases y lineamientos FIRA
│   ├── reference/                # diccionarios/descripción del dataset
│   ├── AI_USAGE.md               # registro de IA/prompts para el reto
│   └── README.md
├── models/                       # artefactos entrenados, no versionados
├── notebooks/                    # EDA y prototipos; no fuente de verdad de producción
├── reports/                      # métricas, figuras, predicciones y entregables
├── src/geocebada/
│   ├── data/                     # carga, validación, geoprocesamiento
│   ├── features/                 # ingeniería de variables
│   ├── models/                   # entrenamiento e inferencia
│   └── evaluation/               # CV, métricas y diagnósticos
├── tests/                        # pruebas automáticas
├── .env.example
├── .gitignore
└── pyproject.toml
```

## Contrato de carpetas

`data/source/` y `docs/official/` representan evidencia/fuentes del reto. No deben modificarse para acomodar el pipeline. Los artefactos reproducibles deben derivarse por código.

Los notebooks son para exploración. Cuando una transformación o modelo se vuelve parte del pipeline, debe migrarse a `src/geocebada/` y, cuando aplique, cubrirse con pruebas.

Los resultados de experimentos deben vivir en `reports/`; los objetos de modelos entrenados en `models/`. No mezclar resultados con datos fuente.

## Pipeline previsto

```text
official sources
      ↓
data audit + schema validation
      ↓
spatial/temporal alignment
      ↓
parcel-level feature engineering
      ↓
model-ready table
      ↓
leakage-safe cross-validation
      ↓
baselines + model comparison
      ↓
feature ablations + explainability
      ↓
final model / ensemble
      ↓
59 hidden-target predictions
      ↓
GeoCebada web app + technical report
```

Una representación conceptual útil es:

\[
X_i = [X_i^{sat}, X_i^{clima}, X_i^{topo}, X_i^{geom}, X_i^{espacio}],
\qquad
\hat y_i=f(X_i).
\]

Las series temporales o rasters deben transformarse en features de parcela con una ventana temporal agronómicamente válida. No colapsar 2022–2025 de forma arbitraria.

## Estrategia de modelado

El reto es regresión tabular/geoespacial con un número pequeño de targets observados (138). La prioridad inicial debe ser una comparación reproducible de baselines y modelos apropiados para bajo \(n\), no deep learning por defecto.

Familias razonables a evaluar, sin asumir ganador:

- media/mediana como baseline trivial;
- Linear/Ridge/Elastic Net;
- Random Forest / Extra Trees;
- gradient boosting: CatBoost, LightGBM, XGBoost;
- otros modelos sólo si la validación muestra una razón clara.

Métricas principales: RMSE, MAE y \(R^2\). La selección final debe basarse en desempeño out-of-fold estable, no en un único split favorable.

También deben compararse grupos de variables para responder qué aportan satélite, clima, topografía y geografía mediante ablations reproducibles.

## Convenciones de experimentos

- usar `random_seed: 42` salvo razón documentada;
- conservar `ID_POLIGONO` como llave canónica end-to-end;
- registrar fecha, commit, configuración, folds, features y métricas;
- guardar predicciones out-of-fold separadas de las predicciones del conjunto oculto;
- no optimizar directamente contra las 59 parcelas de evaluación;
- reportar media y dispersión entre folds cuando sea posible;
- si se introduce una decisión metodológica importante, documentarla en README/handoff antes o junto con el código.

## Aplicación

La interfaz vive en `app/`. El scaffold actual usa Streamlit. La app final debe consumir artefactos reproducibles del pipeline, no reimplementar feature engineering de manera distinta.

Objetivo funcional previsto:

- consulta por `ID_POLIGONO`;
- rendimiento estimado en ton/ha;
- visualización geoespacial de parcelas;
- factores/variables que explican la predicción;
- contexto agroclimático relevante;
- separación clara entre valores observados y predichos.

Ejecutar el scaffold con:

```bash
streamlit run app/main.py
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

## Próximos pasos canónicos

1. Auditar BASIC y PRO: columnas, fechas, cardinalidades, missingness y relación con `ID_POLIGONO`.
2. Extraer/leer las geometrías y verificar su CRS, validez, estado y correspondencia 1:1 con las 197 parcelas.
3. Verificar si el split 70/30 está balanceado/estratificado espacialmente y por área.
4. Resolver el año/ciclo exacto del target antes de crear features temporales.
5. Construir una tabla de features a nivel parcela con trazabilidad a la fuente.
6. Definir y justificar CV aleatoria vs espacial/agrupada; conservar ambas como diagnóstico si son informativas.
7. Entrenar baselines antes de tuning avanzado.
8. Ejecutar ablations por dominio de variables y análisis de importancia.
9. Congelar el pipeline final, reentrenar con las 138 parcelas y generar exactamente 59 predicciones.
10. Integrar el modelo congelado en la aplicación y preparar reporte/video/entregables.

## Entregables y uso de IA

Los documentos oficiales del reto están en `docs/official/`; las referencias técnicas en `docs/reference/`. El uso material de herramientas de IA debe registrarse en `docs/AI_USAGE.md` con propósito, prompt/referencia, resultado incorporado y validación humana.

`.ai_handoff` y `data/.ai_handoff` son documentación operativa para continuidad entre agentes. **No sustituyen** el registro de IA requerido para los entregables.

## Estado

Proyecto en fase de **auditoría de datos y diseño de validación**. Todavía no existe un modelo final ni una definición confirmada de la ventana temporal del target.
