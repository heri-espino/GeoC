# Registro de uso de inteligencia artificial

Las bases del Reto AgroCebada FIRA 2026 solicitan documentar el uso de herramientas de inteligencia artificial generativa y anexar los prompts utilizados.

Registrar cada uso relevante en la tabla siguiente.

| Fecha | Herramienta | Propósito | Prompt / referencia | Resultado incorporado | Validación humana |
|---|---|---|---|---|---|
| 2026-09-14 | ChatGPT | Estructuración inicial del repositorio | Organización del repositorio GeoCebada para datos, modelos, aplicación y entregables | Estructura del repositorio | Revisada por el equipo |
| 2026-09-14 | ChatGPT | Documentación y handoff del dataset | Solicitud de documentar estructura, split 70/30, fechas, CRS, leakage y próximos pasos en `data/` | `data/README.md` y `data/.ai_handoff` | Revisada por el equipo; hechos derivados de archivos oficiales inspeccionados |
| 2026-09-14 | ChatGPT | Continuidad del proyecto para agentes/Codex | Solicitud de extender el handoff a todo el repositorio con hechos conocidos, incógnitas, invariantes, CRS, riesgos de leakage y próximos pasos | `README.md`, `.ai_handoff`, `AGENTS.md` y actualización de `configs/base.yaml` | Pendiente de revisión final del equipo antes de decisiones metodológicas |
| 2026-09-14 | ChatGPT | Arquitectura de librería Python compartida | Diseñar `src/geocebada/` como librería reutilizable para notebooks/app y evitar duplicación de funciones | Utilidades de paths, configuración, ingestión, target split, raster y CRS; tests y convenciones de importación | Pendiente de validación funcional completa por el equipo |
| 2026-09-14 | ChatGPT | Sistema de documentación e índice de funciones | Crear documentación Sphinx y un índice Markdown autogenerado para localizar funciones antes de crear nuevas | `docs/conf.py`, `docs/api/`, `docs/FUNCTION_INDEX.md`, `tools/generate_function_index.py` | Validación automatizada mediante CI |
| 2026-09-14 | ChatGPT | Laboratorio interactivo de análisis | Implementar un dashboard que permita explorar datos, realizar inferencia estadística, revisar supuestos, crear features reproducibles y comparar modelos sin usar targets ocultos | `app/main.py`, módulos `statistics`, `features`, `visualization`, `evaluation`, documentación y tests asociados | Validación automatizada mediante Ruff/pytest/Sphinx; interpretación estadística y decisiones metodológicas requieren revisión del equipo |
| 2026-09-14 | ChatGPT | Exploración del conjunto de predicción y despliegue | Permitir inspeccionar las 59 filas de predicción sin targets, comparar covariables train/test y preparar la app para Streamlit Community Cloud | pestaña `Prediction Set`, `covariate_shift_screen`, tests, `requirements.txt` y `docs/DEPLOYMENT.md` | Pendiente de revisión del equipo; CI valida código/documentación, pero el uso metodológico del conjunto oculto debe mantenerse no supervisado |
| 2026-09-14 | ChatGPT | Exploración espacial y multivariada | Ampliar la aplicación con mapa filtrable, pairplot, correlaciones, exploración bivariada, outliers, resúmenes por grupo y missingness; permitir cargar BASIC/PRO en una página dedicada | `app/pages/1_Visual_Explorer.py`, `app/visual_explorer.py`, utilidades de filtrado, geometrías y visualización geoespacial/multivariada | Pendiente de revisión visual del equipo; CRS nunca se infiere |
| 2026-09-14 | ChatGPT | Auditoría y diccionario de variables | Inspeccionar los documentos oficiales de descripción/diccionario y los esquemas BASIC/PRO para documentar qué significa cada variable | `docs/VARIABLES.md`, sincronización de `README.md`, handoffs, `AGENTS.md` y `configs/base.yaml`; se confirmó el ciclo objetivo abril–octubre 2025 y las fuentes/esquemas BASIC/PRO | Hechos contrastados con `Descripcion_general_variables_DataSet.docx`, `Diccionario_de_indices_satelitales.docx` y los CSV oficiales; CI valida documentación/código, interpretación metodológica final requiere revisión del equipo |

## Criterio de registro

Registrar prompts que hayan contribuido materialmente a:

- metodología;
- generación o modificación de código;
- análisis de datos;
- selección o interpretación de modelos;
- redacción del reporte;
- diseño de la aplicación;
- generación de figuras o tablas;
- arquitectura y documentación reproducible del software;
- documentación operativa que condicione decisiones posteriores del proyecto.

No incluir secretos, contraseñas, tokens ni datos personales.

## Nota de trazabilidad

Los archivos `.ai_handoff` y `AGENTS.md` sirven para continuidad operativa entre asistentes/agentes. No sustituyen este registro ni los prompts que deban anexarse en los entregables oficiales del reto.
