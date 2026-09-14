# Registro de uso de inteligencia artificial

Las bases del Reto AgroCebada FIRA 2026 solicitan documentar el uso de herramientas de inteligencia artificial generativa y anexar los prompts utilizados.

Registrar cada uso relevante en la tabla siguiente.

| Fecha | Herramienta | Propósito | Prompt / referencia | Resultado incorporado | Validación humana |
|---|---|---|---|---|---|
| 2026-09-14 | ChatGPT | Estructuración inicial del repositorio | Organización del repositorio GeoCebada para datos, modelos, aplicación y entregables | Estructura del repositorio | Revisada por el equipo |
| 2026-09-14 | ChatGPT | Documentación y handoff del dataset | Solicitud de documentar estructura, split 70/30, fechas, CRS, leakage y próximos pasos en `data/` | `data/README.md` y `data/.ai_handoff` | Revisada por el equipo; hechos derivados de archivos oficiales inspeccionados |
| 2026-09-14 | ChatGPT | Continuidad del proyecto para agentes/Codex | Solicitud de extender el handoff a todo el repositorio con hechos conocidos, incógnitas, invariantes, CRS, riesgos de leakage y próximos pasos | `README.md`, `.ai_handoff`, `AGENTS.md` y actualización de `configs/base.yaml` | Pendiente de revisión final del equipo antes de decisiones metodológicas |

## Criterio de registro

Registrar prompts que hayan contribuido materialmente a:

- metodología;
- generación o modificación de código;
- análisis de datos;
- selección o interpretación de modelos;
- redacción del reporte;
- diseño de la aplicación;
- generación de figuras o tablas;
- documentación operativa que condicione decisiones posteriores del proyecto.

No incluir secretos, contraseñas, tokens ni datos personales.

## Nota de trazabilidad

Los archivos `.ai_handoff` y `AGENTS.md` sirven para continuidad operativa entre asistentes/agentes. No sustituyen este registro ni los prompts que deban anexarse en los entregables oficiales del reto.
