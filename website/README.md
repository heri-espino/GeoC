# GeoCebada · sitio de presentación (2026)

Sitio editorial **estático, sin dependencias de npm**. Presenta en español el reto, los datos,
la metodología Local04D, dos evaluaciones separadas (Checkpoints 07/08), un explorador de las
59 predicciones congeladas y un anexo con los modelos descartados.

## Abrir localmente

Desde la raíz de GeoC:

```bash
python -m http.server 8000 --directory website
```

Ir a <http://localhost:8000>. **No abrir el HTML vía file://**: los archivos CSV se solicitan
por `fetch` y muchos navegadores bloquean solicitudes locales entre archivos.

El HTML/CSS/JS está en `website/index.html`, `website/styles.css` y `website/app.js`.
No hay llamadas a APIs externas, rastreadores, mapas georreferenciados ficticios ni backend
de aprendizaje. Se usan gráficos esquemáticos etiquetados como conceptuales.

## Origen y sincronización de los datos

| Copia estática | Fuente canónica |
| --- | --- |
| `data/checkpoint07_summary.csv` | `reports/checkpoint_07/shared/three_model_summary.csv` |
| `data/checkpoint08_summary.csv` | `reports/checkpoint_07/shared/checkpoint08_comparison_summary.csv` |
| `data/checkpoint08_bootstrap.csv` | `reports/checkpoint_07/shared/checkpoint08_paired_cluster_bootstrap.csv` |
| `data/predictions.csv` | `reports/checkpoint_05/final_predictions.csv` |

Revisar o actualizar después de cambiar los resultados **sólo si la decisión de selección está documentada**:

```bash
python tools/sync_website_data.py --check
python tools/sync_website_data.py --write
python tools/sync_website_data.py --check
pytest -q tests/test_website_presentation.py
```

`--write` únicamente sincroniza estas cuatro copias. Nunca vuelve a entrenar un modelo,
recalcula predicciones, ni accede a rendimientos ocultos.

**Bancos diferentes**:

- Checkpoint 07: 16 `target_matched` históricos, 97/41, 656 predicciones repetidas,
  125 parcelas únicas. Local04D RMSE pooled 0.495741.
- Checkpoint 08: 16 nuevas particiones `confirm`, 97/41, 656 predicciones repetidas,
  132 parcelas únicas. Local04D fijo RMSE pooled 0.481936.
- Las dos métricas **no** representan mejora de un modelo; provienen de máscaras diferentes.
- Los intervalos bootstrap de 08 están agrupados por `ID_POLIGONO`, con dependencia
  geoespacial/resampling residual; nunca deben publicitarse como garantía para las 59.

El explorador proporciona **consulta/demostración de 59 estimaciones previamente calculadas**,
no predicción arbitraria para una parcela nueva. Nunca mostrar el `y` reservado de las 59.
La figura de campos en el hero y el diagrama de vecinos son ilustraciones, no coordenadas reales.

## Publicación

El workflow `.github/workflows/deploy_website.yml` sube **sólo `website/`** al artefacto de
GitHub Pages, mediante `actions/upload-pages-artifact` y `actions/deploy-pages`, en pushes a
`main` que modifiquen `website/` o al ejecutarlo con `workflow_dispatch`. Es un despliegue
ligero; **no** genera figuras, papers, datasets pesados, ni ejecuta GPU.

Antes de publicar, confirmar en **Settings → Pages → Build and deployment → GitHub Actions**
y revisar los permisos de redistribución de geometrías/datos/predicciones. Los CSV incluidos
son copias de resultados ya versionados públicamente en este repositorio.

## Diferencia respecto de Streamlit

La aplicación analítica existente está en `app/` y su contrato en `app/AGENTS.md`.
El sitio de presentación **no la sustituye ni altera**. Para añadir una capacidad real de
inferir sobre parcelas nuevas se requeriría un servicio Python usando `src/geocebada/`,
el modelo correcto, tests de equivalencia y condiciones de despliegue. El ONNX global 03D
**no es equivalente a Local04D** y no puede presentarse como tal.

## Fuentes

- `docs/METODOLOGIA_Y_RESULTADOS.md` — resumen científico y advertencias.
- `docs/TRANSDUCTIVE_OBJECTIVE.md` — contrato estadístico.
- `reports/checkpoint_05/model_manifest.json` — decisión congelada.
- `docs/AI_USAGE.md` — registro de apoyo mediante IA.

La paleta es una interpretación de investigación agrícola, **no** una especificación oficial de
marca ni un aval institucional de FIRA.
