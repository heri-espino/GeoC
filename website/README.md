# GeoCebada · sitio de presentación (2026)

Sitio editorial **estático, sin dependencias de npm**. Presenta en español el reto, los datos,
la metodología (modelo GeoRidge, `Local04D` en el repositorio), una explicación visual de la
validación cruzada 97/41 y los protocolos de las etapas 4B/7/8, dos evaluaciones separadas
(etapas 7 y 8), referencias de comparación, un explorador de las 59 predicciones congeladas
con su soporte de datos, y un anexo con los modelos descartados, las variables y las
limitaciones.

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

## Resumen de una página para la entrega

Abrir [`website/resumen.html`](resumen.html) para consultar el **resumen ejecutivo en
una página tamaño carta**. Elegir «Imprimir o guardar como PDF» en el navegador y seleccionar
«Guardar como PDF», con tamaño carta, escala 100 %, márgenes ninguno y gráficos de fondo.
El PDF exportado debe revisarse para confirmar que contiene exactamente una página,
sin cortar texto o cifras. Este es un **anexo breve de presentación**: no sustituye los
archivos, videos y demás entregables específicos indicados en las bases del reto.

El PDF de una página también se puede producir a partir de este HTML mediante un navegador
headless para control visual, pero esa construcción de PDF es manual; no se ejecuta por cada push.

## Cómo explicamos la validación cruzada

La sección `#validacion` contiene un diagrama **esquemático** de 97 parcelas
de entrenamiento y 41 pseudo-objetivos en cada una de las 16 particiones. Las 16 barras
representan **sólo tamaños**, nunca identidades de parcelas. La página explica:

- 04B/07: selección X-only de los pseudo-objetivos que imitan el destino, cuotas de
  estados por Hamilton, balance municipal, 80 propuestas por repetición, penalización
  por similitud de Jaccard; 125 parcelas únicas reservadas.
- 08: nuevo banco X-only por estratos estado/municipio, 160 propuestas por repetición,
  selección de superposición mínima y 132 parcelas únicas; las dos familias candidatas
  se reajustan con 3-fold interior sobre los 97 ejemplos y Optuna.
- El RMSE se mide **después** del ajuste sobre los 41 rendimientos reservados. Las
  656 predicciones no son independientes, los bancos 07/08 no se comparan como una
  única muestra, y las 59 etiquetas finales permanecen ocultas.

## Origen y sincronización de los datos

| Copia estática | Fuente canónica |
| --- | --- |
| `data/checkpoint07_summary.csv` | `reports/checkpoint_07/shared/three_model_summary.csv` |
| `data/checkpoint08_summary.csv` | `reports/checkpoint_07/shared/checkpoint08_comparison_summary.csv` |
| `data/checkpoint08_bootstrap.csv` | `reports/checkpoint_07/shared/checkpoint08_paired_cluster_bootstrap.csv` |
| `data/predictions.csv` | `reports/checkpoint_05/final_predictions.csv` |
| `data/checkpoint02_baseline_summary.csv` | `reports/checkpoint_02/baseline_summary.csv` |
| `data/checkpoint03b_expression_stability.csv` | `reports/checkpoint_03b/expression_stability.csv` |
| `data/checkpoint04b_protocol_summary.csv` | `reports/checkpoint_04b/protocol_summary.csv` |
| `data/checkpoint05_protocol_summary.csv` | `reports/checkpoint_05/protocol_summary.csv` |
| `data/final_prediction_diagnostics.csv` | `reports/checkpoint_04f/final_prediction_diagnostics.csv` |
| `data/checkpoint06_feature_family_tail_importance.csv` | `reports/checkpoint_06/feature_family_tail_importance.csv` |
| `data/checkpoint06_feature_family_error_risk.csv` | `reports/checkpoint_06/feature_family_error_risk.csv` |

Revisar o actualizar después de cambiar los resultados **sólo si la decisión de selección está documentada**:

```bash
python tools/sync_website_data.py --check
python tools/sync_website_data.py --write
python tools/sync_website_data.py --check
pytest -q tests/test_website_presentation.py
```

`--write` únicamente sincroniza estas once copias. Nunca vuelve a entrenar un modelo,
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

## Nombres de presentación y dependencias

| Nombre en el sitio | Nombre en el repositorio / informe |
| --- | --- |
| GeoRidge (formal: «GeoRidge Local — Configuración Geográfica (k=24, α=30, p=1)») | `Local04D` = `LocalRidge_C4_all_deterministic_Geo_k24_a30_p1` |
| GeoRidge fijo / reajustado | `Local04D_frozen` / `Local04D_nested` |
| GeoRidge+ | `Local07` |
| Prithvi-EO, CatBoost | `Prithvi07`, `catboost` |

Los nombres sólo cambian en el texto visible; las claves de los CSV y del informe canónico
no se modifican. La equivalencia aparece en el Anexo A y en `resumen.html`. En el sitio los
checkpoints se llaman «etapas».

La ecuación del método está prerrenderizada con **KaTeX 0.16.11** y se sirve desde
`website/vendor/katex/` (CSS y fuentes woff2 locales). No hay JavaScript de KaTeX en el
navegador ni solicitudes a terceros. Para cambiar la fórmula hay que volver a renderizarla con
`katex.renderToString(..., {displayMode: true, output: "htmlAndMathml"})`.

Secciones añadidas, todas leídas de las copias CSV anteriores:

- **Referencias de comparación** (Resultados): RMSE medio de 16 particiones similares al
  destino frente a promedios, modelos globales y vecinos simples.
- **Familias de información** (Resultados): ablaciones de la etapa 2 con modelos de referencia;
  no son la importancia de variables de GeoRidge.
- **Soporte de datos** (Explorador): nivel `x_support_tier` y diferencia con el modelo de grafo;
  no es un intervalo de confianza ni la clasificación de 04A.
- **Anexo D** (variables) y **Anexo E** (limitaciones).

## Publicación

El workflow `.github/workflows/deploy_website.yml` sube **sólo `website/`** al artefacto de
GitHub Pages, mediante `actions/upload-pages-artifact` y `actions/deploy-pages`, en pushes a
`main` que modifiquen `website/` o al ejecutarlo con `workflow_dispatch`. Es un despliegue
ligero; **no** genera figuras, papers, datasets pesados, ni ejecuta GPU.

Antes de publicar, confirmar en **Settings → Pages → Build and deployment → GitHub Actions**
y revisar los permisos de redistribución de geometrías/datos/predicciones. Los CSV incluidos
son copias de resultados ya versionados públicamente en este repositorio. La carpeta
`vendor/katex/` (licencia MIT) se despliega junto con el sitio.

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
