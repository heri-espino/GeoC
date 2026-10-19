# Validación estadística final: Reference26 con media + Compact14

**Estado: IMPLEMENTADO PARA EJECUCIÓN EN LA VM; todavía sin resultados
reentrenados en este repositorio.** Se estudian 138 parcelas con cosecha
observada y se generan pronósticos de las 59 parcelas desconocidas. Objetivo
exclusivo: competencia FIRA AgroCebada. La estructura deja trazabilidad para
la presentación ante el jurado, sin seleccionar retrospectivamente la
métrica más conveniente.

## Receta congelada

Los pesos seleccionados anteriormente con el conjunto completo conocido
están en `configs/reference26_compact14_ensemble.json`:

```math
\widehat y_{\mathrm{ens}} =
0.7049462505599942\widehat y_{\mathrm{Reference26,mean}}+
0.29505374944000595\widehat y_{\mathrm{Compact14}}.
```

**Punto esencial:** Reference26 **debe volver a entrenarse** con
sustituciones por la media aritmética de X **calculada exclusivamente
en el entrenamiento permitido**. El antiguo
`final_train138.joblib` usa la receta nativa con mediana, por
lo que no sirve como representación de Reference26 con media.

Compact14 conserva **su receta original y su imputación original**,
ya que sus predicciones LOOCV originales son las que participaron
en la optimización de estos pesos. Para una comparación reproducible
también se reajusta Compact14 desde cero.

Se generan **dos modelos joblib distintos**, con el ajuste completo
original de sus recetas; el JSON contiene pesos fijos y procedencia.
Ambos comparten solo la tabla X/metadata de 197 parcelas y la geometría
congelada de distancias, siempre que el programa verifique la coincidencia
exacta de los primeros 138 identificadores, etiquetas y representaciones.

El modelo elegido con los pesos indicados no ha demostrado una mejora
generalizable frente a Reference26 medio. El análisis anterior encontró
que la mejora desaparecía al quitar una sola parcela muy favorable.
**No se presenta como nuevo campeón confirmado.**

## 1. Reentrenar desde cero

Desde `C:\Users\175199\GeoC`, con el entorno `geocebada`
y el repositorio hermano `C:\Users\175199\todoseboso`:

~~~powershell
git pull --ff-only origin experiment/nuria-champions-loocv

# Requisitos, conjunto privado de 138 y pesos congelados
python tools/run_reference26_compact14_final.py --check

# Una sola parcela completamente reajustada desde cero
python tools/run_reference26_compact14_final.py --smoke --jobs 1

# Ambas recetas desde cero para cada una de las 138 exclusiones
python tools/run_reference26_compact14_final.py --run --jobs 4

# Extrae métricas de las 138 predicciones, sin nuevos ajustes
python tools/run_reference26_compact14_final.py --summary

# Dos reajustes completos nuevos sobre las 138 etiquetas;
# reanudables si existe un joblib parcialmente generado válido
python tools/run_reference26_compact14_final.py --train-final --jobs 1
~~~

La opción `--run` puede tardar **muchas horas** o más,
porque ajusta dos recetas complejas por fold, con
validaciones internas y correcciones. No confundir las
276 recetas por fold con 276 validaciones independientes:
son 138 predicciones externas emparejadas por modelo.
Los ajustes previos, joblibs antiguos y CSV de cribados
anteriores no se reutilizan. Sólo se reanudan salidas de este
experimento si sus huellas de código y datos coinciden.
`--jobs 4` minimiza el riesgo de agotar la memoria; subir
gradualmente tras observar RAM.

Los archivos individuales se guardan, **privados e ignorados por Git**,
en `reports/reference26_compact14_fresh138/fold_001.csv`, etc.
Cada fila incluye el modelo A, el modelo B, el ensamble, ID,
rendimiento observado, `outer_train=137` y `outer_heldout=1`.

Los artefactos finales se guardan en
`reports/reference26_compact14_fresh138/final_artifacts/`:

- `reference26_mean_train138.joblib`: **nuevo Reference26 con media**;
- `compact14_original_train138.joblib`: **nuevo Compact14 nativo**;
- `predictions59_private.csv`: los **tres** resultados para cada una
  de las 59 parcelas, por ID, sin pretender conocer sus valores reales;
- `final_manifest.json`: hashes, pesos, 138/59 y paridad de
  serialización; referencias a `features197.parquet` y
  `distance197.npy` del repositorio compañero.

El entrenamiento final usa los 138 rendimientos y las mismas
representaciones y distancias auditadas. La verificación de que
`predict_saved` reproduce lo serializado no sustituye la
paridad **Python–navegador**; esta última se realizará antes
de habilitar la nueva interfaz de los tres resultados.

## 2. Informe estadístico académico

Después de completar los 138 folds:

~~~powershell
python tools/report_reference26_compact14_final.py --run
python tools/report_reference26_compact14_final.py --summary
~~~

Incluye:

| Familia | Métricas y pruebas | Interpretación |
|---|---|---|
| Errores | L1, L2, Linf, SSE, MSE, RMSE, MAE, MedAE, P90/P95/P99 | Magnitud promedio, total y errores extremos |
| Asociación y desempeño | R² predictivo, varianza explicada, Pearson, Spearman, concordancia de Lin | Relación y reducción de pérdida frente a referencia |
| Sesgo y calibración | Media/mediana residual, desviación, pendiente e intercepto de calibración | Sobreestimación, subestimación y distorsión de escala |
| Relativos | WAPE, NRMSE, sMAPE, MAPE si y distinto de cero, RMSLE si y/pred no negativos | Comparar errores con la escala de cosecha |
| Pérdidas especializadas | Huber, log-cosh, quantile loss, error cúbico | Sensibilidad diferencial a errores grandes |
| Cobertura por umbral | Fracción con error absoluto ≤0.25/0.50/1.00 t/ha | Proporción de predicciones dentro de tolerancias explícitas |
| Residuos | Cuantiles, IQR, MAD, asimetría, exceso de curtosis | Simetría, colas pesadas, casos problemáticos |
| Normalidad | Shapiro–Wilk, D'Agostino–Pearson K², Anderson–Darling; estadístico JB | Compatibilidad exploratoria con normalidad |
| Varianza condicional | Koenker/Breusch–Pagan, Brown–Forsythe y correlaciones de error | Heterocedasticidad y desempeño por rango de producción |
| Diferencias emparejadas | Sign test, Wilcoxon, Holm; diferencias de MAE/MSE | Comparación por la misma parcela, no tests independientes |
| Espacial | Moran global I, Geary C, p-valores de permutación (k=4/8/12/24), Holm/BH | Autocorrelación geográfica de errores firmados, absolutos y cuadrados |
| Distancia | Semivariograma empírico del error y sus diferencias | Dependencia de las equivocaciones con separación km |
| Regional | RMSE, MAE, sesgo por estado/municipio (n≥5) | Desempeño desigual por zona, NO spatial block-CV |

Las definiciones y fórmulas se encuentran en
`src/geocebada/evaluation/loocv_comprehensive_metrics.py`,
`src/geocebada/evaluation/jury_residual_diagnostics.py` y
`tools/report_reference26_compact14_final.py`. No
interpretar las normas totales L1/L2 como MAE/RMSE.

**R² ajustado:** no se fuerza un valor con un número
inventado de parámetros. En este modelo anidado/local/
regularizado no existe un `p` único defendible; el
reporte conserva `adjusted_r2 = N/A` y su explicación.
Análogamente, AIC/BIC y log-verosimilitud predictiva
no están definidos sin especificar y validar una distribución
probabilística de respuesta. No reportarlos artificialmente.

### Precauciones estadísticas

- Los 138 errores LOOCV proceden de **ajustes solapados**.
  Shapiro, K², tests de medias, Brown–Forsythe, Koenker y
  Wilcoxon tienen hipótesis de muestreo que podrían no cumplirse.
  Sus p-valores se identifican explícitamente como
  **diagnósticos exploratorios**, no pruebas confirmatorias
  de superioridad del ensamble.
- SciPy advierte que el p-valor aproximado de
  Jarque–Bera necesita muestra mucho mayor que 138.
  Se registra el estadístico y la advertencia; no
  se publica un p engañoso.
- Aun cuando haya p pequeña en Shapiro, una predicción
  de regresión no requiere residuos marginales normales para
  calcular RMSE/MAE. Una p grande tampoco prueba normalidad.
- Moran/Geary usan pesos KNN sin auto-vecindad y
  1/k por fila, mediante Haversine en kilómetros.
  La permutación plantea una hipótesis de intercambiabilidad
  espacial bajo nulo; Moran positivo sugiere
  agrupamiento espacial de errores, no causalidad.
  Se reportan Holm y Benjamini–Hochberg en la familia
  de todos los tests espaciales. Las pruebas son de
  sensibilidad frente a k, no 4 validaciones independientes.
- LOOCV permite entrenar con vecinos muy próximos;
  puede sobrevalorar desempeño al extrapolar a regiones
  distantes. **Dejar municipios o bloques geográficos
  completos fuera sería un estudio adicional diferente**.
- No hay inferencia sobre error real de las 59 parcelas,
  dado que sus rendimientos verdaderos no están disponibles.
  La dispersión entre modelos no es un intervalo de confianza.
- El ensamble fue optimizado antes con los 138 y
  conocidos. Repetir los mismos heldouts con modelos
  reentrenados no elimina el sesgo de selección de
  los pesos. La evaluación totalmente externa de ese
  meta-ajuste requeriría aprendizaje del peso en
  validación cruzada interna o datos adicionales.

### Figuras e informe

`reports/reference26_compact14_fresh138/analysis/` contendrá:
`primary_metrics.csv`, `specialized_metrics.csv`,
`normality_and_distribution.csv`,
`heteroscedasticity_and_calibration.csv`,
`paired_loss_tests.csv`,
`global_spatial_autocorrelation.csv`,
`residual_semivariogram.csv`,
`private_regional_errors.csv`,
`INFORME_ESTADISTICO_FINAL.md` y
`figures/` (histograma, Q–Q, observado/predicho,
residuo/predicción, error absoluto acumulado,
Moran vs k, semivariograma y mapa privado).

Las figuras pueden revelar ubicaciones y rendimientos
individuales. **No se exportan automáticamente**;
el proceso genera un JSON de **estadísticas agregadas**,
`jury_metrics_site_ready.json`.

Tras inspeccionar los valores y permisos:
~~~powershell
python tools/report_reference26_compact14_final.py --publish
~~~

Copia las métricas agregadas a
`../todoseboso/site/data/ensemble_evaluation.json`
para futura integración de una **tabla interactiva con filtros
por familia, modelo, estadístico y definición**. El comando no
hace cambios HTML/JS, ni publica archivos automáticamente
en GitHub Pages.

## 3. Web: tres predictores verificables

Actualmente el repositorio `todoseboso` dispone de dos
implementaciones: una API FastAPI que carga
`final_train138.joblib` con verificaciones SHA256, y un
motor JavaScript de `Reference26` que reproduce las 59
predicciones mediante JSON verificado. **Ambas corresponden
todavía al modelo original con mediana**.

No basta con sustituir la etiqueta y publicar
`weights.json`:

1. El nuevo Reference26_mean necesita su **propio**
   `joblib`/exportación numérica verificada y
   el pipeline de imputación por media.
2. Compact14 debe tener un ejecutor de predicción propio
   o una ruta certificada a `predict_saved(pack,...)`.
3. Se deben comprobar las **59 predicciones**
   de Python frente al motor web, para cada
   componente y para la combinación, con tolerancia
   documentada (y hashes de artefactos).
4. La interfaz debe mostrar **Reference26_mean**,
   **Compact14** y **Ensamble sugerido** como tres resultados,
   siempre aclarando que el RMSE de la mezcla
   fue obtenido tras una búsqueda exploratoria.
5. No integrar modelos antes de las pruebas de paridad,
   ni exponer entrenamientos, etiquetas o IDs privados
   salvo autorización explícita del concurso.

## Fuentes

- [NIST: Residual diagnostics](https://www.itl.nist.gov/div898/handbook/pri/section2/pri24.htm)
- [SciPy: Shapiro–Wilk](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.shapiro.html)
- [SciPy: D'Agostino K²](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.normaltest.html)
- [SciPy: Jarque–Bera limitations](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.jarque_bera.html)
- [SciPy: Brown–Forsythe via Levene](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.levene.html)
- [PySAL: Global Moran's I and permutation tests](https://pysal.org/esda/stable/user-guide/global_morans_i.html)
- [PySAL: LISA, multiplicity and interpretation](https://pysal.org/esda/stable/user-guide/local_spatial_autocorrelation.html)
- [Le Rest et al. (2014), Spatial leave-one-out](https://doi.org/10.1111/geb.12161)
- [Roberts et al. (2017), Spatial blocking](https://doi.org/10.1111/ecog.02881)
