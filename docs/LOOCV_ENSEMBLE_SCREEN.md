# Pruebas rápidas de ensambles sobre las predicciones LOOCV guardadas

**Estado:** código de diagnóstico implementado; ejecución con datos reales
pendiente en la VM. No se han entrenado modelos, seleccionado pesos finales
ni cambiado las predicciones de las 59 parcelas objetivo.

## Conclusión del último grid Reference26 con media

El usuario terminó el estudio `tools/run_reference26_ridge_budget_mean.py`,
usando los 138 folds externos y 137 etiquetas de entrenamiento por ajuste.
Resultados proporcionados directamente por la ejecución local:

| Modelo | Imputación | Vecinos | Ridge alpha | RMSE | MAE | Sesgo |
|---|---|---:|---:|---:|---:|---:|
| Reference26 original | mediana | 24 | 30 | 0.4681855146560597 | 0.3336730043055529 | 0.002375 |
| Reference26 ancla | media | 24 | 30 | **0.468138** | 0.333786 | 0.001466 |
| Reference26 retuneado | media | 20 | 30 | 0.470112 | 0.336173 | -0.000110 |
| Reference26 retuneado | media | 12 | 3.2 | 0.474493 | 0.339729 | 0.007165 |

El ancla `k=24,alpha=30` fue la mejor **de las tres finalistas**
en la evaluación completa. Su ventaja de ~0.000048 t/ha frente a
la mediana es minúscula; **no** describe una mejora sólida ni un
nuevo hiperparámetro óptimo global. Los resultados de 12/46
heldouts sobreseleccionaron configuraciones que empeoraron sobre 138,
por lo que no conviene escalar a cientos de modelos o combinaciones.

## Qué buscamos

Identificar modelos que aporten errores **complementarios**, no solo
modelos con un RMSE individual bajo. Priorizamos un ensamble
barato de predicciones existentes, no otro entrenamiento exhaustivo.
Reference26 completo incluye CT3 + experto Ridge local, gate geológico
y corrección R3; el siguiente ensamble añade predictores completos
sobre ese resultado, sin modificar su receta interna.

Candidatos LOOCV completos que el script puede recuperar:

| Nombre | Fuente privada | Condición |
|---|---|---|
| `Reference26_mean` | `reports/reference26_ridge_budget_mean/fold_001..138/k24_a30.csv` | Ancla con media, requiere manifiesto |
| `Reference26_original` | `reports/selected_champions_loo/loo_predictions.csv` | Referencia histórica con mediana |
| `Compact14` | Mismo archivo anterior | Competidor completo |
| `Local04D` | `reports/area_weighted_climate_loo/loo_predictions.csv` | Long: filtro `GeoPCA20_GeographicWeightedRidge`; 137/1 |
| `Reference26_DBSCAN` | `reports/peer_dbscan_loo138/loo_predictions.csv` o 138 archivos `fold_XXX.csv` | Deben estar completos |

Si falta Local04D en su archivo original, el script intenta
`reports/peer_champions_loo/loo_predictions.csv` solo si contiene
las 138 predicciones auténticas. La ausencia de un candidato opcional
no detiene los demás; se informa como `[SKIP]`.

**IMPORTANTE:** las proyecciones gaussianas, CatBoost, Local07,
Prithvi, PLS y modelos GMM de Checkpoints 03–14 tienen múltiples
experimentos con protocolos **diferentes** (16 × 41 heldouts, 97 train,
grupos geográficos, etc.). No se pueden añadir a un conjunto de 138
predicciones LOOCV por el mero hecho de que existan sus archivos.
En concreto, una **proyección aleatoria gaussiana** no es lo mismo que
regresión mediante un **proceso gaussiano**; GMM también es un método de
agrupamiento y no se contabiliza como un pronosticador de rendimiento.

## Cómo ejecutarlo en PowerShell

~~~powershell
cd C:\Users\175199\GeoC
conda activate geocebada
git pull --ff-only origin experiment/nuria-champions-loocv

# Localiza archivos y muestra posibles tablas adicionales, sin entrenar.
python tools/run_loo_ensemble_screen.py --audit

# Usa las predicciones completas compatibles; no reentrena nada.
python tools/run_loo_ensemble_screen.py --run

# Recupera la tabla de resultados sin repetir cálculos.
python tools/run_loo_ensemble_screen.py --summary
~~~

Se imprime:
- RMSE, MAE, sesgo, R² y delta-MSE frente a Reference26 con media;
- correlación de errores firmados de los modelos individuales;
- proporción de parcelas en que el segundo modelo reduce el error
  absoluto del primero;
- todos los ensambles evaluados, ordenados por RMSE.

Solo se mezclan modelos con **exactamente los mismos 138 IDs únicos**
(`ID_POLIGONO`) y observaciones reales coincidentes (`observed`,
tolerancia absoluta `1e-9`). No se toma simplemente la intersección
de IDs, ni se combinan filas por posición.

## Cómo incluir resultados adicionales

Si un nuevo modelo tiene realmente **138 predicciones externas
con 137 train / 1 heldout**, puede añadirse con:

~~~powershell
python tools/run_loo_ensemble_screen.py --run --extra "CatBoost=reports/mi_experimento/catboost_loo_predictions.csv"
~~~

Para modelos externos al runner original se exige un CSV con
`ID_POLIGONO,observed,outer_train,outer_heldout,predicted`
(o una columna con el nombre del modelo), 138 IDs únicos,
`outer_train=137` y `outer_heldout=1` en todas las filas.
Un archivo largo con `model` debe tener una etiqueta que
coincida con el nombre usado en `--extra`.

No basta con convertir artificialmente un archivo de 16×41 a 138 filas
ni cambiar las columnas `outer_train` y `outer_heldout`:
**los 138 ajustes deben haberse ejecutado realmente con 137 etiquetas
y una excluida**. El control de esquema no sustituye la auditoría de
procedencia. Si solo tenemos predicciones de 97/41 para CatBoost y
proyecciones gaussianas, la opción válida es compararlos dentro de
ese protocolo, después de conseguir predicciones de Reference26 con
las **mismas máscaras y las mismas 97 etiquetas de entrenamiento**,
o realizar su LOOCV completo si el diagnóstico lo justifica.

## Ensambles y límites

Primero se calculan los RMSE individuales. Se prueban
pesos predeclarados sobre Reference26 con media:

- Dos modelos: `(1-w) Reference26_mean + w model_j`,
  con `w ∈ {0.05, 0.10, 0.20, 0.30, 0.50}`.
- Tres modelos: `(1-u-v) Reference26_mean + u model_j + v model_k`,
  con `u,v ∈ {0.10, 0.20, 0.30}` y `u+v <= 0.50`.
- Se incluyen los predictores individuales y el ancla sin mezclar
  como controles. No se ejecuta entrenamiento de CatBoost, GP o
  stacking neuronal.

Se guardan únicamente **tablas agregadas** en la ruta ignorada
`reports/loo_ensemble_screen/`:

- `ensemble_scores.csv`
- `residual_correlation.csv`
- `pairwise_diversity.csv`
- `run_spec.json`

El programa lee los rendimientos individuales sin crear un nuevo archivo
de predicciones por parcela, y ninguna de estas rutas debe subirse
sin revisar su confidencialidad.

**Advertencia metodológica:** escoger los pesos con los errores LOOCV de
esas mismas 138 parcelas da una puntuación **exploratoria optimista**.
No se puede considerar validación anidada del meta-modelo: las
predicciones OOF guardadas para otras parcelas suelen haber sido
ajustadas con el y de la parcela que ahora se pretende excluir del
meta-ajuste. La siguiente fase, solo si los resultados son prometedores,
será generar predicciones internas realmente cross-fitted con los
137 entrenamientos disponibles, y seleccionar pesos simples y
regularizados allí. La anterior búsqueda adaptativa de Reference26
también introduce optimismo independiente.

**Regla de decisión:** inspeccionar correlaciones residuales, mejora
emparejada y estabilidad antes de decidir qué predictor adicional
merece un LOOCV nuevo. Que una proyección gaussiana tenga peor RMSE
individual no demuestra que sea inútil para un ensamble, pero tampoco
justifica reentrenarla sin evidencia de complementariedad.

## Próximo paso

Ejecutar `--audit` y `--run` en la VM, compartir las tablas
`[ENSEMBLE]` y comprobar si Local04D, Compact14 o DBSCAN
aporta una mejora material al Reference26 con media. Después decidir
si tiene sentido preparar los 138 folds de CatBoost o un modelo
espacial gaussiano. Ningún modelo es promovido automáticamente.
