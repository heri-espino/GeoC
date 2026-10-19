# Reference26: búsqueda reducida con imputación por media

Este experimento usa la **media aritmética calculada dentro de cada subconjunto
de entrenamiento** para imputar valores X faltantes en cinco preprocesadores
existentes del modelo Reference26. Mantiene el ajuste original de clima,
geología, Ridge y corrección R3. No cambia los datos originales ni los
archivos fuente de `../todoseboso`.

**El experimento de 250 configuraciones con mediana NO se mezcla con este
experimento.** Las predicciones que se calcularon con mediana, aunque sean
640, no son válidas como predicciones de la variante con media. Se conservan
en `reports/reference26_ridge250_loo/` para posibles análisis históricos.

La implementación de la media ya existe en
`src/geocebada/evaluation/peer_mean_imputation.py`, introducida en el
commit `467980a`, con una evaluación independiente de media frente a mediana
(`tools/run_reference26_mean_imputation_loo.py`). Este nuevo script reutiliza
**la misma lógica de imputación** durante todos los ajustes anidados de
Reference26, pero explora los hiperparámetros geográficos.

## Configuraciones y fases

| Fase | Modelos | Parcelas excluidas iguales para todos | Ajustes nuevos sin reutilización |
|---|---:|---:|---:|
| Cribado | 25 | 12 primeras | 300 |
| Ampliación | 8 | 46 primeras | 272 adicionales |
| Comparación final | 3 | 138 | 276 adicionales |

Hiperparámetros: vecinos `k = 12, 20, 24, 32, 48` y Ridge
`alpha = 3.2, 10, 30, 100, 300`. Se conserva la configuración original
`(k=24, alpha=30)` como **referencia de la nueva variante con media**.

Los RMSE de cada fase se calculan únicamente sobre el mismo subconjunto
de parcelas; también se muestra el RMSE del modelo histórico con mediana
*en esas mismas parcelas*. La diferencia de error se etiqueta
`delta_mse_vs_original_median`. Comparar contra la mediana no implica que
las predicciones de mediana se reutilicen para ajustar un modelo con media.

## Ejecutar desde PowerShell

Detener antes el proceso de 250 configuraciones con Ctrl+C y dejar que
terminen los procesos secundarios. No borrar los resultados antiguos.

~~~powershell
conda activate geocebada
git pull --ff-only origin experiment/nuria-champions-loocv

# Sin entrenamiento: verificar bundle y predicciones originales
python tools/run_reference26_ridge_budget_mean.py --check

# Un solo ajuste (k=24, alpha=30) para detectar errores
python tools/run_reference26_ridge_budget_mean.py --smoke --jobs 1

# Primera comparación completa con media: 25 configuraciones, 12 parcelas
python tools/run_reference26_ridge_budget_mean.py --screen --jobs 12

# 8 seleccionadas sobre las primeras 46 parcelas
python tools/run_reference26_ridge_budget_mean.py --advance --jobs 12

# 3 finalistas sobre las 138 parcelas
python tools/run_reference26_ridge_budget_mean.py --final --jobs 12

# Tablas de las fases completadas, sin entrenar
python tools/run_reference26_ridge_budget_mean.py --summary
~~~

`--jobs` admite cualquier entero positivo. Cada tarea se guarda
individualmente bajo `reports/reference26_ridge_budget_mean/` y puede
reanudarse tras detener el entrenamiento. El código guarda un manifiesto
que identifica la imputación por media, los cinco preprocesadores, el bundle
de entrada, el código y el conjunto de modelos.

**No se importan como caché** los resultados del grid de 250 realizado
con mediana. Tampoco se descartan: permanecen intactos.

La búsqueda es **exploratoria** porque selecciona configuraciones a partir de
errores de parcelas previamente observadas. Evaluar las finalistas con
LOOCV sobre las 138 etiquetas no produce una confirmación independiente
del mínimo seleccionado; necesitaríamos validación anidada completa o
etiquetas nuevas para inferencias de generalización no sesgadas.

## Evaluación de la imputación como cambio separado

Antes de atribuir una mejora a la elección de hiperparámetros, se puede
evaluar exclusivamente el cambio de mediana a media usando:

~~~powershell
python tools/run_reference26_mean_imputation_loo.py --check
python tools/run_reference26_mean_imputation_loo.py --smoke --jobs 1
python tools/run_reference26_mean_imputation_loo.py --run --jobs 12
python tools/run_reference26_mean_imputation_loo.py --summary
~~~

Ese experimento mantiene `k=24, alpha=30` y los demás parámetros
originales, por lo que separa el efecto de la imputación del efecto del
grid de hiperparámetros. No es necesario terminarlo antes de empezar
el cribado reducido, pero es deseable para documentar qué mejoró realmente.
