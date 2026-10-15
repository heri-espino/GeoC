# Checkpoint 13 — Búsqueda aleatoria de 10 000 configuraciones con selección anidada

**Estado:** código y pruebas disponibles; corrida sobre el conjunto real pendiente. **No hay nuevos resultados de RMSE.** Este experimento conserva las 59 predicciones finales sin modificarlas.

## Pregunta

Tras el Checkpoint 10, el predictor original **Local04D** con PCA20, k24, alfa30 y p1 sigue en primer lugar: **RMSE 0.481936 t/ha, R² 0.685388** (sobre las 16 máscaras de confirmación de 08, reutilizadas en 10). Las variantes retuneadas en el Checkpoint 10 no ganaron. El usuario pidió ensayar **10 000 configuraciones aleatorias** incluyendo alternativas de PCA y regularización, aprovechando que la búsqueda anterior se ejecutó rápidamente.

La muestra es pequeña (**138 rendimientos reportados**, de los que se utilizan 97 para entrenar cada fold externo). Evaluar muchísimos candidatos contra los mismos rendimientos aumenta el riesgo de sobreajustar la selección. Por eso 13 emplea una búsqueda escalonada y no selecciona modelos sobre las 41 etiquetas externas.

## Diseño exacto

Se muestrean **exactamente 10 000 configuraciones únicas de semilla fija**, aproximadamente repartidas por igual entre seis familias:

| Familia | Reducción o representación | Regresor |
|---|---|---|
| PCA_Ridge | PCA de la representación C4 | Ridge local |
| ICA_Ridge | PCA40 → FastICA | Ridge local |
| RP_Ridge | Proyección aleatoria gaussiana | Ridge local |
| PLS_Ridge | PLS supervisado dentro del fold | Ridge local |
| PCA_ElasticNet | PCA C4 | Elastic Net local |
| PLS_ElasticNet | PLS supervisado dentro del fold | Elastic Net local |

Los candidatos se generan mediante un generador aleatorio fijo, antes de observar los errores. Se conserva **la misma muestra de 10 000 candidatos** en cada una de las 16 máscaras externas: no son 160 000 modelos únicos.

| Dimensión | Valores o distribución |
|---|---|
| Componentes PCA | 10, 20, 30, 40, 60, 80, 120, 160 |
| Componentes PCA→ICA | 8, 16, 32 |
| Componentes RP | 16, 32, 64 |
| Componentes PLS | 4, 8, 16 |
| Vecinos geográficos k | 8, 12, 16, 20, 24, 32, 48 |
| Exponente geográfico p | Uniforme entre 0 y 2.5 |
| alfa Ridge | Log-uniforme entre 0.1 y 1000 |
| alfa Elastic Net | Log-uniforme entre 0.001 y aproximadamente 2 |
| Proporción L1 Elastic Net | Uniforme entre 0.05 y 0.95 |

**Importante:** ICA y proyección aleatoria utilizan únicamente la X de las 197 parcelas y son transductivos. PLS está supervisado, por lo que se **vuelve a ajustar exclusivamente con las etiquetas del fold de entrenamiento** para cada evaluación interna y para el reajuste exterior. Nunca se crea un PLS supervisado global usando y de los 41 pseudo-objetivos o de las 59 consultas reales.

El control `PCA20_frozen` reproduce la configuración original `Local04D`; no compite por hiperparámetros ni se cambia.

## Validación cruzada anidada en dos etapas

Para cada una de las **16 máscaras X-only del Checkpoint 08** (97 parcelas en el entrenamiento exterior, 41 pseudo-objetivos externos):

1. **Cribado de 10 000 candidatos:** se reservan **12 de los 97 rendimientos para el cribado interno** (los mismos para todos los candidatos), y se ajusta cada configuración con los otros 85. No se consulta ningún y externo.
2. **Selección balanceada:** se conservan los **12 candidatos con menor RMSE de cribado de cada una de las seis familias**, máximo 72 finalistas. Esto evita que una familia domine por azar el presupuesto.
3. **Reevaluación de finalistas:** se ejecuta un **4-fold CV con dos repeticiones** sobre las 97 etiquetas exteriores de entrenamiento. Cada candidato obtiene **194 predicciones internas**, y con ellas un RMSE de selección.
4. **Evaluación exterior:** se elige el ganador de cada familia y el ganador conjunto entre todos los finalistas. Después se predicen los **41 pseudo-objetivos externos una vez por procedimiento**, junto con Local04D fijo. Resultado: **8 modelos/procedimientos emparejados**, 16 × 41 = 656 pronósticos repetidos por procedimiento.
5. **Informe:** RMSE pooled, R² pooled, peor RMSE de split, comparaciones de \(\Delta\)MSE y bootstrap por parcela (todas las repeticiones de un ID se remuestrean juntas), además de la familia e hiperparámetros seleccionados en cada split.

La comparación **principal y declarada antes de ver resultados** es `Random_joint_nested` contra `PCA20_frozen`; no se promocionará automáticamente el modelo que tenga el menor RMSE retrospectivo entre los ocho. Los 656 pares de un modelo representan hasta 132 parcelas únicas, no 656 parcelas independientes.

Se trata de **una búsqueda exploratoria** porque el banco exterior de 08 se ha examinado en 08, 10, 11 y 12. Aunque la selección dentro de cada split respeta los límites de y, cualquier resultado debe interpretarse teniendo en cuenta la selección histórica repetida sobre las mismas parcelas.

## Reanudación, costo y seguridad

- Los 10 000 candidatos quedan congelados en `candidates.csv` con una semilla. Si se ejecuta `--trials 10000` se muestrean y se criban los 10 000 por **cada** uno de los 16 splits exteriores.
- Las evaluaciones de cribado se guardan en **bloques de 250 configuraciones** por split. Una interrupción permite reanudar sin rehacer los bloques terminados.
- Cada split registra una tabla de finalistas, el ganador por familia, las 41 predicciones de cada procedimiento y su resumen agregado.
- Un `run_spec.json` almacena las huellas SHA-256 de las entradas, el código y el protocolo. Si se cambia el código, la configuración o las fuentes, debe elegirse **otro directorio `--output`** para evitar mezclar experimentos.
- **Puede ser significativamente más lento** que los Checkpoints 10 y 12: cada configuración representa muchas regresiones locales (una por parcela consultada), no un solo entrenamiento global. Los modelos con PLS y Elastic Net pueden ser más costosos. No existe una estimación de tiempo fiable sin ejecutar el proceso sobre la máquina.
- Los CSV de cribado y predicciones individuales son **locales e ignorados por Git**: `reports/checkpoint_13_randomized/`. No subir rendimientos individuales o etiquetas privadas con `git add -f`.
- **No modifica** las 59 estimaciones de `reports/checkpoint_05/final_predictions.csv` ni los resultados históricos.

## Comandos PowerShell

Desde el directorio raíz del repositorio con conda `geocebada` activado:

```powershell
git switch main
git pull --ff-only origin main
conda activate geocebada

python tools/run_checkpoint_13_randomized.py --check
python tools/run_checkpoint_13_randomized.py --smoke

# 10 000 candidatos por split externo, con salida reanudable
python tools/run_checkpoint_13_randomized.py --run --trials 10000

# Al completar los 16 splits
python tools/run_checkpoint_13_randomized.py --summary --trials 10000
```

Para explorar con menor presupuesto sin tocar los archivos del experimento principal se puede usar `--trials 1000 --output reports/checkpoint_13_pilot`, con la salvedad de que **el pilot debe tener su directorio y huellas propios**. El smoke siempre corre un split y 60 configuraciones y escribe en el subdirectorio `smoke`.

No es necesario ejecutar Checkpoint 12 nuevamente para correr el 13; ambos leen las mismas capas C4 y las máscaras externas ya congeladas.

## Archivos producidos

| Archivo | Contenido |
|---|---|
| `run_spec.json` | Contrato del estudio, fingerprints y presupuesto |
| `candidates.csv` | 10 000 combinaciones predefinidas, sin etiquetas |
| `confirm_XX__screen__YYYYY.csv` | Cribado interno por bloques |
| `confirm_XX__reranked.csv` | Hasta 72 finalistas con CV repetida |
| `confirm_XX__paired.csv` | Predicciones externas emparejadas, con rendimientos observados locales |
| `confirm_XX__selection.json` | Ganador por familia y ganador conjunto |
| `comparison_summary.csv` | RMSE, R², recuentos y peores splits |
| `paired_cluster_bootstrap.csv` | Diferencias de MSE contra Local04D, con IC95% agrupados por parcela |
| `selected_hyperparameters.csv` | Modelos/hiperparámetros elegidos por split |

## Conclusión prevista, no resultado

**No se espera necesariamente mejorar Local04D** por aumentar el número de modelos explorados. Es particularmente probable que una búsqueda de 10 000 candidatos encuentre configuraciones con buen RMSE de cribado interno por simple azar. La CV repetida sirve para filtrar ese ruido; el banco exterior se emplea para una evaluación emparejada exploratoria, no para seguir iterando indefinidamente.

Para afirmar una mejora definitiva es necesario contrastarla con un conjunto genuinamente nuevo de rendimientos o una confirmación independiente que no haya guiado la selección.

Véanse [Checkpoint 10](CHECKPOINT_10_PCA_GRID.md), [Checkpoint 11](CHECKPOINT_11_WIDE_PCA.md), [Checkpoint 12](CHECKPOINT_12_ALTERNATIVES.md) y [Metodología](METODOLOGIA_Y_RESULTADOS.md).
