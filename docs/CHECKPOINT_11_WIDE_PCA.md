# Checkpoint 11: búsqueda amplia y validación interna más estable de Local04D

**Estado:** implementación disponible; evaluación completa pendiente de ejecutar en la estación de trabajo.  
**Protocolo:** exploratorio, sin reemplazar las 59 estimaciones oficiales.

## 1. Motivación: qué mostró Checkpoint 10

Checkpoint 10 completó el grid conjunto de cuatro parámetros sobre las 16 máscaras de Checkpoint 08. Los resultados emparejados fueron:

| Método | RMSE pooled (t/ha) | R² pooled |
|---|---:|---:|
| **Local04D fijo (PCA 20, k 24, alfa 30, p 1)** | **0.481936** | **0.685388** |
| PCA 80, búsqueda anidada | 0.492532 | 0.671400 |
| PCA 20, búsqueda anidada | 0.493880 | 0.669599 |
| PCA 40, búsqueda anidada | 0.494361 | 0.668955 |
| PCA 160, búsqueda anidada | 0.495184 | 0.667853 |
| PCA y parámetros seleccionados conjuntamente | 0.497939 | 0.664146 |

El bootstrap por ID de parcela dio **diferencias de MSE positivas** e intervalos 95 % por encima de cero para las alternativas anidadas frente al control fijo. Las 16 particiones representan 656 predicciones repetidas pero solo 132 parcelas únicas.

**Hipótesis para 11:** con apenas 97 etiquetas por split externo, escoger un ganador entre muchos hiperparámetros mediante una sola partición interior de 3 folds puede ser inestable. Probar más valores tiene sentido solo si se mejora también la selección interior.

## 2. Un nuevo experimento (no reemplaza 10)

La familia de predicción permanece **Local04D**: Ridge independiente para cada parcela de consulta, vecinos geográficos mediante Haversine, pesos inversamente proporcionales a `distancia ** p`, componentes PCA de la matriz C4 determinista de 2,082 variables, sin utilizar los 59 rendimientos ocultos.

| Parámetro | Valores de Checkpoint 11 |
|---|---|
| Componentes PCA | 10, 20, 30, 40, 60, 80, 120, 160 |
| Vecinos locales `k` | 8, 12, 16, 20, 24, 32, 48 |
| Penalización Ridge `alpha` | 1, 3, 10, 30, 100, 300, 1000 |
| Exponente geográfico `p` | 0, 0.5, 1, 1.5, 2 |

Se evalúan **8 × 7 × 7 × 5 = 1,960 configuraciones**, frente a las 192 del Checkpoint 10. Se mantienen las tres variables `k`, `alpha` y `p` conjuntamente con PCA; ninguno queda congelado durante la búsqueda. Para evitar vecindarios impracticables en los folds internos, `k` llega solo hasta 48.

El control sin selección sigue siendo **PCA 20 / k 24 / alfa 30 / p 1**. Es una comparación emparejada y no participa en la selección supervisada de los otros métodos.

## 3. CV de dos etapas dentro de cada split externo

Se preservan exactamente las **16 máscaras congeladas de Checkpoint 08**, con 97 parcelas pseudo-entrenamiento y 41 pseudo-objetivos en cada una. El generador de máscaras no se vuelve a ejecutar. El PCA transductivo se estima solo a partir de las características `X` de las 197 parcelas; ninguna selección de hiperparámetros usa las 41 etiquetas externas de su propio split.

Para cada una de las 16 máscaras:

1. **Cribado amplio:** se calculan RMSE internos mediante **3-fold CV** sobre las 97 parcelas, para las 1,960 combinaciones.
2. **Lista corta predefinida:** se retienen las **4 mejores configuraciones** de cada dimensión PCA (según el primer RMSE) y, además, se incluye la configuración histórica `k24/alpha30/p1` para cada dimensión. Las duplicadas se eliminan.
3. **Selección más estable:** los finalistas se vuelven a evaluar mediante **dos 5-fold KFold repetidos**, con semillas distintas y fijadas antes de evaluar. Se suman los errores cuadrados de las 194 predicciones de validación interna por candidato (97 por repetición), y se calcula un único RMSE de selección.
4. **Evaluación externa:** se selecciona, para cada dimensión PCA, la mejor terna `k/alpha/p`, y también un ganador conjunto entre todos los finalistas. Se reajustan con las 97 etiquetas y se predicen las **41 observaciones externas una sola vez**.
5. **Comparación emparejada:** se publican agregados de RMSE, R², peor split, distribución de hiperparámetros seleccionados y diferencias de MSE con bootstrap agrupado por parcela.

Los procedimientos externos predefinidos son **10**: un Local04D fijo, ocho versiones con dimensión fija y ajustes `k/alpha/p` internos, y una versión que elige simultáneamente PCA y los tres hiperparámetros. La **comparación principal** es el selector conjunto `PCA_joint_nested` frente a `PCA20_frozen`. No se promueve un candidato simplemente por ser el mínimo entre los ocho RMSE externos observados.

El diseño de dos etapas **no transforma el resultado en una confirmación independiente**. Las 138 etiquetas y las 16 máscaras externas ya se examinaron en Checkpoints 08 y 10. La CV repetida puede mejorar la estabilidad *dentro* de esos datos; no inventa nuevos datos agrícolas. Tampoco elimina automáticamente la correlación espacial o los errores en el rendimiento reportado.

## 4. Auditoría, privacidad y reanudación

- Se reutiliza el constructor de **PCA C4 del Checkpoint 10**, incluidas las verificaciones de **paridad de los 20 PCs originales** y, cuando existe el archivo local de 08, de las predicciones del Local04D congelado.
- La nueva representación de 10/30/60/120 componentes es un prefijo del **mismo PCA160** estandarizado; no se mezclan componentes procedentes de ajustes diferentes.
- La puntuación interna usa solo los `y` de las 97 parcelas de entrenamiento de cada fold externo. Una prueba sintética cambia intencionalmente los 41 `y` reservados y verifica que la tabla interna no cambie.
- Las máscaras, archivos fuente, código, cuadrícula y parámetros CV quedan congelados mediante un `run_spec.json` con SHA-256.
- Los CSV de cribado se guardan **por split y por dimensión PCA**. Si se interrumpe la ejecución, no es necesario recomputar los bloques terminados. Los archivos se escriben de modo atómico; los archivos en caché se validan antes de reutilizarse.
- Los pronósticos individuales, etiquetas de referencia y tablas del grid quedan **ignorados por Git** en `reports/checkpoint_11_wide_pca/`. No ejecutar `git add -f` para publicarlos.
- No cambia `reports/checkpoint_05/final_predictions.csv`, el modelo exportado, ni las 59 estimaciones del conjunto de destino.

## 5. Comandos de PowerShell

Desde `C:\Users\175199\GeoC` con conda activado y las fuentes C4 del Checkpoint 04A y las máscaras de Checkpoint 08 en su ubicación original:

```powershell
git switch main
git pull --ff-only origin main
conda activate geocebada

# Comprobación de características C4, PCA y control original
python tools/run_checkpoint_11_wide_pca.py --check

# Prueba de ejecución aislada, un solo split y grid reducido
python tools/run_checkpoint_11_wide_pca.py --smoke

# Evaluación completa con guardado incremental
python tools/run_checkpoint_11_wide_pca.py --run

# Sólo consolida los 16 splits terminados
python tools/run_checkpoint_11_wide_pca.py --summary
```

La ejecución utiliza CPU, no necesita GPU ni descargar más satélites. Si se detiene voluntariamente, se vuelve a ejecutar `--run` **sin cambiar el código, las fuentes o el directorio de salida**. Al modificar esos factores debe elegirse otro `--output` explícito para no mezclar experimentos.

## 6. Archivos resultantes

| Archivo local | Contenido |
|---|---|
| `confirm_XX__coarse_pcaN.csv` | El grid completo de 245 combinaciones de `k/alpha/p` y su RMSE interno para un número de componentes |
| `confirm_XX__rerank.csv` | Lista corta por dimensión re-evaluada en 2×5-fold |
| `confirm_XX__selection.json` | Parámetros ganadores por dimensión y ganador conjunto |
| `confirm_XX__paired.csv` | 41 predicciones externas de cada uno de los 10 modelos |
| `comparison_summary.csv` | RMSE, R², predicciones repetidas y parcelas distintas |
| `paired_cluster_bootstrap.csv` | `delta_mse` e IC percentil 95 % contra Local04D fijo |
| `selected_hyperparameters.csv` | Ganador conjunto en cada uno de los 16 splits |

**Estado correcto al escribir esta guía:** el código está implementado pero no se han ejecutado los resultados completos del Checkpoint 11 sobre los datos de la estación de trabajo. No afirmar una mejora de RMSE hasta ejecutar el experimento y revisar todas las métricas.

Referencia: [Checkpoint 10](CHECKPOINT_10_PCA_GRID.md) y [Metodología consolidada](METODOLOGIA_Y_RESULTADOS.md).
