# Checkpoint 12 — Representaciones alternativas y Elastic Net local

**Estado:** implementado; el experimento completo aún no se ha ejecutado en la estación de trabajo.  
**Alcance:** comparar alternativas ligeras a PCA y a Ridge sin ampliar aún más la rejilla de 1 960 configuraciones del [Checkpoint 11](CHECKPOINT_11_WIDE_PCA.md).

## Motivación

Local04D fijo (PCA20, 24 vecinos, \(\alpha=30\), potencia geográfica \(p=1\)) alcanzó **RMSE 0.481936 t/ha** y **R² 0.685388** en las 16 máscaras del Checkpoint 08 reutilizadas en Checkpoint 10. En el Checkpoint 10, todos los procedimientos de búsqueda anidada empeoraron ese RMSE.

Con 2 082 características de origen, PCA conserva direcciones de mayor varianza de \(X\), no las más predictivas de \(y\). Hay dos preguntas diferentes:

1. ¿Una **representación distinta de PCA** ofrece componentes más adecuados para predecir rendimientos?
2. ¿Un **estimador distinto de Ridge**, como Elastic Net, aprovecha mejor las mismas componentes?

Elastic Net no reduce dimensiones de \(X\) de forma geométrica: penaliza la regresión y puede fijar coeficientes en cero mediante regularización L1. ICA, PLS y proyección aleatoria sí construyen nuevas coordenadas.

## Modelos y costes

Se usa el mismo criterio externo: **16 splits × 97 entrenamiento / 41 pseudo-objetivos** del Checkpoint 08. Cada configuración se escoge por RMSE en **KFold interno de tres particiones** sobre los 97 observados. Se comparan seis métodos, incluida la referencia.

| Nombre del método | Transformación | Dimensiones | Regresor y parámetros interiores |
|---|---|---|---|
| **PCA20_frozen** | C4 PCA original | 20 | Ridge fijo, \(\alpha=30\) |
| **ICA_Ridge** | PCA40 → FastICA | 8, 16, 32 | Ridge; \(\alpha = 10,30,100\) |
| **RP_Ridge** | Proyección aleatoria gaussiana desde 2 082 variables | 16, 32, 64 | Ridge; \(\alpha = 10,30,100\) |
| **PLS_Ridge** | PLS supervisado | 4, 8, 16 | Ridge; \(\alpha = 10,30,100\) |
| **PCA_ElasticNet** | PCA C4 | 20, 40 | Elastic Net; \(\alpha=0.03,0.3\); ratio L1=0.2,0.8 |
| **PLS_ElasticNet** | PLS supervisado | 4, 8, 16 | Elastic Net; \(\alpha=0.03,0.3\); ratio L1=0.2,0.8 |

**Aislamiento de cambios:** para las cinco ramas alternativas, `k=24` y `p=1` se mantienen fijos, al igual que la regla de vecindario geográfico de Local04D. La selección interior se centra en la representación, dimensión y penalización. No se mezclan estas familias con la rejilla amplia de Checkpoint 11.

La rama **ICA** es deliberadamente una *PCA40 seguida de ICA* para reducir coste computacional: no es FastICA directa sobre las 2 082 variables. ICA se calcula con los \(X\) de las 197 parcelas, sin rendimientos; se anotan las advertencias de no convergencia.

**PLS es supervisado.** Debe ajustarse con \((X,y)\) de las parcelas de **entrenamiento del fold interior**; los \(y\) reservados no intervienen. Tras escoger la configuración interior, PLS se vuelve a ajustar con los 97 \(y\) disponibles antes de pronosticar los 41 del fold externo. PCA, ICA y RP usan exclusivamente \(X\), así que el diseño transductivo autoriza su construcción sobre las 197 parcelas. Las 59 etiquetas ocultas no se inspeccionan.

La regresión Elastic Net se ajusta **de nuevo para cada una de las 24 vecinas** de cada parcela consultada (igual estructura local de Local04D), mediante los pesos geográficos originales. Los valores de \(\alpha\) no son comparables directamente con los de Ridge porque los estimadores usan convenciones distintas para normalizar el término de pérdida y `sample_weight`.

## Evidencia y límites

- **Objetivo primario:** RMSE pooled emparejado, y \(R^2\) pooled sobre los mismos 656 pares \((split,parcela)\) de cada modelo.
- Bootstrap de diferencias de MSE con remuestreo del **ID único de parcela**, manteniendo juntas las apariciones repetidas.
- Se informan también RMSE promedio y peor split, configuración elegida y advertencias de convergencia ICA.
- Los datos fuente siguen siendo **138 rendimientos reportados**, no mediciones de verdad física garantizada. Tampoco los 656 errores son independientes ni se han observado los rendimientos reales de las 59 parcelas destino.
- La validación externa de 08 ha sido inspeccionada en 08, 10 y ahora 11/12; por tanto, incluso si una alternativa ganara aquí, el hallazgo sería **exploratorio**, no una confirmación independiente. No sustituir las 59 estimaciones sin una evaluación adicional defendible.

El archivo del experimento contiene únicamente modelos locales ligeros de scikit-learn; no necesita GPU, Prithvi, HLS nueva ni entrenamiento profundo. Es independiente del experimento amplio de 11.

## Ejecución en PowerShell

```powershell
cd C:\Users\175199\GeoC
git switch main
git pull --ff-only origin main
conda activate geocebada

python tools/run_checkpoint_12_representations.py --check
python tools/run_checkpoint_12_representations.py --smoke
python tools/run_checkpoint_12_representations.py --run
python tools/run_checkpoint_12_representations.py --summary
```

El directorio local `reports/checkpoint_12_representations/` conserva los estudios internos por split/familia, pronósticos individuales, selección y tablas agregadas. **Está ignorado por Git para proteger los rendimientos reportados**; no subirlo íntegro.

La corrida permite reanudar tras una interrupción: el CSV interior de cada familia se guarda por split, las predicciones y selecciones son atómicas y `run_spec.json` exige coincidencia de huellas de entrada y código. Una modificación requiere usar otro directorio `--output`.

## Resultados esperados

La salida imprimirá `comparison_summary.csv` con RMSE/R² y `paired_cluster_bootstrap.csv` con los intervalos para las cinco alternativas. Asimismo guardará `selected_hyperparameters.csv` y los archivos por split.

**No existe aún un RMSE de Checkpoint 12 sobre los datos reales.** Ninguno de estos procedimientos se promoverá automáticamente y el archivo `reports/checkpoint_05/final_predictions.csv` permanece inalterado.

Referencia: [Metodología y resultados 02–08](METODOLOGIA_Y_RESULTADOS.md), [Checkpoint 10](CHECKPOINT_10_PCA_GRID.md), [Checkpoint 11](CHECKPOINT_11_WIDE_PCA.md).
