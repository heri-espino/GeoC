# Checkpoint 14 — CV completo para 1 000 configuraciones próximas a Local04D

**Estado:** implementación en el repositorio; falta ejecutar los 16 splits con datos reales.  
**Objetivo:** corregir el filtro de 12 parcelas del Checkpoint 13 y centrarse en las familias que mejor se comportaron en los experimentos 04D, 10 y 13.

## Hallazgos que motivan el cambio

| Experimento | Procedimiento | RMSE pooled (t/ha) |
|---|---|---:|
| Checkpoint 08/10/13 | Local04D fijo: PCA20, k24, alfa30, p1 | **0.481936** |
| Checkpoint 10 | PCA80, parámetros anidados | 0.492532 |
| Checkpoint 13 | Selección conjunta de 100 000 candidatos por filtro de 12 + CV de finalistas | 0.522300 |
| Checkpoint 13 | PLS + Ridge por el mismo selector | 0.539687 |
| Checkpoint 13 | PCA + Ridge por el mismo selector | 0.554651 |

Checkpoint 13 analizó 100 000 modelos **en solo 12 pseudo-objetivos internos** y después avanzó 12 finalistas por familia a 2×4-fold CV. La auditoría de Local04D sobre esas mismas 12 parcelas confirmó que **el incumbente habría quedado fuera del top-12 PCA+Ridge en 16/16 splits** aunque ganó el RMSE exterior, lo cual es evidencia fuerte de una selección demasiado sensible a esos 12 ejemplos.

No debemos inferir de ello que *todos* los 100 000 candidatos fallaban. Tampoco usar resultados externos reiteradamente observados como confirmación independiente: el mismo banco de 16 splits se examinó antes en 08, 10, 11, 12 y 13.

## Protocolo congelado, sin filtrar por solo 12 parcelas

Se generan **1 000 propuestas predefinidas**, con semilla fija, incluyendo configuraciones históricas cercanas a Local04D. Las cantidades por familia son:

| Familia | Candidatos | Motivo |
|---|---:|---|
| **PCA + Ridge local** | **700** | Mejor método histórico; contiene obligatoriamente el Local04D original |
| PLS + Ridge local | 200 | Segundo mejor procedimiento ajustado en 13 |
| PCA + Elastic Net local | 60 | Regularización dispersa alternativa |
| PLS + Elastic Net local | 40 | Combinación supervisada exploratoria |
| **Total** | **1 000** | No usa ICA ni proyección aleatoria |

El mayor presupuesto se asigna a PCA + Ridge, ya que fue la opción más sólida. Dentro de esta familia se fijan anclas históricas, entre ellas **PCA20, k24, alfa30, p1**, además de PCA40 y PCA80, y pequeñas variaciones de vecinos, penalización y distancia. Las otras muestras se distribuyen aleatoriamente alrededor de valores razonables: PCA 5–192 componentes (más peso alrededor de 20/40/80), PLS 4–32, k 8–48, Ridge alfa aproximadamente 3–300, y p aproximadamente 0–2.5. Elastic Net prueba ratios L1 continuos y penalizaciones log-uniformes dentro de un intervalo predefinido.

**Importante:** el ancla de Local04D es un miembro explícito de la lista de 1 000 candidatos, no solo una fila de referencia externa; llega obligatoriamente a la segunda etapa aunque su RMSE de la primera sea malo. La primera etapa **no puede descartarla definitivamente**.

Para cada una de las 16 máscaras congeladas Checkpoint 08, con 97 parcelas de entrenamiento y 41 de pseudo-validación externa:

1. **CV completa de los 1 000 candidatos:** tres folds internos que cubren exactamente las **97 etiquetas** del split exterior. Cada candidato produce 97 predicciones de validación interna, no solo 12.
2. **Lista corta balanceada:** 20 PCA+Ridge, 12 PLS+Ridge, 8 PCA+Elastic Net, 6 PLS+Elastic Net, más Local04D histórico si no figuraba entre los seleccionados; como máximo **47 finalistas**.
3. **CV más robusta:** reevaluación de todos esos finalistas con **2 × 4-fold**, dos semillas y **194 predicciones internas** por candidato. El histórico Local04D también recibe este RMSE interior y su rango.
4. **Evaluación externa emparejada:** los cuatro ganadores de familia y un selector conjunto (capaz de elegir el incumbente) se ajustan con las 97 etiquetas y predicen las mismas 41 parcelas que el modelo original fijo.
5. **Informe:** RMSE pooled, R² pooled, RMSE promedio/peor split, CI percentil 95 % sobre diferencias de MSE con bootstrap agrupado por ID único de parcela. Se registra también el número de splits en que la selección conjunta elige la configuración original.

**La comparación principal preespecificada** es el selector conjunto `Focused_joint_nested` vs `PCA20_frozen`. No se seleccionará a posteriori como campeón simplemente la fila con el mínimo RMSE entre todas las familias.

Las representaciones PCA usan solo la información `X` de las 197 parcelas (protocolo transductivo), mientras que PLS se ajusta **dentro de cada fold interior**, con las etiquetas del entrenamiento de ese fold exclusivamente. Las 59 etiquetas destino permanecen estrictamente ocultas.

## Ejecución

En PowerShell, desde la VM con conda:

```powershell
cd C:\Users\175199\GeoC
conda activate geocebada
git switch main
git pull --ff-only origin main

python tools/run_checkpoint_14_focused_cv.py --check
python tools/run_checkpoint_14_focused_cv.py --smoke

# Experimento completo: 16 splits × 1 000 modelos × 3-fold en la primera etapa.
python tools/run_checkpoint_14_focused_cv.py --run

# Mostrar únicamente los resultados completos
python tools/run_checkpoint_14_focused_cv.py --summary
```

No se necesita GPU ni descargar imágenes adicionales. La CV de 1 000 candidatos puede ser costosa porque cada candidato entrena un estimador **local por parcela consultada**, pero los bloques terminados se guardan y se pueden reanudar ejecutando exactamente el mismo comando.

Se utiliza un **directorio nuevo**, `reports/checkpoint_14_focused_1k/`, ignorado por Git porque los archivos contienen rendimientos individuales. No borramos `reports/checkpoint_13_100k/`; tampoco modificamos las 59 predicciones oficiales.

Cada corrida genera `run_spec.json` con SHA-256 de archivos de entrada y del código, `candidates.csv`, `confirm_XX__cv3__YYYY.csv` para las CV completas por bloques, `confirm_XX__rerank.csv`, `confirm_XX__paired.csv`, `confirm_XX__selection.json` y los CSV agregados `comparison_summary.csv`, `paired_cluster_bootstrap.csv`, `selected_hyperparameters.csv`.

## Límites

La evaluación usa **el mismo banco ya explorado de 16 × 41 pseudo-targets** de Checkpoint 08. Repetir CV interior reduce algunas fuentes de selección ruidosa, pero **no equivale a disponer de nuevas 138 etiquetas ni a una confirmación verdaderamente independiente**. Una mejora aparente debe interpretarse como exploratoria y no justificaría sustituir automáticamente las 59 estimaciones.

La hipótesis verificable es que preservar Local04D y medir todos los candidatos sobre más observaciones internas mejora la *calidad de selección*, no que forzosamente un modelo más sofisticado reduzca el RMSE.

Documentación asociada: [Checkpoint 13](CHECKPOINT_13_RANDOMIZED.md) y [Checkpoint 10](CHECKPOINT_10_PCA_GRID.md).
