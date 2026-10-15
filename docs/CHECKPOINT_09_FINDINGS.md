# Checkpoint 09 — Resultados y cierre (ruido de etiquetas)

**Estado:** CERRADO / NO PROMOCIÓN  
**Fecha de corrida comunicada:** 2026-10-12  
**Objetivo:** sensibilidad de Local04D ante posibles errores de captura de rendimientos; **no** demostrar qué registros son incorrectos.  
**Método conservado:** `LocalRidge_C4_all_deterministic_Geo_k24_a30_p1`.  
**Predicciones canónicas:** `reports/checkpoint_05/final_predictions.csv` (59 filas; sin modificación).

## 1. Qué se ejecutó y qué se verificó

El operador ejecutó localmente:

```powershell
python tools/run_checkpoint_09_robust.py --prepare
python tools/run_checkpoint_09_robust.py --smoke
python tools/run_checkpoint_09_robust.py --run
python tools/run_checkpoint_09_robust.py --summary
```

El programa informó:

- Contrato de **197 parcelas** (138 etiquetadas / 59 objetivos desconocidos): PASS.
- Paridad numérica del control C4/Local04D histórico: diferencia máxima **8.88 × 10⁻¹⁶ t/ha**.
- `--prepare`: 16 máscaras de confirmación nuevas congeladas con semilla **20261013** antes de puntuar desarrollo.
- `--smoke`: 1 split × 41 pseudo-objetivos × 5 modelos: PASS.
- `--run`: 16 splits de desarrollo target-matched × 41 pseudo-objetivos = **656 predicciones repetidas por modelo**, correspondientes a **125 parcelas únicas**.
- Las 59 etiquetas desconocidas no se utilizaron.

Se compararon cinco métodos emparejados, todos con C4/Geo, k=24 y Ridge α=30. El único cambio fue la función/peso robusto asignado a las **etiquetas del entrenamiento**, con dos variantes Huber IRLS y dos ponderaciones conservadoras basadas en errores LOO internos.

## 2. Tabla definitiva de desarrollo

| Modelo | RMSE pooled (t/ha) | RMSE medio split | Peor split | MAE pooled | RMSE centro | RMSE colas |
|---|---:|---:|---:|---:|---:|---:|
| **Local04D_frozen** | **0.495741** | **0.487845** | **0.694700** | 0.342501 | **0.332269** | **0.809414** |
| LOOReliability_c3p0 | 0.498899 | 0.490915 | 0.695345 | 0.341664 | 0.338472 | 0.809450 |
| Huber_c2p5 | 0.500934 | 0.492582 | 0.695730 | **0.341366** | 0.341101 | 0.811166 |
| LOOReliability_c2p0 | 0.502146 | 0.494001 | 0.696373 | 0.341989 | 0.339575 | 0.816105 |
| Huber_c1p5 | 0.506344 | 0.498057 | 0.697114 | 0.344905 | 0.347770 | 0.816107 |

Los números proceden de la salida de terminal publicada por el operador y se muestran con seis decimales. **No son cinco campañas distintas ni 656 parcelas independientes**.

Hallazgos:
1. El control original tiene el RMSE más bajo de todos los candidatos, tanto global como central y de colas.
2. El rival de menor RMSE, `LOOReliability_c3p0`, obtuvo 0.498899, **0.003158 t/ha peor** que Local04D.
3. Algunas variantes disminuyeron muy ligeramente el MAE; la más baja fue Huber 2.5 con 0.341366 frente al 0.342501 de Local04D. Esa reducción no compensa el empeoramiento en RMSE y no permite una promoción.
4. Ninguna técnica robusta probada permite concluir que un rendimiento extremo sea un error de captura.

## 3. Selección LOSO y decisión predeclarada

La salida `--summary` comunicó:

```text
LOSO_STABILITY_GATE = FAIL
development_best_challenger = LOOReliability_c3p0
baseline_rmse_pooled = 0.4957414287107349
loso_rmse_pooled = 0.4957414287107349
baseline_mae_pooled = 0.3425011444879952
loso_mae_pooled = 0.3425011444879952
baseline_tail_rmse = 0.8094140980228774
loso_tail_rmse = 0.8094140980228774
candidate_to_freeze_for_new_mask_confirmation = null
can_confirm = false
official_59_predictions_changed = false
```

Las condiciones se definieron antes de la corrida: la regla seleccionada LOSO debía mejorar al control al menos 0.003 t/ha en RMSE pooled, sin empeorar MAE más de 0.002 ni RMSE de colas más de 0.005 t/ha. No las cumplió.

**Decisión final: cerrar Checkpoint 09, no utilizar el banco 09 de confirmación y no reemplazar Local04D.** Las máscaras de confirmación, aunque congeladas antes del experimento, no se puntuaron.

LOSO sólo prueba sensibilidad a la elección de splits históricos; las máscaras comparten el universo de 138 etiquetas. Ninguna métrica identifica el verdadero rendimiento agrícola ni prueba errores en registros.

## 4. Procedencia de resultados y privacidad

Archivos públicos, transcritos directamente de la salida que compartió el operador:

- [Resumen agregado de desarrollo](../reports/checkpoint_09/shared/development_summary.csv)
- [Decisión completa](../reports/checkpoint_09/shared/decision.json)
- [Nota de publicación y limitaciones](https://github.com/heri-espino/GeoC/blob/main/reports/checkpoint_09/shared/README.md)

El CSV público conserva los números **redondeados** de la terminal; para una auditoría con precisión original, conservar el `reports/checkpoint_09/development_summary.csv` local.

Se mantienen fuera de Git la tabla de predicciones individuales, los errores por ID, los pesos de confiabilidad, la lista `human_label_review_candidates.csv` y las máscaras de confirmación. Esta última lista es sólo una cola de revisión humana, **no** evidencia de rendimientos falsos.

No se modifican documentos oficiales, datos fuente, etiquetas, código histórico, las 59 estimaciones del Checkpoint 05 ni la presentación principal. El Checkpoint 09 concluye con un resultado negativo reproducible en la misma computadora, condicionado a los rendimientos **reportados**.

**Sin acciones pendientes de modelado para Checkpoint 09.**
