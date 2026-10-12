# Checkpoint 09 — Robustez de Local04D frente a rendimientos posiblemente ruidosos

**Fecha de diseño:** 12 de octubre de 2026  
**Estado:** CERRADO / NO PROMOCIÓN; corrida de desarrollo completada 2026-10-12. LOSO_STABILITY_GATE = FAIL. Consulte [resultados y decisión](CHECKPOINT_09_FINDINGS.md).  
**Modelo vigente:** `Local04D = LocalRidge_C4_all_deterministic_Geo_k24_a30_p1`.  
**Predicciones oficiales congeladas:** `reports/checkpoint_05/final_predictions.csv`.

## 1. Por qué iniciamos este experimento

En el Checkpoint 06 encontramos que aproximadamente el 62.2 % de la suma
de errores cuadrados se concentraba en el 10 % de parcelas con mayor error
de Local04D. Se observó regresión hacia la media (rendimientos muy bajos
sobreestimados y muy altos subestimados). Los especialistas 06E/06F mejoraron
ligeramente el centro en la tabla de desarrollo, pero **fallaron LOSO** y nunca
se promovieron. Véase `docs/CHECKPOINT_06_FINDINGS.md`.

La nueva pregunta no es «¿qué nuevas colas debo forzar?», sino:

> ¿La regresión local original es demasiado sensible a unas pocas etiquetas
> registradas con error? ¿Puede una pérdida robusta reducir esa sensibilidad
> sin sacrificar rendimientos extremos auténticos?

El registro observado se modela conceptualmente como

\[
y_i^{\mathrm{reportado}}=y_i^{\mathrm{real}}+\varepsilon_i.
\]

No sabemos si \(\varepsilon_i\) es grande para alguna parcela. Un residuo
importante **no demuestra** un error de captura, ni autoriza a modificar
o borrar una etiqueta. Variaciones de suelo, lluvia, fecha de siembra,
manejo y cosecha también podrían producir rendimientos extremos reales.

**Limitación de identificación:** sin una fuente independiente de verdad,
no podemos distinguir rigurosamente un rendimiento real sorprendente de
una medición incorrecta. Este experimento mide estabilidad predictiva
*respecto de etiquetas reportadas*, no exactitud con una verdad oculta.

## 2. Candidatos congelados antes de ejecutar

Se reutiliza **exactamente** el vector PCA C4 original, la normalización
de distancias geográficas de 04B, los \(k=24\) vecinos etiquetados, la
ponderación \(w_i=(d_i+10^{-6})^{-1}\) y Ridge con \(\alpha=30\).

| Método | Única modificación al Local04D |
|---|---|
| `Local04D_frozen` | Ninguna; referencia exacta |
| `Huber_c1p5` | IRLS: reduce influencia de residuales locales grandes; umbral 1.5 × escala MAD |
| `Huber_c2p5` | Como arriba, umbral 2.5 × escala MAD |
| `LOOReliability_c2p0` | Reduce peso de etiquetas con desacuerdo en predicción leave-one-out **del entrenamiento**; umbral 2 × escala MAD |
| `LOOReliability_c3p0` | Como arriba, umbral 3 × escala MAD |

Las variantes Huber mantienen la misma **penalización cuadrática Ridge**:
se aplica un algoritmo IRLS con factores
\(\min(1,c\,s/|r_i|)\) multiplicando los pesos geográficos; la
escala \(s\) se calcula con la MAD robusta de residuos del ajuste inicial
y tiene un piso técnico de 0.20 t/ha. Se permiten hasta 10 iteraciones.
No se usa un `HuberRegressor` con otra convención de regularización.

El esquema LOO calcula para cada etiqueta de entrenamiento \(j\)
una predicción Local04D **sin utilizar \(y_j\)**, y usa el residual
para ponderar esa etiqueta posteriormente. Todas estas operaciones se
realizan nuevamente **dentro de los 97 registros visibles de cada split**.
Los pesos se acotan en [0.35, 1.00] para evitar descarte total de un
rendimiento auténtico pero atípico.

La escala MAD y sus factores son decisiones de sensibilidad;
**no estimaciones calibradas de la probabilidad de mala captura**.

## 3. Validación y criterios de decisión

### 3.1 Desarrollo: Checkpoint 04B/07 congelado

- 16 particiones `target_matched` históricas, cada una con 97 etiquetas
  disponibles y 41 pseudo-objetivos ocultados.
- Todos los métodos predicen **los mismos 41 IDs por split**.
- Se exigen paridad numérica con Local04D histórico antes de analizar
  candidatos, y paridad de RMSE pooled 0.49574142871073495.
- Las etiquetas externas sirven únicamente para calcular las métricas
  de evaluación de la partición. Nunca el ajuste, pesos LOO, escala de Huber.
- Se informan RMSE pooled, RMSE medio y peor split, MAE, RMSE central y
  RMSE de colas; las regiones central y de colas usan q15/q85 de los
  **97 rendimientos de entrenamiento**.
- LOSO selecciona método usando los otros 15 splits y evalúa el split
  excluido. *Como una parcela puede participar en muchos splits, esto
  es un chequeo de estabilidad y **no** validación estadísticamente
  independiente.* No se usa para inferir ausencia de sesgo en el mundo real.

### 3.2 Reglas predeclaradas

El examen de estabilidad sólo pasa si la regla seleccionada LOSO:

1. supera a Local04D en RMSE pooled por al menos **0.003 t/ha**;
2. no empeora MAE por más de **0.002 t/ha**;
3. no empeora RMSE de colas por más de **0.005 t/ha**;
4. y el mejor rival individual en desarrollo mejora al control.

Si no pasa, se conserva Local04D y **no se gasta** la confirmación.
Si pasa, se congela **un solo candidato**, el mejor del desarrollo,
antes de abrir un banco de confirmación.

### 3.3 Confirmación 09

Se congelan al inicio 16 máscaras `confirm_` a partir **sólo de X**,
semilla 20261013, utilizando el protocolo municipal/estatal del
Checkpoint 08 con esa nueva semilla. Se conservan los mismos 138
rendimientos históricos, por lo que **no constituyen nueva evidencia
independiente de medición**. Su utilidad es comprobar sensibilidad
a otros pseudo-objetivos no seleccionados durante el desarrollo 09.

En confirmación se ejecutan sólo `Local04D_frozen` y **un rival
precongelado**. Se calcula diferencia emparejada de MSE con un bootstrap
de 1000 remuestreos agrupado por ID de parcela, además de MAE y RMSE
de colas. Un intervalo superior estrictamente negativo, y MAE/colas
dentro de tolerancias, puede justificar una **revisión humana**;
nunca promoverá predicciones automáticamente.

Los intervalos bootstrap no corrigen automáticamente dependencia espacial,
solapamiento del entrenamiento entre splits ni errores en etiquetas.

## 4. Auditoría de calidad sin falsas acusaciones

Después de desarrollo se produce una lista local
`reports/checkpoint_09/human_label_review_candidates.csv`,
con los 138 registros de entrenamiento y su desacuerdo LOO.
Sus columnas incluyen ID, rendimiento reportado, entidad/municipio,
residuo, magnitud de desacuerdo y campos vacíos para verificación.

La lista **no** dice que las primeras posiciones están mal: sólo
prioriza una posible revisión con recibos de compra, tickets de báscula,
rendimiento oficial del productor, superficie cosechada, unidades,
fecha/ciclo, GPS, o fuente que sea independiente del modelo.

Si se detecta un error respaldado por otra fuente, conservar
**ambos** valores, metadatos de procedencia y motivo de corrección,
en una tabla nueva y versionada bajo política de privacidad; no
editar el archivo oficial original. Una evaluación con etiquetas
corregidas requeriría nuevos protocolos previamente establecidos,
con las revisiones ocultas a quien puntúe el modelo.

## 5. Ejecución local en PowerShell

Desde la raíz del repositorio, después de `git pull --ff-only`:

```powershell
conda activate geocebada
git lfs pull
python tools/run_checkpoint_09_robust.py --preflight
python tools/run_checkpoint_09_robust.py --prepare
python tools/run_checkpoint_09_robust.py --smoke
python tools/run_checkpoint_09_robust.py --run
python tools/run_checkpoint_09_robust.py --summary
```

**`--prepare` debe ejecutarse antes de `--run`** para congelar
las máscaras de confirmación sin observar los resultados nuevos.
El código rechaza reemplazar salidas de resultados ya calculadas o
continuar una corrida cuyos datos/código/configuración difieran de las
huellas SHA-256 archivadas.

Sólo si en `decision.json` aparece `LOSO_STABILITY_GATE=PASS`:

```powershell
python tools/run_checkpoint_09_robust.py --confirm
python tools/run_checkpoint_09_robust.py --summary
```

La comprobación inicial usa fuentes pequeñas de 04A/04B/06 y las
tablas oficiales; **no requiere HLS/SMAP, Prithvi ni GPU**. La operación
es CPU/SciKit-Learn local. Las salidas `reports/checkpoint_09/` contienen
etiquetas reportadas y se mantienen fuera de Git.

## 6. Límites y entregables

Este experimento no altera los checkpoints anteriores, no
recalcula las 59 predicciones finales, no limpia/elimina observaciones,
no usa ni intenta revelar las 59 etiquetas desconocidas y no escribe
en `reports/checkpoint_05/final_predictions.csv`.

El desarrollo se ejecutó con los rendimientos disponibles: **Local04D sigue ganando**
(RMSE 0.495741 frente a 0.498899 del mejor candidato robusto) y el gate LOSO
falló. El checkpoint queda cerrado, sin ejecutar confirmación ni modificar
las 59 predicciones. Detalles: [CHECKPOINT_09_FINDINGS.md](CHECKPOINT_09_FINDINGS.md).
Estos resultados no demuestran que las observaciones de cola estén bien o mal capturadas.
