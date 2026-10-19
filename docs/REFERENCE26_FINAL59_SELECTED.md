# Cierre del modelo: 138 LOOCV anidados y 59 predicciones corregidas

**Fecha:** 2026-10-19. **Ámbito:** solamente competencia FIRA
AgroCebada, 197 parcelas X observables, 138 rendimientos reales y
59 parcelas con rendimiento desconocido.

**Estado:** el usuario concluyó y compartió los resultados de los 138
folds del selector anidado; el código de predicción final está
implementado, pero todavía debe ejecutarse en su VM con el bundle
privado. No se han generado ni compartido aquí las 59 cifras finales.

## Resultado medido: modelos corregidos

Se conserva el modelo nativo Reference26 con imputación aritmética
de X calculada solamente en el conjunto de entrenamiento.

| Modelo | RMSE LOOCV (t/ha) | MAE (t/ha) | Sesgo (t/ha) | R² CV |
|---|---:|---:|---:|---:|
| Reference26 mean original | 0.468138 | 0.333786 | 0.001466 | 0.695292 |
| **Selector anidado top-five** | **0.462892** | **0.331956** | **-0.005062** | **0.702083** |
| Robust26 k3 blend .25 supported FIJO | **0.462418** | 0.332019 | -0.004103 | 0.702693 |
| PCA8 k3 blend .25 supported fijo | 0.464956 | 0.334091 | -0.000414 | 0.699421 |
| PCA8 k8 blend .25 supported fijo | 0.465506 | 0.334439 | -0.004101 | 0.698709 |
| Robust26 k3 blend .10 supported fijo | 0.465538 | 0.332972 | -0.000762 | 0.698668 |
| Robust26 k3 blend .10 none fijo | 0.466518 | 0.335959 | -0.004757 | 0.697397 |

La ganancia observada del selector anidado contra Reference26 sin
corregir es 0.005246 t/ha (≈1.12% relativo), y su MAE
desciende 0.001830 t/ha.

En los **138 folds externos, cada uno con 137 etiquetas reales de
train y un test real**, el selector se entrenó sobre **4 folds
internos exclusivos de las 137 etiquetas disponibles**. Hubo 552
ajustes adicionales nativos de Reference26. Elecciones:

| Opción | Folds seleccionados |
|---|---:|
| Robust26, 3 vecinos, blend .25, supported | **128** |
| PCA8, 3 vecinos, blend .25, supported | 5 |
| Reference26 sin corrección | 5 |
| Las otras tres variantes | 0 |

Las 5 variantes fijas ya tienen LOOCV auténtico de 138 casos:
el anidamiento valida adicionalmente una **regla de selección**
entre ellas. Es importante que **las cinco se seleccionaron
post-hoc** tras examinar repetidamente estos mismos 138 yields;
la política anidada elimina el uso directo del fold externo para
su elección pero **no constituye confirmación independiente
de todo el proceso de investigación**. No hay una nueva muestra de
138 casos independiente. No se conoce el error verdadero de los 59.

## Regla final para las 59 parcelas sin etiquetas

La estrategia operativa coherente con el experimento anidado
NO consiste en seleccionar siempre la configuración fija con RMSE
0.462418, sino en **repetir la elección con los 138 datos reales
disponibles**, sin las 59 etiquetas ocultas:

1. Requiere el artefacto **original** `reference26_mean_train138.joblib`,
   producido por `tools/run_reference26_compact14_final.py --train-final`
   con imputación por **media**, y su CSV privado original de
   59 pronósticos, hash SHA y orden de ID verificados.
2. Crea un **nuevo 4-fold** sobre los 138 rendimientos reales.
   Por fold interno se ajusta nativamente Reference26 sobre solo
   103/104 etiquetas de entrenamiento, ocultando explícitamente
   las 34/35 etiquetas de validación. Esto suma solamente **4 ajustes
   nativos adicionales** de Reference26 para la predicción final;
   no son los 552 ajustes del estudio anidado ya ejecutado.
3. En cada fold interno calcula las cinco correcciones a partir de
   los rendimientos de las parcelas de **ese train interno**,
   con las mismas funciones, 26 features/robust PCA, k, coeficiente
   de mezcla, `supported` y `max_delta=0.5` que durante LOOCV.
4. Elige una sola regla entre las cinco y Reference26 sin corrección:
   requiere que el MSE interno baje al menos 1% relativo para
   introducir una corrección. De lo contrario elige Reference26
   original. Las 59 etiquetas ocultas **no participan** en
   esa decisión.
5. Aplica la regla elegida a los 59 valores originales del
   Reference26 nativo entrenado con **los 138 rendimientos reales**.
   Los vecinos de cada parcela desconocida se buscan entre
   las 138 parcelas reales; no hay interpolaciones de y ocultos,
   datos sintéticos ni cambios en la implementación original.
6. Genera una comparación privada que conserva la predicción
   original, las cinco correcciones candidatas, la seleccionada
   y la diferencia; también produce un CSV de dos columnas como
   candidato de entrega y un manifiesto agregado reproducible.

El modelo elegido por validación interna **puede diferir**
del robust26 que ganó 128/138 folds históricos. No se debe inventar
la elección final hasta completar los cuatro ajustes en la VM.

## Comandos en la VM Windows

```powershell
cd C:\Users\175199\GeoC
conda activate geocebada
git fetch origin
git switch experiment/reference26-top5-final59
git pull --ff-only

python -m py_compile tools/run_reference26_top5_final59.py
python -m pytest tests/test_reference26_top5_final59.py -q

python tools/run_reference26_top5_final59.py --check
python tools/run_reference26_top5_final59.py --smoke
python tools/run_reference26_top5_final59.py --run
python tools/run_reference26_top5_final59.py --summary
```

Si `--check` indica que no existen el joblib original entrenado
con 138 y sus 59 predicciones verificadas, ejecutar
`python tools/run_reference26_compact14_final.py --train-final`
**solo para crear los artefactos finales originales faltantes**;
el script nativo rechaza resultados anteriores incompatibles.

## Archivos generados (siempre privados)

Ruta: `reports/reference26_top5_final59/`.

- `inner_01.csv`, ..., `inner_04.csv`: predicciones de las
  138 etiquetas originales en 4 folds internos, con yields observados,
  excluidos de Git y reciclables de forma segura.
- `run_spec.json`: huellas del código, datos, modelo original y
  configuración, para impedir reutilizar resultados incompatibles.
- `predictions59_final_nested_private.csv`: las 59 parcelas con
  Reference26 original, cinco candidatos corregidos, predicción
  final seleccionada, diferencia y nombre del método.
- **`submission59_candidate_private.csv`**: tabla de 59 IDs y un
  único campo `RENDIMIENTO_T_HA` con la predicción final
  seleccionada. **Solo formato candidato; cotejar las bases
  oficiales antes de enviar a FIRA.**
- `final_selection_manifest.json`: método elegido por cuatro
  folds internos, MSE/RMSE de las seis opciones, conteos del
  estudio anidado previo, hashes verificables y estadísticas
  agregadas de las 59 predicciones. No contiene observaciones
  por parcela.

Ninguno de estos ficheros debe añadirse al GitHub público:
`.gitignore` cubre toda la carpeta. El original
`reports/reference26_compact14_fresh138/final_artifacts/` no
se sobrescribe. El script rechaza la escritura de un segundo
resultado final que ya exista, conservando la procedencia.
No se fusiona ni despliega el PR automáticamente.

## Cómo interpretar la cifra final

La comparación global observada en 138 parcelas es RMSE 0.462892
para la política de selección. **No** se puede reportar ese RMSE
como error medido de las 59 parcelas: sus rendimientos
desconocidos aún no son verificables. La mejora medida, pequeña
y post-hoc, justifica un candidato corregido distinto de
Reference26 original, pero no garantiza que sea mejor para las 59.

La predicción de los 59 casos es un ajuste con 138 etiquetas reales,
más una corrección determinista basada en vecinos reales.
No debe decirse que se entrenó Reference26 sobre 197 rendimientos
observados ni que se crearon 59 rendimientos reales nuevos.

## 2026-10-19 — Corregido: choque de huella del LOOCV anidado

El primer intento del usuario en la rama de predicción final
falló en `--check`, `--smoke` y `--run` antes de hacer ningún
ajuste o generar una predicción. Mensaje:
`Nested run fingerprint does not match current original 138 screen,
candidate family or training source.` La causa técnica está en que
el exportador exigía igualdad **byte por byte** del manifiesto
generado previamente por `run_reference26_top5_nested.py`, lo cual
incluye el SHA del código Python. Es posible que cambios posteriores
del script (formato/mantenimiento) alteren ese SHA aunque los folds
guardados sean exactamente los mismos.

La revisión del exportador permite divergencia de SHA **solo** en
`source_sha` (script anidado) y `screen_source_sha` (script de
correcciones). Mantiene coincidencia **estricta** de huellas de
datos y modelos (`screen_private_sha`, `native_run_spec_sha`,
`peer_source_sha`, `bundle_sha` y `frozen_weights_sha`),
los cinco candidatos, límites de corrección y umbral de selección.
Después verifica individualmente los 138 folds, identificadores,
etiquetas conocidas y predicciones de los cinco modelos contra el
screen original antes de aceptar los resultados.

Si persiste alguna diferencia importante, el nuevo error identifica
los campos exactos que difieren. Nunca eliminar la verificación de
fuentes para evitar el error. El fallo de `--summary` simplemente
indica que `--run` no llegó a producir el manifiesto final.

```powershell
git pull --ff-only
python -m pytest tests/test_reference26_top5_final59.py -q
python tools/run_reference26_top5_final59.py --check
python tools/run_reference26_top5_final59.py --smoke
python tools/run_reference26_top5_final59.py --run
python tools/run_reference26_top5_final59.py --summary
```

Si la nueva salida menciona campos incompatibles, preservar los
datos privados y revisar esos campos; **no borrar los 138 resultados**.
Los nuevos tests rechazan explícitamente alteraciones de fuentes,
candidatos y umbrales. Aún no hay 59 predicciones ejecutadas
en el entorno privado de la VM.
