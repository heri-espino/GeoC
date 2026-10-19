# Experimento pendiente — Ensamble Reference26 + Local04D

**Estado:** PLANIFICADO; **no implementado ni evaluado**.
**Fecha de planificación:** 2026-10-19.
**Ámbito:** competencia AgroCebada (138 parcelas etiquetadas y 59 objetivos ocultos).
Este documento define el siguiente experimento **después** de cerrar la búsqueda
de Reference26 con imputación por media. No promueve ningún modelo ni modifica
las predicciones para las 59 parcelas.

## Objetivo

Determinar si el modelo Local04D aporta errores complementarios a Reference26,
de modo que una mezcla mejore el rendimiento predictivo frente al mejor
Reference26 por sí solo. Se busca una mejora emparejada y suficientemente
estable, no ganar unas milésimas de RMSE tras escoger a posteriori muchos pesos.

Reference26 ya combina dos predictores (CT3 climático y un experto Ridge
local de siete variables) mediante una mezcla geológica; después aplica su
corrección residual R3. Local04D es otro predictor distinto: PCA20 sobre la
representación C4 y Ridge geográfico ponderado, k=24, alpha=30, p=1.
**Primer experimento:** mezclar el resultado FINAL de Reference26 con la
predicción completa de Local04D, no sustituir el modelo CT3 ni contar R3 como
un predictor independiente.

La mezcla convexa simple será

\[
\widehat y_{\lambda,i}
  =(1-\lambda)\widehat y_{\mathrm{Ref26},i}
    +\lambda\widehat y_{\mathrm{Local04D},i},
\qquad 0\leq\lambda\leq 1.
\]

- `lambda=0`: Reference26 completo (control obligatorio).
- `lambda=1`: Local04D completo (segundo control).
- Probar de forma exploratoria una **malla pequeña predefinida** de pesos:
  0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.
- Una posible extensión posterior, **no incluida en esta primera fase**,
  es combinar directamente CT3 + experto local + Local04D con pesos no
  negativos y que sumen uno, manteniendo una corrección residual compatible.
  Exigiría nuevas predicciones por componente y controles adicionales contra
  sobreajuste.

## Dependencias y fuentes de resultados

1. **Esperar a que termine** el `--final` de
   `tools/run_reference26_ridge_budget_mean.py`. Recuperar
   `reports/reference26_ridge_budget_mean/final_scores.csv` y las
   predicciones privadas `fold_XXX/kYY_aZ.csv` del Reference26 ganador,
   más la configuración (k=24,alpha=30) imputada con **media**.
   No registrar un ganador final antes de comprobar sus 138 scores.
   Como hubo selección en 12 y 46 parcelas, el RMSE del ganador es
   **exploratorio**, no prueba independiente.
2. **Local04D congelado**: recuperar las 138 predicciones LOOCV de
   `reports/area_weighted_climate_loo/loo_predictions.csv`,
   columna `GeoPCA20_GeographicWeightedRidge`. Si faltan archivos
   individuales, recuperar/regenerar únicamente este control usando el
   mismo protocolo de exclusión (137 train / 1 heldout); no reinterpretar
   como LOOCV las predicciones in-sample del modelo entrenado con 138 y.
3. **Reference26 original**: conservar
   `reports/selected_champions_loo/loo_predictions.csv` como control
   histórico para el mismo conjunto de 138 IDs.
4. **Resultados agregados conocidos (no recalculados aquí):**
   Reference26 original RMSE 0.4681855146560597,
   Local04D original RMSE ~0.485133 y Compact14 ~0.469673.
   No inferir con ellos el RMSE de un ensamble ni su correlación residual.
5. Mantener todos los CSV que contienen IDs, valores observados o
   predicciones individuales **fuera de Git**. Publicar únicamente
   métricas agregadas, protocolo y código sin datos privados.

## Auditoría antes de combinar

- Unificar por `ID_POLIGONO`, **nunca por índice/orden de filas**.
  Confirmar exactamente 138 IDs únicos e idénticos; verificar que
  `RENDIMIENTO_T_HA` / `observed` coincidan con tolerancia numérica
  estricta y no haya NaN o predicciones no finitas.
- Confirmar para **cada** modelo un verdadero fold externo 137/1 con
  rendimiento excluido antes de todo ajuste dependiente de y.
- Comprobar que Local04D mantiene su representación transductiva X-only
  original y que ninguna transformación supervisada utiliza y de la
  parcela excluida. Se pueden usar X visibles de las 197 parcelas si el
  protocolo de la competencia lo permite; no los 59 rendimientos ocultos.
- Verificar la reproducibilidad y los hashes de archivos/modelo de
  Reference26 **con media**: jamás tratar los CSV con mediana como
  predicciones con media.
- Confirmar que el objetivo de evaluación, unidades t/ha, definición de
  RMSE y población de 138 parcelas coinciden realmente.
- Si no se pueden alinear por ID y y observado, **detener** el experimento
  en lugar de combinar resultados de protocolos incompatibles.

## Fase A — Diagnóstico sin nuevos entrenamientos

Con las 138 parejas de predicciones LOOCV ya calculadas:

1. Calcular errores firmados `e_ref = yhat_ref - y` y
   `e_local = yhat_local - y`, sus RMSE, MAE, sesgo y correlación,
   y estudiar el error cuadrático emparejado por parcela.
2. Evaluar la malla de pesos pequeña descrita arriba. Mostrar los
   pesos con su `RMSE`, `MAE`, `bias`,
   `R²` y `delta_MSE` frente a Reference26 **en las mismas 138
   parcelas**; generar una curva RMSE–lambda y una tabla de errores
   emparejados agregados. Si lambda > 0 no mejora o la curva es plana,
   detener el ensamble.
3. Examinar estabilidad: intervalos de confianza mediante bootstrap
   pareado por parcela como diagnóstico; reconocer dependencia espacial
   y considerar bootstrap por grupos espaciales como sensibilidad.
4. Reportar claramente que escoger el peso con esos mismos 138 errores
   produce un **resultado exploratorio optimista**: los OOF base
   individuales NO equivalen a validación externa del peso seleccionado.

## Fase B — Validación honesta del peso (si la fase A es prometedora)

Para cada parcela excluida i:

1. Construir únicamente con las **otras 137 etiquetas** predicciones
   cross-fitted internas de ambos modelos, manteniendo todos los
   preprocesamientos dependientes de y en los pliegues internos.
2. Elegir `lambda_i` (de la malla preespecificada, o con una pequeña
   regularización hacia 0) minimizando error *interno*, sin mirar
   `y_i`. Importante: **no** elegir el peso con los 137 errores LOOCV
   globales guardados directamente; sus modelos pueden haber sido
   entrenados utilizando `y_i` y filtrar información al meta-ajuste.
3. Reajustar cada predictor base sobre las 137 etiquetas externas y
   producir dos predicciones para i. Combinar usando únicamente
   `lambda_i`, congelado previamente.
4. Repetir sobre las 138 parcelas. Comparar contra Reference26 y
   Local04D originales con el mismo conjunto de observaciones.

Como la búsqueda previa de hiperparámetros de Reference26 observó ya
errores de LOOCV externos, la elección previa del candidato introduce
optimismo. Para reclamar generalización no sesgada de **toda la cadena**
(imputación + hiperparámetros + peso), habría que anidar también esas
selecciones o conseguir etiquetas verdaderamente nuevas. La fase B
puede comprobar principalmente la selección del peso, **condicionada**
a haber congelado la variante de Reference26, no eliminar ese sesgo
histórico.

## Reglas de decisión y costes

- No entrenar de nuevo nada pesado para la **fase A** si los dos CSV
  emparejados de 138 predicciones ya existen.
- Priorizar una combinación constante y simple. No lanzar un modelo de
  tres expertos con mezcla espacial flexible antes de encontrar
  complementariedad en los residuos.
- Mantener exactamente las predicciones históricas y el benchmark
  sin alteraciones.
- No promover un ensamble solo porque se alcance un RMSE ligeramente
  menor después de probar muchos pesos. Examinar tamaño del efecto,
  errores emparejados, sensibilidad espacial, robustez y costo.
- La predicción final de 59 parcelas se considerará **únicamente después**
  de completar auditoría, evaluaciones y una decisión explícita.
- Conservar `competition` como único objetivo del estudio.

## Entregables previstos (NO creados todavía)

- Script de auditoría y ensamble barato de predicciones OOF;
- prueba de contrato ID/y/137-1 y tests sintéticos;
- tabla agregada RMSE-versus-lambda y correlación de residuos;
- informe con decisión continuar/detener;
- opcionalmente un segundo script de selección de peso anidada.

**Próximo paso inmediato:** esperar y revisar la tabla
`[MEAN FINAL]` del Reference26 con media, localizar el CSV de
predicciones LOOCV originales de Local04D, y decidir si hay evidencia
para la fase A. Nada de esto se ha ejecutado por la mera creación del plan.
