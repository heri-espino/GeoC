# Reference26 — búsqueda masiva de ensambles convexos (138 parcelas)

**Fecha:** 2026-10-19. **Estado:** evaluación exploratoria concluida en la VM;
métricas facilitadas por el usuario desde `tools/run_loo_ensemble_massive.py
--summary`. Solo se archivan métricas agregadas, nunca los 138 rendimientos,
identificadores o pronósticos individuales.

## Diseño

- Cinco predictores alineados sobre exactamente 138 parcelas, cada uno con
  predicción LOOCV externa y 137 etiquetas para su ajuste:
  `Reference26_mean`, `Reference26_original`, `Compact14`,
  `Local04D` y `Reference26_DBSCAN`.
- Búsqueda exhaustiva de **4,598,126** mezclas convexas en una malla de
  1 punto porcentual de resolución (pesos no negativos, suma 1).
- Matriz de productos cruzados de errores `G = E.T @ E / 138`;
  RMSE de cada mezcla igual a `sqrt(w.T @ G @ w)`. No se entrenó
  de nuevo ninguna base.
- También se aplicó optimización cuadrática continua de pesos,
  barrido de subconjuntos, regularización hacia Reference26 y
  diagnóstico de bootstrap pareado sobre diferencias de MSE.

## Mejores resultados completos

| Candidato | Pesos efectivos | RMSE (t/ha) | ΔMSE vs Reference26_mean |
|---|---|---:|---:|
| Reference26_mean | 100% Reference26_mean | 0.468138 | 0 |
| Mejor de 4,598,126 mezclas | 70% Reference26_mean, 30% Compact14 | **0.467812** | -0.000306 |
| Óptimo convexo continuo (QP) | 70.494625% Reference26_mean, 29.505375% Compact14 | **0.467811** | -0.000306 |

La diferencia de RMSE es solamente ~0.000326–0.000327 t/ha,
aproximadamente **0.07%** del error de Reference26_mean.

- Se evaluaron mezclas que daban peso a los cinco modelos; el
  **óptimo matemático global convexo** asignó peso cero o numéricamente
  despreciable a Local04D, Reference26_original y DBSCAN.
- Entre los primeros 20 resultados de la malla los pesos más competitivos
  se concentraron alrededor de 69–72% Reference26_mean y
  28–31% Compact14, con sustituciones pequeñas por Reference26_original.
- El óptimo continuo mejora el RMSE de la malla solo alrededor de
  `1e-6` t/ha al redondear seis decimales. No tiene valor práctico
  aumentar más la resolución de la búsqueda con estos mismos cinco modelos.

## Robustez del ensamble 70/30

La cantidad `delta_mse` se define como
`MSE(ensamble) - MSE(Reference26_mean)`: **valores negativos favorecen
al ensamble**.

| Diagnóstico | Valor |
|---|---:|
| ΔMSE de la mezcla 70/30 sobre las 138 parcelas | -0.000306 |
| Fracción de parcelas donde disminuye el error cuadrático | 0.557971 = 77/138 |
| Percentiles bootstrap pareado ingenuo 95% para ΔMSE | [-0.002755, +0.002347] |
| Fracción de remuestreos con ΔMSE menor que cero | 0.605333 |
| ΔMSE quitando la parcela con mayor contribución positiva | +0.000082 |
| ΔMSE quitando las cinco con mayor contribución positiva | +0.001088 |
| ΔMSE quitando las diez con mayor contribución positiva | +0.001991 |

El signo del cambio en MSE **se invierte tras retirar la parcela más
beneficiosa para el ensamble**. El intervalo bootstrap ingenuo contiene
cero; la mejora es débil y sensible a un solo caso. La fracción
bootstrap favorable de 0.605333 **no es** una probabilidad posterior
de mejora ni un p-valor.

El bootstrap sobre parcelas supone independencia, una aproximación
discutible con dependencia espacial. Además, calcular un intervalo para
pesos escogidos con los mismos 138 valores de y **no corrige** el
sesgo por búsqueda adaptativa. Las bases generan predicciones LOOCV,
pero los pesos no se eligieron dentro de un meta-LOOCV anidado.

## Decisión

**NO PROMOVER** este ensamble a nuevo modelo oficial ni usar el
RMSE 0.467811 como validación independiente. Mantener Reference26
con media (k=24, alpha=30) como referencia y el ensamble 70/30 como
candidato exploratorio previamente registrado.

Cerrar los barridos de pesos convexos globales sobre estos mismos cinco
predictores: ya se encontró su óptimo matemático. El siguiente experimento
debe aportar **diversidad predictiva real**, no más combinaciones numéricas.

Opciones prácticas:

1. Auditar las predicciones OOF guardadas de CatBoost, proyecciones
   aleatorias gaussianas, Prithvi y modelos alternativos. Sus RMSE
   individuales peores no descartan complementariedad, pero NO mezclar
   directamente resultados de 16×41 (97 train) con 138 LOOCV (137 train).
2. Si existen comparaciones sobre idénticas particiones antiguas,
   calcular matrices de errores y ensambles en ese protocolo por
   separado para decidir si alguna familia merece 138 folds nuevos.
3. Solo si aparece una mejora material y estable, abordar un
   entrenamiento anidado adecuado para seleccionar pesos o compuertas
   geográficas dentro de las 137 etiquetas por fold.
4. No inferir rendimiento para las 59 parcelas ocultas ni actualizar
   el modelo de producción/competencia con esta búsqueda exploratoria.

## Archivos privados de origen

- `reports/loo_ensemble_screen/massive/grid_best.csv`
- `reports/loo_ensemble_screen/massive/optimized_subsets.csv`
- `reports/loo_ensemble_screen/massive/robustness.csv`
- `reports/loo_ensemble_screen/massive/run_spec.json`

Estos archivos permanecen en la VM e ignorados por Git.
