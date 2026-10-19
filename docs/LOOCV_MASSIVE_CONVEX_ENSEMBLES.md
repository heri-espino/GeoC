# Búsqueda masiva de combinaciones convexas con Reference26

**Estado:** código añadido para ejecución local; no es una nueva validación
independiente. Se emplean **únicamente** las predicciones LOOCV ya
calculadas de las mismas 138 parcelas. No se ajusta ningún modelo base.

## Motivación y resultados de partida

Resultados previos sobre los mismos 138 identificadores y rendimientos:

- Reference26 con media: RMSE 0.468138 t/ha.
- Mezcla exploratoria Reference26 con media (0.70) +
  Compact14 (0.30): RMSE **0.467812** t/ha, MAE **0.332477**.
- Mezcla ancla (0.60) + Compact14 (0.30) +
  Reference26 original (0.10): RMSE 0.467816.
- Correlaciones entre residuos firmados:
  Reference26_mean–Compact14: 0.992,
  Reference26_mean–Local04D: 0.966,
  Reference26_mean–DBSCAN: 0.997.
- La mejora del ensamble 70/30 frente al ancla es solo ~0.000326
  t/ha en RMSE, ~0.07%; no es un salto grande ni se puede atribuir
  a generalización independiente.

El usuario pidió probar MUCHAS más combinaciones y estrategias
sin volver a entrenar los modelos; las combinaciones lineales son
lo suficientemente baratas para analizarlas de forma exhaustiva.

## Aceleración matemática: matriz de errores

Sea E de tamaño 138 × M, donde cada columna es el error firmado
de un predictor, e_ij = yhat_ij - y_i. Un ensamble convexo requiere
w_j ≥ 0, suma de w_j = 1. Su MSE exacto es

```math
G = E^T E / 138,
MSE(w) = w^T G w.
```

Se calcula G UNA vez. Con M=5, es una matriz **5 × 5**;
evaluar millones de pesos requiere solo productos de matrices muy
pequeñas. No implica millones de fits ni millones de validaciones.
El RMSE de la mezcla es la raíz de MSE(w).

## Tres familias de experimentos

### I. Malla exhaustiva

Evaluar **todos** los pesos no negativos que suman 100%, en incrementos
de 1 punto porcentual, para **los cinco modelos disponibles**:

```text
Reference26_mean
Reference26_original
Compact14
Local04D
Reference26_DBSCAN
```

Número exacto de combinaciones para pasos de 1%:

```math
C(100 + 5 - 1, 5 - 1) = C(104,4) = 4,598,126.
```

El script genera fragmentos de hasta 50,000 combinaciones en memoria
y guarda únicamente las mejores 30; no escribe millones de filas ni
predicciones por parcela. Puedes variar `--step`, `--top`,
`--chunk-size` y `--max-evals`. Con paso 2%,
hay C(54,4) = 316,251 combinaciones para cinco modelos.

### II. Optimización convexa y regularizada

Además de la búsqueda discreta, se calcula el óptimo continuo mediante
**programación cuadrática**:

```math
min_{w >= 0, 1^T w = 1} w^T G w.
```

El óptimo no puede ser peor que el mejor punto de una malla discreta.
Para evitar que cinco modelos casi idénticos reciban pesos
arbitrariamente grandes por el sobreajuste, comparar

```math
min_{w >= 0, 1^T w = 1}
    w^T G w + gamma ||w - (1,0,0,0,0)||_2^2,
```

donde `gamma` ∈ {0, 0.0001, 0.001, 0.01, 0.1}.
También se evalúan **todos los subconjuntos no vacíos** de
predictores (2^M−1 subconjuntos; 31 para cinco modelos), y la
restricción adicional w(Reference26_mean) ≥ 0.5 en la familia principal.
Este tipo de mezcla restringida remite a Breiman (1996), mientras
que elegir pesos a través de validación cruzada es el enfoque de
Super Learner (van der Laan, Polley y Hubbard, 2007).

### III. Diagnósticos de robustez

Para pesos **congelados**, registrar

- ΔMSE emparejado contra Reference26_mean en las 138 parcelas;
- fracción de parcelas cuya pérdida cuadrática mejora;
- bootstrap pareado descriptivo sobre parcelas (por defecto 3,000
  réplicas; no reoptimiza los pesos);
- cómo cambia ΔMSE al quitar 1, 5 o 10 parcelas con mayores
  beneficios individuales;
- comparar el ensamble 70/30 fijado antes del nuevo grid contra
  el ganador exhaustivo y el óptimo continuo.

**Advertencia:** intervalos bootstrap ordinarios no corrigen la
elección retrospectiva de pesos, ni necesariamente la dependencia
espacial. No etiquetar sus intervalos como confirmación del
rendimiento externo.

## Cómo correrlo (PowerShell)

~~~powershell
conda activate geocebada
git pull --ff-only origin experiment/nuria-champions-loocv

# Millones de mezclas, sin entrenamientos de modelos base
python tools/run_loo_ensemble_massive.py --run

# Leer la clasificación y los experimentos regularizados
python tools/run_loo_ensemble_massive.py --summary
~~~

Si deseas verificar el flujo con menos combinaciones:

~~~powershell
python tools/run_loo_ensemble_massive.py --run --step 0.02
~~~

**Atención:** el segundo comando escribe en el mismo directorio
`reports/loo_ensemble_screen/massive/` y reemplaza sus tablas
agregadas. Para conservar ambos conjuntos, especifica
`--output reports/loo_ensemble_screen/massive_2pct`.

Para añadir un predictor REAL con 138 predicciones LOOCV equivalentes:

~~~powershell
python tools/run_loo_ensemble_massive.py --run --extra "CatBoost=reports/mi_modelo/catboost_loo.csv" --step 0.02 --max-evals 5000000
~~~

El archivo adicional se valida mediante el cargador
`tools/run_loo_ensemble_screen.py`, que requiere los mismos 138 IDs,
los mismos valores observados y la evidencia 137-train/1-heldout por fila.
Para seis o más modelos, el número de combinaciones crece rápidamente:
el programa se detiene antes de calcular una malla excesiva, salvo
que se eleve explícitamente `--max-evals`. El optimizador cuadrático
sigue siendo la opción más eficiente.

Salidas agregadas, todas ignoradas por Git:

- `reports/loo_ensemble_screen/massive/grid_best.csv`:
  las mejores combinaciones discretas;
- `optimized_subsets.csv`: óptimos convexos y regularizados,
  subconjuntos y pesos;
- `robustness.csv`: diagnósticos de estabilidad pareada;
- `run_spec.json`: orden de modelos, resolución, rutas y advertencias.

No se generan archivos de etiquetas o pronósticos por parcela.

## Límite metodológico fundamental

**Evaluar 4.6 millones de combinaciones con 138 etiquetas conocidas
no equivale a validar 4.6 millones de modelos independientes.**
Se incrementa el riesgo de elegir retrospectivamente pesos que
se adapten al ruido de ese conjunto. Las predicciones base son
individualmente OOF, pero se entrenaron sobre conjuntos que pueden
incluir la etiqueta del outer-heldout de una eventual validación de
pesos; NO se debe simular una meta-validación anidada eliminando solo
una fila de los CSV de predicciones base.

Requerimientos para una validación honesta de la mezcla:
predicciones internas cross-fitted de CADA base dentro de los
137 entrenamientos disponibles, selección de parámetros/pesos sin
y de la parcela externa y ajuste externo final. La elección
anterior de Reference26 con datos LOOCV ya inspeccionados también
introduce sesgo de selección.

La búsqueda masiva es un diagnóstico barato para definir una
familia de ensambles pequeña, regularizada y verificable, **no**
para adoptar automáticamente su mejor RMSE como predicción final
de la competencia.

## Bibliografía

- Breiman, L. (1996). *Stacked Regressions*. Machine Learning 24,
  49–64. https://doi.org/10.1023/A:1018046112532
- van der Laan, M. J., Polley, E. C. y Hubbard, A. E. (2007).
  *Super Learner*. Statistical Applications in Genetics and Molecular
  Biology 6(1), Article 25.
  https://doi.org/10.2202/1544-6115.1309
- Wolpert, D. H. (1992). *Stacked Generalization*.
  Neural Networks 5(2), 241–259.
  https://doi.org/10.1016/S0893-6080(05)80023-1
- scikit-learn, `StackingRegressor`:
  https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingRegressor.html
  Su meta-estimador se ajusta usando predicciones de validación cruzada;
  no debe extrapolarse esa propiedad a un ajuste directo sobre la tabla
  OOF global previamente inspeccionada.
