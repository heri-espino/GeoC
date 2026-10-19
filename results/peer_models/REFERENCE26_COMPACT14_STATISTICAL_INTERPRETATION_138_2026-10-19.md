# Interpretación del estudio estadístico LOOCV final — 138 parcelas

**Fuente:** salida real de la VM del usuario, 2026-10-19, scripts tools/report_reference26_compact14_final.py --run y --summary. Las 138 predicciones corresponden a exclusión externa de una parcela y 137 etiquetas por ajuste. El ensamble usa los pesos ya optimizados sobre esas mismas 138 etiquetas: 70.494625% Reference26 con media (k=24, alpha=30), 29.505375% Compact14 original. Por tanto, las métricas del ensamble NO evalúan independientemente el procedimiento de selección de pesos. Las 59 parcelas reservadas no tienen rendimiento real conocido.

## Desempeño y normas de error

| Métrica | Reference26 (media) | Compact14 | Ensamble |
|---|---:|---:|---:|
| RMSE (t/ha) | 0.468138 | 0.469673 | **0.467811** |
| MAE (t/ha) | 0.333786 | 0.333044 | **0.332489** |
| Mediana error absoluto (t/ha) | 0.244645 | **0.239316** | 0.247911 |
| P95 error absoluto (t/ha) | 0.779618 | 0.820040 | **0.779594** |
| Máximo error absoluto (t/ha) | **2.139958** | 2.146351 | 2.141844 |
| L1 total (t/ha) | 46.062466 | 45.960097 | **45.883453** |
| L2 total (t/ha) | 5.499378 | 5.517408 | **5.495541** |
| SSE ((t/ha)²) | 30.243156 | 30.441796 | **30.200967** |
| MSE ((t/ha)²) | 0.219153 | 0.220593 | **0.218848** |
| R² predictivo | 0.695292 | 0.693291 | **0.695717** |
| WAPE (%) | 8.438822 | 8.420068 | **8.406026** |
| Concordancia de Lin | **0.818817** | 0.815820 | 0.818435 |

**Lectura:** el ensamble mejora RMSE frente a Reference26 por ~0.000327 t/ha (**0.07%**), y reduce ligeramente MAE, pero no es siempre mejor: Compact14 conserva la menor mediana de error absoluto; Reference26 conserva un máximo absoluto algo menor. Estas diferencias son mínimas y dependen de la pérdida utilizada.

Si e_i = predicho_i − observado_i, las normas L1 = suma |e_i|, L2 = raíz de suma e_i², L∞ = máximo |e_i|; MAE=L1/138, RMSE=L2/raíz(138). El R² predictivo se calcula como 1−SSE/TSS de los 138 observados: **no equivale a porcentaje de aciertos**. El R² ajustado es N/A porque las recetas compuestas y regularizadas no admiten un único número defendible de parámetros efectivos.

## Colas de error y tolerancias

Para el ensamble: mediana |e| **0.247911 t/ha**, P90 **0.679927**, P95 **0.779594**, P99 **1.680626**; error máximo **2.141844 t/ha**, que corresponde a sobreestimación (e positivo). La mayor subestimación absoluta es **0.877516 t/ha**. Hay 69/138 parcelas con |e| ≤0.25 t/ha, 104/138 ≤0.50 y 133/138 ≤1.0. Estas proporciones son descriptivas, no coberturas predictivas calibradas.

RMSE respecto de cosecha media **11.827%**, sMAPE **8.937%**, MAPE **9.551%**, WAPE **8.406%**. Son normalizaciones de pérdidas distintas: no se convierten en «porcentaje de precisión». El error máximo mucho mayor que la mediana muestra la importancia de los extremos. Sin auditar las etiquetas no podemos atribuir los errores grandes a ruido de medición.

## Sesgo y calibración

Media observada **3.955362 t/ha**, estimada **3.954745 t/ha**, sesgo **−0.000617 t/ha** (casi nulo). Mediana residual **−0.046580 t/ha**; 60 errores positivos y 78 negativos. Pendiente de calibración descriptiva observado vs predicho **1.014672** e intercepto **−0.057409 t/ha**. Pearson r **0.834184**, Spearman rho **0.769006**, CCC de Lin **0.818435**.

La alineación global de medias es buena; no implica calibración uniforme entre zonas, ni que los errores positivos y negativos se compensen en cada parcela. La mediana negativa y la asimetría positiva sugieren subestimaciones relativamente frecuentes compensadas por algunas sobreestimaciones grandes.

## Normalidad, asimetría y curtosis

| Modelo | Shapiro W | Shapiro p | D’Agostino–Pearson p | Anderson–Darling al 5% |
|---|---:|---:|---:|---|
| Reference26 media | 0.910171 | 1.401988e−7 | 8.411629e−12 | Rechaza |
| Compact14 | 0.911359 | 1.644084e−7 | 9.983071e−12 | Rechaza |
| Ensamble | 0.909641 | 1.306209e−7 | 7.347903e−12 | Rechaza |

Para el ensamble: asimetría **+1.439436**, exceso de curtosis **+4.467670**, Anderson **2.284912** vs crítico **0.748**. Jarque–Bera **151.179791** (solo estadístico, sin p asintótica por limitación muestral n=138).

**Lectura:** patrones fuertemente incompatibles con una normal marginal, cola positiva y valores extremos más frecuentes que bajo la referencia normal. No invalida los indicadores predictivos; Shapiro/K²/Anderson en residuos LOOCV tampoco son p-valores iid exactos porque los modelos comparten gran parte del entrenamiento.

## Heterocedasticidad y relación error–predicción

| Modelo | Koenker–BP LM | p asintótica | Brown–Forsythe p |
|---|---:|---:|---:|
| Reference26 media | 0.705933 | 0.400797 | 0.929793 |
| Compact14 | 1.009336 | 0.315062 | 0.833443 |
| Ensamble | 0.793133 | 0.373155 | 0.855512 |

Para el ensamble: correlación Pearson(error firmado, pronóstico) **−0.021868** (p 0.799048); Spearman(error absoluto, pronóstico) **−0.007582** (p 0.929667).

**Lectura:** no se observa evidencia suficiente de heterocedasticidad respecto al nivel estimado con estas pruebas; tampoco queda demostrada la homocedasticidad ni ausencia de dependencia por geología, municipio o época. Los contrastes iid son exploratorios y predicción y residuo tienen acoplamiento algebraico.

## Diagnóstico geoespacial: Moran I global de errores firmados

Distancias Haversine, grafo KNN dirigido con pesos 1/k por fila, 1,999 permutaciones:

| Vecinos k | Reference26 I / p bilateral | Compact14 I / p bilateral | Ensamble I / p bilateral |
|---:|---|---|---|
| 4 | −0.055864 / 0.3455 | −0.038332 / 0.5525 | −0.051225 / 0.4150 |
| 8 | −0.076861 / 0.0630 | −0.066386 / 0.1095 | −0.074302 / 0.0720 |
| 12 | −0.062976 / 0.0640 | −0.053345 / 0.1035 | −0.060552 / 0.0670 |
| 24 | −0.041802 / 0.0645 | −0.037813 / 0.1025 | −0.040814 / 0.0735 |

Para estas filas, los p Holm ajustados dentro de la familia completa de tests espaciales son **1.0**. No se detectó señal bilateral al 5%. Moran negativo es compatible con cierta disimilitud entre vecinos, pero **no fue significativo**. La ausencia de significación no prueba independencia espacial, buen desempeño en regiones alejadas ni ausencia de clústeres en errores absolutos/cuadráticos.

El programa también calcula Geary C, Moran sobre |e| y e², semivariogramas y análisis regional. **Los valores de esos indicadores no fueron incluidos en la transcripción disponible**, por lo que no se inventan conclusiones. El JSON agregado de la VM permitirá explorar sus resultados una vez publicado.

## Contrastes emparejados con Reference26 media

| Modelo y pérdida | Diferencia media | Mejora en parcelas | Wilcoxon p | Holm p |
|---|---:|---:|---:|---:|
| Compact14, absoluta | −0.000742 t/ha | 52.90% | 0.621214 | 1.000000 |
| Compact14, cuadrática | +0.001439 (t/ha)² | 52.90% | 0.642378 | 1.000000 |
| Ensamble, absoluta | −0.001297 t/ha | 55.80% | 0.178868 | 0.715473 |
| Ensamble, cuadrática | −0.000306 (t/ha)² | 55.80% | 0.253311 | 0.759932 |

El ensamble reduce el error en **77 de 138 parcelas**, pero los contrastes emparejados exploratorios no proporcionan evidencia de superioridad estadística tras Holm. Eso **no implica equivalencia** de los modelos. Como se usaron los mismos 138 rendimientos para elegir pesos, ningún p-valor aquí valida independientemente la selección.

En el experimento previo de búsqueda de 4.6 millones de mezclas con estos mismos datos, la ventaja en MSE se invirtió de **−0.000306** a **+0.000082** al retirar la sola parcela más favorable; el intervalo bootstrap pareado ingenuo de ΔMSE fue **[−0.002755, +0.002347]**. Esto refuerza la cautela, sin que ese intervalo corrija dependencia espacial ni selección adaptativa.

## Conclusión propuesta al jurado

> En una evaluación leave-one-parcel-out de 138 parcelas, las predicciones de Reference26 con media y Compact14 fueron muy cercanas. Su ensamble convexo preseleccionado alcanzó el menor RMSE (0.467811 t/ha; R² predictivo 0.695717), aunque la diferencia frente a Reference26 fue pequeña y sensible a algunas parcelas. Los residuos presentaron asimetría positiva, colas pronunciadas y rechazo de normalidad marginal; las pruebas aplicadas no detectaron heterocedasticidad ni autocorrelación espacial significativa de errores firmados a los tamaños de vecindario considerados. Los resultados son diagnósticos exploratorios, no pruebas de causalidad, independencia espacial o mejora externamente confirmada.

## Advertencias para sitio y documento

- No equiparar R² con porcentaje correcto.
- No interpretar p>0.05 como prueba de normalidad, varianza constante o independencia espacial.
- No usar 26/14 como «número de parámetros» para fabricar R² ajustado.
- No reportar significación del ensamble elegido con las mismas 138 etiquetas.
- No presentar predicciones de las 59 parcelas como cosechas observadas.
- No describir AIC/BIC, probabilidades de cobertura ni intervalos calibrados: no se calcularon bajo una distribución predictiva apropiada.

## Referencias de método

- [NIST residual analysis](https://www.itl.nist.gov/div898/handbook/pri/section2/pri24.htm).
- [SciPy Shapiro–Wilk](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.shapiro.html), [normaltest](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.normaltest.html), [Jarque–Bera](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.jarque_bera.html).
- [SciPy Levene/Brown–Forsythe](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.levene.html).
- [PySAL Moran I](https://pysal.org/esda/stable/user-guide/global_morans_i.html).
- [Roberts et al. (2017): spatial cross-validation](https://doi.org/10.1111/ecog.02881).
