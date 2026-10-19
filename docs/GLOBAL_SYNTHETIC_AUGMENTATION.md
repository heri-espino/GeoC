# Experimento: un modelo global con muchas observaciones sinteticas

**2026-10-19 — implementado en rama experimental; ejecucion real pendiente.**
Competencia AgroCebada; 197 parcelas X observables, 138 rendimientos
observados y 59 rendimientos ocultos. No se cambian modelos publicados,
predicciones oficiales ni el repositorio de Todoseboso.

## Por que es diferente del experimento de tres especialistas

La ejecucion anterior con dos Ridge sinteticos + Random Forest clasificador
NO supero Reference26 mean (RMSE LOOCV **0.468138** t/ha):
el clasificador conservador no cambio observaciones y las configuraciones
que si cambiaron empeoraron el RMSE. Eso NO prueba que una aumentacion
directa de datos a un unico modelo global sea inutil.

**Nueva hipotesis:** dos observaciones extremadamente dificiles tal vez
representan una region de la distribucion de X,y poco muestreada. Evaluar
un solo regresor global con TODAS las observaciones reales permitidas en
cada fold MAS puntos sinteticos derivados exclusivamente de dos anclas
extremas presentes en entrenamiento. No hay clasificador ni gating.
Tambien entrenar EL MISMO regresor sin aumentacion en los mismos folds.

Se comparan dos familias de modelo global (Ridge y ExtraTrees). Ninguna
es el Reference26 nativo: el original Reference26 se conserva como
TERCER control, con sus 138 predicciones OOF autenticas y sin refit.
Por tanto, toda mejora sobre Reference26 y sobre el control homologo
debe interpretarse separadamente.

## Generacion sintetica

Las dos anclas reales se recuperan de la lista privada, seleccionada
previamente por la interseccion top-2 entre modelos, en
`reports/loocv_hard_parcels/diagnostic_manifest_private.json`.
No se incorporan rendimientos de las 59 parcelas desconocidas.

Primero transformar las 26 covariables numericas originales de las
197 parcelas a una representacion X-only robustamente estandarizada.
El ruido esta en **este espacio de caracteristicas**, no en latitud/longitud.

5 metodos, manteniendo disponible al menos una ancla real para las variantes iniciales:

1. **repeat**: muestreo con reemplazo de las anclas reales, X e y
   exactamente repetidos. Control de simple reponderacion.
2. **local**: X artificial cerca de cada ancla con perturbacion gaussiana
   cuya matriz de covarianza es estimada a partir de 20 vecinos X-only
   de las 197 parcelas (65% local, 35% identidad). Su pseudo-y es el
   rendimiento real de su ancla. No se asigna rendimiento real nuevo.
3. **bridge**: si ambas anclas permanecen en entrenamiento, usar un
   puente browniano multivariante entre z_a y z_b, con ruido nulo en
   t=0,1 y covarianza maxima en t=.5. Se interpola pseudo-y
   linealmente: y(t)=(1-t)y_a+t*y_b. Si una ancla es heldout,
   **retroceder a local alrededor del ancla restante**. Nunca leer
   el y del heldout para generar otra observacion.
4. **mixed**: mitad de filas `local`, mitad `bridge`;
   tambien sigue el fallback sin filtracion.

5. **gaussian_hard (nuevo, preferente para la nueva hipotesis):**
   identificar dentro de las 137 etiquetas de entrenamiento los **8
   puntos con mayor residuo LOOCV absoluto de un Ridge auxiliar**
   mediante el calculo exacto y eficiente PRESS. Cada punto se
   convierte en el centro de una distribucion normal multivariante de
   X, con covarianza local estimada usando vecinos X observables.
   Generar 256 o 1,024 filas desde esas ocho normales; cada fila
   sintetica hereda el y real de su propio centro, sin agregar ruido
   ficticio al rendimiento. El numero de centros es configurable con
   `--hard-centers 8`.

   Esta seleccion se **recalcula dentro de cada fold**. No se
   reutilizan las listas de residuos de Reference26 calculadas
   usando las 138 etiquetas, porque esas predicciones historicas de
   otras parcelas podrian haber sido ajustadas con el rendimiento
   del fold externo que estamos evaluando. El ranking Ridge PRESS
   es un indicador auxiliar de dificultad, **no exactamente el
   ranking residual de Reference26**. Esta separacion permite
   investigar la hipotesis sin filtracion directa del y heldout
   a la eleccion de los centros.

Los *datos sinteticos no son mediciones independientes*. Aumentar de
137 a 1161 filas artificialmente NO equivale a tener 1161 etiquetas
reales. Los metodos solo codifican supuestos sobre la vecindad o la
interpolacion del rendimiento.

Dos tamanos iniciales: **256** y **1,024** filas por entrenamiento.
Dos pesos totales del conjunto sintetico:

- **32**: cada fila recibe peso 32/N_syn, limitando la fuerza total
  de puntos redundantes;
- **full**: cada fila recibe peso 1 y su masa total es N_syn,
  para probar la version mas agresiva solicitada por el usuario.

Las 137 filas reales tienen peso 1. Para `--predict59` son 138.

## Evaluacion

El script usa las 138 predicciones LOOCV genuinas de Reference26 mean
(137/1) ya calculadas y verifica paridad del bundle de 26 covariables.

En cada uno de 138 folds:

1. Retirar el rendimiento real de la parcela heldout de TODOS los
   ajustes y generadores de pseudoy.
2. Ajustar Ridge y ExtraTrees globales en las 137 etiquetas reales
   (controles **plain**).
3. Por cada metodo, tamaño y masa sintetica, generar SOLO a partir
   de las anclas que quedan en esas 137 observaciones y ajustar el
   mismo regresor global con filas reales + artificiales.
4. Predecir la parcela real heldout, almacenar exclusivamente en
   resultados locales privados.

El grid estandar es **5 metodos x 2 tamanos x 2 masas x 2 modelos = 40**
candidatos aumentados, mas dos modelos reales sin aumentacion
(**42 series predictivas x 138 folds**). Los 8 centros gaussianos
cambian en cada fold de 137 etiquetas reales. La carga es mayor que en el
experimento de expertos; el procesamiento usa varios workers,
cache por fold y huella SHA de fuente/datos/parametros.

Metricas: RMSE/MAE de los 138, RMSE de las dos anclas problematicas,
RMSE de las otras 136, delta-MSE vs Reference26 original y vs
**el mismo regresor sin sinteticos**, casos que empeoran y maxima
diferencia de prediccion. El analisis de errores firmados se ilustra
sobre las 197 X en PCA y versus rendimiento observado, sin fijar
grupos imaginarios ni usar K-Means.

**Limite de inferencia:** las anclas se escogieron a posteriori mirando
los propios residuos LOOCV. Aunque no se filtre el y heldout a la
generacion en cada fold, la seleccion global de anclas es post-hoc.
Un resultado positivo seria exploratorio, no una confirmacion
independiente de precision FIRA.

## Instrucciones de VM

~~~powershell
cd C:\Users\175199\GeoC
conda activate geocebada

git fetch origin
git switch experiment/global-synthetic-augmentation
git pull --ff-only

python -m pytest tests/test_global_synthetic_augmentation.py -q
python -m py_compile tools/run_global_synthetic_augmentation.py

python tools/run_global_synthetic_augmentation.py --check
python tools/run_global_synthetic_augmentation.py --plot
python tools/run_global_synthetic_augmentation.py --smoke
python tools/run_global_synthetic_augmentation.py --run --jobs 4
python tools/run_global_synthetic_augmentation.py --summary
~~~

Si una primera comparacion rapida es preferible a las 34 series,
ejecutar solo Ridge en **otra carpeta** para no confundir caches:

~~~powershell
python tools/run_global_synthetic_augmentation.py --run --jobs 4 --models ridge --output reports/global_synthetic_augmentation_ridge_only
~~~

El directorio `global_synthetic_augmentation_ridge_only` tambien
debe permanecer privado y gitignored. No mezclar estos resultados
con los del experimento completo.

Para una variante muy agresiva (4096 sinteticos por fold, masa
completa), **usar otro output** y limitar primero a Ridge:

~~~powershell
python tools/run_global_synthetic_augmentation.py --run --models ridge --methods gaussian_hard --sizes 4096 --masses full --hard-centers 8 --output reports/global_synthetic_augmentation_4096
~~~

### Predicciones candidatas para las 59 parcelas (solo despues del CV)

Se exige un `final_manifest.json` original 138/59 y su CSV
verificado de predicciones Reference26 mean. No se sustituye nada:

~~~powershell
python tools/run_global_synthetic_augmentation.py --predict59 --predict-model ridge --predict-method gaussian_hard --predict-size 1024 --predict-mass 32
~~~

Los flags elegidos deben estar presentes en el grid del script;
si se usa un grid alternativo, pasarlo tambien en la llamada.
El archivo `candidate59_private.csv` no se sobreescribe y no se
publica. Elegir parametros con la misma CV solo da comparacion
post-hoc, no garantiza mejor desempeno en 59.

### Archivos privados

- `reports/global_synthetic_augmentation/comparison_aggregate.csv`
- `reports/global_synthetic_augmentation/private_fold_predictions.csv`
- `reports/global_synthetic_augmentation/fold_001.csv` ... `fold_138.csv`
- `reports/global_synthetic_augmentation/run_spec.json`
- `reports/global_synthetic_augmentation/signed_residual_pca_private.png`
- `reports/global_synthetic_augmentation/signed_residual_vs_y_private.png`
- `reports/global_synthetic_augmentation/candidate59_private.csv` (opcional)

No subir archivos por parcela ni rendimientos observados al GitHub
publico. Si se desea incorporar metricas agregadas, revisarlas
primero y copiarlas explicitamente a results/.

## Interpretacion esperada

- Si `repeat` y `local` dan ganancias similares, el efecto
  principal sera **reponderar los dos y extremos**.
- Si `local` gana a `repeat` dentro del mismo algoritmo,
  hay evidencia exploratoria de que la continuidad local en X
  aporta regularizacion util.
- Si `bridge` gana al resto, la interpolacion entre anclas
  puede ser compatible con los casos restantes, aunque NO prueba
  una distribucion generadora real entre ellas.
- Si los modelos globales empeoran el centro al mejorar las colas,
  el tradeoff contradice el objetivo de no perjudicar predicciones
  ordinarias. Medirlo explicitamente.
- Si ninguna combinacion supera el modelo sin aumento y Reference26,
  conservar Reference26 mean. No crear etiquetas sinteticas con
  rendimientos desconocidos ni inferir que FIRA elimino parcelas
  intencionalmente.

## Como interpretar la nueva variante normal centrada en casos dificiles

Para cada fold externo, el generador Ridge PRESS auxiliar encuentra los
indices de hasta 8 mayores errores internos usando SOLO entrenamiento.
Una nube centrada en cada caso se define como

\[
\widetilde x\mid c \sim \mathcal N(x_c,\sigma^2\widehat\Sigma_c),
\qquad
\widetilde y=y_c.
\]

El estimador \(\widehat\Sigma_c\) proviene de 20 vecinos de X
(representacion estandarizada sobre 197 covariables visibles) y su
matriz esta regularizada hacia la identidad. Los valores X
sinteticos se truncan a [-9,9] unidades estandarizadas. Cuando
sigma=0 todos los X son simples repeticiones de sus centros.

Importante: los puntos seleccionados como dificiles pueden ser
rendimientos *reales pero extremos*, *ruidosos* o simplemente mal
explicados por el auxiliar Ridge; la normal centrada en ellos es
un supuesto generativo. Si los resultados no mejoran frente a
los controles del MISMO regresor, la nueva aumentacion no sera
promovida. Mantener siempre el RMSE de Reference26 como comparacion.

Tests unitarios comprueban identidad exacta del PRESS frente al
LOOCV Ridge bruto, invariancia al modificar el rendimiento heldout,
ausencia de centros excluidos y recuento exacto de filas.


## Resultados auténticos comunicados desde VM — 2026-10-19

**Estado actualizado:** ejecución real `--run` de 138 folds completada en la
máquina del usuario; `--summary` reprodujo la tabla. La agente de GitHub
NO pudo inspeccionar los CSV privados, por lo que los valores siguientes son
los números agregados copiados del log enviado por el usuario. No se subieron
etiquetas individuales ni archivos de predicciones por parcela.

El experimento ajustó dos familias globales diferentes de Reference26,
**Ridge** y **ExtraTrees**, con sus correspondientes versiones sin
augmentación, y 40 recetas sintéticas (cinco métodos × dos tamaños
× dos masas × dos regresores). Se conservaron las 138 predicciones
LOOCV auténticas de Reference26 con imputación media como comparador
**sin reentrenarlo con los datos sintéticos**.

| Receta y familia | RMSE (t/ha) | MAE (t/ha) | RMSE del mismo regresor sin augmentación |
|---|---:|---:|---:|
| Reference26 mean (control histórico, no reentrenado) | **0.468138** | **0.333786** | n/a |
| Ridge global sin augmentación | 0.492732 | 0.361595 | 0.492732 |
| ExtraTrees global sin augmentación | 0.496662 | 0.368799 | 0.496662 |
| ExtraTrees + repeat 256, masa full (mejor receta sintética) | 0.506404 | 0.374848 | 0.496662 |
| ExtraTrees + gaussian_hard 256, masa 32 | 0.538480 | 0.395175 | 0.496662 |
| ExtraTrees + gaussian_hard 1024, masa 32 | 0.542316 | 0.393803 | 0.496662 |
| Ridge + gaussian_hard 256, masa 32 | 0.563545 | 0.455681 | 0.492732 |

Las salidas están ordenadas por RMSE; incluso el mejor modelo
**aumentado** (`repeat` con ExtraTrees) fue peor tanto que
Reference26 (+0.038266 t/ha de RMSE, ≈+8.2%) como que su ExtraTrees
sin augmentación (+0.009742 t/ha). Las gaussianas centradas en
parcelas con gran residuo PRESS tampoco mejoraron sus controles.
No se logró mejora LOOCV global dentro del grid examinado.

### Evidencia de intercambio entre extremos y parcelas ordinarias

El candidato `aug__ridge__gaussian_hard__n256__m32` bajó el RMSE
descriptivo en las dos históricamente más difíciles de 2.065
(Reference26) a **1.767889 t/ha**, pero elevó el RMSE en las otras
136 de 0.399583 a **0.525634 t/ha**. La ganancia en dos observaciones
no compensó el deterioro de la población ordinaria; el RMSE global
fue 0.563545 t/ha. La selección histórica de estos dos casos a partir
de los mismos residuos hace esa reducción exploratoria y susceptible
a sesgo de selección.

### Lo que estos resultados demuestran, y lo que no

1. Bajo estos hiperparámetros, aumentar agresivamente datos alrededor
   de puntos con errores elevados **perjudicó** tanto a Ridge como a
   ExtraTrees. El conjunto sintético de 256/1024 ejemplos no aporta
   observaciones de rendimiento nuevas, sino hipótesis locales sobre
   la relación X→y; puede distorsionar el compromiso global de errores.
2. No se ha probado **Reference26 nativo entrenado directamente con
   filas sintéticas**: su receta original combina múltiples componentes
   y requiere geometría, índices y distancias de parcelas reales, no
   admite simplemente concatenar filas al dataframe del Ridge global.
   Por ello el resultado NO descarta toda augmentación posible de
   Reference26, pero sí rechaza la promoción de estos dos regresores.
3. `gaussian_hard` seleccionó centros vía Ridge PRESS *dentro del fold*,
   no según los residuos de Reference26; las dos poblaciones pueden
   diferir. No se confirmó que las dos parcelas extremas constituyan
   un mismo régimen agronómico ausente de los datos.
4. **Decisión:** no modificar las 59 predicciones ni reemplazar
   Reference26 mean (RMSE 0.468138). Conservar el experimento como
   resultado negativo documentado, sin cambiar ninguna etiqueta.
