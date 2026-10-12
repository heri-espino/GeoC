# GeoCebada: metodología, validación cruzada y resultados (Checkpoints 02–08)

**Corte documental:** 12 de octubre de 2026.  
**Estado:** experimentos 07 y 08 completados; modelo final conservado: **Local04D**.  
**Alcance:** explicar qué se hizo, cómo se construyeron las particiones de validación, qué modelos se compararon, qué se aprendió de los resultados y qué queda —y qué no queda— validado.

> **Lectura breve:** contamos con 197 parcelas de cebada, 138 con rendimiento reportado y 59 sin rendimiento disponible. La estimación se construye sobre esas 59 parcelas fijas. Después de probar familias globales, modelos locales, grafos, ensambles, correcciones por residuos, nuevas variables satelitales y una comparación final con validación anidada, la evidencia disponible favorece **mantener Local04D**. Los RMSE de este documento provienen de rendimientos *reportados y temporalmente ocultados dentro de las 138 parcelas conocidas*, no de revelar el rendimiento real de las 59 parcelas de destino.

## 1. Pregunta que resolvemos

Se quiere estimar el rendimiento agrícola por hectárea de 59 parcelas georreferenciadas correspondientes al ciclo de 2025, aprovechando las mediciones y características observables de las 197 parcelas.

| Elemento | Definición |
|---|---|
| Identificador | \`ID_POLIGONO\` |
| Variable objetivo | \`RENDIMIENTO_T_HA\`, toneladas por hectárea (t/ha) |
| Universo total | 197 parcelas |
| Parcela con rendimiento conocido | 138, identificadas como \`ENTRENAMIENTO\` |
| Parcela por estimar | 59, identificadas como \`PREDICCION\` |
| Evidencia observable | Geometría, satélites, índices de vegetación, clima, suelo, relieve y variables derivadas, según disponibilidad y trazabilidad |

Formalmente, si \(L\) representa las parcelas con rendimiento conocido y \(U\) las 59 pendientes, disponemos de \(X_L\), \(X_U\) y \(y_L\), pero **no** de \(y_U\). Buscamos una estimación de \(y_U\) con el menor error posible respecto de la referencia aún no observada. Esto es una **regresión transductiva de objetivos fijos**: la identidad y las covariables de las 59 parcelas ya se conocen; no estamos afirmando haber construido un predictor validado para cualquier parcela futura de México.

**Límite estricto:** usar las covariables de las 197 parcelas para aprender representaciones, distancias, PCA o estructura espacial es compatible con el planteamiento transductivo. Utilizar los rendimientos ocultos de las 59, o los rendimientos de una partición reservada para entrenamiento supervisado, no lo es.

Para el contrato completo, véase [Objetivo transductivo](TRANSDUCTIVE_OBJECTIVE.md).

## 2. Datos: de observaciones repetidas a características por parcela

### 2.1 Fuentes originales y primera tabla

- **BASIC:** aproximadamente 107 666 observaciones × 96 columnas, de Sentinel-2/Landsat y sus índices, entre 2022 y 2025.
- **PRO:** aproximadamente 47 804 observaciones × 20 columnas, principalmente Planet y variables de vegetación en 2025.
- Datos geográficos, climáticos, edáficos, topográficos y administrativos; cada integración tiene sus reglas de unidad, fecha, cobertura y unión espacial.
- La primera etapa produjo tablas de **una fila por parcela** (\`parcel_features_*.csv\`), con agregaciones temporales, resúmenes, índices y metadatos. La tabla convencional simplifica el entrenamiento, pero comprime información temporal.

La distinción \`clean\` / \`competition\` en los archivos históricos **describe conjuntos de características y disponibilidad temporal**, no dos grupos independientes de parcelas para entrenamiento/prueba. En los experimentos posteriores se utilizó el universo de características autorizado por el planteamiento transductivo.

### 2.2 Información adicional de Checkpoint 07

Para comprobar si nuevas mediciones físicas aportaban señal se preparó una capa adicional:

| Fuente | Uso propuesto/implementado | Precaución |
|---|---|---|
| NASA HLS v2 | Secuencias y resúmenes espectrales por parcela; chips multitemporales para Prithvi | Resolución y nubes; una observación por fecha no equivale a un rendimiento nuevo |
| SMAP L3 Enhanced | Humedad superficial y estadísticas temporales/rezagos, con QA | Resolución ~9 km; varias parcelas comparten celda |
| Sentinel-1 GRD | Radar y propiedades de retrodispersión como nueva fuente disponible | No atribuir su inclusión en un modelo si no consta en su matriz final |
| AgERA5 v2 | Variables meteorológicas diarias disponibles para enriquecimiento | Distinguir descarga de uso efectivo en los predictores |
| Prithvi-EO-2.0-300M-TL | Extractor preentrenado de representaciones HLS, con pesos congelados | No se entrenan 300 millones de parámetros con 138 etiquetas |

La construcción 07B generó aproximadamente **21 445 características numéricas por parcela**, incluyendo **180 variables semilla** y **12 000 productos de interacción**, con selección inicial basada en \(X\), sin utilizar rendimientos. Los artefactos locales registraron **29 592 observaciones HLS por parcela/fecha** y **84 316 registros parcela/sobrevuelo SMAP**; no son 29 592 ni 84 316 etiquetas supervisadas. La disponibilidad de Sentinel-1 y AgERA5 dentro del proyecto **no autoriza** a afirmar que todos esos productos se usaron efectivamente en los cuatro modelos de la comparación publicada.

Los archivos crudos y los pesos pesados permanecieron en la estación de trabajo; los CSV ligeros de resultados sí se publicaron. Véanse [Checkpoint 07A](CHECKPOINT_07A_PROCESSING.md), [Checkpoint 07B](CHECKPOINT_07B_HIGH_DIMENSIONAL.md) y [compartición compacta de 07](CHECKPOINT_07_COMPACT_SHARING.md).

## 3. Recorrido de los experimentos: por qué llegamos a Local04D

| Etapa | Pregunta central | Resultado que guió la siguiente etapa |
|---|---|---|
| 02 | ¿Qué variables y particiones tenemos? | Inventario, controles de integridad y primeras referencias de RMSE |
| 03A–03B | ¿Aportan rasgos agronómicos, trayectorias y expresiones empíricas? | Capas deterministas y descubrimiento de expresiones dentro del fold |
| 03C–03D | ¿Hasta dónde llegan PLS, Ridge, CatBoost, árboles y ensambles globales? | Alternativas globales útiles, pero sin ventaja concluyente bajo la validación espacial objetivo |
| 04A | ¿Cómo se relacionan las 59 parcelas con las 138 etiquetadas? | Distancias espaciales, similitud agronómica/temporal, densidad y soporte |
| **04B** | ¿Cómo simular el conjunto de 59 sin conocer sus rendimientos? | Diseño repetido de **pseudo-objetivos target-matched** y pruebas de robustez |
| **04D.1** | ¿Funcionan mejor regresiones locales y grafos ajustados a vecindades? | Selección de **Local04D**: Ridge local con vecindad geográfica |
| 04E–04F | ¿Ayudan información externa, grafos o mezclas? | Pequeñas mejoras en tablas de desarrollo, sin selección robusta; Local04D congelado |
| 05 | ¿Mejoran un *stack* o mixture-of-experts global/local? | Ganancias nominales insuficientes tras leave-one-split-out; sin promoción |
| 06A–06F | ¿Son corregibles los errores concentrados en los rendimientos extremos y centrales? | Existen patrones, pero sus correcciones no mejoran de forma estable |
| **07** | ¿Ayudan más datos HLS/SMAP, CatBoost, Local07 y Prithvi preentrenado? | Los tres rivales y los blends probados quedaron detrás de Local04D en desarrollo |
| **08** | ¿Era injusta la comparación porque Local04D y Local07 se seleccionaron distinto? | Nueva validación con **búsqueda interna para ambos**; Local04D fijo vuelve a quedar primero |

Los documentos de cada checkpoint explican todos los experimentos descartados. Este archivo integra sus conclusiones, pero **no sustituye** sus manifiestos ni scripts ejecutables.

## 4. Qué es la validación con pseudo-objetivos

No conocemos el rendimiento de las 59 parcelas reales; por tanto, no podemos calcular todavía su RMSE. Para aproximarnos a esa situación reservamos temporalmente valores **dentro de las 138 parcelas con \(y\) conocido**.

Un split de desarrollo contiene:

\`\`\`text
138 parcelas con y observado antes de particionar
               |
           split X-only
               |
      97 pseudo-entrenamiento        41 pseudo-objetivos
      X e y disponibles              X disponible; y oculto al ajuste
               |                         |
               +-------- entrenar --------+
                              |
                     predecir los 41 y
                              |
             revelar sus y SOLO para evaluar
\`\`\`

Por diseño, un modelo entrenado en esos 97 rendimientos tiene que predecir otros 41 sin utilizar sus valores durante el ajuste. Se permite estudiar la geometría y características \(X\) de los 197, incluidas las 41 reservadas, **sólo en operaciones no supervisadas** autorizadas. Cualquier transformación *supervisada*, selección de parámetros mediante RMSE o entrenamiento del modelo debe permanecer dentro del conjunto de entrenamiento de su fold correspondiente.

### 4.1 Métricas reportadas

Para un conjunto de predicciones observadas, el error cuadrático medio en raíz es:

\`\`\`text
RMSE = sqrt( mean( (y_observado - y_estimado)^2 ) )
\`\`\`

Las salidas distinguen:

- **RMSE pooled:** se juntan las predicciones de los 16 splits y se calcula un único RMSE.
- **RMSE mean split:** media aritmética de los 16 RMSE individuales.
- **RMSE worst split:** el mayor RMSE de uno de los 16 splits.

En nuestro diseño, cada split aporta 41 pronósticos, de modo que **16 × 41 = 656 evaluaciones por modelo**. Sin embargo, **las parcelas se repiten**: hubo 125 parcelas distintas entre los pseudo-objetivos de Checkpoint 07 y 132 en Checkpoint 08. No hay 656 parcelas independientes ni 16 muestras experimentales independientes. El RMSE pooled pondera más a las parcelas que aparecen más veces entre los splits.

**Regla de comparación:** los RMSE son directamente comparables entre modelos solamente si se calculan sobre las **mismas predicciones objetivo, mismos splits y rendimientos de referencia**. No se comparan automáticamente los RMSE de 03D, 07 y 08 como si procedieran de una única muestra.

## 5. Cómo se generó la validación original: Checkpoint 04B

El generador está en \`src/geocebada/evaluation/checkpoint04b.py\`, función \`generate_target_matched_splits\`. Su contrato quedó congelado en \`reports/checkpoint_04b/pseudo_competition_splits.csv\` y en \`configs/checkpoint04b.yaml\`.

### 5.1 Por qué no usar sólo K-fold aleatorio

Las 59 parcelas de destino pueden tener una distribución distinta de localidades, características agrícolas o distancias respecto de las etiquetadas. Separar al azar 30 % de las 138 podría producir un conjunto fácil, pero **poco parecido al conjunto de destino**. El protocolo principal buscó que los 41 pseudo-objetivos fueran similares a las 59 parcelas reales **en variables observables**, sin utilizar sus rendimientos.

### 5.2 Construcción verificable de los 16 splits

1. Se calculan descriptores \(X\)-only de todas las parcelas: distancia geográfica al conjunto etiquetado, distancia agronómica, mejor similitud temporal y probabilidad adversarial de pertenecer al conjunto de destino. Se estandariza el perfil utilizado para el emparejamiento.
2. Se determina la composición de **estados** entre las 59 parcelas reales y se asignan cuotas enteras para 41 pseudo-objetivos mediante el método de Hamilton.
3. Dentro de cada estado, los candidatos etiquetados reciben pesos de muestreo según la cercanía de su perfil a los perfiles de las 59 reales y el ajuste de frecuencias municipales; no interviene \(y\).
4. Para cada repetición se generan **80 máscaras candidatas**. Se evalúa su discrepancia respecto de la distribución de destino —incluida la distancia del perfil y la discrepancia municipal— y se penaliza la superposición con máscaras elegidas anteriormente (Jaccard).
5. Se conserva una máscara y se repite hasta completar **16 splits** (\`target_matched_01\` a \`target_matched_16\`), siempre con **97 pseudo-entrenamiento / 41 pseudo-objetivos**.

**Semilla del generador:** \`20260920\`. Los parámetros exactos, como temperatura \`0.85\`, suavizado municipal \`0.02\` y penalización de diversidad \`0.08\`, están en el YAML correspondiente.

El diseño se contrastó contra un muestreo \`state_random\`, además de pruebas de estrés \`fold_state_stratified\` y \`fold_municipality_grouped\`. La discrepancia municipal de variación total fue aproximadamente **0.0547** para \`target_matched\` y **0.0889** para \`state_random\`; por eso se adoptó el primero para la selección principal. La validación agrupada por municipio sigue siendo útil como prueba de transferencia a grupos geográficos ausentes del entrenamiento, pero **no** reemplaza el objetivo principal.

**Importante:** los 16 splits fueron seleccionados para imitar las covariables de las 59 parcelas; **no** son un muestreo probabilístico independiente de fincas nacionales.

## 6. Qué es exactamente Local04D y cómo se eligió

En 04D.1 se evaluaron **626 métodos/variantes** de predicción local, grafos y baselines sobre las máscaras del 04B. Se evaluaron también familias de estrés y una selección leave-one-split-out (LOSO).

El predictor finalmente conservado se identifica como:

\`\`\`text
LocalRidge_C4_all_deterministic_Geo_k24_a30_p1
\`\`\`

Su construcción para una parcela de consulta \(q\) es:

1. Representar las parcelas mediante los componentes **C4** deterministas de 04A, obtenidos con procedimientos \(X\)-only transductivos (PCA en el universo de 197 \(X\)).
2. Medir distancias geográficas de los centroides mediante **haversine**.
3. Elegir los **24** vecinos más cercanos que sí tienen \(y\) disponible en ese fold.
4. Ajustar una regresión Ridge ponderada, **una regresión local distinta para cada parcela a predecir**, con \(\alpha=30\) y pesos inversamente proporcionales a la distancia elevada a \(p=1\):

\`\`\`text
w_j(q) = 1 / (d_geo(q,j) + 1e-6)^p

Ridge local: minimizar
    suma_{j en 24 vecinos} w_j(q) * (y_j - b0 - X_j b)^2
    + alpha * ||b||²
\`\`\`

No se está ajustando un único Ridge global sobre las 138 parcelas: se construye **un vecindario y un ajuste específico para cada predicción**.

La configuración fija es \(k=24\), \(\alpha=30\), \(p=1\), con las distancias geográficas normalizadas según la implementación original.

En la evaluación original:

- **RMSE promedio de los 16 splits:** 0.487845 t/ha.
- **RMSE pooled:** 0.495741 t/ha.
- **Peor RMSE por split:** 0.694700 t/ha.

Se detectaron modelos cercanos de rendimiento parecido, y Graph04D mostró mejor resistencia en algunos escenarios agrupados por municipio. Pese a ello, el peso de Graph y otras correcciones escogidas mirando la tabla completa no mejoraron de manera consistente bajo LOSO. Por eso se congeló **Local04D sin mezcla**.

**Diferencia entre selección y evaluación:** como la configuración 04D surgió de explorar muchas opciones en estas mismas 16 máscaras, su RMSE histórico puede contener **optimismo por selección**. El LOSO empleado para algunas reglas de selección/mezcla reduce el optimismo de escoger una regla con el split que se puntúa; no convierte automáticamente toda la exploración anterior en un experimento anidado independiente. Esta limitación motivó 08.

Implementaciones: \`src/geocebada/evaluation/checkpoint04d1.py\`, \`tools/run_checkpoint_04d1.py\`; análisis: [04D.1](CHECKPOINT_04D1_FINDINGS.md), [04F](CHECKPOINT_04F_FINDINGS.md).

## 7. Qué se aprendió antes del nuevo bloque satelital: Checkpoints 05 y 06

**Checkpoint 05** cruzó especialistas globales (PLS, Ridge, CatBoost y ExtraTrees) con Local04D/Graph04D mediante ensambles y stacking cross-fitted. El blend más favorable en desarrollo era **90 % Local04D + 10 % E13** (E13 = PLS + CatBoost): RMSE pooled **0.495261** frente a **0.495741**. La ventaja era pequeña y desapareció en la selección LOSO (pooled del selector: **0.495933**); no se promovió.

**Checkpoint 06** analizó residuos y propuestas de calibración para centro/colas. Entre sus diagnósticos:

- El **10 %** de parcelas con mayor error acumulaba aproximadamente **62.2 %** de la suma de errores cuadrados bajo la agregación del análisis.
- Había sobreestimación de rendimientos bajos y subestimación de rendimientos altos.
- Algunos clasificadores identificaban regímenes extremos usando señales observables, pero eso no demostró que los errores pudieran corregirse de forma estable.
- Los clusters agronómicos \(X\)-only no explicaron los residuos.
- Los expertos de centro, correcciones por colas y abstención selectiva llegaron a mostrar mejoras en la tabla completa, pero **fallaron las pruebas LOSO**.

No se promovió una corrección. Es importante distinguir **rendimiento reportado** de verdad física: errores de medición, estimación o registro podrían explicar parte de los residuos, pero **no han sido demostrados** como causa de las dos observaciones anómalas. Además, mejorar una estimación del rendimiento físico latente no necesariamente reduce el RMSE contra rendimientos reportados con error.

Fuentes: [Checkpoint 05](CHECKPOINT_05_FINDINGS.md), [Checkpoint 06](CHECKPOINT_06_FINDINGS.md), [Checkpoint 06F](CHECKPOINT_06F_FINDINGS.md).

## 8. Checkpoint 07: nuevas variables y tres rivales bajo la MISMA validación externa

07 utiliza **exactamente las 16 máscaras del 04B** para comparar Local04D con tres nuevas ramas. Cada rama usa los mismos 41 pseudo-objetivos de un split. Los rendimientos de estos 41 no se utilizan en la búsqueda interna del rival.

| Modelo | Qué hace | Selección interna |
|---|---|---|
| **Local04D congelado** | Predicciones guardadas de la regla seleccionada en 04D | No se vuelve a seleccionar en 07 |
| **CatBoost07** | Árboles boosting sobre características HLS/SMAP de 07B | Optuna, típicamente **48 trials** por split, con **3-fold interno** de los 97 |
| **Local07** | Ridge local por parcela, con vecindad mixta por geografía y características de 07B | Optuna, típicamente **48 trials**, **3-fold interno** |
| **Prithvi07** | Prithvi-EO-2.0-300M-TL preentrenado como encoder HLS congelado; cabezal Ridge | Rejilla interna de componentes PCA y \(\alpha\), con **3-fold interno** |

Local07 busca entre representaciones \`raw256\`, \`pca16\`, \`pca32\`, \`pca64\`; escoge número de vecinos, regularización Ridge y proporción de distancia geográfica. En su modelo, las distancias geográficas y de representación se escalan y se combinan; **no es el mismo procedimiento de vecinos/pesos que Local04D**.

CatBoost examina representaciones \`raw256\`/\`raw512\` o PCA de dimensión 16/32/64; sólo las etiquetas del pseudo-entrenamiento pueden entrar en la búsqueda. Prithvi no se reajusta a las 138 etiquetas en su encoder; se entrenan sus cabezales Ridge con fold interno. El bloque Prithvi explora rangos PCA **8, 16, 32, 64** y \(\alpha\) **0.1, 1, 10, 100, 1000**.

El diseño es **anidado para la selección de los nuevos rivales de 07**, pero la referencia Local04D fue elegida históricamente sobre la misma familia de splits. Por ello la comparación de desarrollo es emparejada en las parcelas, aunque **no perfectamente simétrica en historia de selección**.

### 8.1 Ensambles sin nuevos parámetros supervisados

Para cada rival \(m\) también se probaron pesos **fijos**, no ajustados con los rendimientos pseudo-objetivo:

\`\`\`text
estimacion_blend = (1 - w) * Local04D + w * rival
w = 0.10, 0.25, 0.50
\`\`\`

Estos 9 blends no son modelos de corrección residual entrenados con los 41 resultados reservados.

### 8.2 Resultados completos de Checkpoint 07

**Protocolo:** 16 splits \`target_matched\`, 656 pronósticos repetidos, **125 parcelas únicas**. Tabla de origen: [\`three_model_summary.csv\`](../reports/checkpoint_07/shared/three_model_summary.csv). Todos los valores son RMSE en t/ha y corresponden al mismo banco de desarrollo.

| Modelo | RMSE pooled | RMSE medio por split | Peor split |
|---|---:|---:|---:|
| **Local04D** | **0.495741** | **0.487845** | **0.694700** |
| Local07 | 0.523081 | 0.515800 | 0.712418 |
| Prithvi07 | 0.556672 | 0.551219 | 0.721866 |
| CatBoost07 | 0.561325 | 0.551931 | 0.750689 |
| Local04D + 10 % Local07 | 0.496387 | 0.488477 | 0.694174 |
| Local04D + 25 % Local07 | 0.498252 | 0.490357 | 0.694352 |
| Local04D + 50 % Local07 | 0.503715 | 0.495939 | 0.697223 |
| Local04D + 10 % Prithvi07 | 0.496391 | 0.488610 | 0.694036 |
| Local04D + 25 % Prithvi07 | 0.499773 | 0.492255 | 0.694471 |
| Local04D + 50 % Prithvi07 | 0.511663 | 0.504758 | 0.698998 |
| Local04D + 10 % CatBoost07 | 0.496591 | 0.488606 | 0.696274 |
| Local04D + 25 % CatBoost07 | 0.500403 | 0.492314 | 0.699169 |
| Local04D + 50 % CatBoost07 | 0.513328 | 0.505068 | 0.705401 |

**Lectura:** todos los rivales autónomos y sus blends subieron el RMSE pooled respecto de Local04D. Alguno redujo mínimamente el peor RMSE de split, pero **no mejoró** la métrica principal. Se mantuvo la predicción congelada de las 59 parcelas.

Código y parámetros: \`tools/run_checkpoint_07_three.py\`, \`src/geocebada/evaluation/checkpoint07c.py\`, \`checkpoint07_local.py\`, \`checkpoint07_prithvi_head.py\`; detalles: [tres modelos de 07](CHECKPOINT_07_THREE_MODELS.md).

## 9. Por qué hicimos Checkpoint 08 y qué cambia en la validación

La pregunta pendiente no era «¿Local07 puede ganar después de probar otra mezcla?», sino:

> ¿Qué sucede cuando **Local04D y Local07 reciben ambos búsqueda de hiperparámetros dentro del entrenamiento**, y después se evalúan en máscaras que no se utilizaron al seleccionar la familia Local04D?

### 9.1 Un **nuevo banco de confirmación**, pero no la misma construcción de 04B

08 **no copió** las 16 máscaras de 04B. Generó otras **16**, de nuevo con **97/41**, usando \`src/geocebada/evaluation/checkpoint08_local.py\`, función \`label_blind_memberships\`, y semilla \`20261012\`.

Su generador funciona así:

1. Define estratos observables \`estado|municipio\` en las 138 parcelas etiquetadas y las 59 de destino.
2. Calcula cuotas de 41 pseudo-objetivos para aproximar las frecuencias de estratos de las 59, con asignación entera voraz y límite de disponibilidad de parcelas etiquetadas.
3. Genera **160 máscaras candidatas por repetición**, con muestreo aleatorio dentro de cada estrato.
4. Elige la máscara que minimiza la **máxima similitud Jaccard** con las 16 máscaras históricas y con las de confirmación ya elegidas.
5. Guarda \`confirm_01\` a \`confirm_16\` antes de entrenar modelos; el mismo banco se usa para ambos rivales y la referencia fija.

**Diferencia relevante:** 04B usaba afinidad basada en **perfiles \(X\)** además de cuotas estatales y distribución municipal. 08 utiliza **estratificación geográfica por estado/municipio y control de superposición**, pero **no** reoptimiza las distancias de perfil temporal/agronómico de 04B. Por tanto, 08 es una confirmación con un **protocolo emparentado, no idéntico**. La comparación de modelos **dentro de 08** sí es estrictamente emparejada.

En estas máscaras aparecen **132 parcelas diferentes** entre los 656 pronósticos de cada modelo. La generación fue independiente de \(y\), pero **sigue reutilizando el mismo universo de 138 rendimientos reportados**: no equivale a disponer de 16 campañas agrícolas nuevas.

### 9.2 Validación anidada: cada nivel responde una pregunta distinta

\`\`\`text
Para cada uno de los 16 splits de confirmación:

       138 parcelas originalmente etiquetadas
                         |
              EXTERIOR: 97 / 41
                         |
            97 y disponibles para ajustar
                         |
             INTERIOR: KFold de 3 partes
                 /       |       \
             fold 1    fold 2    fold 3
                 \       |       /
         Optuna busca el hiperparámetro que
         minimiza RMSE usando sólo los 97
                         |
         Elegir y volver a ajustar con los 97
                         |
              Predecir los 41 externos
                         |
         Calcular RMSE externo una única vez
\`\`\`

El **fold interno** no mide el rendimiento final del modelo: sirve para **escoger su configuración** sin consultar los 41 rendimientos externos. El **fold externo** mide cómo se comporta ese *procedimiento de selección + reentrenamiento* cuando se le presentan pseudo-objetivos reservados.

### 9.3 Búsqueda amplia, pero no rejilla exhaustiva

En cada split de 08 se utilizan por defecto **96 trials de Optuna TPE por familia**, sobre espacios definidos antes de obtener sus resultados:

| Dimensión | Local04D_nested | Local07_nested |
|---|---|---|
| Representación | C4 original congelada | \`raw256\`, \`raw512\`, \`pca16\`, \`pca32\`, \`pca64\` |
| Vecinos \(k\) | 8, 12, 16, 20, 24, 30, 36, 48 | Los mismos |
| \(\alpha\) | 0.3, 1, 3, 10, 30, 100, 300, 1000 | Los mismos |
| Cuarto parámetro | Potencia de distancia: 0, 0.5, 1, 1.5, 2 | Peso geográfico: 0, 0.25, 0.5, 0.75, 1 |
| Espacio total | **320** configuraciones posibles | **1 600** configuraciones posibles |
| Configuraciones ensayadas por split | **96 trials**, no las 320 | **96 trials**, no las 1 600 |

La configuración Local04D original \`k=24, alpha=30, power=1\` se incluye explícitamente como un trial inicial. El preprocesamiento de las representaciones nuevas de 07B —imputación, selección de variables, estandarización y PCA— se construye a partir de las muestras de entrenamiento de cada fold, no de sus etiquetas de prueba.

Se comparan tres salidas:

- **Local04D_frozen:** regla original fija, ajustada usando los 97 rendimientos de cada split de confirmación; es el control principal.
- **Local04D_nested:** la misma familia de regresión local original, pero con \((k,\alpha,p)\) seleccionado por 3-fold interno.
- **Local07_nested:** la nueva familia y representación 07B, con representación, \(k\), \(\alpha\) y peso geográfico seleccionados por el mismo procedimiento anidado.

El runner comprueba la **paridad numérica** de la reconstrucción de Local04D contra las predicciones históricas guardadas antes de permitir la búsqueda extensa. Archiva estudios Optuna por split/familia en SQLite y usa huellas SHA-256 de entradas/código/configuración para impedir mezclas entre corridas incompatibles.

Código: \`tools/run_checkpoint_08_local.py\`; especificación: [Checkpoint 08](CHECKPOINT_08_NESTED_LOCAL.md).

## 10. Resultados definitivos de la comparación de Checkpoint 08

**Protocolo:** 16 máscaras \`confirm_\`, 656 pronósticos repetidos, 132 parcelas únicas. Fuente publicada: [\`checkpoint08_comparison_summary.csv\`](../reports/checkpoint_07/shared/checkpoint08_comparison_summary.csv).

| Modelo | RMSE pooled (t/ha) | RMSE medio por split | Peor split |
|---|---:|---:|---:|
| **Local04D_frozen** | **0.481936** | **0.477557** | **0.581761** |
| Local04D_nested | 0.493943 | 0.488551 | 0.653956 |
| Local07_nested | 0.510070 | 0.506025 | 0.608748 |

**Resultado principal:** en estas máscaras de confirmación, la configuración fija original superó tanto a su variante retuneada como a Local07 retuneado.

El cambio **0.495741 (07) → 0.481936 (08)** del control fijo **no** significa que hayamos entrenado una versión mejor: son **particiones diferentes**, obtenidas bajo procedimientos de selección de pseudo-objetivos diferentes. Es una segunda medición de una misma regla, no una «reducción de RMSE» imputable a un nuevo modelo.

### 10.1 Comparación emparejada e incertidumbre

Para cada predicción de la misma pareja \((split, parcela)\), se calculan los errores cuadrados de cada modelo. Se define la diferencia:

\`\`\`text
delta_MSE = mean( error_cuadrado_del_rival
                  - error_cuadrado_de_Local04D_frozen )
\`\`\`

Si \`delta_MSE < 0\`, el rival mejora el promedio cuadrático; si es positivo, empeora.

El programa realizó **1 000 remuestreos bootstrap** con semilla \`20261012\`, usando como unidad el **ID único de parcela**. En cada remuestreo se arrastran *todas* las apariciones de una parcela a través de los splits; por eso no se simulan 656 observaciones independientes. El intervalo es percentil 2.5 % a 97.5 % de esos remuestreos.

Fuente publicada: [\`checkpoint08_paired_cluster_bootstrap.csv\`](../reports/checkpoint_07/shared/checkpoint08_paired_cluster_bootstrap.csv).

| Rival frente al control fijo | \(\Delta\) MSE (t/ha)² | Intervalo bootstrap 95 % |
|---|---:|---|
| Local04D_nested | **+0.011718** | **[+0.003461, +0.020479]** |
| Local07_nested | **+0.027909** | **[+0.008857, +0.048316]** |

Ambos intervalos son **positivos**, apoyando la conclusión de que, en este experimento, ajustar de nuevo los hiperparámetros no produjo una mejora estable respecto del control fijo.

**Límite estadístico del bootstrap:** agrupar por parcela evita tratar sus predicciones repetidas como independientes. No corrige automáticamente dependencia espacial entre parcelas, dependencia por reutilización del conjunto de entrenamiento entre splits, sesgos en los rendimientos reportados ni la selección previa de familias y parámetros. Los intervalos expresan **incertidumbre de esta comparación sobre estas parcelas**, no garantía del error real de las 59 parcelas ocultas.

### 10.2 Interpretación de los resultados negativos

La selección de hiperparámetros es una decisión estadística también sujeta a variación. Con sólo 97 etiquetas por fold externo, una búsqueda amplia puede escoger alternativas que ganan por azar o por particularidades de sus folds internos. La hipótesis de un mejor vecindario en la representación HLS/SMAP no quedó apoyada por RMSE en los splits externos.

**No concluimos** que HLS, SMAP, Prithvi, CatBoost o PCA carezcan de valor agronómico en general; concluimos que **las implementaciones y protocolos evaluados aquí no mejoraron el modelo elegido para las 59 parcelas fijas**.

## 11. Controles de integridad, trazabilidad y reproducción

| Qué se controla | Cómo se controla |
|---|---|
| El universo de parcelas | IDs únicos; 197 registros; 138 con \(y\), 59 sin \(y\) |
| Que los pseudo-objetivos no entrenen el modelo | Membresía explícita \`pseudo_train\` / \`pseudo_target\` o \`confirm\` |
| Que el ajuste de hiperparámetros no use \(y\) externo | 3-fold interno restringido a los 97 \(y\) disponibles |
| Que todos los modelos se comparen en el mismo banco | Cobertura emparejada por \`split_id\` e \`ID_POLIGONO\` |
| Que se reproduzca el predictor original | Paridad numérica Local04D con archivo histórico |
| Que una interrupción no cambie el experimento | Optuna SQLite, salidas parciales atómicas y comprobación de huellas |
| Que los artefactos crudos no se confundan con resultados públicos | Directorios locales ignorados por Git; exportación ligera seleccionada |
| Que las 59 etiquetas desconocidas no se puntúen | No existen en la tabla de etiquetas utilizada para calcular RMSE |

### 11.1 Archivos del repositorio que permiten auditarlo

| Pregunta | Archivo |
|---|---|
| ¿Cuáles fueron exactamente los 16 splits de desarrollo? | \`reports/checkpoint_04b/pseudo_competition_splits.csv\` |
| ¿Cómo se generan? | \`src/geocebada/evaluation/checkpoint04b.py\` y \`configs/checkpoint04b.yaml\` |
| ¿Cómo ajusta Local04D? | \`src/geocebada/evaluation/checkpoint04d1.py\` |
| ¿De dónde salen las predicciones congeladas de Local04D? | \`reports/checkpoint_06/parcel_oof_residuals.csv\` |
| ¿Cómo se ajustan los tres nuevos modelos? | \`src/geocebada/evaluation/checkpoint07c.py\`, \`checkpoint07_local.py\`, \`checkpoint07_prithvi_head.py\` |
| ¿Dónde está el resumen completo de 07? | \`reports/checkpoint_07/shared/three_model_summary.csv\` |
| ¿Cómo se generan los 16 splits nuevos y la búsqueda anidada? | \`src/geocebada/evaluation/checkpoint08_local.py\` |
| ¿Cuál es el comando reproducible de 08? | \`tools/run_checkpoint_08_local.py\` |
| ¿Dónde están las métricas agregadas de 08? | \`reports/checkpoint_07/shared/checkpoint08_comparison_summary.csv\` |
| ¿Dónde están los intervalos emparejados de 08? | \`reports/checkpoint_07/shared/checkpoint08_paired_cluster_bootstrap.csv\` |
| ¿Dónde está la estimación final de las 59? | \`reports/checkpoint_05/final_predictions.csv\` |
| ¿Cómo se valida su procedencia? | \`reports/checkpoint_05/model_manifest.json\` |

Los archivos detallados de Checkpoint 08, como los estudios \`.sqlite\`, los pronósticos por split y las máscaras \`confirmation_membership.csv\`, se conservan **localmente** bajo \`reports/checkpoint_08/\`, que está ignorado por Git. Las métricas agregadas sí fueron publicadas. Por tanto, el repositorio contiene el **algoritmo para regenerar la confirmación** y sus resultados compartidos, pero no todas las predicciones individuales de esa corrida.

### 11.2 Comandos para reproducir el experimento final

En la carpeta raíz de GeoC y dentro del ambiente \`geocebada\`, con la tabla local completa 07B:

\`\`\`powershell
git pull --ff-only origin main
conda activate geocebada

python tools/run_checkpoint_08_local.py --prepare
python tools/run_checkpoint_08_local.py --check
python tools/run_checkpoint_08_local.py --smoke
python tools/run_checkpoint_08_local.py --run --trials 96
python tools/run_checkpoint_08_local.py --summary
\`\`\`

**La regeneración completa de 07 no es requisito para leer este reporte.** Para repetir 08 sí debe estar disponible \`reports/checkpoint_07/highdim/expanded_parcel_features.csv\` y las características C4 de 04A. No debe confundirse un smoke test de un split con el resultado completo de 16 splits.

## 12. Decisión final y cómo se generan las 59 estimaciones

El método **no cambió** después de 07 ni 08:

\`\`\`text
Local04D
= LocalRidge_C4_all_deterministic_Geo_k24_a30_p1
\`\`\`

Para las **59 consultas reales**, se usan los **138 rendimientos disponibles** como conjunto de candidatos a vecinos. Para cada parcela destino se eligen sus **24 vecinos etiquetados más cercanos geográficamente**, se estima su Ridge local ponderado y se genera una predicción. La preparación \(X\)-only puede incorporar las características de las 197. No se produce una etiqueta artificial para entrenar con las 59 consultas.

La salida canónica sigue siendo:

\`\`\`text
reports/checkpoint_05/final_predictions.csv

ID_POLIGONO,RENDIMIENTO_T_HA
(59 filas; sin columna de rendimiento observado real de las 59)
\`\`\`

El archivo y \`reports/checkpoint_05/model_manifest.json\` documentan que se usaron 138 registros de entrenamiento y 59 de predicción, y que el método publicado es \`Local04D\`. Checkpoint 08 **no recalculó ni sustituyó automáticamente** estas 59 predicciones.

### Qué podemos afirmar

- Diseñamos un protocolo de pseudo-objetivos para aproximar la composición observable de las parcelas reales.
- Comparamos alternativas sobre los mismos splits y el mismo RMSE, evitando puntuarlas contra \(y\) que no estaba disponible para entrenamiento.
- Repetimos una comparación metodológicamente más simétrica en un banco nuevo de máscaras.
- La configuración Local04D fija quedó primero en ambos bancos, y en 08 la comparación por bootstrap frente a ambas variantes ajustadas favoreció al control.
- Por eso conservar la predicción canónica es una **decisión sustentada por la evidencia disponible**, no únicamente una preferencia por el modelo más simple.

### Qué todavía no podemos afirmar

- El RMSE exacto que obtendrán las **59 parcelas reales**: sus rendimientos de referencia no se han observado.
- Que las 656 predicciones de una corrida proceden de 656 parcelas independientes.
- Que las confirmaciones de 07 y 08 constituyen muestras agrícolas independientes entre sí.
- Que Local04D sea universalmente superior a otros métodos para todos los cultivos, años, regiones o protocolos.
- Que los residuos grandes prueban por sí solos errores de medición en el dato de referencia.
- Que toda fuente satelital descargada fue incorporada efectivamente en cada modelo evaluado.

## 13. Material recomendado para el informe y la página del proyecto

Para divulgación breve, conviene narrar **problema → datos satelitales y agronómicos → 138/59 parcelas → validación de 97/41 repetida 16 veces → modelos comparados → comparación final anidada → selección de Local04D → limitaciones**.

Una gráfica de barras de 07 y otra de 08 deberán aparecer **separadas**, con sus respectivos nombres de banco y métricas; no deben dibujarse todas juntas como si compartieran los mismos 41 pseudo-objetivos. Una figura de los folds anidados y otra con el mapa de vecinos de Local04D ayudan a explicar por qué el sistema es transductivo.

**Resumen de una frase para materiales públicos:** *Se contrastaron modelos locales, globales y representaciones satelitales mediante validación geoespacial de pseudo-objetivos y un experimento final con búsqueda anidada; el predictor local geográfico ya seleccionado mantuvo el menor error de validación, por lo que se conservaron sus estimaciones para las 59 parcelas sin rendimiento observado.*

---

### Lecturas y referencias internas

- [Objetivo transductivo y particiones](TRANSDUCTIVE_OBJECTIVE.md)
- [Resultados y diseño de 04B](CHECKPOINT_04B_FINDINGS.md)
- [Refinamiento local/grafos de 04D.1](CHECKPOINT_04D1_FINDINGS.md)
- [Cierre y regla final de 04F](CHECKPOINT_04F_FINDINGS.md)
- [Ensambles de Checkpoint 05](CHECKPOINT_05_FINDINGS.md)
- [Diagnóstico centro/colas de Checkpoint 06](CHECKPOINT_06_FINDINGS.md)
- [CatBoost, Local07, Prithvi y blends de 07](CHECKPOINT_07_THREE_MODELS.md)
- [Protocolo completo anidado de 08](CHECKPOINT_08_NESTED_LOCAL.md)
- [CSV agregado de 07](../reports/checkpoint_07/shared/three_model_summary.csv)
- [CSV agregado de 08](../reports/checkpoint_07/shared/checkpoint08_comparison_summary.csv)
- [Intervalos de 08](../reports/checkpoint_07/shared/checkpoint08_paired_cluster_bootstrap.csv)
