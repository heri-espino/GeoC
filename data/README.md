# Datos — Reto AgroCebada FIRA 2026

Esta carpeta contiene las fuentes oficiales del reto y los artefactos derivados del pipeline de datos. El objetivo predictivo es estimar el rendimiento agrícola de parcelas de cebada en Hidalgo, Puebla y Tlaxcala.

> **Estado actual:** el cierre reproducible de la fase de datos está documentado en `checkpoints/01_data/README.md`. Para el inventario operativo actualizado de fuentes oficiales/externas, joins y caveats, leer `docs/DATA_SOURCES.md`. Ese documento corrige y amplía cualquier descripción histórica de este README.

## Resumen del problema

La unidad de observación final es una **parcela georreferenciada**. El archivo oficial de partición contiene 197 parcelas:

- **138 parcelas (70.05 %) de entrenamiento**: el rendimiento real está disponible.
- **59 parcelas (29.95 %) de predicción/evaluación**: el rendimiento está oculto y debe ser estimado.

El split es **por parcela completa**, no un 70/30 dentro de cada parcela. Es decir, FIRA entrega el rendimiento de unas parcelas y oculta por completo el de otras.

Formalmente, para cada parcela \(i\):

\[
\hat y_i=f(X_i),
\]

con:

- \(y_i\): rendimiento real en toneladas por hectárea (`RENDIMIENTO_T_HA`),
- \(X_i\): variables satelitales, climáticas, topográficas, geométricas y otras covariables de la parcela,
- \(\hat y_i\): rendimiento predicho.

El rendimiento real del conjunto de predicción permanece resguardado por FIRA y se utilizará para la evaluación final.

---

## Estructura de `data/`

```text
data/
├── README.md
├── .ai_handoff
├── source/                 # fuentes oficiales, inmutables
│   ├── README.md
│   ├── tabular/
│   │   ├── Conjunto_datos_BASICO_AgroCebada2026.csv
│   │   ├── Conjunto_datos_PRO_AgroCebada.csv
│   │   └── ID_area_rendimiento_70_30_Reto_AgroCebada.csv
│   ├── geospatial/
│   │   └── Parcelas_Reto_AGC_CONJUNTO_70_30.zip
│   ├── climate/
│   │   ├── precipitation/
│   │   │   └── Reto_AgroCebada_CHIRPS_Precipitacion_2022_2025/...
│   │   └── temperature/
│   │       └── Reto_AgroCebada_Temperatura_2022_2025/...
│   └── topography/
│       └── Reto_AgroCebada_Topografia_INEGI_CEM4/...
├── raw/                    # ingestión/descompresión; no versionado
├── interim/                # limpieza, joins y features intermedias; no versionado
└── processed/              # tablas/matrices finales para modelado; no versionado
```

### Contrato de carpetas

| Carpeta | Propósito | ¿Se modifica manualmente? | ¿Se versiona? |
|---|---|---:|---:|
| `source/` | Archivos oficiales exactamente como fueron entregados | No | Sí |
| `raw/` | Copias descomprimidas, ingestión y formatos de trabajo | No; generar por código | No |
| `interim/` | Limpieza, uniones geoespaciales, agregaciones y features intermedias | No; generar por código | No |
| `processed/` | Dataset final model-ready y tablas de evaluación/predicción | No; generar por código | No |

Toda transformación reproducible debe vivir en `src/geocebada/` o scripts controlados por el proyecto. No colocar modelos entrenados, predicciones o resultados dentro de `source/`.

### Datos externos locales y manifiesto reproducible

Todo archivo generado o descargado bajo `data/raw/`, `data/interim/` y
`data/processed/` está excluido de Git. En particular,
`data/raw/external/` contiene insumos locales que pueden incluir GeoTIFF y CSV
grandes; no se deben añadir al repositorio ni borrar para limpiar el worktree.

El downloader obtiene sólo el entorno espacial de las parcelas cuando la fuente
permite subconjuntos:

```powershell
python tools\download_external_data.py `
  --sources inegi chirps soilgrids wapor `
  --workers 12
```

Para reintentar exclusivamente SoilGrids después de una interrupción o error:

```powershell
python tools\download_external_data.py --sources soilgrids --workers 4
```

El conjunto externo prioritario se considera adquirido localmente a 2026-09-17: municipios INEGI de Hidalgo/Puebla/Tlaxcala, CHIRPS diario, WaPOR, SoilGrids, CEM de alta resolución y SIAP incluyendo 2023. ERA5-Land es opcional y puede contener descargas parciales; no debe asumirse completo.

El estado versionable de los archivos locales se reconstruye sin versionar los
binarios con:

```powershell
python tools\build_external_data_manifest.py
```

Este comando actualiza `data/external_manifest.json` con rutas relativas,
conteos, tamaños, SHA256 y metadatos ligeros de ráster/NetCDF cuando están
disponibles. El script tolera fuentes ausentes para que cualquier clon del
repositorio pueda ejecutarlo.

| Fuente | Método | Credenciales / acción manual |
|---|---|---|
| INEGI municipios | API pública por terminal | No requiere cuenta |
| CHIRPS diario | COG público por terminal | No requiere cuenta |
| SoilGrids | WCS público por terminal | No requiere cuenta |
| WaPOR v3 | API pública mediante `wapordl` | No requiere cuenta |
| ERA5-Land | API CDS por terminal | Requiere cuenta CDS, aceptar licencia y `.cdsapirc` local no versionado |
| SIAP | Interfaz web | Descarga manual; conservar bajo `data/raw/external/SIAP/` |
| FIRA Agrocostos | Interfaz web | Descarga manual si se incorpora |
| INEGI CEM 4.0 | Área de descarga interactiva | Descarga manual; conservar bajo `data/raw/external/Inegi/` |

Nunca incluir tokens, claves ni el archivo `.cdsapirc` en el repositorio.

---

## 1. Tabla oficial de objetivo y split 70/30

Archivo:

```text
data/source/tabular/ID_area_rendimiento_70_30_Reto_AgroCebada.csv
```

Columnas observadas:

| Columna | Significado |
|---|---|
| `ID_POLIGONO` | Identificador oficial de parcela, por ejemplo `AGC_001` |
| `AREA_HA` | Superficie de la parcela en hectáreas |
| `RENDIMIENTO_T_HA` | Rendimiento real en ton/ha; ausente para predicción |
| `CONJUNTO` | `ENTRENAMIENTO` o `PREDICCION` |

Ejemplo conceptual:

```text
ID_POLIGONO,AREA_HA,RENDIMIENTO_T_HA,CONJUNTO
AGC_001,7.61,3.50,ENTRENAMIENTO
AGC_002,2.31,4.63,ENTRENAMIENTO
AGC_003,18.68,,PREDICCION
```

### Interpretación correcta

Para entrenamiento:

\[
y_i = \texttt{RENDIMIENTO\_T\_HA}
\]

es conocido.

Para predicción:

\[
y_i = ?
\]

pero sí conocemos el identificador, el área y las covariables asociadas a la parcela.

**No interpretar el 70/30 como 70 % de observaciones dentro de cada parcela.** La partición ocurre a nivel de parcela.

### Identificador canónico

En las fuentes oficiales la llave observada es `ID_POLIGONO`. Si internamente se decide renombrarla a `ID_parcela`, conservar una transformación explícita y reversible. Nunca perder el identificador original porque la entrega final debe mapear inequívocamente cada predicción con la parcela evaluada.

---

## 2. Geometría de las parcelas

Archivo:

```text
data/source/geospatial/Parcelas_Reto_AGC_CONJUNTO_70_30.zip
```

Contiene las parcelas georreferenciadas utilizadas por el reto. Las parcelas pertenecen a:

- Hidalgo,
- Puebla,
- Tlaxcala.

La geometría permite relacionar cada parcela con información ráster mediante estadísticas zonales. Para un ráster \(R_t(s)\) y una parcela \(P_i\), por ejemplo:

\[
\bar R_{i,t}=\frac{1}{|P_i|}\int_{P_i}R_t(s)\,ds.
\]

A partir de esto pueden obtenerse media, mediana, desviación estándar, mínimos, máximos, cuantiles u otras estadísticas espaciales por parcela.

---

## 3. Precipitación — CHIRPS

Ubicación:

```text
data/source/climate/precipitation/
```

Contenido verificado:

- fuente: **CHIRPS**,
- periodo: **enero de 2022 a diciembre de 2025**,
- 48 GeoTIFF mensuales,
- variable: precipitación acumulada mensual,
- unidad: `mm/mes`,
- resolución espacial: `0.05° × 0.05°` (aprox. 5 km),
- CRS: `EPSG:4326`,
- dimensiones de entrega: `65 × 72` píxeles.

Nomenclatura:

```text
PREC_AAAA_MM.tif
```

Ejemplos:

```text
PREC_2022_01.tif
PREC_2025_12.tif
```

Cada capa mensual fue calculada sumando los rásteres diarios correspondientes al mes.

Posibles features derivadas, sólo después de definir correctamente el ciclo agrícola:

- precipitación acumulada por etapa o temporada,
- media mensual,
- máximo mensual,
- variabilidad temporal,
- número/proporción de periodos secos,
- anomalías respecto de climatología, si se incorpora una referencia externa.

---

## 4. Temperatura — CHIRTS-ERA5

Ubicación:

```text
data/source/climate/temperature/
```

Contenido verificado:

- fuente: **CHIRTS-ERA5**,
- periodo: **enero de 2022 a diciembre de 2025**,
- 48 GeoTIFF de `Tmin`,
- 48 GeoTIFF de `Tmax`,
- total: **96 GeoTIFF**,
- unidad: grados Celsius (`°C`),
- resolución espacial: `0.05° × 0.05°` (aprox. 5 km),
- CRS: `EPSG:4326`,
- dimensiones: `65 × 72` píxeles,
- NoData: `-9999`.

Variables:

\[
T_{\min} = \text{promedio mensual de la temperatura mínima diaria},
\]

\[
T_{\max} = \text{promedio mensual de la temperatura máxima diaria}.
\]

Nomenclatura:

```text
Tmin_AAAA_MM.tif
Tmax_AAAA_MM.tif
```

Ejemplos:

```text
Tmin_2022_01.tif
Tmax_2025_12.tif
```

---

## 5. Topografía — INEGI CEM

Ubicación:

```text
data/source/topography/
```

Fuente: **INEGI, Continuo de Elevaciones Mexicano (CEM)**, utilizando insumos de Hidalgo, Puebla y Tlaxcala.

Productos principales:

### Elevación

```text
Elevacion_INEGI_CEM4_120m.tif
```

- variable: elevación del terreno,
- unidad: metros sobre el nivel del mar,
- resolución: 120 m.

### Pendiente

```text
Pendiente_INEGI_CEM4_120m_grados.tif
```

- variable: pendiente derivada del DEM,
- unidad: grados,
- resolución: 120 m,
- método: Horn 3×3.

Sistema de referencia de los productos topográficos:

```text
EPSG:6372 — México ITRF2008 / LCC
```

NoData: `-9999.0`.

Estas capas son esencialmente estáticas en el horizonte del reto. Pueden convertirse a variables por parcela como elevación media, dispersión de elevación, pendiente media, pendiente máxima, etc.

---

## 6. Datos tabulares satelitales / sensores remotos

Ubicación:

```text
data/source/tabular/
```

Archivos:

```text
Conjunto_datos_BASICO_AgroCebada2026.csv
Conjunto_datos_PRO_AgroCebada.csv
```

Estos archivos ya fueron auditados a nivel de esquema:

- **BASIC:** 107,666 × 96, 197 parcelas, Sentinel-2 + Landsat, 2022–2025, filas longitudinales parcela/fecha/sensor, 23 familias de índices × cuatro estadísticas, más ID/fecha/sensor/nubosidad.
- **PRO:** 47,804 × 20, 197 parcelas, Planet, 2025, NDVI/EVI/LAI/MSAVI × cuatro estadísticas, más ID/fecha/sensor/nubosidad.
- Las fechas son day-first.
- El missingness es fuertemente específico por sensor.
- Las filas satelitales no son muestras supervisadas independientes.
- Índices con el mismo nombre entre sensores no deben fusionarse como si fueran mediciones homogéneas sin una estrategia explícita.

Ver `docs/VARIABLES.md` y `docs/DATA_SOURCES.md` para el detalle actualizado.

---

## 7. Estructura temporal: punto crítico

El objetivo oficial corresponde al **ciclo abril–octubre 2025**. BASIC cubre 2022–2025, PRO cubre 2025 y el clima oficial cubre 2022–2025.

La unidad objetivo sigue siendo:

$$
\text{1 parcela} \rightarrow \text{1 rendimiento observado u oculto}.
$$

Las covariables satelitales y climáticas son longitudinales:

$$
\text{1 parcela} \rightarrow \{X_{i,t}\}.
$$

Para evaluación científicamente prospectiva todavía debe declararse un **cutoff operativo de predicción** y excluir información posterior a ese cutoff. Para el reto también se mantendrá un modo de features **competition** separado, donde puede evaluarse información pública de temporada completa si las reglas lo permiten y se documenta explícitamente.

Nunca utilizar los targets ocultos de las 59 parcelas.


---

## 8. ¿El split está estratificado?

No está documentado todavía que el 70/30 haya sido estratificado por estado, año, área u otra variable.

El archivo de split sólo proporciona `ID_POLIGONO`, `AREA_HA`, `RENDIMIENTO_T_HA` y `CONJUNTO`.

Por tanto, no debe suponerse que existe aproximadamente 70/30 dentro de Hidalgo, Puebla y Tlaxcala.

Debe verificarse mediante un `spatial join` de los polígonos con límites estatales y una tabla como:

| Estado | Train | Test | % Test |
|---|---:|---:|---:|
| Hidalgo | por calcular | por calcular | por calcular |
| Puebla | por calcular | por calcular | por calcular |
| Tlaxcala | por calcular | por calcular | por calcular |

También conviene comparar las distribuciones de `AREA_HA`, topografía y demás covariables entre entrenamiento y predicción para detectar **covariate shift**.

---

## 9. Forma lógica del dataset final

Una representación conceptual útil es:

```text
parcela
├── ID y área
├── geometría
├── variables satelitales temporales
├── precipitación temporal
├── Tmin / Tmax temporales
├── topografía estática
└── rendimiento
    ├── conocido → ENTRENAMIENTO
    └── oculto   → PREDICCION
```

Dependiendo de la estrategia de modelado, el dataset procesado puede tomar al menos dos formas.

### A. Tabla de features agregadas

Una fila por parcela:

```text
ID_POLIGONO | AREA_HA | precip_* | tmin_* | tmax_* | topo_* | sat_* | y
```

Adecuada para modelos tabulares como regresión regularizada, Random Forest, XGBoost, LightGBM o CatBoost.

### B. Tensor parcela × tiempo × variable

\[
X \in \mathbb{R}^{N\times T\times p},
\]

con:

- \(N\): número de parcelas,
- \(T\): número de pasos temporales válidos,
- \(p\): número de variables por paso.

Adecuada si la frecuencia y alineación temporal justifican modelos secuenciales o representaciones temporales explícitas.

Con sólo 138 etiquetas de entrenamiento, cualquier modelo de alta capacidad debe validarse con especial cuidado para evitar sobreajuste.

---

## 10. Evaluación y entrega

Las bases del reto indican que el desempeño predictivo es el componente principal de la primera etapa y puede evaluarse mediante métricas como:

- RMSE,
- MAE,
- \(R^2\),
- u otras definidas por el Comité Organizador.

La entrega debe mantener una correspondencia inequívoca entre el identificador oficial de cada parcela evaluada y su rendimiento predicho en `ton/ha`.

Nunca usar las 59 parcelas de predicción para selección supervisada de hiperparámetros o validación, ya que sus etiquetas reales no están disponibles.

---

## 11. Reglas de integridad de datos

1. **No editar manualmente `data/source/`.**
2. Mantener los archivos oficiales sin modificar y conservar los ZIP originales para trazabilidad.
3. Toda transformación debe ser reproducible desde código.
4. Mantener `ID_POLIGONO` como llave fuente estable; cualquier alias interno debe ser explícito.
5. No mezclar datos fuente, features procesadas, modelos ni predicciones.
6. Registrar CRS antes de realizar operaciones geoespaciales y reproyectar explícitamente cuando sea necesario.
7. Respetar NoData (`-9999` / `-9999.0`) y convertirlo a missing values antes de calcular estadísticas.
8. Ajustar transformaciones aprendidas —imputación, escalamiento, selección de features, etc.— sólo con folds de entrenamiento durante validación.
9. No utilizar información temporal posterior al horizonte predictivo una vez que éste quede definido.
10. Mantener un pipeline determinista y trazable desde `source/` hasta `processed/`.

---

## 12. Siguiente fase: Audit + Data Contract v2

Antes de construir la tabla maestra, ejecutar el audit descrito en `docs/DATA_SOURCES.md`. Debe incluir IDs/CRS/geometrías, cobertura temporal BASIC/PRO, municipios, duplicados SIAP, etiquetas reales de cebada, unidades/escalas SoilGrids, continuidad CHIRPS, semántica temporal WaPOR y cobertura espacial de todos los rásteres.

Después crear fixtures de integración con las mismas ~12 parcelas en todas las fuentes y construir la tabla maestra de **197 filas, una por `ID_POLIGONO`**, con namespaces por fuente, trazabilidad y separación explícita entre features `clean` y `competition`.


---

## Referencias internas

- `data/source/README.md`: reglas para fuentes oficiales.
- `data/.ai_handoff`: contexto operativo compacto para agentes de IA que trabajen en esta carpeta.
- `src/geocebada/`: código reproducible de procesamiento/modelado.

> Principio rector: las 138 etiquetas disponibles sirven para aprender y validar el modelo; las 59 restantes son el conjunto objetivo que FIRA evaluará externamente.
