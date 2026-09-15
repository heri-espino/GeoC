# Diccionario de variables — GeoCebada

Este documento resume las variables oficiales entregadas para el Reto AgroCebada FIRA 2026 y cómo deben interpretarse dentro de GeoCebada.

Las descripciones se basan en los archivos oficiales de referencia:

- `docs/reference/Descripcion_general_variables_DataSet.docx`
- `docs/reference/Diccionario_de_indices_satelitales.docx`
- los esquemas observados directamente en los CSV de `data/source/tabular/`
- los metadatos entregados con los rasters climáticos y topográficos.

> **Regla importante.** El diccionario oficial de índices es conceptual. Para índices que tengan varias formulaciones publicadas o una implementación específica del proveedor, **no debe suponerse una fórmula distinta de la utilizada por la fuente de datos**. Este documento explica qué representa cada variable, pero no inventa fórmulas no entregadas por FIRA.

## 1. Unidad de análisis y target

La unidad final de predicción es una parcela agrícola identificada por `ID_POLIGONO`.

El archivo oficial de target/split es:

`data/source/tabular/ID_area_rendimiento_70_30_Reto_AgroCebada.csv`

| Variable | Tipo conceptual | Significado |
|---|---|---|
| `ID_POLIGONO` | Identificador | Identificador único de parcela, desde `AGC_001` hasta `AGC_197`. Es la llave canónica para relacionar tablas, geometrías, rasters y features derivados. |
| `AREA_HA` | Geométrica | Área de la parcela en hectáreas. Está disponible para las 197 parcelas. |
| `RENDIMIENTO_T_HA` | Target | Rendimiento agrícola de cebada en toneladas por hectárea. Está observado para las 138 parcelas de entrenamiento y oculto para las 59 de predicción. |
| `CONJUNTO` | Split | Define si la parcela pertenece a `ENTRENAMIENTO` o `PREDICCION`. |

### Ciclo agrícola del target

La documentación oficial de descripción general establece que `RENDIMIENTO_T_HA` corresponde al **ciclo de producción abril–octubre de 2025**.

Esto resuelve el año/ciclo representado por el target, pero no define por sí solo el instante exacto en el que una predicción operativa debe considerarse disponible. Para modelos de pronóstico antes de cosecha todavía debe definirse una ventana temporal que no incorpore información posterior al horizonte de predicción.

## 2. Cómo leer las columnas satelitales

Las tablas BASIC y PRO son **longitudinales**: una parcela aparece repetidamente para distintas fechas de captura y, en BASIC, para distintos sensores.

Las columnas de índices siguen normalmente este patrón:

```text
<indice>_<estadistico>
```

Cada estadístico describe la distribución espacial del índice dentro de la parcela en una fecha determinada.

| Sufijo | Significado |
|---|---|
| `_promedio` | Valor medio del índice dentro de la parcela para esa observación. |
| `_std` | Desviación estándar espacial dentro de la parcela; mide heterogeneidad interna. |
| `_max` | Valor máximo observado dentro de la parcela. |
| `_min` | Valor mínimo observado dentro de la parcela. |

Por ejemplo, una observación puede contener simultáneamente:

```text
ndvi_promedio
ndvi_std
ndvi_max
ndvi_min
```

Estos cuatro campos no son cuatro índices diferentes; son cuatro resúmenes espaciales del mismo índice.

## 3. Variables comunes de BASIC y PRO

| Variable | Significado |
|---|---|
| `ID_POLIGONO` | Parcela a la que pertenece la observación. |
| `fecha_captura` | Fecha de adquisición/captura de la observación satelital. |
| `sensor` | Sensor/plataforma de origen de la observación. Debe conservarse porque mediciones con el mismo nombre de índice no son necesariamente equivalentes entre sensores. |
| `porcentaje_nubosidad` | Variable de calidad que representa el porcentaje de nubosidad asociado con la observación. Puede utilizarse para definir criterios de filtrado, calidad y tratamiento de faltantes. |

## 4. Dataset BASIC

Archivo:

`data/source/tabular/Conjunto_datos_BASICO_AgroCebada2026.csv`

Esquema auditado:

- 107,666 observaciones;
- 96 columnas;
- 197 parcelas;
- observaciones de **Sentinel-2 y Landsat**;
- cobertura temporal **2022–2025**;
- los registros de sensores distintos permanecen separados;
- VI6T está asociado con las observaciones Landsat; la mayor parte de los otros índices proviene de Sentinel-2.

BASIC contiene 23 familias de índices. Cada familia aparece con `_promedio`, `_std`, `_max` y `_min`, lo que produce 92 columnas numéricas de índices, además de `ID_POLIGONO`, `fecha_captura`, `sensor` y `porcentaje_nubosidad`.

### 4.1 Índices BASIC

| Prefijo | Nombre / categoría oficial | Qué representa | Columnas del CSV |
|---|---|---|---|
| `ndvi` | Normalized Difference Vegetation Index (NDVI) | Indicador general de presencia, vigor y actividad de vegetación verde. | `ndvi_promedio`, `ndvi_std`, `ndvi_max`, `ndvi_min` |
| `evi` | Enhanced Vegetation Index (EVI) | Índice de vigor vegetal diseñado para reducir parte de la influencia del suelo y de efectos atmosféricos. | `evi_promedio`, `evi_std`, `evi_max`, `evi_min` |
| `crc` | Crop Residue Cover Index (CRC) | Respuesta espectral asociada con cobertura de residuos de cultivo sobre la superficie. | `crc_promedio`, `crc_std`, `crc_max`, `crc_min` |
| `mcrc` | Modified Crop Residue Cover (MCRC) | Variante modificada de índices orientados a detectar residuos de cultivo. | `mcrc_promedio`, `mcrc_std`, `mcrc_max`, `mcrc_min` |
| `ndsvi` | Normalized Difference Senescent Vegetation Index (NDSVI) | Índice sensible a vegetación senescente y material vegetal no fotosintético. | `ndsvi_promedio`, `ndsvi_std`, `ndsvi_max`, `ndsvi_min` |
| `srndi` | Shortwave Red Normalized Difference Index (SRNDI) | Contrasta información del infrarrojo de onda corta con la región roja del espectro. | `srndi_promedio`, `srndi_std`, `srndi_max`, `srndi_min` |
| `mra` | MRA, nomenclatura del proveedor | Variable espectral entregada bajo la abreviatura MRA. La documentación no proporciona una expansión o fórmula adicional; no debe inventarse. | `mra_promedio`, `mra_std`, `mra_max`, `mra_min` |
| `lai` | Leaf Area Index (LAI) | Variable biofísica que representa la cantidad de superficie foliar respecto de la superficie del terreno. | `lai_promedio`, `lai_std`, `lai_max`, `lai_min` |
| `fapar` | Fraction of Absorbed Photosynthetically Active Radiation (FAPAR) | Fracción de la radiación fotosintéticamente activa absorbida por la vegetación. | `fapar_promedio`, `fapar_std`, `fapar_max`, `fapar_min` |
| `savi` | Soil-Adjusted Vegetation Index (SAVI) | Índice de vegetación que reduce la influencia del brillo del suelo. | `savi_promedio`, `savi_std`, `savi_max`, `savi_min` |
| `vi6t` | VI6T | Índice espectral-térmico de Landsat. Combina información de una banda espectral con información térmica de la superficie y puede aportar señal complementaria sobre condiciones espectrales y térmicas. | `vi6t_promedio`, `vi6t_std`, `vi6t_max`, `vi6t_min` |
| `ndti` | Normalized Difference Tillage Index (NDTI) | Índice basado en infrarrojo de onda corta para caracterizar suelo, residuos y condiciones de labranza. | `ndti_promedio`, `ndti_std`, `ndti_max`, `ndti_min` |
| `sti` | Simple Tillage Index (STI) | Relación espectral utilizada para caracterizar residuos y condiciones de labranza. | `sti_promedio`, `sti_std`, `sti_max`, `sti_min` |
| `ndwi` | Normalized Difference Water Index (NDWI) | Índice relacionado con agua o humedad a partir del contraste entre bandas espectrales. | `ndwi_promedio`, `ndwi_std`, `ndwi_max`, `ndwi_min` |
| `endwi` | Enhanced Normalized Difference Water Index (ENDWI) | Variante mejorada de un índice normalizado de agua/humedad. | `endwi_promedio`, `endwi_std`, `endwi_max`, `endwi_min` |
| `msi` | Moisture Stress Index (MSI) | Índice sensible al contenido de agua de la vegetación y al estrés hídrico. | `msi_promedio`, `msi_std`, `msi_max`, `msi_min` |
| `emsi` | Enhanced Moisture Stress Index (EMSI) | Variante orientada a ampliar la sensibilidad a cambios de humedad o estrés de la vegetación. | `emsi_promedio`, `emsi_std`, `emsi_max`, `emsi_min` |
| `dswi2` | Disease-Water Stress Index 2 (DSWI2) | Índice espectral desarrollado para detectar cambios relacionados con estrés hídrico y condición vegetal. | `dswi2_promedio`, `dswi2_std`, `dswi2_max`, `dswi2_min` |
| `dswi3` | Disease-Water Stress Index 3 (DSWI3) | Variante de la familia Disease-Water Stress Index con una combinación distinta de bandas. | `dswi3_promedio`, `dswi3_std`, `dswi3_max`, `dswi3_min` |
| `dswi4` | Disease-Water Stress Index 4 (DSWI4) | Índice de la misma familia que utiliza un contraste espectral diferente. | `dswi4_promedio`, `dswi4_std`, `dswi4_max`, `dswi4_min` |
| `dswi5` | Disease-Water Stress Index 5 (DSWI5) | Índice multibanda orientado a caracterizar condición vegetal y estrés hídrico. | `dswi5_promedio`, `dswi5_std`, `dswi5_max`, `dswi5_min` |
| `nddi` | Normalized Difference Drought Index (NDDI) | Integra información de vegetación y humedad para caracterizar condiciones de sequedad. | `nddi_promedio`, `nddi_std`, `nddi_max`, `nddi_min` |
| `enddi` | Enhanced Normalized Difference Drought Index (ENDDI) | Variante mejorada de un índice de sequedad que incorpora una señal hídrica complementaria. | `enddi_promedio`, `enddi_std`, `enddi_max`, `enddi_min` |

### 4.2 Missingness en BASIC

La tabla BASIC presenta faltantes relevantes y éstos no deben interpretarse automáticamente como cero. En la auditoría inicial, VI6T es una de las familias con mayor cantidad de valores ausentes; también existen faltantes sustanciales en otras familias como FAPAR y NDVI.

La combinación de `sensor`, `fecha_captura`, `porcentaje_nubosidad` y disponibilidad específica de cada índice debe considerarse al construir features.

## 5. Dataset PRO

Archivo:

`data/source/tabular/Conjunto_datos_PRO_AgroCebada.csv`

Esquema auditado:

- 47,804 observaciones;
- 20 columnas;
- 197 parcelas;
- fuente **Planet**;
- cobertura **2025**.

PRO contiene cuatro familias de índices, cada una con `_promedio`, `_std`, `_max` y `_min`.

| Prefijo | Nombre / categoría oficial | Qué representa | Columnas del CSV |
|---|---|---|---|
| `ndvi` | Normalized Difference Vegetation Index (NDVI) | Indicador general de vigor y presencia de vegetación verde. | `ndvi_promedio`, `ndvi_std`, `ndvi_max`, `ndvi_min` |
| `evi` | Enhanced Vegetation Index (EVI) | Indicador de vigor vegetal con ajustes para reducir parte de los efectos del fondo y la atmósfera. | `evi_promedio`, `evi_std`, `evi_max`, `evi_min` |
| `lai` | Leaf Area Index (LAI) | Variable biofísica relacionada con la superficie foliar del dosel. | `lai_promedio`, `lai_std`, `lai_max`, `lai_min` |
| `msavi` | Modified Soil-Adjusted Vegetation Index (MSAVI) | Índice de vegetación diseñado para reducir de forma adaptativa la influencia del suelo. | `msavi_promedio`, `msavi_std`, `msavi_max`, `msavi_min` |

### BASIC y PRO no son intercambiables

Aunque BASIC y PRO compartan nombres como NDVI, EVI y LAI, la documentación oficial advierte que provienen de fuentes distintas y **no deben fusionarse automáticamente como si fueran mediciones equivalentes**.

Las diferencias entre sensores pueden incluir resolución espacial, respuesta espectral, frecuencia de observación y procesamiento. Por ello debe conservarse `sensor` y documentarse cualquier armonización posterior.

## 6. Variables climáticas

Los datos climáticos no llegan como columnas tabulares por parcela. Se entregan como rasters mensuales y deben extraerse espacialmente para cada polígono mediante un procedimiento reproducible, por ejemplo zonal statistics.

### 6.1 Precipitación

| Propiedad | Valor |
|---|---|
| Variable | Precipitación mensual acumulada |
| Fuente | CHIRPS |
| Cobertura | enero 2022–diciembre 2025 |
| Archivo | `PREC_AAAA_MM.tif` |
| Unidad entregada | mm/mes |
| Interpretación | Suma de la precipitación registrada durante el mes; permite caracterizar disponibilidad y distribución temporal de lluvia. |
| CRS entregado | EPSG:4326 |
| Resolución aproximada | 0.05° × 0.05° (~5 km) |

Features derivados posibles, una vez fijado el horizonte temporal, incluyen acumulados por etapa del cultivo, número de meses secos, anomalías y distribución intraestacional. Estos campos serían **features de GeoCebada**, no variables originales de FIRA.

### 6.2 Temperatura mínima

| Propiedad | Valor |
|---|---|
| Variable | Temperatura mínima mensual |
| Fuente | CHIRTS-ERA5 |
| Cobertura | enero 2022–diciembre 2025 |
| Archivo | `Tmin_AAAA_MM.tif` |
| Unidad | °C |
| Interpretación | Promedio mensual de la temperatura mínima; caracteriza las condiciones térmicas más bajas del periodo. |
| CRS | EPSG:4326 |
| Resolución aproximada | 0.05° × 0.05° (~5 km) |

### 6.3 Temperatura máxima

| Propiedad | Valor |
|---|---|
| Variable | Temperatura máxima mensual |
| Fuente | CHIRTS-ERA5 |
| Cobertura | enero 2022–diciembre 2025 |
| Archivo | `Tmax_AAAA_MM.tif` |
| Unidad | °C |
| Interpretación | Promedio mensual de la temperatura máxima; permite caracterizar diferencias térmicas entre meses y zonas. |
| CRS | EPSG:4326 |
| Resolución aproximada | 0.05° × 0.05° (~5 km) |

La documentación oficial menciona explícitamente que pueden construirse indicadores adicionales como temperatura media, amplitud térmica y acumulados por periodos específicos. Esas transformaciones deben documentarse y respetar el horizonte de predicción.

## 7. Variables topográficas

También son rasters y no columnas del CSV original.

| Variable | Archivo entregado | Qué representa | Unidad / escala operativa |
|---|---|---|---|
| Elevación | `Elevacion_INEGI_CEM4_120m.tif` | Altitud del terreno; ayuda a caracterizar diferencias ambientales y topográficas entre parcelas. | metros sobre el nivel del mar; raster entregado a 120 m |
| Pendiente | `Pendiente_INEGI_CEM4_120m_grados.tif` | Inclinación del terreno; describe condiciones topográficas que pueden relacionarse con drenaje, exposición y microclima. | grados; raster entregado a 120 m |

Los productos entregados usan EPSG:6372. Deben reproyectarse de forma explícita antes de combinarlos con capas en otro CRS.

## 8. Geometría de parcela

La capa vectorial oficial contiene las 197 geometrías. La documentación de referencia indica que incluye identificador, área y `CONJUNTO`.

La geometría permite crear variables derivadas como centroides, perímetro, compactación, vecindad o agregados raster, pero esas variables no deben confundirse con columnas originales del dataset.

El CRS de la capa de parcelas debe verificarse directamente desde el archivo antes de calcular áreas, distancias o realizar overlays.

## 9. Qué significa una fila

### Archivo de target/split

Una fila representa **una parcela**:

```text
ID_POLIGONO → una parcela → un rendimiento objetivo
```

### BASIC y PRO

Una fila representa una **observación satelital de una parcela en una fecha y sensor**:

```text
ID_POLIGONO + fecha_captura + sensor → una observación
```

Por ello, las 107,666 filas BASIC y las 47,804 filas PRO **no son muestras independientes de rendimiento**. Las parcelas se repiten muchas veces.

En validación supervisada nunca debe hacerse un split aleatorio fila-a-fila que permita que observaciones de una misma parcela aparezcan simultáneamente en train y validation.

## 10. Cómo pasan las variables longitudinales al modelo

El target existe una vez por parcela, mientras que los índices satelitales y el clima tienen múltiples observaciones temporales. El modelado final requiere transformar las series a features por parcela o usar un método temporal que respete la agrupación por parcela.

Ejemplos de features derivados potenciales:

```text
ndvi_mean_apr_jun
ndvi_peak
ndvi_date_peak
ndvi_auc
ndvi_slope
ndvi_std_temporal
precip_sum_apr_jun
tmin_mean_apr_jun
tmax_mean_apr_jun
thermal_range_mean
elevation_mean
slope_mean
```

Estos nombres son ejemplos metodológicos de GeoCebada; **no son columnas oficiales entregadas por FIRA**.

## 11. Reglas de interpretación

1. No existen umbrales universales para los índices aplicables a todas las parcelas, fechas y etapas fenológicas.
2. La evolución temporal del cultivo importa: un valor aislado puede significar cosas distintas según la etapa fenológica.
3. Conservar siempre `sensor` durante auditoría, exploración y construcción inicial de features.
4. No asumir que NDVI/EVI/LAI de BASIC y PRO son directamente equivalentes.
5. `porcentaje_nubosidad` es una señal de calidad y puede explicar parte del missingness.
6. No tratar valores faltantes como cero sin justificación.
7. No inventar fórmulas para MRA, VI6T u otros índices cuya implementación exacta no haya sido proporcionada.
8. El target corresponde al ciclo abril–octubre de 2025, pero el corte temporal exacto para una predicción pre-cosecha debe definirse explícitamente antes de congelar features.
9. Las transformaciones derivadas deben registrarse en código bajo `src/geocebada/features/` cuando dejen de ser exploratorias.

## 12. Resumen de fuentes de información

| Componente | Fuente | Cobertura | Papel en GeoCebada |
|---|---|---|---|
| Target / split | Datos del reto | ciclo abril–octubre 2025 | Target, área y partición train/prediction |
| BASIC | Sentinel-2 + Landsat | 2022–2025 | Series temporales de índices satelitales |
| PRO | Planet | 2025 | Series temporales de NDVI, EVI, LAI y MSAVI |
| Precipitación | CHIRPS | 2022–2025 | Disponibilidad y distribución temporal de lluvia |
| Tmin/Tmax | CHIRTS-ERA5 | 2022–2025 | Condiciones térmicas |
| Elevación | INEGI CEM | estática | Contexto topográfico |
| Pendiente | derivada de CEM | estática | Inclinación del terreno |
| Geometrías | parcelas del reto | 197 parcelas | Integración espacial, mapas y extracción raster |

## 13. Archivos relacionados

Para esquema físico y missingness observado:

`reports/data_schema_official.md`

Para estructura general del dataset:

`data/README.md`

Para localizar funciones de carga/procesamiento:

`docs/FUNCTION_INDEX.md`
