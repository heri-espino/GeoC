# Official tabular schema audit

Audited from the immutable CSV files under `data/source/tabular/` on 2026-09-14.

## Yield/split table

`ID_area_rendimiento_70_30_Reto_AgroCebada.csv`

- 197 rows
- 4 substantive columns: `ID_POLIGONO`, `AREA_HA`, `RENDIMIENTO_T_HA`, `CONJUNTO`
- one row per parcel
- 138 `ENTRENAMIENTO`, 59 `PREDICCION`

This table is metadata/target/split information. It is **not** the full predictor dataset.

## BASIC remote-sensing table

`Conjunto_datos_BASICO_AgroCebada2026.csv`

- shape: **107,666 rows × 96 columns**
- 197 unique `ID_POLIGONO` values
- repeated observations per parcel
- identifier/context columns: `ID_POLIGONO`, `fecha_captura`, `sensor`
- cloud column: `porcentaje_nubosidad`
- 92 numeric index/statistic columns organized as four statistics (`promedio`, `std`, `max`, `min`) for the following index families:
  - `ndvi`
  - `evi`
  - `crc`
  - `mcrc`
  - `ndsvi`
  - `srndi`
  - `mra`
  - `lai`
  - `fapar`
  - `savi`
  - `vi6t`
  - `ndti`
  - `sti`
  - `ndwi`
  - `endwi`
  - `msi`
  - `emsi`
  - `dswi2`
  - `dswi3`
  - `dswi4`
  - `dswi5`
  - `nddi`
  - `enddi`

The table contains substantial missingness for several index families, so missing-data patterns must be inspected before modeling.

## PRO remote-sensing table

`Conjunto_datos_PRO_AgroCebada.csv`

- shape: **47,804 rows × 20 columns**
- 197 unique `ID_POLIGONO` values
- repeated observations per parcel
- identifier/context columns: `ID_POLIGONO`, `fecha_captura`, `sensor`
- cloud column: `porcentaje_nubosidad`
- numeric index/statistic columns:
  - `ndvi_promedio`, `ndvi_std`, `ndvi_max`, `ndvi_min`
  - `evi_promedio`, `evi_std`, `evi_max`, `evi_min`
  - `lai_promedio`, `lai_std`, `lai_max`, `lai_min`
  - `msavi_promedio`, `msavi_std`, `msavi_max`, `msavi_min`

## Joining with parcel metadata

Both BASIC and PRO use `ID_POLIGONO` and cover all 197 parcels. They can therefore inherit the parcel-level `AREA_HA`, `CONJUNTO`, and observed training `RENDIMIENTO_T_HA` through a many-to-one merge. Hidden prediction targets remain missing.

This produces up to:

- BASIC + split metadata: **99 columns**
- PRO + split metadata: **23 columns**

The resulting tables remain longitudinal/repeated-observation tables. Repeated rows for the same parcel must not be treated as independent labeled samples in random cross-validation.
