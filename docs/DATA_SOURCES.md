# Data sources and integration contract

Updated: 2026-09-17  
Project: GeoCebada / Reto AgroCebada FIRA 2026  
Audience: humans, AI agents and Codex sessions working on data ingestion, feature engineering or modeling.

This document records the current shared understanding of the data. It separates **confirmed facts**, **local acquisition status**, **known caveats** and **work that still requires audit**. When this document conflicts with official files or directly inspected metadata, the official/source files win and this document must be updated.

## Read order for data work

1. `docs/DATA_SOURCES.md` — this document: source inventory, joins, caveats and integration plan.
2. `docs/VARIABLES.md` — source-backed variable/index dictionary.
3. `data/.ai_handoff` — compact operational handoff.
4. `data/README.md` — directory contract and general data rules.
5. `data/data_catalog.json` and `docs/DATA_CATALOG.md` — machine-generated/local schema inventory.
6. Official/reference files under `docs/official/`, `docs/reference/` and immutable `data/source/`.

## Core prediction unit

The final supervised unit is the **parcel**, not an individual satellite observation.

Official target/split file:

`data/source/tabular/ID_area_rendimiento_70_30_Reto_AgroCebada.csv`

Confirmed contract:

- 197 parcels total.
- 138 `ENTRENAMIENTO` parcels with observed `RENDIMIENTO_T_HA`.
- 59 `PREDICCION` parcels with hidden yield.
- canonical key: `ID_POLIGONO`.
- target: `RENDIMIENTO_T_HA` in t/ha.
- split column: `CONJUNTO`.
- target cycle: April–October 2025.
- geographic scope: Hidalgo, Puebla and Tlaxcala.
- the split is by complete parcel.
- an empty `Unnamed: 4` column has been observed in the split CSV and should be dropped.

Independent labeled sample size is therefore **138**, even though satellite tables contain tens of thousands of rows.

## Official parcel geometry

Source:

`data/source/geospatial/Parcelas_Reto_AGC_CONJUNTO_70_30.zip`

Observed catalog facts:

- 197 Polygon features.
- fields include `Cultivo`, `ID_POLIGON`, `área_ha`, `Municipio`, `Estado`, `CONJUNTO`, geometry.
- the vector field `ID_POLIGON` is a truncated form of the canonical `ID_POLIGONO`.
- `área_ha` should be normalized internally to `area_ha`.
- observed WGS84-like geographic extent is approximately longitude -98.613 to -98.197 and latitude 19.524 to 19.984.
- direct catalog inspection records the official parcel archive as `EPSG:4326`; loaders must still read and validate this CRS rather than assigning it blindly.

Canonical internal parcel contract should become:

`ID_POLIGONO, area_ha, estado, municipio, cve_ent, cve_mun, cvegeo, conjunto, geometry`

Required loader normalization:

- `ID_POLIGON` -> `ID_POLIGONO`
- `área_ha` -> `area_ha`

## Official satellite tables

### BASIC

`data/source/tabular/Conjunto_datos_BASICO_AgroCebada2026.csv`

Audited shape and coverage:

- 107,666 rows × 96 columns.
- 197 parcels.
- years 2022–2025.
- Sentinel-2 + Landsat.
- row grain is approximately `(parcel, capture date, sensor)`.
- dates are day-first.
- metadata: `ID_POLIGONO`, `fecha_captura`, `sensor`, `porcentaje_nubosidad`.
- 23 index families × four statistics (`promedio`, `std`, `max`, `min`) = 92 feature columns.

Index families:

`ndvi, evi, crc, mcrc, ndsvi, srndi, mra, lai, fapar, savi, vi6t, ndti, sti, ndwi, endwi, msi, emsi, dswi2, dswi3, dswi4, dswi5, nddi, enddi`

Important:

- missingness is strongly sensor-specific and must not be treated as random missingness.
- do not globally fill missing satellite values.
- preserve sensor provenance.
- same-named indices from different sensors are not automatically equivalent.
- `MRA` is provider nomenclature; do not invent a formula.
- the source material does not provide a formula for `VI6T`; do not invent one.

### PRO

`data/source/tabular/Conjunto_datos_PRO_AgroCebada.csv`

Audited shape and coverage:

- 47,804 rows × 20 columns.
- 197 parcels.
- Planet observations.
- 2025.
- metadata: `ID_POLIGONO`, `fecha_captura`, `sensor`, `porcentaje_nubosidad`.
- NDVI, EVI, LAI and MSAVI, each with `promedio`, `std`, `max`, `min`.

BASIC and PRO are longitudinal repeated observations. Never create supervised folds by random satellite rows; any supervised validation must be parcel-aware.

## Temporal alignment already implemented

Reusable code:

`src/geocebada/features/temporal.py`

Current functions:

- `make_temporal_grid`
- `temporal_coverage_summary`
- `temporal_backend_available`
- `align_temporal_knn`
- `aligned_to_wide`

Current alignment concept for a variable at target time `tau_m`:

$$
\tilde x_{ij}(\tau_m)=
\frac{\sum_{r\in N_{k,j}(\tau_m)} w_r x_{ijr}}
{\sum_r w_r},
\qquad
w_r=e^{-|t_r-\tau_m|/h}q_r.
$$

with cloud quality weight

$$
q_r=1-\frac{\mathrm{cloud}_r}{100}.
$$

Notebook `notebooks/02_cobertura_alineacion_temporal.ipynb` currently uses April 1–October 31 2025, a 14-day grid, `k=3`, bandwidth 14 days, maximum distance 30 days and cloud weighting. These are exploratory defaults, not frozen final hyperparameters.

## Official climate

### Monthly CHIRPS precipitation

Under `data/source/climate/precipitation/`:

- 48 monthly rasters for 2022–2025.
- EPSG:4326.
- approximately 0.05°.
- precipitation units: mm/month.

### Monthly temperature

Under `data/source/climate/temperature/`:

- 96 monthly rasters total for 2022–2025.
- 48 Tmax + 48 Tmin.
- EPSG:4326.
- approximately 0.05°.
- units: °C.

The official climate package also includes inventory CSV files. Catalog count observed for official climate is 146 files total: 48 precipitation + 96 temperature + 2 inventory files.

Useful derived features include April–October 2025 summaries and historical 2022–2024 climatology/anomaly features.

## Official topography

Under `data/source/topography/`:

- elevation from INEGI CEM4 at 120 m product resolution.
- slope in degrees at 120 m product resolution.
- inventory metadata.
- CRS: EPSG:6372.
- NoData observed as -9999 / -9999.0.

This is the official baseline topography source.

# External public data

External raw inputs live under:

`data/raw/external/`

They are intentionally gitignored. Do not commit large raw rasters, NetCDFs or CSVs. The tracked `data/external_manifest.json` records reproducible inventory metadata but is not a substitute for the local files.

## Acquisition status as of 2026-09-17

The required/prioritized external inputs have been acquired locally according to the latest project session:

- INEGI municipal boundaries for Hidalgo, Puebla and Tlaxcala.
- CHIRPS v3 daily precipitation for the target season.
- WaPOR v3 NPP and AETI.
- SoilGrids 250 m properties/depths.
- INEGI CEM state rasters at approximately 15 m.
- SIAP municipal agricultural closure history including the previously missing 2023 file.

ERA5-Land is **optional and not a completeness requirement**. Attempts through CDS encountered request-cost/backend errors; the `data/raw/external/era5_land/` directory may contain partial/incomplete files and must not be assumed complete. Do not make downstream features depend on ERA5 unless a later audit confirms a complete reproducible acquisition.

## INEGI municipal boundaries

Canonical expected files:

`data/raw/external/inegi_municipios/hidalgo_municipios.geojson`  
`data/raw/external/inegi_municipios/puebla_municipios.geojson`  
`data/raw/external/inegi_municipios/tlaxcala_municipios.geojson`

Tlaxcala was previously inspected with 60 MultiPolygon features and fields including:

`cvegeo, cve_ent, cve_mun, nomgeo, cve_cab` plus population-related fields.

Hidalgo and Puebla must be schema-checked during the audit rather than assumed identical.

For administrative/SIAP joins, preserve the official parcel `Estado`/`Municipio`
attributes as primary. Use INEGI largest-overlap assignment as spatial QA/fallback, not
centroid. The v2 audit found clear dominant overlaps for boundary-crossing AGC_020 and
AGC_129, and one documented official-vs-spatial mismatch for AGC_048
(`Nanacamilpa de Mariano Arista` official vs `Calpulalpan` spatial).

## CHIRPS v3 daily

Local directory:

`data/raw/external/chirps_v3_daily/`

Expected/acquired target-period inventory:

- 214 daily GeoTIFFs.
- 2025-04-01 through 2025-10-31 inclusive.
- EPSG:4326.
- approximately 0.05°.
- cropped to the challenge region.

Candidate parcel features:

- total April–October precipitation.
- monthly totals.
- early/mid/late-season totals.
- rain-day counts for documented thresholds.
- dry-day count.
- maximum consecutive dry spell.
- maximum 1-day rainfall.
- maximum rolling 5-day rainfall.

Because official monthly CHIRPS also exists, source-prefix all features and test whether daily extremes add information beyond the official monthly source.

## WaPOR v3

Local directory:

`data/raw/external/wapor_v3/`

Acquired products:

- AETI NetCDF.
- NPP NetCDF.
- 21 dekadal bands per product for April–October 2025.
- grid observed around 227 × 212.

Metadata recovered from files:

- AETI: units `mm/day`; long name `Actual EvapoTranspiration and Interception`.
- NPP: units `gC/m²/day`; long name `Net Primary Production`.
- each band has `start_date`, `number_of_days` (10 or 11) and `temporal_resolution="Dekad"`.

Rates must be aggregated with band duration. For example:

$$
AETI_{total}=\sum_k AETI_k d_k.
$$

Do not treat a simple unweighted sum of daily rates as a seasonal total.

Potential features:

- seasonal total / mean.
- monthly or phenological-phase summaries.
- peak magnitude and timing.
- NPP/AETI ratios or water-productivity proxies only if clearly defined and physically defensible.

## SoilGrids 250 m

Local directory:

`data/raw/external/soilgrids_250m/`

Acquired inventory:

- 36 GeoTIFFs.
- 9 properties × 4 depths.
- properties: `phh2o, clay, sand, silt, soc, nitrogen, cec, bdod, cfvo`.
- depths: `0-5cm, 5-15cm, 15-30cm, 30-60cm`.

Important CRS caveat:

- the downloaded TIFF headers have been observed with missing/null CRS metadata.
- the downloader knows the source is SoilGrids WCS.
- SoilGrids uses pseudo identifier 152160 in WCS requests.
- locally, PROJ recognizes the equivalent Interrupted Goode Homolosine CRS as `ESRI:54052`.
- assign `ESRI:54052` **only when the file is positively identified as this SoilGrids download**. Never guess that CRS for arbitrary CRS-less rasters.

Verified SoilGrids semantics:

- raw rasters are integer-valued;
- Data Contract v2 freezes the official ISRIC divisors and conventional units in `configs/data_contract_v2.yaml`;
- `phh2o/clay/sand/silt/soc/cec/cfvo` divide by 10; `nitrogen/bdod` divide by 100;
- do not feed raw INT16 values into the final model without applying/documenting those conversions.

Official source: https://docs.isric.org/globaldata/soilgrids/SoilGrids_faqs_01.html

Candidate features:

- parcel mean, median and standard deviation by property/depth.
- depth-weighted 0–30 cm and 0–60 cm summaries after units/scales are verified.

## SIAP municipal agricultural closure

Local directory:

`data/raw/external/SIAP/`

Observed schema has 24 columns including:

`Anio, Idestado, Nomestado, Idddr, Nomddr, Idcader, Nomcader, Idmunicipio, Nommunicipio, Idciclo, Nomcicloproductivo, Idmodalidad, Nommodalidad, Idunidadmedida, Nomunidad, Idcultivo, Nomcultivo, Sembrada, Cosechada, Siniestrada, Volumenproduccion, Rendimiento, Preciomediorural, Valorproduccion`.

Dictionary semantics confirmed for area variables:

- `Sembrada`: hectares.
- `Cosechada`: hectares.
- `Siniestrada`: hectares.
- `Rendimiento`: normally t/ha, but crop-specific unit conventions must still be checked for the selected barley records.

Municipal key must be constructed as:

```text
CVEGEO = zfill(Idestado, 2) + zfill(Idmunicipio, 3)
```

Never join on `Idmunicipio` alone.

Historical local inventory originally contained duplicate file copies for 2004, 2005 and 2025 and was missing 2023. The 2023 file has now been downloaded, but duplicate-year files still need checksum/content comparison and a deterministic keep/drop rule during the audit. Do not concatenate duplicate copies blindly.

The audit discovers the exact barley labels `Cebada grano` and
`Cebada forrajera en verde`, with `Otoño-Invierno` / `Primavera-Verano` and
`Riego` / `Temporal` represented in the target states. Do not mix grain and forage
barley. SIAP 2015–2020 use `Nomcultivo Sin Um`, which is a source alias normalized to
canonical `Nomcultivo`.

### Competition-mode caution

Municipal 2025 yield can be a very strong proxy for parcel 2025 yield. Treat contemporaneous SIAP 2025 yield as a **competition-mode feature / potential leakage proxy**, document it explicitly, and compare results with a clean feature set. Never use hidden parcel targets.

A useful competition residual formulation to test later is:

$$
\hat y_i^{(0)}=Y_{SIAP,muni(i),2025},
\qquad
r_i=y_i-\hat y_i^{(0)}.
$$

Then learn parcel-level residuals `r_i` from remote-sensing/soil/topography features.

## INEGI CEM high-resolution external rasters

Local state directories under:

`data/raw/external/Inegi/`

Known local state products cover Hidalgo, Puebla and Tlaxcala at approximately 15 m. Direct raster inspection records CRS `EPSG:6365`.

Potential features:

- elevation mean/min/max/std.
- slope.
- aspect.
- roughness/local relief.

Because official 120 m elevation/slope is already provided, namespace these features separately and evaluate marginal value rather than silently replacing the official source.

## ERA5-Land

ERA5-Land is optional. The project attempted CDS downloads for April–October 2025. Authentication/licensing were resolved, but large NetCDF requests hit request-cost limits and later smaller requests encountered repeated CDS HTTP 500 errors.

Current policy:

- do not block the pipeline on ERA5.
- do not assume `data/raw/external/era5_land/` is complete.
- if revisited, prefer small resumable requests or the ERA5-Land time-series product at unique grid points.
- CHIRPS + WaPOR already provide strong water/climate information for the first feature set.

# Join and aggregation rules

## Parcel is the master key

Every final feature table must have exactly one row per `ID_POLIGONO`.

Expected final master shape before target filtering:

- 197 rows.
- `ID_POLIGONO` unique.
- all official parcel IDs present.
- target populated only for the 138 training parcels.
- `CONJUNTO` preserved.

## Municipality mapping

Recommended order:

1. normalize parcel geometry and CRS.
2. normalize municipal `cve_ent`, `cve_mun`, `cvegeo`.
3. intersect parcel polygons with municipal polygons.
4. choose municipality by largest overlap area.
5. keep overlap fraction and ambiguity diagnostics.
6. compare against official `Municipio`/`Estado` attributes rather than overwriting silently.

## Raster extraction

For every raster source:

1. inspect/assign source CRS only with source-backed evidence.
2. reproject parcel geometry to raster CRS.
3. mask NoData.
4. compute documented zonal statistics.
5. store coverage/pixel-count diagnostics.
6. prefix feature names by source.
7. never infer field-scale precision from coarse climate rasters.

## Satellite features

Preserve:

- source dataset (BASIC/PRO).
- sensor.
- date.
- cloud percentage.
- statistic suffix.
- missingness indicators where useful.

Do not naively concatenate overlapping NDVI/EVI/LAI values across Sentinel/Landsat/Planet as one homogeneous series.

# Clean vs competition feature modes

GeoCebada should maintain two explicit feature groups.

## Clean / scientifically defensible

Use information that would plausibly be available at the declared prediction cutoff and fit preprocessing only inside training folds.

Examples:

- parcel geometry/topography/soil.
- historical climate.
- satellite observations up to the frozen cutoff.
- contemporaneous weather only up to that cutoff.

## Competition / rule-permitted public information

May include public information that improves leaderboard error even if it would not be available in a strict prospective forecast, provided the challenge rules permit it and it is disclosed.

Examples to evaluate separately:

- full April–October 2025 remote sensing.
- full-season 2025 WaPOR/CHIRPS.
- SIAP 2025 municipal yield proxy.
- transductive transforms learned from all 197 unlabeled `X` rows.

Never cross the boundary of using hidden `RENDIMIENTO_T_HA` labels.

# Data catalog and fixtures

Existing tooling:

- `tools/build_data_catalog.py`
- `tools/_netcdf_catalog_worker.py`
- `tools/build_external_data_manifest.py`
- `data/data_catalog.json`
- `data/external_manifest.json`
- `data/samples/`

The catalog currently records tabular/vector/raster/NetCDF metadata and representative fixtures.

Known catalog limitations to fix in Data Contract v2:

1. `available` is not the same as `complete`; add expected/observed counts and missing files.
2. critical ID/date coverage should inspect full files, not only small row samples.
3. BASIC/PRO date ordering must parse `%d/%m/%Y`; string sorting is incorrect.
4. current schema samples use unrelated parcel IDs, so they cannot test joins end-to-end.
5. SoilGrids scale factors/units need semantic metadata.
6. partial ERA5 files must not make ERA5 appear complete.

# Data Contract v2 — executable audit gate

The formal machine-readable contract now exists at `configs/data_contract_v2.yaml`, with executable audit `tools/audit_data_contract_v2.py` and documentation in `docs/DATA_CONTRACT_V2.md`. Run the full local audit and resolve its findings before the master feature table.

## Integration fixtures

Create a shared set of about 12 fixed parcels spanning:

- all three states.
- both `ENTRENAMIENTO` and `PREDICCION`.
- small and large parcel areas.
- BASIC sensors.
- low/high cloud conditions.
- representative missingness.

Recommended layout:

```text
data/samples/
  schema/
    tabular/
    raster/
    vector/
    netcdf/
  integration/
    parcels.geojson
    split.csv
    basic.csv
    pro.csv
    siap.csv
    climate/
    chirps_daily/
    soilgrids/
    wapor/
    cem/
```

Raster fixtures should crop around those same parcels instead of arbitrary raster centers.

Integration tests should assert at minimum:

```python
len(features) == 12
features["ID_POLIGONO"].is_unique
set(features["ID_POLIGONO"]) == expected_ids
```

## Audit checklist

The v2 audit should verify:

- exactly 197 official IDs.
- exactly 138 train + 59 prediction.
- ID coverage across split, geometry, BASIC and PRO.
- normalized parcel field names and geometry validity.
- parcel CRS.
- BASIC/PRO date ranges and per-sensor coverage.
- cloud/missingness relationships.
- train-vs-prediction temporal coverage differences.
- municipal polygon completeness for 13/21/29.
- parcel-to-municipality overlap diagnostics.
- SIAP year inventory, duplicate checksums and presence of 2023.
- actual barley labels/cycles/modalities in SIAP.
- SoilGrids units and scale factors.
- raster/parcel overlap for every external raster family.
- CHIRPS daily count = 214 and date continuity.
- WaPOR variables/band dates/day counts.
- CEM state coverage.
- ERA5 status explicitly partial/unused unless later completed.

# Master feature table — task after Data Contract v2

After the audit passes, build a deterministic table with exactly one row per parcel, likely under:

`data/processed/parcel_features.*`

Recommended feature namespaces:

- `base_*`: area, coordinates and stable parcel metadata.
- `admin_*`: state/municipality/CVEGEO.
- `sat_basic_*`
- `sat_pro_*`
- `clim_official_*`
- `chirps_daily_*`
- `wapor_*`
- `soilgrids_*`
- `topo_official_*`
- `cem15_*`
- `siap_hist_*`
- `siap_2025_*` only in competition mode.

Required invariants:

- 197 rows.
- unique `ID_POLIGONO`.
- no accidental many-to-many joins.
- source-prefixed columns.
- feature provenance recorded.
- clean vs competition feature groups machine-identifiable.
- no hidden target leakage.

# Modeling implications

With only 138 labels, feature engineering and validation discipline matter more than fitting extremely high-capacity models.

Planned comparisons can include:

- Ridge / ElasticNet.
- PLS.
- PCA/fPCA-based phenology summaries.
- CatBoost.
- ExtraTrees.
- carefully validated transductive representations.
- SIAP municipal baseline + learned residual.

Use consistent parcel folds across feature/model comparisons. Compare random parcel CV with spatial/group diagnostics because nearby parcels may be correlated.

# Do not guess

Future agents must not invent:

- SoilGrids scale factors/physical units.
- exact SIAP barley label/cycle/modalities.
- MRA formula.
- VI6T formula.
- equivalence between sensor-specific indices.
- completeness of ERA5.
- a parcel CRS that has not been read from the source.
- hidden target values.

When uncertain, audit the local files or official source documentation and update this document.
