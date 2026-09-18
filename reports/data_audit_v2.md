# Data Audit v2

Generated: 2026-09-18T15:38:40.048763+00:00
Overall status: **PASS**

## Summary

| Status | Checks |
|---|---:|
| PASS | 46 |
| WARN | 0 |
| FAIL | 0 |
| SKIP | 1 |

## Checks

| Status | Check | Summary |
|---|---|---|
| PASS | official_split.columns | Required target/split columns are present. |
| PASS | official_split.rows | Split contains 197 parcel rows. |
| PASS | official_split.ids | ID_POLIGONO is non-null and unique. |
| PASS | official_split.counts | Official split is 138 training + 59 prediction parcels. |
| PASS | official_split.target_visibility | Targets are observed only for training parcels. |
| PASS | official_basic.shape | BASIC shape matches 107,666 × 96. |
| PASS | official_basic.columns | Required longitudinal metadata columns are present. |
| PASS | official_basic.parcel_coverage | BASIC covers all 197 official parcels. |
| PASS | official_basic.dates | Dates parse strictly as %d/%m/%Y and span 2022-01-01 to 2025-12-31. |
| PASS | official_basic.sensors | Observed sensors match contract: ['Landsat', 'Sentinel-2']. |
| PASS | official_basic.observation_keys | Parcel/date/sensor keys are unique. |
| PASS | official_basic.coverage_profile | Recorded sensor-specific missingness and train/prediction coverage diagnostics. |
| PASS | official_pro.shape | PRO shape matches 47,804 × 20. |
| PASS | official_pro.columns | Required longitudinal metadata columns are present. |
| PASS | official_pro.parcel_coverage | PRO covers all 197 official parcels. |
| PASS | official_pro.dates | Dates parse strictly as %d/%m/%Y and span 2025-01-01 to 2025-12-30. |
| PASS | official_pro.sensors | Observed sensors match contract: ['Planet']. |
| PASS | official_pro.observation_keys | Parcel/date/sensor keys are unique. |
| PASS | official_pro.coverage_profile | Recorded sensor-specific missingness and train/prediction coverage diagnostics. |
| PASS | official_parcels.rows | Parcel archive contains 197 features. |
| PASS | official_parcels.crs | Parcel CRS is EPSG:4326. |
| PASS | official_parcels.geometry | All parcel geometries are present, non-empty and valid. |
| PASS | official_parcels.ids | Parcel IDs are unique and match the official split. |
| PASS | official_parcels.area_alias | Parcel area field normalizes to area_ha. |
| PASS | official_climate.months | Official climate has complete monthly PREC/Tmin/Tmax coverage for 2022–2025. |
| PASS | official_climate.crs | Representative climate rasters use EPSG:4326. |
| PASS | official_topography.count | Official topography contains 2 GeoTIFFs. |
| PASS | official_topography.crs | Official topography CRS is EPSG:6372. |
| PASS | external_municipios.13 | Hidalgo: 84 municipalities with required keys. |
| PASS | external_municipios.21 | Puebla: 217 municipalities with required keys. |
| PASS | external_municipios.29 | Tlaxcala: 60 municipalities with required keys. |
| PASS | external_municipios.parcel_join | Every parcel has a dominant spatial municipality above the 80% QA threshold. |
| PASS | external_municipios.name_crosscheck | Known parcel/INEGI municipality name discrepancies are documented; official parcel labels remain primary for administrative joins. |
| PASS | external_chirps.continuity | CHIRPS has all 214 daily files from 2025-04-01 through 2025-10-31. |
| PASS | external_chirps.crs | Representative CHIRPS rasters use EPSG:4326. |
| PASS | external_wapor.files | Both expected WaPOR NetCDF products are present. |
| PASS | external_wapor.metadata | WaPOR AETI/NPP have 21 dekads with expected units and metadata. |
| PASS | external_soilgrids.inventory | SoilGrids contains 9 properties × 4 depths = 36 rasters. |
| PASS | external_soilgrids.semantics | ISRIC conversion divisors/units are frozen in Data Contract v2. |
| PASS | external_soilgrids.crs_headers | SoilGrids headers are compatible with the source-specific CRS fallback. |
| PASS | external_siap.years | SIAP contains every year 2003–2025. |
| PASS | external_siap.duplicates | Duplicate SIAP yearly copies are byte-identical; canonical unsuffixed files are safe to use. |
| PASS | external_siap.schema | All SIAP yearly files expose the required columns. |
| PASS | external_siap.barley_discovery | Discovered SIAP barley labels: ['Cebada forrajera en verde', 'Cebada grano']. |
| PASS | external_cem15.inventory | CEM 15 m inventory contains one raster for each target state. |
| PASS | external_cem15.crs | CEM 15 m rasters use EPSG:6365. |
| SKIP | external_era5.policy | ERA5-Land is absent and intentionally not required. |

## Interpretation

- FAIL blocks feature-table construction.
- WARN requires an explicit documented decision before downstream use.
- SKIP is intentional or dependency/source-limited; it is not evidence of completeness.
- Presence of a source is not equivalent to satisfying the contract.

Machine-readable details: reports/data_audit_v2.json
