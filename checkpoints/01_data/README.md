# Checkpoint 01 — Data foundation

**Checkpoint date:** 2026-09-18  
**Project:** GeoCebada — Reto AgroCebada FIRA 2026  
**Status:** Data acquisition/audit complete; integration fixtures validated and versioned.  
**Reference fixture commit:** `6df329bcd12ec083cb88c3667c14e2ca12624f3f` — *Add validated shared integration fixtures*

This checkpoint records everything established before building the final parcel-level
feature matrix. Its purpose is to let another person or AI understand **what data exist,
why they were acquired, what was changed, what was deliberately not changed, what has
already been validated and what still remains to be built**.

---

## 1. What "data complete" means at this checkpoint

The data stage is complete in the following sense:

- all prioritized official and external sources needed for the first modeling pipeline
  have been acquired;
- their schemas, counts, dates, CRS assumptions and key joins have been audited;
- Data Contract v2 closes without `FAIL` findings on the full workstation dataset;
- known source inconsistencies have explicit policies instead of silent fixes;
- a small, real, joinable 12-parcel integration dataset has been generated, validated and
  committed to Git.

It does **not** mean that the final clean/model-ready 197-row table already exists.

The next stage is to extract features from the complete sources and build that table.

### What is actually stored in Git

| Data class | In Git? | Meaning |
|---|---|---|
| Official challenge sources under `data/source/` | Yes | Immutable source evidence. |
| Data documentation/contracts | Yes | Reproducible interpretation of the sources. |
| `data/samples/integration/` | Yes | Small real-data fixtures for end-to-end tests. |
| Full external raw downloads under `data/raw/external/` | **No** | Large/local source downloads; intentionally gitignored. |
| Final 197-row processed feature table | **Not yet** | This is the next major artifact to build. |
| Trained model artifacts | Not yet | Modeling has not reached the frozen production stage. |

Therefore: **the repository contains the official data and the small validated integration
fixtures, but not all full external raw data and not yet the final "clean" master table.**

---

## 2. Core challenge contract

The supervised prediction unit is the parcel.

| Item | Confirmed contract |
|---|---|
| Canonical ID | `ID_POLIGONO` |
| Target | `RENDIMIENTO_T_HA` |
| Split column | `CONJUNTO` |
| Total parcels | 197 |
| Training parcels | 138 |
| Prediction parcels | 59 |
| Target cycle | April–October 2025 |
| States | Hidalgo, Puebla, Tlaxcala |

The 59 `PREDICCION` targets remain hidden and must never be reconstructed or used as
pseudo-ground-truth.

---

## 3. Official data supplied by the challenge

### 3.1 Target/split table

Source:

```text
data/source/tabular/ID_area_rendimiento_70_30_Reto_AgroCebada.csv
```

Why it matters:

- defines the one-row-per-parcel supervised target space;
- identifies the 138 labeled and 59 hidden-target parcels;
- provides official parcel area.

Audited:

- 197 rows;
- `ID_POLIGONO` unique/non-null;
- exactly 138 `ENTRENAMIENTO` + 59 `PREDICCION`;
- training targets present;
- prediction targets absent.

An empty `Unnamed: 4` source column is treated as disposable CSV noise.

### 3.2 Parcel geometries

Source:

```text
data/source/geospatial/Parcelas_Reto_AGC_CONJUNTO_70_30.zip
```

Why it matters:

- spatial extraction of raster covariates;
- municipality assignment/QA;
- area/location/topographic features.

Audited:

- 197 valid polygons;
- CRS `EPSG:4326`;
- IDs match the split exactly.

Canonical loader normalization:

```text
ID_POLIGON -> ID_POLIGONO
área_ha    -> area_ha
```

The source geometry is never edited in place.

### 3.3 BASIC satellite table

Source:

```text
data/source/tabular/Conjunto_datos_BASICO_AgroCebada2026.csv
```

Audited structure:

- 107,666 rows × 96 columns;
- all 197 parcels;
- 2022-01-01 through 2025-12-31;
- Landsat + Sentinel-2;
- dates are `DD/MM/YYYY`;
- parcel/date/sensor keys are unique;
- 23 index families × four statistics, plus date/sensor/cloud metadata.

Why it matters:

- multi-year phenology;
- vegetation/water/stress signals;
- temporal dynamics before and during the target cycle.

Important decision:

- rows are repeated longitudinal observations, **not 107,666 independent supervised
  samples**;
- same-named indices from different sensors are not silently merged;
- missingness is strongly sensor-specific and is preserved as information;
- no global blanket fill is allowed.

Provider-specific `MRA` and `VI6T` formulas must not be invented.

### 3.4 PRO satellite table

Source:

```text
data/source/tabular/Conjunto_datos_PRO_AgroCebada.csv
```

Audited structure:

- 47,804 rows × 20 columns;
- all 197 parcels;
- 2025-01-01 through 2025-12-30;
- Planet;
- NDVI/EVI/LAI/MSAVI × mean/std/max/min;
- no duplicate parcel/date/sensor keys.

Why it matters:

- denser/high-resolution 2025 temporal signal.

Again, PRO rows are longitudinal observations, not independent targets.

### 3.5 Official climate

Under:

```text
data/source/climate/
```

Audited:

- 48 monthly precipitation rasters, 2022–2025;
- 48 Tmin rasters;
- 48 Tmax rasters;
- continuous monthly coverage;
- CRS `EPSG:4326`;
- precipitation in mm/month;
- temperature in °C.

Why it matters:

- direct climate context;
- 2025 seasonal totals;
- historical anomalies relative to 2022–2024.

### 3.6 Official topography

Under:

```text
data/source/topography/
```

Audited:

- elevation;
- slope;
- 120 m product resolution;
- CRS `EPSG:6372`.

Why it matters:

- terrain/elevation effects on moisture, temperature and productivity.

---

## 4. External public data acquired

External full-resolution downloads live locally under:

```text
data/raw/external/
```

They are intentionally excluded from Git because they are large/reproducible raw inputs.

### 4.1 INEGI municipal boundaries

Local source:

```text
data/raw/external/inegi_municipios/
  hidalgo_municipios.geojson
  puebla_municipios.geojson
  tlaxcala_municipios.geojson
```

Audited counts:

- Hidalgo: 84 municipalities;
- Puebla: 217;
- Tlaxcala: 60.

Why acquired:

- stable `CVEGEO` construction;
- SIAP municipal joins;
- spatial QA of the official parcel municipality labels.

Policy:

- official parcel `Estado`/`Municipio` remains primary for administrative/SIAP joins;
- INEGI largest polygon overlap is QA/fallback, not a silent replacement.

Known boundary cases:

- `AGC_020`: 98.65% Emiliano Zapata, Hidalgo; 1.35% Calpulalpan, Tlaxcala.
- `AGC_129`: 85.43% Singuilucan; 14.57% Cuautepec de Hinojosa.
- `AGC_048`: official municipality = Nanacamilpa de Mariano Arista; spatial overlay =
  Calpulalpan. The official label is preserved for administration/SIAP and the spatial
  result is retained as QA metadata.

### 4.2 CHIRPS v3 daily precipitation

Local source:

```text
data/raw/external/chirps_v3_daily/
```

Audited:

- 214 daily GeoTIFFs;
- 2025-04-01 through 2025-10-31 with no missing dates;
- CRS `EPSG:4326`;
- approximately 0.05° resolution.

Why acquired:

Official monthly precipitation cannot directly express short dry spells or rainfall
extremes. Daily CHIRPS enables features such as:

- seasonal/monthly totals;
- rain-day counts;
- dry-day counts;
- maximum consecutive dry spell;
- maximum 1-day rain;
- maximum rolling 5-day rain.

It is source-prefixed separately because official monthly CHIRPS already exists.

### 4.3 WaPOR v3

Local source:

```text
data/raw/external/wapor_v3/
```

Products:

- AETI — Actual EvapoTranspiration and Interception;
- NPP — Net Primary Production.

Audited:

- both NetCDFs present;
- 21 dekadal bands each, April–October 2025;
- AETI units `mm/day`;
- NPP units `gC/m²/day`;
- each dekad includes `number_of_days`.

Why acquired:

- crop water-use signal;
- vegetation productivity signal;
- phase/timing features that are complementary to spectral indices.

Important transformation rule:

[
AETI_{total}=\sum_k AETI_k d_k
]

and the same duration logic applies to other rate-based dekadal aggregations.

The known malformed UTF-8 NPP metadata is repaired only in derived/versioned fixtures;
the raw source file is never modified.

### 4.4 SoilGrids 250 m

Local source:

```text
data/raw/external/soilgrids_250m/
```

Audited inventory:

- 36 GeoTIFFs;
- 9 properties × 4 depths.

Properties:

```text
phh2o
clay
sand
silt
soc
nitrogen
cec
bdod
cfvo
```

Depths:

```text
0-5cm
5-15cm
15-30cm
30-60cm
```

Why acquired:

Soil properties are static field-context variables that can explain persistent yield
differences not captured by one season of remote sensing.

Source-specific CRS policy:

- downloaded TIFF headers can lack CRS;
- only positively identified SoilGrids WCS downloads may receive the known
  `ESRI:54052` fallback;
- the WCS pseudo identifier is 152160.

The official ISRIC integer conversion factors are frozen in
`configs/data_contract_v2.yaml`. Raw INT16 values are never treated as final physical
units without those conversions.

### 4.5 SIAP municipal agricultural closure

Local source:

```text
data/raw/external/SIAP/
```

Audited history:

- 2003–2025 inclusive;
- previously missing 2023 was acquired;
- duplicate copies for 2004, 2005 and 2025 are byte-identical by SHA256;
- canonical unsuffixed files are used.

Why acquired:

- historical municipal barley yield context;
- municipal baseline/trend/anomaly features;
- possible strong contemporaneous 2025 proxy for competition-mode modeling.

Stable municipal key:

```text
CVEGEO = zfill(Idestado, 2) + zfill(Idmunicipio, 3)
```

Never join on `Idmunicipio` alone.

Schema normalization:

```text
Nomcultivo        -> Nomcultivo
Nomcultivo Sin Um -> Nomcultivo   # 2015–2020
```

Discovered barley categories:

- `Cebada grano`;
- `Cebada forrajera en verde`.

Observed cycles/modalities include:

- `Otoño-Invierno`;
- `Primavera-Verano`;
- `Riego`;
- `Temporal`.

Do **not** mix grain and forage barley in yield features.

SIAP 2025 municipal yield is explicitly a `competition` feature because it can be a
contemporaneous outcome proxy. It must remain separate from scientifically prospective
`clean` features.

### 4.6 INEGI CEM high-resolution elevation

Local source:

```text
data/raw/external/Inegi/
```

Audited:

- one state raster each for Hidalgo, Puebla and Tlaxcala;
- approximately 15 m;
- CRS `EPSG:6365`.

Why acquired:

- finer local relief/elevation than the official 120 m source;
- potential terrain heterogeneity, roughness and aspect features.

It must stay source-prefixed separately from the official topography.

### 4.7 ERA5-Land

ERA5-Land was investigated and download tooling was implemented, but repeated CDS
request-cost/backend failures made acquisition unreliable.

Decision:

- ERA5 is **not required** for the current pipeline;
- it remains optional/partial;
- downstream code must not consume it unless a future completeness audit explicitly
  certifies it.

Why it was acceptable to stop:

- CHIRPS already covers precipitation;
- WaPOR provides water-use/productivity information;
- official temperature data are already available.

---

## 5. Data Contract v2

Machine-readable contract:

```text
configs/data_contract_v2.yaml
```

Audit implementation:

```text
tools/audit_data_contract_v2.py
```

Human documentation:

```text
docs/DATA_CONTRACT_V2.md
docs/DATA_SOURCES.md
docs/VARIABLES.md
```

The full workstation audit was run after all missing/ambiguous sources were resolved.
The user confirmed the final audit contains **no FAIL findings**.

The audit verifies, among other things:

- exact parcel counts and split;
- hidden-target visibility;
- BASIC/PRO shapes, sensors, strict date parsing and ID coverage;
- parcel CRS/validity;
- official climate continuity;
- municipal source completeness;
- CHIRPS daily continuity;
- WaPOR metadata;
- SoilGrids inventory/CRS semantics;
- SIAP annual coverage/schema/duplicates/barley discovery;
- CEM inventory/CRS;
- ERA5 optional policy.

This contract is now the gate that must be rerun whenever raw/source data change.

---

## 6. Shared integration fixtures

Versioned location:

```text
data/samples/integration/
```

Purpose:

The original tiny catalog samples were useful for inspecting one file at a time, but they
did not share parcel IDs and therefore could not test real joins. The integration fixture
solves that by using the **same 12 parcels across the pipeline**.

Selected IDs:

```text
AGC_020
AGC_048
AGC_050
AGC_054
AGC_065
AGC_076
AGC_090
AGC_108
AGC_119
AGC_129
AGC_183
AGC_186
```

Selection design:

- two parcels per `Estado × CONJUNTO` stratum;
- both training and prediction partitions;
- target-free diversity using parcel area, cloudiness, missingness and observation
  coverage;
- known edge cases `AGC_020`, `AGC_048`, `AGC_129` forced into the set;
- `RENDIMIENTO_T_HA` is never used for choosing the fixtures.

Validated snapshot:

| Artifact | Count |
|---|---:|
| Parcels | 12 |
| BASIC rows | 192 |
| PRO rows | 96 |
| SIAP rows | 201 |
| Files | 63 |
| Total size | 3.77 MiB |
| Builder validation | PASS |

Included families:

```text
split.csv
parcels.geojson
basic.csv
pro.csv
admin.csv
municipalities.geojson
siap.csv
climate/
chirps_daily/
topography/
soilgrids/
wapor/
cem/
```

Builder:

```text
tools/build_integration_fixtures.py
```

The integration fixtures are deliberately committed to Git so CI and future agents can
test end-to-end extraction/join behavior without possessing the full local raw archive.

---

## 7. What cleaning/normalization has actually happened

At this checkpoint we have **source-aware normalization rules and QA**, not a monolithic
"cleaned CSV".

Completed normalization decisions include:

- canonical parcel ID `ID_POLIGONO`;
- canonical parcel area `area_ha`;
- strict BASIC/PRO date parsing as `%d/%m/%Y`;
- source/sensor provenance retained;
- official parcel CRS verified as `EPSG:4326`;
- explicit CRS transformations before spatial operations;
- municipality administrative-vs-spatial policy frozen;
- `CVEGEO` construction frozen;
- SIAP historical crop-name alias normalized;
- SIAP duplicate years resolved by hashes;
- SoilGrids physical-unit conversion policy frozen;
- WaPOR duration-weighted aggregation semantics frozen;
- ERA5 excluded unless later certified.

What has **not** happened yet:

- no final one-row-per-parcel feature matrix for all 197 parcels;
- no final imputation/scaling/PCA;
- no learned preprocessing fitted to the full labeled data;
- no final feature selection;
- no production model;
- no final prediction cutoff/horizon decision.

Those belong to later checkpoints.

---

## 8. Clean vs competition data modes

Two explicit feature modes are required.

### clean

Scientifically prospective features available by a declared prediction cutoff. Learned
preprocessing must be fitted inside CV folds.

### competition

Public/transductive information permitted by challenge rules but potentially unavailable
in a strict real-time forecast.

Examples:

- full-season 2025 public covariates;
- SIAP 2025 municipal yield;
- unsupervised transforms fitted on all 197 predictor rows.

Neither mode may ever use the 59 hidden parcel targets.

---

## 9. Reproducibility commands

Audit the complete local data tree:

```powershell
python tools\audit_data_contract_v2.py
```

Generate integration fixtures:

```powershell
python tools\build_integration_fixtures.py
```

Run tests without OneDrive pytest cache issues:

```powershell
python -m pytest -q -p no:cacheprovider
```

Regenerate structural data catalog:

```powershell
python tools\build_data_catalog.py
```

Regenerate external-file manifest:

```powershell
python tools\build_external_data_manifest.py
```

---

## 10. Files that explain this checkpoint

Read in this order for data work:

1. `checkpoints/01_data/README.md` — this milestone narrative.
2. `docs/DATA_SOURCES.md` — detailed current source inventory.
3. `docs/DATA_CONTRACT_V2.md` — audit semantics and policies.
4. `configs/data_contract_v2.yaml` — machine-readable contract.
5. `data/.ai_handoff` — current operational handoff.
6. `docs/VARIABLES.md` — source-backed variable dictionary.
7. `data/samples/integration/fixture_manifest.json` — exact fixture snapshot.

Immutable source evidence always outranks derived documentation.

---

## 11. Next checkpoint

The next milestone should be **Checkpoint 02 — Features**.

Before that checkpoint can be created, the project should:

1. implement reusable extraction functions for each source family;
2. exercise them first against the 12 shared integration parcels;
3. build a deterministic table with exactly 197 unique `ID_POLIGONO` rows;
4. namespace columns by source;
5. attach feature provenance and `clean`/`competition` membership;
6. assert no many-to-many joins or hidden-target leakage;
7. freeze the resulting schema and tests.

A reasonable next directory will be:

```text
checkpoints/02_features/
```

That checkpoint should describe the final feature table, transformations, feature
families, leakage classification and validation invariants before model selection begins.
