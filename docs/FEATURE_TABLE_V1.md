# Parcel Feature Table v1

Updated: 2026-09-18

This stage converts the audited GeoCebada source data into deterministic one-row-per-parcel
tables. It is the first model-ready representation after Checkpoint 01.

Implementation:

```text
configs/features_v1.yaml
src/geocebada/features/parcel.py
src/geocebada/data/external.py
tools/build_parcel_feature_table.py
```

## Outputs

The full build writes under:

```text
data/processed/features_v1/
  parcel_features_all.csv
  parcel_features_clean.csv
  parcel_features_competition.csv
  feature_manifest.json
  build_report.json
```

Every table keeps one unique `ID_POLIGONO` row. The full workstation build must produce
exactly 197 rows: 138 `ENTRENAMIENTO` and 59 `PREDICCION`.

`RENDIMIENTO_T_HA` is retained as metadata/target: it is populated only for training
rows and remains missing for all prediction rows.

## Why there are clean and competition tables

The operational prediction cutoff inside the April–October 2025 cycle is not frozen yet.
For that reason, v1 does not pretend that all 2025 information is prospective.

### clean

Contains static and historical information that does not require full 2025 knowledge:

- parcel geometry/area/location;
- administrative keys;
- BASIC historical April–October summaries through 2024;
- historical official climate climatology through 2024;
- SoilGrids;
- official topography;
- high-resolution INEGI CEM;
- SIAP grain-barley history through 2024.

### competition

Contains every clean feature plus public 2025 full-season information that may be useful
for the challenge but is not necessarily available in a strict prospective forecast:

- BASIC April–October 2025 summaries;
- PRO April–October 2025 summaries;
- official 2025 climate;
- CHIRPS daily April–October 2025;
- WaPOR AETI/NPP April–October 2025;
- SIAP 2025 municipal grain-barley proxy.

The hidden 59 parcel targets are never used in either table.

## Base features

The base block includes:

- official area;
- geometry-derived area in hectares;
- area difference/ratio QA;
- projected-geometry centroid transformed to WGS84;
- official state/municipality labels;
- normalized INEGI administrative keys;
- spatial-overlap QA fields.

Administrative/SIAP joins use official parcel municipality labels as primary and retain
INEGI overlap results as QA/fallback.

## Satellite features

BASIC and PRO remain sensor-aware.

For every usable numeric source variable, v1 builds:

- April–October historical summaries: mean/std/min/max;
- historical cloud/record/valid-value coverage;
- monthly historical means for `*_promedio` variables;
- April–October 2025 summaries: mean/std/min/max;
- 2025 coverage;
- monthly 2025 means for `*_promedio` variables.

Sensor names are embedded in feature names, so Sentinel-2, Landsat and Planet values are
never silently merged.

The historical BASIC block is `clean`. Full-season 2025 BASIC/PRO blocks are
`competition`.

## Official climate

For PREC/Tmin/Tmax, the pipeline computes:

- 2022–2024 month-of-year climatology for April–October;
- 2025 monthly parcel values;
- 2025 monthly anomalies versus 2022–2024;
- seasonal precipitation sum;
- seasonal Tmin/Tmax means.

Historical climatology is clean. 2025 values/anomalies are competition features.

## CHIRPS daily

Daily CHIRPS is summarized to:

- April–October precipitation total;
- daily mean;
- maximum 1-day precipitation;
- maximum rolling 5-day precipitation;
- rain-day count (>=1 mm);
- heavy-rain-day count (>=10 mm);
- dry-day count (<1 mm);
- maximum consecutive dry spell;
- monthly totals.

These are full-season 2025 competition features.

## SoilGrids

All 36 property/depth rasters are extracted by parcel with source-backed
`ESRI:54052` fallback only for positively identified SoilGrids files.

Raw integer maps are divided by the Data Contract v2 conversion factor before modeling.

v1 includes:

- mean and standard deviation per property/depth;
- thickness-weighted 0–30 cm mean;
- thickness-weighted 0–60 cm mean.

All SoilGrids features are clean/static.

## Topography

Official topography:

- elevation mean/std/min/max;
- slope mean/std/min/max.

External CEM 15 m:

- state-appropriate elevation mean/std/min/max/range.

All are clean/static.

## SIAP

Only `Cebada grano` is used.

Municipal annual yield is reconstructed consistently as:

[
\text{yield}=
\frac{\text{Volumenproduccion}}{\text{Cosechada}}.
]

Historical features include:

- long-run yield mean/std/trend;
- 2020–2024 mean/std/trend;
- annual 2020–2024 yield;
- 2024 yield, harvested area, production and damage rate.

Competition features include:

- 2025 yield;
- 2025 harvested area;
- 2025 production;
- 2025 damage rate;
- 2025 yield anomaly versus recent historical mean.

## WaPOR

AETI and NPP are extracted from all 21 dekads. Parcel means are calculated on the WaPOR
grid using all-touched geometry masks.

Rates are duration weighted using each band's `number_of_days`.

Features:

- seasonal total;
- duration-weighted mean rate;
- peak rate;
- peak dekad index;
- monthly totals.

These are full-season 2025 competition features.

## Provenance

`feature_manifest.json` records for every feature:

- column name;
- source family;
- mode: `clean` or `competition`;
- human-readable description.

This is the authoritative way to select clean versus competition predictors. Do not infer
mode from ad hoc column lists in notebooks.

## Validation

The builder fails if:

- row count differs from the expected 197;
- `ID_POLIGONO` is null/duplicated;
- split counts differ from 138/59;
- any training target is missing;
- any prediction target is exposed;
- a feature block creates a many-to-many join;
- infinite numeric values appear.

All-all-null generated feature columns are dropped and listed in `build_report.json`.

## Integration-fixture test

The same pipeline is tested against the 12 committed integration parcels with WaPOR
optionally skipped in the minimal CI environment:

```powershell
python tools\build_parcel_feature_table.py --fixture --skip-wapor
```

This validates joins/extractors without needing the workstation's full gitignored raw
archive.

## Full build

On the complete workstation:

```powershell
python tools\build_parcel_feature_table.py
```

The full build requires the GeoCebada `geo` and `external` optional dependencies so
WaPOR NetCDF can be read.

After the full 197-row build passes, create **Checkpoint 02 — Features** documenting the
observed schema, feature counts, missingness and source-family inventory before model
selection begins.
