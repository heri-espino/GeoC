# Data Contract v2

> **Checkpoint 04 note:** this contract remains the source-integrity gate. Its historical
> `clean`/prospective rules must not be confused with the active transductive modeling
> objective. In Checkpoint 04, all rule-permitted `competition` covariates for the 197 parcels
> may participate in X-only transductive inference; hidden target y remains unavailable.


Updated: 2026-09-17

Data Contract v2 is the machine-checkable boundary between raw GeoCebada inputs
and feature engineering. Its purpose is to prevent silent schema drift, incomplete
downloads, accidental many-to-many joins, CRS mistakes and hidden-target leakage.

The machine-readable contract is:

```text
configs/data_contract_v2.yaml
```

The executable audit is:

```text
tools/audit_data_contract_v2.py
```

## Run the audit

From the repository root with the `geocebada` environment active:

```powershell
python tools\audit_data_contract_v2.py
```

It writes:

```text
reports/data_audit_v2.json
reports/data_audit_v2.md
```

The JSON contains full machine-readable details. The Markdown report is a compact
human-readable status table.

For CI or a checkout without local external inputs:

```powershell
python tools\audit_data_contract_v2.py --official-only
```

## Status semantics

| Status | Meaning |
|---|---|
| `PASS` | Observed data satisfy the declared contract. |
| `WARN` | Data can exist, but a documented decision or cleanup is still required. |
| `FAIL` | Blocks construction of the master feature table. |
| `SKIP` | Check was intentionally omitted or the source is optional; this is not evidence of completeness. |

A directory being present does **not** imply that a source is complete.

## Core parcel contract

The final prediction unit is one parcel.

| Item | Contract |
|---|---|
| Canonical key | `ID_POLIGONO` |
| Target | `RENDIMIENTO_T_HA` |
| Split | `CONJUNTO` |
| Total parcels | 197 |
| Training parcels | 138 |
| Prediction parcels | 59 |
| Target cycle | 2025-04-01 through 2025-10-31 |

The hidden target must remain missing for all 59 `PREDICCION` parcels.

## Official geometry normalization

The delivered parcel vector uses source field names that differ from the canonical
internal contract.

Normalize explicitly:

```text
ID_POLIGON -> ID_POLIGONO
área_ha    -> area_ha
```

The directly inspected parcel archive is expected to contain:

- 197 polygons;
- CRS `EPSG:4326`;
- one unique geometry per official parcel ID.

No feature pipeline should rely on aliases after the canonical loading step.

## BASIC and PRO

### BASIC

Expected:

- 107,666 rows × 96 columns;
- 197 parcel IDs;
- exact date format `%d/%m/%Y`;
- date span 2022-01-01 through 2025-12-31;
- sensors `Landsat` and `Sentinel-2`.

### PRO

Expected:

- 47,804 rows × 20 columns;
- 197 parcel IDs;
- exact date format `%d/%m/%Y`;
- date span 2025-01-01 through 2025-12-30;
- sensor `Planet`.

The audit records duplicate parcel/date/sensor keys, sensor-specific missingness and
train-vs-prediction observation coverage. Satellite rows are longitudinal observations,
not independent supervised examples.

## Climate and topography

Official climate must contain complete monthly coverage from January 2022 through
December 2025:

- 48 `PREC` rasters;
- 48 `Tmin` rasters;
- 48 `Tmax` rasters;
- climate CRS `EPSG:4326`.

Official topography expects two GeoTIFF products:

- elevation;
- slope;
- CRS `EPSG:6372`.

## INEGI municipalities

Expected canonical files:

```text
hidalgo_municipios.geojson
puebla_municipios.geojson
tlaxcala_municipios.geojson
```

Expected municipality counts:

| State | Code | Municipalities |
|---|---:|---:|
| Hidalgo | 13 | 84 |
| Puebla | 21 | 217 |
| Tlaxcala | 29 | 60 |

Required join fields:

```text
cvegeo
cve_ent
cve_mun
nomgeo
```

Administrative municipality features use the **official Estado/Municipio labels delivered
with each parcel** as the primary source. INEGI polygons are a spatial QA/fallback layer:
the audit computes largest polygon intersection in a projected CRS and requires a dominant
spatial municipality with at least 80% overlap.

The audit identified two genuine boundary-crossing parcels but both have a clear dominant
municipality: AGC_020 (98.65% Emiliano Zapata, Hidalgo) and AGC_129 (85.43% Singuilucan,
Hidalgo). One documented source discrepancy exists: AGC_048 is labelled
`Nanacamilpa de Mariano Arista` in the official parcel attributes while the INEGI spatial
overlay places it in `Calpulalpan`. Preserve the official label for administrative/SIAP
joins and retain the spatial assignment as QA metadata rather than silently overwriting it.

## CHIRPS daily

Expected external target-season inventory:

- 214 daily GeoTIFFs;
- 2025-04-01 through 2025-10-31 inclusive;
- no missing dates;
- CRS `EPSG:4326`.

This is separate from the official monthly CHIRPS source.

## WaPOR

Expected NetCDF products:

```text
bb_L1-AETI-D_NONE_none.nc
bb_L1-NPP-D_NONE_none.nc
```

Each should contain 21 dekadal bands covering April–October 2025.

Expected rate units:

- AETI: `mm/day`;
- NPP: `gC/m²/day`.

Every dekad carries `number_of_days`. Seasonal totals must therefore use duration
weighting, e.g.

[
AETI_{total}=\sum_k AETI_k d_k.
]

A simple unweighted sum of rate bands is not a seasonal total.

## SoilGrids 250 m

Expected inventory:

- 9 properties;
- 4 depths;
- median prediction `Q0.5`;
- 36 GeoTIFFs.

Depths:

```text
0-5cm
5-15cm
15-30cm
30-60cm
```

Data Contract v2 freezes the official SoilGrids integer conversion factors. SoilGrids
stores integer map values; divide by the factor below to obtain the conventional unit.

| Property | Divide by | Conventional unit |
|---|---:|---|
| `bdod` | 100 | kg/dm³ |
| `cec` | 10 | cmol(c)/kg |
| `cfvo` | 10 | vol% |
| `clay` | 10 | % |
| `nitrogen` | 100 | g/kg |
| `phh2o` | 10 | pH |
| `sand` | 10 | % |
| `silt` | 10 | % |
| `soc` | 10 | g/kg |

Source: ISRIC SoilGrids documentation, “SoilGrids layers”:
https://docs.isric.org/globaldata/soilgrids/SoilGrids_faqs_01.html

The WCS service uses pseudo identifier `152160`; local PROJ uses the equivalent
`ESRI:54052`. If the downloaded GeoTIFF header has no CRS, the fallback is permitted
**only** for files positively identified as these SoilGrids downloads.

## SIAP

Expected annual coverage is 2003–2025 inclusive.

The audit does not hard-code a barley label. It discovers records matching `cebada`
inside the actual SIAP files for state codes 13, 21 and 29 and reports:

- exact `Nomcultivo` labels;
- productive cycles;
- modalities;
- production unit labels.

SIAP schema changes are normalized source-aware: `Nomcultivo` is the standard crop-name
field, while 2015–2020 use `Nomcultivo Sin Um`; both map to canonical `Nomcultivo`.
The audit currently discovers `Cebada grano` and `Cebada forrajera en verde`; downstream
grain-yield features must not mix those crop categories.

Municipal joins must construct:

```text
CVEGEO = zfill(Idestado, 2) + zfill(Idmunicipio, 3)
```

Never join on `Idmunicipio` alone.

Duplicate yearly files are hashed. Duplicate copies are a `WARN` until a deterministic
keep/drop rule is frozen; conflicting duplicate hashes must be investigated.

SIAP 2025 municipal yield belongs to the explicit `competition` feature group because
it can act as a contemporaneous yield proxy.

## High-resolution INEGI CEM

Expected:

- one state raster for codes 13, 21 and 29;
- three GeoTIFFs total;
- source CRS `EPSG:6365`.

These external high-resolution elevation products must remain namespaced separately from
the official 120 m elevation/slope source.

## ERA5-Land

ERA5-Land is not required by Data Contract v2.

The local directory may contain partial files from failed CDS requests. The audit marks
any such files as `WARN`; downstream feature code must not consume ERA5 until a future
dedicated completeness check certifies it.

## Clean vs competition feature groups

Every generated feature must be traceable to one of two modes.

### clean

Prospective/scientifically defensible features that would be available at a declared
prediction cutoff. Learned preprocessing must be fitted inside training folds.

### competition

Public/transductive information allowed by the challenge but kept separate because it may
not correspond to a strict prospective forecast. Examples include:

- full-season 2025 public covariates;
- SIAP 2025 municipal yield;
- unsupervised transforms fitted on predictor rows from all 197 parcels.

Neither mode may use hidden parcel targets.

## Gate before the master table

The 197-row master feature table must not be built until:

1. there are no `FAIL` checks;
2. every `WARN` has a documented resolution or explicit accepted policy;
3. source IDs and geometry are canonical;
4. municipality assignment is deterministic;
5. SIAP duplicates are resolved;
6. SoilGrids conversions are applied explicitly;
7. temporal/raster inventories are complete;
8. ERA5 is either excluded or separately certified.

The next stage after this gate is the shared integration-fixture set and then the
one-row-per-parcel feature table.
