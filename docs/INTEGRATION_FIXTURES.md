# Integration fixtures

Updated: 2026-09-18

The Data Contract v2 audit has been closed locally without `FAIL` findings. The next
stage is a small **joinable real-data fixture** built around the same representative
parcels across all major source families.

These fixtures are for development/testing only. They are not a training subset and must
not be used to estimate model performance.

## Builder

Run from the repository root:

```powershell
python tools\build_integration_fixtures.py
```

The builder requires the complete local workstation data tree, including gitignored
external sources.

Configuration is frozen in:

```text
configs/data_contract_v2.yaml
```

under `integration_fixtures`.

## Parcel selection

The fixture contains exactly 12 parcels:

- two per `Estado × CONJUNTO` stratum;
- all three challenge states;
- both `ENTRENAMIENTO` and `PREDICCION`;
- selection based only on target-free metadata:
  - parcel area;
  - BASIC/PRO cloud level;
  - BASIC/PRO missingness;
  - sensor/row coverage.

Within each stratum the algorithm chooses parcels that are far apart in rank-scaled
coverage/quality space.

Three known spatial edge cases are forced into the fixture:

- `AGC_020`;
- `AGC_048`;
- `AGC_129`.

The target is never used to select parcels.

## Generated structure

```text
data/samples/integration/
  fixture_manifest.json
  selected_parcels.csv
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

### Tabular/vector fixtures

- `split.csv`: the official split rows for the 12 IDs.
- `parcels.geojson`: the exact parcel geometries.
- `basic.csv`: time-spanning rows per parcel and BASIC sensor.
- `pro.csv`: time-spanning Planet rows per parcel.
- `admin.csv`: canonical administrative join keys plus spatial-QA metadata.
- `municipalities.geojson`: relevant INEGI municipalities.
- `siap.csv`: historical SIAP barley records for the selected municipalities.
- `selected_parcels.csv`: target-free selection metrics used to make the fixture.

### Raster/NetCDF fixtures

The builder masks/compresses real rasters around the selected parcel geometries.

It includes:

- representative official `PREC/Tmin/Tmax` rasters for Apr/Jul/Oct 2025;
- representative CHIRPS daily rasters for Apr 1, Jul 16 and Oct 31;
- both official topography rasters;
- all 36 SoilGrids property/depth rasters;
- the three high-resolution INEGI CEM state products, cropped to selected parcels;
- both WaPOR AETI/NPP NetCDFs with all 21 dekads.

WaPOR files are rewritten through the isolated NetCDF metadata-sanitizing worker so the
known NPP UTF-8 metadata issue is not propagated into the versioned integration fixture.

## Administrative join policy represented by the fixture

Administrative/SIAP joins preserve the official parcel `Estado` and `Municipio`
attributes as primary.

The INEGI polygon overlay is QA/fallback metadata.

Known cases deliberately represented:

- `AGC_020`: 98.65% Emiliano Zapata, 1.35% Calpulalpan.
- `AGC_129`: 85.43% Singuilucan, 14.57% Cuautepec de Hinojosa.
- `AGC_048`: official municipality is Nanacamilpa de Mariano Arista while the spatial
  overlay gives Calpulalpan; official municipality remains primary for SIAP/admin joins.

## SIAP normalization represented by the fixture

The source-era crop-name aliases are normalized as:

```text
Nomcultivo        -> Nomcultivo
Nomcultivo Sin Um -> Nomcultivo
```

The fixture retains barley rows from the actual source history. Downstream grain-yield
logic must distinguish `Cebada grano` from `Cebada forrajera en verde`.

Duplicate 2004/2005/2025 source copies were confirmed byte-identical by the audit; the
canonical unsuffixed files are used.

## Validation performed by the builder

Generation fails unless:

- exactly 12 representative IDs are selected;
- all forced edge-case IDs are present;
- split, parcel geometry and admin tables have one row per parcel;
- BASIC/PRO/admin/parcels contain the exact same 12 IDs;
- hidden `PREDICCION` targets remain absent;
- SIAP rows can be linked through selected municipal `CVEGEO` values.

After generation, `fixture_manifest.json` records the exact selected IDs, row counts,
file list and total fixture size.

## Next gate

Once the generated fixture is versioned and its tests pass, the next implementation stage
is the deterministic **197-row parcel master feature table**. Production extraction logic
should first be exercised against this integration fixture before running against the full
raw data tree.


## Generated fixture snapshot

The first complete workstation generation passed validation on 2026-09-18.

Selected parcel IDs:

`AGC_020`, `AGC_048`, `AGC_050`, `AGC_054`, `AGC_065`, `AGC_076`, `AGC_090`, `AGC_108`, `AGC_119`, `AGC_129`, `AGC_183`, `AGC_186`

Observed fixture size/counts:

- 12 parcels;
- 192 BASIC rows;
- 96 PRO rows;
- 201 SIAP rows;
- 63 generated files;
- 3.77 MiB total;
- builder validation: **PASS**.

These values are now the expected reference snapshot unless the fixture-selection contract
is intentionally changed.
