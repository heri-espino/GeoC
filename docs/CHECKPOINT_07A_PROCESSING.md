# Checkpoint 07A.2 — From 180 GiB of raw imagery to parcel-level data

**Status:** HLS COMPLETED ON WORKSTATION; SMAP v006 COORDINATE FALLBACK FIX IMPLEMENTED / RETEST PENDING  
**Track:** competition-only; no use of the 59 hidden yields  
**Purpose:** convert HLS and SMAP source archives into small X-only inputs for
Local07 and Prithvi07. This phase does not train or promote any model.

## Acquisition inventory — observed on the university workstation (2026-10-09)

User-provided output from the tracked metadata-only audit:

| Source | GiB | Files | Current interpretation |
|---|---:|---:|---|
| HLS v2 | 41.846 | 3,250 | 197 parsed complete granules; all Prithvi bands and Fmask present |
| SMAP v006 | 135.492 | 215 | Largest footprint (~75% of raw); global daily HDF5 product |
| Sentinel-1 GRD | 2.801 | 66 | Available locally, acquisition status previously reported running |
| AgERA5 v2 | ~0 | 2 | Reported complete, source data quality still needs validation |
| Acquisition manifests | ~0 | 9 | Provenance/status metadata |
| **Total raw** | **180.140** | | |
| Prithvi weights | 3.693 | | Separately from raw |
| Disk free | 935.644 | | Enough to keep raw + extracted data during audit |

A zero missing-band count does NOT establish cloud-free HLS coverage.
The acquisition status JSON saying "running" does not itself establish whether
a downloader process is still alive.

## Why not train with all 180 GiB?

The supervised task has just 138 known yields and 59 target parcels. HLS tiles
and SMAP files cover much larger areas than the parcels.

The processing strategy reads only relevant raster windows or SMAP grid cells.
The 180 GiB are immutable evidence, not the in-memory modeling matrix.

HLS is a 30 m product; SMAP is nominally 9 km. Do not attribute SMAP's
regional estimates to sub-field spatial detail.

## Implemented stage A — HLS scene-first processing

Code:

- src/geocebada/data/checkpoint07_hls.py
- tools/run_checkpoint_07_processing.py
- tests/test_checkpoint07_processing.py

Processing order:

1. Parse filenames and group all six reflectance bands + Fmask by scene.
2. Process **scene-first**, opening each band's COG once per scene and reading
   only polygon windows for the parcels intersecting it.
3. Reproject parcel polygons into the image CRS without changing their source.
4. Apply HLS v2 Fmask, excluding clouds (bit 1), adjacent cloud/shadow (2),
   shadows (3), snow/ice (4), water (5) and fill.
5. Calculate per-date, per-sensor and per-parcel Blue/Green/Red/NIR/SWIR1/SWIR2
   mean, std and quantiles, NDVI and NIR-SWIR1 index, valid pixel counts and
   fractions. Preserve per-date observations; do not collapse into one mean.
6. Save each scene's results atomically in a resumable JSON shard, then merge
   into the compact longitudinal CSV.
7. Use only scene quality from X to select one HLS granule within each of the
   frozen four temporal windows for each parcel.
8. Read each selected scene on a **fixed projected 30 m grid around that
   parcel**, producing aligned 224 x 224 frames.
9. Save packed HLS int16 DN values (not pre-normalized floats), valid masks,
   polygon mask, acquisition dates, centroid latitude/longitude, source CRS
   and fixed-grid affine transform as compressed parcel-level NPZ.
10. If any quality window is missing, record an incomplete sequence; do not
    silently repeat another date or impute a fake observation.

L30 semantic mapping:
B02/B03/B04/B05/B06/B07.

S30 semantic mapping:
B02/B03/B04/B8A/B11/B12.

HLS digital numbers have a 0.0001 reflectance scaling factor. For the
longitudinal Local07 panel, use physical reflectance. For the Prithvi chips,
preserve raw HLS DN so the later encoder loader can apply **exactly the
normalization from its official pretrained config once**, avoiding double
scaling or double normalization.

The 224 x 224 chip spans 6.72 x 6.72 km and may include neighboring parcels
and non-crop pixels. Do not treat it as an all-parcel image; the polygon mask
is saved to support parcel-only token pooling vs parcel-plus-context.

### HLS commands (Windows PowerShell)

    git pull --ff-only
    conda activate geocebada
    python -m pip install -e ".[dev,geo,checkpoint07]"

    python tools\run_checkpoint_07_processing.py --preflight

Test the pipeline without contaminating full outputs:

    python tools\run_checkpoint_07_processing.py --stage panel --max-scenes 3 --max-parcels 3

Run all 197 scenes, preserve dates, and create Prithvi chips:

    python tools\run_checkpoint_07_processing.py --stage all

Output (gitignored):

    data/processed/checkpoint_07/hls_parcel_observations.csv
    data/processed/checkpoint_07/hls_scene_shards/
    data/processed/checkpoint_07/prithvi_chips/<ID_POLIGONO>.npz
    data/processed/checkpoint_07/prithvi_chip_manifest.json
    data/processed/checkpoint_07/processing_report.json

The pipeline is resumable per scene and per parcel. An interrupted run can be
re-executed with the same command. The --force switch rebuilds all output.

### Strict quality limitations

A scene with valid bands but with no cloud-free pixels inside a polygon does
not become a valid Prithvi frame. Tiny polygons may intersect just one or a few
30 m pixels; coverage stats therefore require scrutiny. We need to inspect:

- distribution of valid HLS observation count per parcel;
- parcel-window missingness for each of the four periods;
- how many parcels have all four Prithvi frames;
- date coverage of labeled versus 59 prediction parcels;
- any source/sensor/municipality imbalance.

An incomplete Prithvi sequence is not a training example until a defensible
X-only missing-frame strategy has been defined, tested and documented.

## Workstation HLS completion and SMAP v006 fix (2026-10-10)

Confirmed from workstation execution log:
- All 197/197 HLS granules processed successfully, producing **29,592**
  parcel-date observation rows; all 197 parcels were covered.
- All 197/197 parcels have **four HLS quality-filtered frames** for Prithvi.
- The original HLS archive was not changed.
- 214 downloaded SMAP HDF5 files were found in preflight, but both
  smoke and full SMAP originally failed because the code required explicit
  `latitude` and `longitude` datasets inside the HDF5 AM group.
- The resulting missing `smap_parcel_daily.csv` blocked Checkpoint 07B;
  this is NOT an HLS or PCA bug.

The revised SMAP reader supports:
- the standard global **EASE-Grid 2.0 EPSG:6933** geolocation when
  latitude/longitude arrays are absent, validated by exact full-global
  HDF5 shape 1624 x 3856; unsupported subset/polar grids are rejected;
- native `_pm` dataset suffixes for the PM overpass in v006;
- QA bit-flag validity, HDF5 missing/fill data checks, individual cell
  access without loading global moisture arrays;
- reporting the selected geolocation method and distinct 9-km grid cells.

Official references:
- https://nsidc.org/sites/default/files/documents/user-guide/spl3smp_e-v006-userguide.pdf
- https://nsidc.org/data/ease
- https://nsidc.org/data/spl3smp_e/versions/6

**Important grid caveat:** the NSIDC v006 user guide labels the 9-km
grid cell size as 9,024.31 m, but the canonical NSIDC EASE-Grid 2.0
reference lists approximately 9,008.05 m for 3856 global columns. The
implementation uses the canonical extent and column count to derive
spacing, with no empirically guessed shifts. The final cell rows/columns
must be checked against actual lat/lon metadata or a small known-point
subset where possible. SMAP is regional; many parcels share 9-km cells.

Do **not** rerun HLS. The next workstation commands are:

```powershell
git pull --ff-only
conda activate geocebada
python tools\run_checkpoint_07_smap.py --max-files 2
python tools\run_checkpoint_07_smap.py
python tools\run_checkpoint_07_highdim.py --preflight --require-hls --require-smap
python tools\run_checkpoint_07_highdim.py --smoke --require-hls --require-smap
```

If an HDF5 file has a non-global shape or unexpected naming after this
fix, report the new exact error and inspect its group layout; do not
silently invent coordinates or redownload everything.

## Implemented stage B — SMAP regional daily extraction

Code:

- src/geocebada/data/checkpoint07_smap.py
- tools/run_checkpoint_07_smap.py

SMAP HDF5 daily files are **not** concatenated into enormous arrays.

1. Read the latitude/longitude grid from one reference daily HDF5 file.
2. Locate the nearest valid 9-km cell for every parcel centroid using a
   spatial KD-tree limited to the challenge area.
3. Read only those unique cells from subsequent daily AM and PM datasets.
4. Preserve retrieval quality flags. Keep soil moisture only if the
   recommended-quality flag is 0 or 8, and data are physically plausible.
5. Save date, parcel ID, AM/PM, value, quality, native grid row/column,
   source filename, and distance to grid center.

Command sequence:

    python tools\run_checkpoint_07_smap.py --preflight
    python tools\run_checkpoint_07_smap.py --max-files 2
    python tools\run_checkpoint_07_smap.py

Output (gitignored):

    data/processed/checkpoint_07/smap_parcel_daily.csv
    data/processed/checkpoint_07/smap_processing_report.json

The two-file smoke test is isolated under
data/processed/checkpoint_07/smoke_smap/.

Many parcels will share the same 9-km cell. These measurements must never be
treated as 197 spatially independent soil-moisture estimates.

## What follows this execution

**07A.4 — Coverage and QC gate:** read both generated processing reports and
the HLS chip manifest, inspect counts/dates/quality by labeled vs target
parcel. Do not train until extraction correctness is verified.

**07B — Local07:** transform the longitudinal HLS/SMAP data, and later
Sentinel-1/AgERA5, into time-aware fold-safe features; compare frozen Local04D
to enriched local Ridge on exactly the same target-matched splits.

**07C — Prithvi07:** use the actual HLS four-frame chips, normalized according
to the downloaded Prithvi config, to produce frozen 300M/600M embeddings on GPU.
Fit small cross-fitted Ridge/PLS/GP heads on the 138 visible yields.

**07D — Fusion and fresh confirmation:** compare independent models and
honest-OOF blends; freeze finalists, score a new untouched target-matched
confirmation bank, only then replace the 59 predictions if supported.

No processing stage may access the 59 hidden FIRA yields. Raw HLS/SMAP files
are never automatically deleted. All new derived outputs are reproducible and
gitignored.
