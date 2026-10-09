# Checkpoint 07 — RMSE Offensive: New Information + Prithvi-EO-2.0

**Status:** ACQUISITION COMPLETED FOR 4/5 SOURCES; 07A2 HLS/SMAP PROCESSORS IMPLEMENTED / WORKSTATION RUN PENDING  
**Track:** competition-only  
**Primary objective:** minimize RMSE on the fixed 59 FIRA parcels  
**Incumbent:** Local04D  
**Hidden 59 FIRA y may be read/scored:** NO

## Why Checkpoint 07 exists

Checkpoints 04–06 extracted substantial signal from the existing deterministic
feature table. Checkpoint 06 also showed that calibration/routing on the same
information can look better on the complete development table while failing to
remain stable across pseudo-competitions.

Checkpoint 07 therefore changes the source of information instead of continuing
to tune the same residual corrections.

The objective is explicitly:

    choose the model with the lowest expected RMSE on the fixed 59 hidden FIRA yields

Validation is evidence used to make that decision. LOSO remains a useful
diagnostic, but it is not itself the optimization target and is not an automatic
veto in Checkpoint 07.

## New data

Checkpoint 07 uses five new resources.

### 1. NASA HLS v2

Download all HLSL30 and HLSS30 granules intersecting the challenge parcels
during April–October 2025.

HLS is the native input family used to pretrain Prithvi-EO-2.0. It also gives
access to the actual raster distribution inside and around each parcel rather
than only the existing parcel-level index summaries.

Prithvi semantic band order:

    Blue, Green, Red, Narrow NIR, SWIR1, SWIR2

Sensor-specific HLS bands must be mapped to that common semantic order during
chip construction.

### 2. SMAP Enhanced L3 9 km soil moisture

Download `SPL3SMP_E` version 6 for the complete April–October 2025 season.

This daily L-band microwave product is spatially coarse relative to a parcel,
so it is not treated as a parcel-resolution measurement. It is used as a
regional soil-moisture/drought-state signal and combined with finer Sentinel-1,
optical and weather measurements.

### 3. Sentinel-1 GRD

Download every available dual-polarization IW acquisition over the challenge
region in the target season.

The downloader requests terrain-corrected, orthorectified regional rasters with:

    VV
    VH
    local incidence angle
    scattering area
    shadow mask
    data mask

at approximately 30 m output resolution.

This is genuinely new measurement physics relative to the current optical
BASIC/PRO tables.

### 4. AgERA5 v2 daily weather

Use the official time-chunked ARCO Zarr and extract only the challenge region
and April–October 2025.

Retain daily variables for:

- Tmin/Tmean/Tmax;
- dew point;
- RH min/max;
- precipitation and precipitation duration;
- FAO56 reference evapotranspiration;
- solar radiation;
- vapour pressure and VPD;
- 10 m wind;
- cloud cover.

Derived parcel features will include GDD, dry spells, water deficit,
heat/cold events, stage-specific precipitation/ET0/VPD/radiation and
interactions with satellite phenology.

### 5. Prithvi-EO-2.0 weights

Prepare both:

    Prithvi-EO-2.0-300M-TL
    Prithvi-EO-2.0-600M-TL

The TL variants include temporal and location embeddings and are appropriate
for the four-frame parcel sequences planned here.

## Phase 07A — acquisition, audit and streaming preparation

**Observed workstation audit, 2026-10-09:** 180.140 GiB total raw;
HLS 41.846 GiB (197 scenes, 0 missing 6-band/Fmask sets),
SMAP 135.492 GiB (215 HDF5), Sentinel-1 2.801 GiB (66 files),
Prithvi weights 3.693 GiB separately; 935.644 GiB free disk.
HLS/SMAP/AgERA5/Prithvi download statuses completed. Sentinel-1 was
still marked running, not assumed complete.

The user has all the HLS band files necessary to begin.
**DO NOT download anything again for 07A2.**

Next runner instructions and protocol:
`docs/CHECKPOINT_07A_PROCESSING.md`.

HLS processing now writes a resumable X-only longitudinal parcel-date panel
and four-frame aligned masked Prithvi input chips. SMAP processing samples
only unique nearby regional cells from its very large global HDF5 files.
Both pipelines have isolated smoke modes.

## Historical acquisition commands (do not rerun if files are present)

### Acquisition and audit

Run:

    python tools\download_checkpoint_07_data.py --preflight
    python tools\download_checkpoint_07_data.py --catalog-only --sources hls sentinel1
    python tools\download_checkpoint_07_data.py --sources all

Raw files are local-only under:

    data/raw/checkpoint_07/
    models/checkpoint_07/

They must not be committed to Git.

The tracked acquisition logic and config are the provenance source of truth.

07A is complete only when:

- HLS catalogue/download inventory is recorded;
- Sentinel-1 date/orbit inventory is recorded;
- SMAP daily inventory is recorded;
- AgERA5 has continuous daily coverage through the target period;
- both Prithvi model snapshots are present;
- raster/parcel overlap is verified;
- source dates, CRS, bands and missingness are audited.

## 07A.2 — Processing 180+ GiB without loading full satellite scenes

The raw HLS tiles are **archive inputs**, not an ML training matrix.
The system must never materialize every pixel/date/band in RAM.

Before extraction, run the metadata-only audit:

    python tools/audit_checkpoint_07_storage.py --sample-raster-headers 8

Review the generated report:

    reports/checkpoint_07/storage_inventory.json

Its source totals, free disk space, and required-band gaps determine whether
acquisition is actually complete. A source-status exit_code=0 alone is NOT
a parcel-coverage/quality guarantee. Sentinel-1 may still be in progress;
the audit is safe to run concurrently.

The next processing deliverables are separate:

1. **07A2 scene catalog** — map each HLS/S1 granule to dates, grid/tile, cloud
   QA, bands and parcel intersections using coordinates only. Do not read full
   scene pixels.
2. **07A3 original temporal parcel panel** — group parcels by scene/tile,
   read raster windows with Rasterio; intersect true parcel masks; record
   valid-pixel count, date, sensor, orbit, band statistics and spatial
   quantiles. Preserve daily/acquisition dates and masks, then write Parquet.
3. **07A4 Prithvi chips** — for each of 197 parcels select four X-only
   Fmask-qualified HLS dates from the predeclared windows. Read six bands
   plus Fmask only within needed local windows, save 197 x 4 x 6 x 224 x 224
   as compressed int16/chunked storage, plus masks, dates and metadata.
   Read as small batches (1–4) into GPU for frozen embedding extraction.
4. **07A5 climate/SMAP context** — subset only the grid cells/timestamps
   containing each parcel, preserving native spatial resolution. Store a
   compact date/parcel panel and derived agronomic phase summaries.

Order of execution is **scene-first** (read a tile/date once, process every
parcel within it) rather than reopening every image 197 times.

For scale, exactly 197 parcels x 4 dates x 6 bands x 224 x 224 pixels uses
about 0.44 GiB of raw int16 reflectance values (before masks/compression).
This is not a promise about actual output size: extra frames, masks, context
and compression change storage requirements.

HLS reflectance values are stored as scaled int16 (scale factor 0.0001).
For Prithvi, preserve the raw encoding as appropriate to the official model
normalization and document every conversion; do not inadvertently normalize
the image twice. Respect HLS Fmask, NoData and sensor-specific band IDs.

A 224 x 224 HLS chip spans roughly 6.72 x 6.72 km at 30 m, often much wider
than the true parcel. Preserve the polygon mask and compare parcel-only token
pooling vs parcel-plus-context to avoid learning mainly adjacent land.

**Deletion policy:** No automatic deletion. After QA verifies chips, temporal
panels and provenance, raw granules can be archived or deliberately removed by
a human, not by the processing runner.

## Phase 07B — Local07 with new information

**Implementation update 2026-10-09:** X-only high-dimensional temporal
feature factory, balanced nonlinear cross-source products and fold-local
PCA+Ridge/MLP benchmark now exist. See
`docs/CHECKPOINT_07B_HIGH_DIMENSIONAL.md` and
`tools/run_checkpoint_07_highdim.py`. Workstation results pending;
feature count/variance explained do not establish RMSE improvement.



The first model line stays close to the strongest incumbent so the effect of
the new data can be measured cleanly.

Build new deterministic parcel features from:

### Sentinel-1

Per orbit direction and jointly:

    VV and VH linear + dB summaries
    VV/VH and VV-VH
    AUC / min / max / amplitude
    temporal slopes
    change points
    early/mid/late summaries
    within-parcel quantiles and IQR
    spatial heterogeneity
    valid/shadow fractions

### SMAP

    daily AM/PM surface soil moisture
    seasonal mean/min/max
    dry-down slopes
    low-moisture duration
    anomalies relative to the 2025 parcel-region season
    interactions with Sentinel-1 and precipitation

### AgERA5

    GDD
    seasonal and stage precipitation
    rain-day counts
    maximum dry spell
    P - ET0 and cumulative water deficit
    VPD summaries/extremes
    heat days
    cold/frost days
    radiation
    wind
    phenology x stress interactions

### HLS raster distributions

Use the six Prithvi-compatible reflectance bands plus parcel-level spatial
distribution/texture features. Do not simply recreate the already-existing
NDVI means.

Evaluate:

1. frozen Local04D;
2. Local04D-style local Ridge with the augmented representation;
3. a small predeclared refresh of local k and Ridge alpha because new
   modalities alter feature-space geometry.

This is not a broad model zoo.

## Phase 07C — Prithvi-EO-2.0

### Four-frame input

For each parcel select one HLS observation from each frozen window:

    2025-04-15 .. 2025-05-31
    2025-06-01 .. 2025-07-15
    2025-07-16 .. 2025-08-31
    2025-09-01 .. 2025-10-15

Scene choice uses only X-derived image quality: prefer the observation with the
largest valid, non-cloud/non-shadow parcel coverage from HLS Fmask.

Create a 224 x 224, 30 m chip centered on each parcel for each frame. Preserve
the parcel mask inside the larger spatial context.

### Stage 1: frozen encoders

First run both pretrained encoders without target-aware fine-tuning.

Extract token embeddings and create at least:

- mean of tokens overlapping the parcel;
- mean of parcel + surrounding context;
- temporal mean/change embeddings.

Fit compact heads using only the 138 labels:

    Ridge
    PLS
    Gaussian Process

All head fitting and dimensionality reduction must be honest within each
pseudo-competition split.

### Stage 2: conditional fine-tuning

Do not fine-tune hundreds of millions of parameters merely because GPU is
available.

Only if frozen Prithvi embeddings beat or materially complement the current
feature representation should 07C continue to:

- last-block/last-stage unfreezing;
- parameter-efficient adapters/LoRA where supported;
- strong weight decay and early stopping;
- multiple seeds.

The validation unit remains the parcel.

## Phase 07D — Local + Prithvi fusion

The final hypothesis is that the two predictors contain different information:

    Local07:
        structured parcel similarity + explicit physical/agronomic features

    Prithvi07:
        learned spatial/spectral/temporal HLS representation

Generate honest OOF predictions for both and evaluate:

    Local04D
    Local07
    Prithvi07
    fixed Local07/Prithvi blends
    non-negative OOF-learned Local07/Prithvi blend

No stacking feature may use in-sample predictions.

## Validation and final selection

### Development

The existing 16 target-matched splits remain the development laboratory.

Use them to compare representations, detect broken approaches and reduce each
family to at most two finalists.

LOSO is reported as a stability diagnostic, not an absolute veto.

### Fresh confirmation

Before scoring the final shortlisted candidates, create a fresh X-only
target-matched split bank with a new frozen seed.

Fresh confirmation should contain only:

- Local04D;
- at most two Local07 finalists;
- at most two Prithvi07 finalists;
- at most two frozen fusion rules.

No tuning occurs after looking at fresh-confirmation y.

### Competition decision

The ranking priority is:

1. fresh pooled RMSE;
2. fresh mean RMSE;
3. fresh worst-split RMSE;
4. development pooled RMSE as secondary evidence.

A new final predictor should beat Local04D on both fresh pooled and fresh mean
RMSE.

The real 59 hidden yields are never inspected. Once the rule is frozen, use all
138 labels and all rule-permitted X from the 197 parcels to generate the 59
competition predictions.

## Stop conditions

Checkpoint 07 should not be killed merely because one LOSO selector is noisy.

Stop a branch when:

- the new modality adds no honest OOF signal;
- Prithvi embeddings do not beat or complement compact baselines;
- a finalist loses on the untouched fresh confirmation bank;
- data quality/coverage makes the modality unreliable.

The goal is the best defensible estimate of the 59 values, not the simplest
model.
