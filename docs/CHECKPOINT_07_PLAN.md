# Checkpoint 07 — RMSE Offensive: New Information + Prithvi-EO-2.0

**Status:** DATA ACQUISITION IMPLEMENTED / MODELING PENDING  
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

Checkpoint 07 acquires four new resources.

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

### 2. Sentinel-1 GRD

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

### 3. AgERA5 v2 daily weather

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

### 4. Prithvi-EO-2.0 weights

Prepare both:

    Prithvi-EO-2.0-300M-TL
    Prithvi-EO-2.0-600M-TL

The TL variants include temporal and location embeddings and are appropriate
for the four-frame parcel sequences planned here.

## Phase 07A — acquisition and audit

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
- AgERA5 has continuous daily coverage through the target period;
- both Prithvi model snapshots are present;
- raster/parcel overlap is verified;
- source dates, CRS, bands and missingness are audited.

## Phase 07B — Local07 with new information

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
