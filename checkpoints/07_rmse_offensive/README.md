# Checkpoint 07 — RMSE Offensive

**Status:** DATA ACQUISITION IMPLEMENTED / MODELING PENDING  
**Canonical plan:** docs/CHECKPOINT_07_PLAN.md  
**Incumbent:** Local04D

Checkpoint 07 opens a genuinely new-information line after the closure of
Checkpoint 06. It downloads HLS v2, SMAP soil moisture, Sentinel-1 GRD and AgERA5 v2 and prepares
Prithvi-EO-2.0-TL weights.

The two primary model families are:

1. **Local07** — the Local04D/local-Ridge idea with SAR + daily climate + HLS
   spatial information;
2. **Prithvi07** — Prithvi-EO-2.0 frozen embeddings first, with conditional
   partial fine-tuning only if frozen representations show honest signal.

The final stage tests Local07 + Prithvi07 fusion.

Run acquisition with:

    python tools\download_checkpoint_07_data.py --preflight
    python tools\download_checkpoint_07_data.py --catalog-only --sources hls sentinel1
    python tools\download_checkpoint_07_data.py --sources all

Large raw files and model weights are local-only and must not be committed.

Local04D and reports/checkpoint_05/final_predictions.csv remain canonical until
Checkpoint 07 fresh confirmation supports a replacement.


## Raw-data storage audit

With HLS/SMAP/AgERA5/Prithvi downloaded and Sentinel-1 potentially running,
audit the size and scene-band inventory without loading raster pixels:

    python tools/audit_checkpoint_07_storage.py --sample-raster-headers 8

Inspect reports/checkpoint_07/storage_inventory.json for per-source GiB,
available disk space and missing HLS bands/Fmask. This metadata-only tool is
safe during acquisition. Then implement the scene-first streaming extraction
described in docs/CHECKPOINT_07_PLAN.md. Do not train on 180 GB of raw tiles.
