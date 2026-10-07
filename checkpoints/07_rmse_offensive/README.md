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
