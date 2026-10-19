# Full 138-fold LOOCV: DBSCAN versus saved champions and Local04D

**Status:** Running code, private inputs unavailable to GitHub. This document
gives exact PowerShell commands and how to safely publish aggregate results.

## Known benchmark before running DBSCAN

The user recovered the completed two-model LOOCV:

| Model | RMSE (t/ha) | MAE | R² |
|---|---:|---:|---:|
| Reference26 | 0.4681855146560597 | 0.3336730043055529 | approx 0.69523 (terminal clipped) |
| Compact14 | 0.4696729969475559 | 0.333044183790906 | approx 0.69329 (terminal clipped) |

The old frozen Local04D was reported at RMSE 0.485133 / R² 0.672766
from a separate 138-fold evaluation. Its per-parcel predictions must be
aligned to exactly the peer 138 held-out IDs/y before paired comparisons.

The exploratory fixed-first-46 ablation ranked Reference26 DBSCAN first
(RMSE 0.545343) versus Reference26 KMeans original (0.547536). These
46-fold errors are NOT directly comparable with 138-fold RMSE.

See [recorded historical numbers](PEER_LOOCV_RECORDED_RESULTS.md).

## Objective and model identity

Fit exactly one **new** model recipe: the teammate's Reference26 with
geological-composition DBSCAN replacing only the original KMeans geological
mixing gate. The same 10 input geological dimensions, role transformations,
climate model, local Ridge expert, correction and 26-feature input table are
retained from the teammate's original source.

The DBSCAN gate's training-only inner search uses eight candidates:
`min_samples in {3,5}` and relative `eps in {.75,1.25,2,3}`. For
every held-out parcel, select the gate within its other 137 training parcels;
the held-out yield is masked *before* fitting. No GPU or CatBoost is
involved.

Full 138 LOOCV produces 138 independent 137/1 predictions; processes can
run concurrently and resume from validated saved fold CSV files.

## Run (PowerShell, from GeoC root)

~~~powershell
conda activate geocebada
git pull --ff-only origin experiment/nuria-champions-loocv

# Inspect saved scores WITHOUT training (already available):
Import-Csv reports/selected_champions_loo/loo_summary.csv | Format-Table
Import-Csv reports/area_weighted_climate_loo/loo_summary.csv | Format-Table

# Safety check for peer original bundle:
python tools/run_peer_dbscan_loo138.py --check

# Smoke one folded DBSCAN fit (independent smoke folder):
python tools/run_peer_dbscan_loo138.py --smoke --jobs 1

# IMPORTANT: reuse the first 46 verified DBSCAN predictions, if LOO46 exists
python tools/run_peer_dbscan_loo138.py --reuse-loo46

# Complete only the remaining 92 folds, unrestricted positive worker count
python tools/run_peer_dbscan_loo138.py --run --jobs 12

# Read DBSCAN-only aggregate again without retraining
python tools/run_peer_dbscan_loo138.py --summary

# Compare all four on the SAME 138 parcel IDs and held-out labels
python tools/run_peer_dbscan_loo138.py --compare
~~~

**If `--reuse-loo46` says the hashes differ, do not bypass its checks.**
Just run `--run --jobs 12` without reuse; it trains all missing folds. A
changed teammate model file, bundle manifest, DBSCAN gate, or fold protocol
could legitimately invalidate the old 46 predictions. The importer
validates *all 46* before writing any, and never overwrites disagreed rows.

The comparison reuses these verified private files:

1. `reports/selected_champions_loo/loo_predictions.csv`: Reference26,
   Compact14, 138 holdouts. The `run_spec.json` hashes and 4-fold nested
   protocol must match the DBSCAN run.
2. `reports/area_weighted_climate_loo/loo_predictions.csv`: long-format
   original Local04D predictions stored under the model alias
   `GeoPCA20_GeographicWeightedRidge`, 138 parcels, 137/1 each.
   Matched by ID and exact observed yield. If this file is absent,
   `--compare` tries the saved original C4 PCA20 model only after
   its historical parity guard passes.
3. Alternatively use `--local04d-file path/to/local04d.csv` with exactly
   three columns `ID_POLIGONO,observed,Local04D` and 138 rows.

The code refuses silently incomparable parcels, yields, code hashes,
incomplete folds or changes to the original peer model.

## Outputs and GitHub publishing

Private ignored folder: `reports/peer_dbscan_loo138/`.

| File | Contains | Git |
|---|---|---|
| `fold_001.csv` through `fold_138.csv` | Parcel IDs, observed yields, DBSCAN predictions | **Never commit** |
| `gate_001.json` etc. | Training-only hyperparameters | Keep private |
| `loo_predictions.csv` | All per-parcel DBSCAN heldout predictions | **Never commit** |
| `loo_summary.csv` | DBSCAN model aggregate scores | Publish via exporter |
| `four_model_predictions_private.csv` | Matched 138 heldout records | **Never commit** |
| `four_model_summary.csv` | Four model RMSE/MAE/R² and delta MSE | Publish via exporter |
| `run_spec.json`, `loo46_reuse_audit.json` | Exact code/data fingerprint, import audit | Keep private |

Export **only** whitelisted aggregate metrics from all completed private
runs to a tracked directory:

~~~powershell
python tools/publish_peer_results.py --export
git status --short
git add docs/PEER_LOOCV_RECORDED_RESULTS.md results/peer_models/
git commit -m "Archive aggregate peer and DBSCAN LOOCV results"
git push origin experiment/nuria-champions-loocv
~~~

This export can also be run **now**, before DBSCAN finishes: it preserves
the exact previously computed Reference26/Compact14 RMSE and R² without
depending on terminal history. Running it again after DBSCAN finishes
updates tracked aggregate tables. Publishing does not happen automatically.

**Methodological caveat:** A model selected using the first 46 outer
errors is exploratory even when evaluated later on all 138 familiar
labels. First-third comparison used for selection means the 138-fold
score is not a fully independent generalization estimate.

No competition submission for the 59 unlabeled parcels is changed.
