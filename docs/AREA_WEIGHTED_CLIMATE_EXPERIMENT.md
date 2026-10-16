# Area-weighted climate and temporal parcel features

**Status:** runnable pipeline, no independent confirmation and no automatic model promotion.

## Goal and names

This is a **data construction** experiment rather than another hyperparameter sweep. The former frozen "Local04D" is called `GeoPCA20_GeographicWeightedRidge` in new outputs: geographic k=24 weighted local Ridge, alpha=30, inverse-distance exponent=1, on 20 standardized PCA components of the existing 2,082-feature C4 matrix. This is a descriptive alias of the exact same historical model, not a new algorithm.

Our official monthly 0.05° (~5 km) climate rasters are spatially coarser than parcel boundaries. The original extraction in `src/geocebada/features/parcel.py` uses `rasterio.mask(..., all_touched=True)` and **unweighted** means of intersected pixels. The new extraction computes each intersection polygon's area with an actual climate-grid cell, projects that overlap to the equal-area EPSG:6933 coordinate system, and assigns weight proportional to the **valid overlapping area**. Invalid and NoData cells do not contribute. It also computes an unweighted comparator on the *same valid, positive-overlap cells*. Exact boundary-only contacts receive zero weight in both comparators.

Mathematical estimator:

```text
parcel_weather(t) = Σ_over_valid_cells [ overlap_area(parcel, cell) * weather(cell, t) ]
                  / Σ_over_valid_cells [ overlap_area(parcel, cell) ]
```

**Resolution constraint:** if the entire parcel lies within one 5 km raster cell, all its climate values still come from that one cell. More transformations **cannot invent subpixel climate observations or more independent yield labels**. Extracted weighted weather and monthly temporal transforms are X-only.

## What is compared

| New model label | Feature matrix | Fixed method |
|---|---|---|
| `GeoPCA20_GeographicWeightedRidge` | Original 20 standardized C4 PCs only | Geographic weighted local Ridge |
| `GeoPCA20_PlusTouchedPixelClimate` | Same 20 PCs + six label-blind PCs of unweighted 2025 climate | **Same** Ridge, k, p, alpha |
| `GeoPCA20_PlusAreaWeightedClimate` | Same 20 PCs + six label-blind PCs of area-weighted 2025 climate | Same |
| `GeoPCA20_PlusAreaWeightedTemporalClimate` | Same 20 PCs + weighted-climate six PCs + six PCs of climate temporal transforms | Same |
| `FourCovariate_GlobalRidge` (optional) | Four numeric columns supplied by teammate | Ordinary global Ridge (alpha 10; fold-train scaler) |
| `GeoPCA20_FourCovariateOOFBlend` (optional) | Inner-OOF convex blend of frozen control and optional four-covariate Ridge | 3-fold inner-CV weight selected strictly on 97 outer-training labels |

**Crucial distinction:** the optional `FourCovariate_GlobalRidge` is only a reproducible four-variable proxy. **It is not Nuria's original model.** No one should report this as her trained model or her prior blend without receiving her variables, code and fold-level predictions.

These comparisons are **augmentation ablations**: the original 2,082-feature matrix and its original 20 PCs are frozen and left untouched; we append weather summaries instead of claiming to replace the official C4 climate block. This means the ablations isolate whether extra climate X helps prediction, but some redundant climate features are possible.

2025 monthly feature parents are April–October rainfall, minimum temperature and maximum temperature (21 values per parcel), with full 2022–2024 data used to calculate year-matched monthly anomaly features. Temporal derivatives include early/mid/late rain totals, rainfall timing, wettest/driest months, largest 2-month rainfall, seasonal mean temperatures, diurnal range, and **monthly-temperature-based** GDD proxy bases 0/5/10 °C. These GDDs are not true daily heat accumulation; the raster does not contain actual daily temperatures.

## Validation, data governance, interpretation

Exactly **197 parcels**, **138 reported yields / 59 unlabeled prediction parcels**. Confirm the source rasters for all 4 × 7 × 3 = **84** mandatory monthly weather files; no silent partial download. The process requires ≥95% valid spatial coverage in all target-year parcel-month rasters before fitting any model. Output includes pixel coverage, effective number of contributing cells, monthly weather matrices, unique complete weather profile counts, and mean disagreement between unweighted vs weighted pixel averages.

For all four core model procedures, *identical* hyperparameters and outer masks are used, and the 20 original C4 PCs are literally the same. The optional 4-covariate model uses only the supplied X-only four columns, never supplied yield/prediction labels. The optional blend chooses from weights 0, .25, .50, .75, 1 using only 3-fold inner OOF predictions inside the outer 97 fit parcels. No outer-held-out yield leaks into feature transforms, model fitting or blend selection. In contrast, the earlier pipeline's target-free transductive PCA does use all 197 X rows by design.

The **16 × 41 historical outer pseudo-targets have already been inspected repeatedly**. Scores and parcel-cluster bootstrap intervals are exploratory, **not independent confirmation**. Check whether groups of parcels share effectively identical coarse climate profiles before interpreting spatial generalization.

## Commands (PowerShell, conda)

```powershell
cd C:\Users\175199\GeoC
conda activate geocebada
git switch main
git pull --ff-only origin main

python tools/run_area_weighted_climate_experiment.py --check
python tools/run_area_weighted_climate_experiment.py --build
python tools/run_area_weighted_climate_experiment.py --smoke
python tools/run_area_weighted_climate_experiment.py --run
python tools/run_area_weighted_climate_experiment.py --summary
```

Requires the repository's existing geospatial dependencies from the conda `geocebada` environment, the complete source ZIP `data/source/geospatial/Parcelas_Reto_AGC_CONJUNTO_70_30.zip`, and the `data/source/climate` 2022–2025 monthly rasters. No new external data download, GPU or 100k-model rerun is required.

All generated files and label-bearing held-out predictions are private and ignored by Git:
`reports/area_weighted_climate_experiment/`. The `run_spec.json` SHA256 checks forbid mixing changed code, rasters, or peer CSV with old outcomes; re-run the same command to resume after interruption and use `--output` to start a new experiment if provenance changes.

Optional four-covariate pathway, **only once the actual X-only features are received**:

```powershell
python tools/run_area_weighted_climate_experiment.py --run --peer-features data/local/peer_four_covariates.csv --output reports/area_weighted_climate_with_peer
python tools/run_area_weighted_climate_experiment.py --summary --peer-features data/local/peer_four_covariates.csv --output reports/area_weighted_climate_with_peer
```

Peer file contract: CSV with `ID_POLIGONO` plus **exactly four numeric non-target columns**; one row per each of the 197 official parcels, complete, no known yields. Actual teammate predictions cannot be safely combined without correct fold IDs and out-of-fold provenance.

Experiment outputs: `weighted_monthly.csv`, `touched_monthly.csv`, `weighted_temporal.csv`, `pixel_coverage_diagnostics.csv`, `climate_extraction_summary.json`, `confirm_XX__paired.csv`, `confirm_XX__selection.json`, `comparison_summary.csv`, `paired_cluster_bootstrap.csv`. No generated report files or predicted yields are auto-committed.

## Decision criteria

Check first (1) meaningful differences between area-weighted and touched-pixel climate values, (2) climate profile duplication, (3) paired RMSE and grouped bootstrap, and (4) performance in each of the 16 outer splits. If raw weather differs only slightly between the methods, improvement may come from added variables, not from reweighting. A weather ablation that wins the same repeatedly examined masks should still **not** automatically replace the final 59 predictions. Freeze a genuine held-out confirmation protocol, or seek genuinely new labeled parcels where possible.


## Additional check: leave-one-parcel-out (LOOCV)

The historical four-arm comparisons above already use exactly matched 16 outer splits,
and provide a fair like-for-like comparison at **97 training + 41 heldout** labels per split.
LOOCV is an **additional** diagnostic, not a replacement or a directly comparable RMSE.

This new evaluation predicts **each of the 138 known-yield parcels exactly once**,
training each frozen candidate on the other **137** observed labels. It applies
the **same leave-one-out masks** to all four feature arms with the **same
fixed geographic Ridge settings** (k=24, alpha=30, inverse-distance power=1).
It reports pooled RMSE, MAE, global R² (computed across 138 predictions, not
per one-observation fold), mean prediction error, and paired per-parcel RMSE/MSE
differences with descriptive bootstrap intervals. These scores are not comparable
numerically with the historical 0.481936 because that score comes from
97-label training fits and repeated 41-parcel test folds.

This standalone runner does **not** alter the earlier implementation or saved
runs and does **not** re-extract any raster. It needs the three already saved
target-free climate matrices. From the repository root in the geocebada environment:

```powershell
git switch main
git pull --ff-only origin main

python tools/run_area_weighted_climate_loo.py --check
python tools/run_area_weighted_climate_loo.py --smoke
python tools/run_area_weighted_climate_loo.py --run
python tools/run_area_weighted_climate_loo.py --summary
```

The private, ignored output directory is
reports/area_weighted_climate_loo, containing loo_predictions.csv
(138 x 4 rows), loo_summary.csv, loo_paired_differences.csv and run_spec.json.
The latter fingerprints all relevant inputs and implementation code to prevent
accidentally mixing a stale run with changed data or procedures.

**Statistical limitations:** Nearby parcels may share conditions, coarse climate
pixels and other spatial signals, so leave-one-parcel-out may overstate accuracy
for genuinely new geographic areas. LOOCV also does not erase selection bias
from years of comparing model candidates against the same 138 labels. Paired
parcel bootstraps ignore residual geographic dependence. This round evaluates
only **pre-fixed** candidates; picking/tuning hyperparameters using the
outer heldout labels and then reporting the same score would be biased.
