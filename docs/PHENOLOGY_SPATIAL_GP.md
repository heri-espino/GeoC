# Phenology-conditioned spatial Gaussian process — experimental model

**Date:** 2026-10-18  
**Status:** implemented on experiment branch; VM evaluation pending; **no promotion**.

## Question

Does stage-conditioned 2025 weather, vegetation, and soil moisture explain
variation in the 138 observed parcel yields beyond the incumbent 20-PC
geographic Ridge, and is there complementary residual spatial dependence?

This is fixed-target transductive regression: 197 observable X rows,
138 observed training labels and 59 hidden-y parcels. Hidden targets are never
used for training, preprocessing or validation.

## Data and X-only representation

The runner reuses the frozen Checkpoint10 C4 PCA20 loader and its numerical
parity check; area-weighted official monthly weather, quality-filtered HLS
2025 optical observations, and valid SMAP 2025 daily moisture.
It does **not** load 180 GiB of raw source imagery:

| Modality | VM input |
|---|---|
| Frozen C4 PCA20 | Existing source and processed features, manifests and Checkpoint 04A embedding |
| 2025 precipitation, Tmin, Tmax | reports/area_weighted_climate_experiment/weighted_monthly.csv |
| HLS dates, sensor, NDVI and QA | data/processed/checkpoint_07/hls_parcel_observations.csv |
| SMAP AM/PM daily regional water | data/processed/checkpoint_07/smap_parcel_daily.csv |
| 16 outer split masks | reports/checkpoint_08/confirmation_membership.csv |
| Parcel coordinates and 138 labels | Official repository processed/target files |

The optical NDVI peak month is a **proxy**, not verified sowing, heading,
grain-filling or harvest phenology. A peak requires three observed NDVI months.
Weather is summarized before/through versus after that peak: monthly rainfall,
approximate GDD5 derived from monthly Tmin/Tmax and actual month duration,
rainfall concentration and temperature features. Gap-aware HLS AUC and
coverage, SMAP preceding/subsequent 30-day means and regional variability are
also used. Missing feature values are retained and imputed *within* each
supervised training fold. SMAP is roughly 9 km and monthly weather roughly
5 km; they are not parcel-scale measurements.

**AgERA5 and Sentinel-1 are not inputs in this first version**; they require
verification of processing completeness and sensor QA before expansion.
SIAP 2025 outcome proxies are excluded.

## Estimator

For each training fold, the environmental mean is Ridge on 20 frozen X-only
PCA components plus the physical features, with train-fold-only imputation
and standardization:

\[
\hat f(x)=\hat b+x^\top\hat\beta,\qquad
\arg\min_{b,\beta}\sum_{i\in L}(y_i-b-x_i^\top\beta)^2+
\alpha\|\beta\|^2.
\]

Train an additive Matérn 3/2 Gaussian process on in-sample environmental
residuals r = y - f(X), using geographic distance in kilometres:

\[
K_{ij}=\rho(1+\sqrt{3}d_{ij}/\ell)e^{-\sqrt{3}d_{ij}/\ell}
+\mathbf 1_{i=j}[(1-\rho)v_i+10^{-5}].
\]
\[
\hat y_q=\hat f(X_q)+K_{qL}K_{LL}^{-1}r_L.
\]

Rho is the spatial share; ell the lengthscale. With rho=0 this is a
Ridge-only ablation. With robust_nugget enabled, the positive diagonal noise
is inflated for large *training* residuals using a clipped median absolute
deviation heuristic. This is NOT a full heteroscedastic or Student-t GP
likelihood or jointly estimated universal kriging.

Inner tuning grid: Ridge alpha 10/100, lengthscale 0.75/3/12 km,
spatial share 0/0.5/0.9, robust diagonal on/off (nonzero shares only).
Threefold inner CV on each outer split's 97 available labels selects the
configuration. The 41 outer labels are revealed only to score the frozen
candidate and the **numerically unchanged Local04D** (PCA20/k24/alpha30/p1).
The outer splits are the same 16 historically reused Checkpoint08 masks,
so scores are exploratory rather than independent confirmation.

## VM commands (PowerShell)

~~~powershell
cd C:\Users\175199\GeoC
conda activate geocebada
git fetch origin
git switch experiment/phenology-spatial-gp-20261018
git pull --ff-only origin experiment/phenology-spatial-gp-20261018
git lfs pull
python -m pip install -e ".[dev,geo,external]"

# Required only if prior area-weighted 2025 weather was not built:
python tools/run_area_weighted_climate_experiment.py --check
python tools/run_area_weighted_climate_experiment.py --build

python tools/run_phenology_spatial_gp.py --check
python tools/run_phenology_spatial_gp.py --smoke
python tools/run_phenology_spatial_gp.py --run
python tools/run_phenology_spatial_gp.py --summary

# OPTIONAL candidate-only 59-target file; never auto-promoted:
python tools/run_phenology_spatial_gp.py --predict-targets
~~~

If compact HLS/SMAP files are missing, follow docs/CHECKPOINT_07A_PROCESSING.md;
do not redownload raw files automatically. If checkpoint08 confirmation
membership is missing, recover its documented frozen generation procedure;
do not silently substitute new splits. All generated files remain in
gitignored reports/phenology_spatial_gp/. The runner fingerprints exact data,
code and label-bearing inputs; changed source requires a new output folder.

Outputs: x_only_coverage.json; fold_XX_predictions.csv,
fold_XX_inner_scores.csv, fold_XX_selection.json, comparison_summary.csv,
paired_delta_mse.csv, optional candidate_59_predictions_NOT_PROMOTED.csv.

## Limitations and decision rule

There are only 138 independent observed yields. Optical peak month is uncertain,
monthly climate grids have duplicated profiles, and nearby SMAP samples can be
identical. The model may double-count spatial structure already captured by
Local04D. Report actual unique climate profiles, missingness, RMSE paired on
identical parcels, selected spatial shares and unique parcel counts. The 16x41
historical splits have been repeatedly evaluated, so apparent improvements are
subject to selection optimism. Treat repeated predictions as clustered by
parcel, not 656 independent labels.

The final 59 official forecasts remain frozen in the checkpoint05 output.
No claim of better RMSE or independent confirmation is made before actual VM
execution and further evaluation. The first model iteration deliberately uses
compact existing data rather than promising to exploit sources not yet validated.
