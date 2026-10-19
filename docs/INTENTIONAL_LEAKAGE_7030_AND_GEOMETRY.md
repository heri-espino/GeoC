# Intentional test contamination and feature-space geometry (2026-10-19)

**Status: source, synthetic unit tests, and methodology committed on a separate
experimental branch; real workstation runs pending.** This is intentionally
invalid cross-validation requested for a diagnostic curiosity, **not** a
candidate unbiased score or competition submission. The official
Reference26 mean model, saved authentic 138-fold LOOCV (RMSE 0.468138 t/ha),
and 59 published-target predictions are unchanged.

## Why this experiment exists

The genuine 138-label full LOOCV established that Reference26 mean
(0.468138 t/ha) outperformed global Ridge (0.492732) and global ExtraTrees
(0.496662). Forty synthetic-augmentation candidates performed worse than
their unaugmented counterparts. The user explicitly requested an
**intentional leakage stress test** in which synthetic test observations
are generated from the actual test parcels and their labels, to see
how the reported RMSE can change, plus new X-geometry visual diagnostics.

This branch DOES NOT make a scientific claim that test labels can be
known during an actual forecast. Never report the leaky score as CV,
LOOCV, model improvement, honest external validation, or hidden-59 quality.

## A. Repeated random 70/30 protocols

Use exactly **100 fixed-seed independent random partitions**, each of
the original **138 REAL labeled parcels**:
- 96 original training rows;
- 42 genuine held-out test rows;
- configurable 512 *synthetic children* derived from the **42 test
  parcels** and the very test yields that should normally be withheld.

The synthetic X child from held-out parent i is

\[
\widetilde X_i=X_i+\sigma L_i\varepsilon,
\quad\varepsilon\sim\mathcal N(0,I),
\quad \widetilde y_i=y_i,
\]
where \(L_iL_i^\top\) estimates X-only neighbor covariance from the
197 observable feature vectors, and the default \(\sigma\) is 0.08.
The synthetic 512 children cover **every test parent** at least once.
Their copied target values make this *explicit target leakage* if
those children are added to training.

Five important interventions (six scored scenarios) are trained/scored
using identical real train/test splits for each of two global learners:

| CSV scenario | Training | Scored on | Meaning |
|---|---|---|---|
| `clean_real_test` | 96 real training parcels only | 42 real tests | Only proper 70/30 control |
| `clean_training_augmented_real_test` | 96 real + 512 Gaussian children centered at top 8 TRAINING PRESS errors | 42 real tests | Fold-safe augmentation; valid split-level diagnostic |
| `clean_model_synthetic_expanded_test` | 96 real only | 42 real + 512 test-born pseudo-test rows | Artificially inflated, dependent test sample |
| `leaky_test_gaussian_training_real_test` | 96 real + 512 children of 42 TEST parcels with copied TEST yields | 42 real tests | Deliberate target leakage |
| `leaky_test_gaussian_training_expanded_test` | Same contaminated training | 42 real + separately generated 512 test-born pseudo-test rows | Double contamination: training + evaluation |
| `leaky_test_exact_copies_training_real_test` | 96 real + 512 exact X/y copies of the 42 TEST parents | 42 real tests | Obvious oracle/memorization upper-bound control |

The two global learners (same for each scenario):
- Ridge, alpha=30;
- ExtraTrees, 64 trees, minimum leaf=3.

The scripts do **not** refit Reference26 (a much more expensive nested
model with a different training recipe). Do not pass the historic
Reference26 LOOCV result as if it were a clean random-70/30 baseline.
It used 137 real labels per fold, not 96.

A valid training-only augmentation control uses Ridge PRESS to pick
hard original TRAINING parcels inside each split; it never touches y
of the 42 TEST parents. For the leaky scenarios, however, the test
y values are copied into pseudo-training targets **on purpose**.

### Computed metrics

For each of 100 splits, each model and each scenario, report RMSE, MAE,
bias, original training/test counts, unique source test-parent counts,
numbers of artificial test/training rows, and explicit Boolean flags
indicating whether test labels entered the model or the evaluation.
The main summary reports:

1. mean RMSE;
2. median RMSE;
3. sample standard deviation of the 100 split RMSEs;
4. RMSE percentiles 5, 25, 50, 75 and 95;
5. mean RMSE **by model and by contamination protocol**;
6. mean *paired difference* from the identical model trained/evaluated
   without contamination on the identical 100 partitions.

There are **4,200 real test appearances** per model over 100 splits,
but they repeatedly sample from the **same 138 parcels**, so they
are not 4,200 independent real observations.
With 512 synthetic test cases per split there are 51,200
**synthetic test appearances per model**, each one copied from
a real test-parent y: also *not* independent.
A low contaminated score only measures the exploitability of leaked
test information.

## B. Diagnostic plots of the actual 197 parcel X

Run `tools/plot_reference26_error_geometry_197.py`. It checks the
original 197 X-only champion bundle and exact 138 saved LOOCV
Reference26 mean predictions before deriving any errors.

1. PCA, t-SNE and UMAP of the **197 genuine feature vectors** (26
   numeric standardized features, not synthetic X). Plot twice
   per projection: signed residual \(\widehat y^{(-i)}-y_i\)
   and absolute residual, coloring only the 138 known parcels.
   The other 59 are neutral gray squares: their yields/errors
   are NOT known.
2. Mark the **10 largest absolute Reference26 LOOCV errors** with
   prominent crosses. Annotate the worst two IDs. No hardcoded IDs.
3. Calculate for every parcel: nearest-neighbor X distance, mean
   five-nearest-X distance, shrinkage Mahalanobis distance,
   continuous Local Outlier Factor, Isolation Forest score.
4. Compare nearest geographic neighbor according to verified
   `distance197.npy` with nearest neighbor in standardized X.
   For each parcel compute the X-rank of the closest geographically
   located parcel. A large rank would support the hypothesis of
   locally surprising features.
5. Diagnostic scatterplots: |error| vs 5-NN X distance, and
   |error| vs X-rank of nearest geographic neighbor.
6. If confirmed WGS84 longitude/latitude columns are present, draw
   geographical coordinates with the same error colors. Otherwise,
   show classical 2D MDS of the original **spatial-distance matrix**,
   explicitly labeled as a *distance projection*, NOT an authentic
   geographical map. Never infer kilometers from undocumented units.
7. Private top-ten parcel table: ID, signed/absolute error, feature
   distances, nearest spatial and feature IDs, LOF, IsolationForest,
   Mahalanobis and neighborhood mismatch metrics.
8. Aggregate, exploratory Spearman correlations of absolute LOOCV
   residual with these X/spatial isolation measures.

A projection is NOT a distance-preserving ground truth: check
full-dimensional nearest-neighbor metrics before concluding that
Puebla parcels have inconsistent attributes.

## Run on the same GeoC VM

~~~powershell
cd C:\Users\175199\GeoC
conda activate geocebada
git fetch origin
git switch analysis/intentionally-leaky-70-30-and-x-geometry
git pull --ff-only

python -m pytest tests/test_intentionally_leaky_7030.py tests/test_reference26_error_geometry_197.py -q
python -m py_compile tools/run_intentionally_leaky_7030.py tools/plot_reference26_error_geometry_197.py

# Original 197 X, signed LOOCV errors, 10 crosses, geometric neighbors:
python tools/plot_reference26_error_geometry_197.py --check
python tools/plot_reference26_error_geometry_197.py --run

# 100 repetitions; use a 3-split smoke test first
python tools/run_intentionally_leaky_7030.py --check
python tools/run_intentionally_leaky_7030.py --smoke
python tools/run_intentionally_leaky_7030.py --run --jobs 4
python tools/run_intentionally_leaky_7030.py --summary
~~~

For UMAP, if needed:

~~~powershell
conda install -c conda-forge umap-learn
~~~

UMAP is optional only if that dependency is absent: the script
still creates PCA, t-SNE and spatial diagnostics, then prints the
installation instruction. Once installed, rerun `--run` for UMAP.
If native longitude/latitude columns exist under different names,
pass `--lat-column NAME --lon-column NAME` to the plotting program.

The split runner saves a SHA-locked experiment manifest and one set
of private files per repetition; rerunning with the same immutable
inputs resumes completed splits. It rejects changed source or
parameters in the same output directory.

### Files: private, gitignored

Folder `reports/intentionally_leaky_70_30/`:
- `summary_rmse.csv`: requested means, medians, SD and five percentiles.
- `mean_rmse_by_model.csv`: wide mean RMSE matrix.
- `all_split_metrics.csv`: paired 100-split scores.
- `rmse_distributions_leakage_comparison.png`: scenario RMSE boxplots.
- `scores_NNN.csv`: per-repetition aggregate; no parcel IDs.
- `real_test_NNN_private.csv`: original test IDs, true yields, predictions.
- `fake_test_NNN_private.csv`: individual synthetic children,
  their real test parent IDs and COPIED test-y; MUST NOT publish.
- `run_spec.json`: exact recipes, code/data fingerprints, explicit
  intentional leakage warning.

Folder `reports/intentionally_leaky_70_30/feature_geometry/`:
- `pca_signed_private.png`, `pca_absolute_private.png`;
- `tsne_signed_private.png`, `tsne_absolute_private.png`;
- `umap_signed_private.png`, `umap_absolute_private.png`
  if umap-learn installed;
- geographic or spatial-distance-MDS signed and absolute plots;
- `error_vs_x_5nn_private.png`;
- `error_vs_geographic_neighbor_feature_rank_private.png`;
- `all_197_feature_outlier_private.csv`,
  `top10_feature_outlier_private.csv`;
- `error_distance_correlations.csv`, `geometry_manifest.json`.

Do not run `git add reports/` or publish any per-parcel values.
For a public scientific report, only publish clearly labeled,
aggregate summaries of the clean scenarios; otherwise an
unqualified contaminated RMSE could be misrepresented as validity.

## Limits and next decisions

The user intentionally requested the misleading scoring effect,
so the fake/heldout-y-derived protocols are separate, conspicuously
marked invalid, and contrasted against the proper 70/30 real-only
control on the same splits. They should never be chosen as a final
model or used to generate the 59 true unknown targets, whose y
cannot be synthesized from real hidden responses.

If signed errors are large despite normal X-neighbor similarity,
the likely issue is target variation not resolved by the 26 predictors,
target noise, or a latent regime. If the top-ten is isolated in X
and/or mismatched with geographically neighboring parcels, focus
on feature provenance and agronomic variable coverage rather than
adding pseudo-labeled fake rows as though they were independent evidence.
