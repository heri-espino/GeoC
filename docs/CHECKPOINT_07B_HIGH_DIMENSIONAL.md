# Checkpoint 07B — High-dimensional trajectories, PCA and neural networks

**Status:** X-only feature factory and honest validation benchmark implemented;
workstation data processing/benchmark run pending.

**Scientific objective:** improve out-of-sample parcel yield RMSE over the frozen
Local04D baseline, without treating feature count or explained X variance as
proof of predictive value.

## Why a high-dimensional pipeline?

Preserving full HLS acquisition dates (rather than only monthly averages) and
SMAP daily regional surface moisture permits thousands of quality-aware
temporal/spatial descriptors. We may then add pairwise cross-domain products,
nonlinear transforms and a neural regression head.

    raw HLS / SMAP / existing features
      -> observed parcel-date trajectories and temporal QA
      -> deterministic X-only nonlinear transforms and interactions
      -> within-training-fold imputation + scaling
      -> within-training-fold PCA (small dual Gram matrix / SVD equivalent)
      -> (a) Ridge
         (b) Ridge with selected quadratic principal-component interactions
         (c) regularized MLP neural network
      -> honest held-out parcel RMSE

The code **does not** require a human interpretation of each component.

### Important statistical limits

- We have 138 measured parcel yields, NOT tens of thousands of independent
  labels. Each labeled parcel's repeated satellite measurements are correlated
  X observations, not new supervised yields.
- On each 97-label pseudo-training split, the centered PCA rank is at most
  96 regardless of whether the initial feature matrix has 2,000 or 20,000
  columns. Dual PCA therefore avoids a huge p-by-p covariance matrix.
- Large X variance is not necessarily related to y. Compare retained
  variance thresholds of 0.80, 0.95 and 0.99 with 16/32/64 fixed ranks.
  The final criterion is validation RMSE, not PCA explained variance.
- Numerical feature families can dominate PCA simply by having many
  correlated columns. A later, predeclared family-weighting/block-PCA
  ablation is warranted, rather than presuming vanilla PCA is optimal.
- The MLP is intentionally small (16, or 32→16 hidden neurons) with
  L2 regularization and early stopping on an **internal split of pseudo-train
  only**. A huge tabular network from scratch is not supported by 138 labels.
- The large pretrained neural-network experiment is separately Prithvi-EO-2.0,
  using aligned actual HLS chips. We will first freeze its encoder and fit
  low-capacity heads, then compare honest OOF RMSE.
- Features can be computed on all 197 X records, but no known/held-out y is
  used for seed selection, nonlinear interactions, PCA, normalization or
  model fitting outside the corresponding pseudo-training set.
- Current development uses the frozen Checkpoint 04B 16 target-matched
  pseudo-splits. They overlap and must NOT be treated as independent
  statistical replicates. A new untouched confirmation bank is still
  required to decide whether a model actually improves over Local04D.
- SIAP 2025 contemporary municipal outcome proxies are excluded by default
  as a potential outcome-information leakage source; optional enabling is
  explicit and must be declared in any comparison.

## Implemented X-only families

### HLS scene-wise time series (30 m)

For every available reflectance band and spectral index (Blue, Green, Red,
NIR, SWIR1, SWIR2, NDVI, NIR/SWIR1 moisture-sensitive index), separately
for each scene sensor and pooled:

- temporal mean/weighted mean, median, min/max, variance, RMS, range;
- quantiles 5/10/25/75/90/95, IQR, median absolute deviation;
- skewness, excess kurtosis, signed change from first to last;
- date of peak/trough relative to first valid observation;
- slopes, quadratic curvature, growth/decline velocities;
- total variation, autocorrelation of successive observations, acceleration;
- observed trapezoidal integral **excluding intervals with large gaps**;
- coverage, observation count, maximum and median observation gaps;
- separate early (Apr–May), development (Jun–Jul), late (Aug–Sep),
  October means/min/max; fixed NDVI threshold fractions;
- QA/polygon-pixel fraction and sensor counts as distinct features.

No signal is invented for missing dates. Long observed gaps are not
filled by a continuous interpolation for feature-AUC.

### SMAP 9-km regional time series

AM, PM and combined daily regional moisture trajectories are summarized
using the same irregular series statistics, plus fixed moisture threshold
fractions, season slices, measurement QA and distance to sampled SMAP cell.

Regional SMAP data should not be interpreted as true parcel-root-zone
moisture or 197 independent field measurements.

### HLS × SMAP antecedent lags

For every observed HLS NDVI date, compute preceding 7/14/30-day SMAP
moisture means. Then derive correlation and mean NDVI×antecedent moisture.
Future SMAP dates do not contribute to an earlier HLS observation.

### X-only interaction expansion

Starting from a maximum number of well-covered numeric seed variables
balanced across available source families, construct:

- signed log1p and signed-square transforms;
- pair products between different source/family classes;
- deterministic caps on seed count and product count so memory remains
  bounded (defaults: up to 180 seeds, 12,000 products).

Seeds use completeness, variance and column names only. They do not
depend on yield or score on the 138 labels.

This is **not** an assertion that every interaction will be useful.
A very large redundant matrix may damage both PCA geometry and prediction.
The benchmark explicitly measures that possibility.

## Execution (PowerShell)

First complete the prerequisite HLS and SMAP compact data extraction
described in docs/CHECKPOINT_07A_PROCESSING.md.

    git pull --ff-only
    conda activate geocebada
    python -m pip install -e ".[dev,geo,checkpoint07]"
    git lfs pull

    python tools\run_checkpoint_07_highdim.py --preflight --require-hls --require-smap

Smoke test: isolated feature table, single historical pseudo-validation
split, reduced interaction/model grid, including one neural architecture:

    python tools\run_checkpoint_07_highdim.py --smoke --require-hls --require-smap

After the smoke report looks sound:

    python tools\run_checkpoint_07_highdim.py --require-hls --require-smap

All outputs are local and gitignored under:

    reports/checkpoint_07/highdim/

Files:

    expanded_parcel_features.csv   X-only 197-parcel wide table
    feature_manifest.json          generator settings/seed names
    oof_predictions.csv            held-out development predictions
    fold_metrics.csv               RMSE, retained rank and X variance
    pca_diagnostics.csv            actual fold-local PCA rank
    benchmark_summary.csv          pooled, mean and worst split RMSE
    run_report.json                status, input modality availability, hashes

Interrupted after completed feature generation? Continue without rebuilding X:

    python tools\run_checkpoint_07_highdim.py --stage evaluate --require-hls --require-smap

Replace an existing table only intentionally:

    python tools\run_checkpoint_07_highdim.py --stage build --force

The stage **never** overwrites Local04D, Prithvi predictions, official yields
or the source images. Running without --require-hls/--require-smap is allowed
for development when the new processed data are not ready, but the report
will label the missing modalities; do not describe that as full 07B.

## After the workstation run

1. Read and check feature counts, NA percentages, coverage by parcel and
   sensor; inspect spectral units and signs before treating large
   correlations as real effects.
2. Compare PCA Ridge, PCA quadratic Ridge and PCA MLP on the *same*
   pseudo-targets against the frozen Local04D predictions, by split.
3. Try a predeclared family-block weighting and a no-PCA regularized baseline,
   because low-variance predictive structure could be discarded by PCA.
4. Freeze at most a small shortlist. Compare on fresh untouched target-matched
   confirmation; do not tune on that confirmation.
5. Only if frozen Prithvi embeddings add independent out-of-sample information,
   attempt fusion. Trainable deep networks on 138 y labels are not a default.

This is a research workflow for estimating measured barley yields,
not an assumption that more parameters automatically improve accuracy.
