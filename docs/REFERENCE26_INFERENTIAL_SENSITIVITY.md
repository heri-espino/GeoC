# Reference26: inferential scope and spatial sensitivity

**Date:** 2026-10-19. **Primary model:** Reference26 with training-fold arithmetic-mean imputation (local k=24, Ridge alpha=30). Compact14 and the outcome-selected blend remain historical secondary comparisons, not confirmed improvements.

**Evidence status:** The actual 138-fold residual diagnostics have already been transcribed and interpreted in [the statistical interpretation report](https://github.com/heri-espino/GeoC/blob/experiment/nuria-champions-loocv/results/peer_models/REFERENCE26_COMPACT14_STATISTICAL_INTERPRETATION_138_2026-10-19.md). The additional cluster-bootstrap and parcel-influence analysis introduced here **has not yet been executed on confidential VM data**. No new test statistic or interval is asserted as observed.

## 1. Scope and notation

For parcel i, define the leave-one-parcel-out error (signed prediction minus observation):

\[
e_i = \widehat y_{-i}(x_i)-y_i,\qquad n=138.
\]

Each forecast excludes the evaluated yield and fits using 137 labeled parcels, plus permissible transductive X-only inputs. The 138 training sets overlap extensively. Spatial dependence in yields and adaptive algorithm/model choices further limit iid interpretations.

Separate four levels of claim:

1. **Descriptive:** RMSE, MAE, bias, residual quantiles, influence, fitted calibration, semivariogram.
2. **Exploratory:** conditional block-bootstrap intervals and paired differences among frozen predictions.
3. **Model-based null probes:** geographic Moran I/Geary C with a specified graph and a spatial-label exchangeability assumption. Multiplicity correction is necessary for the k/model/residual combinations.
4. **Not identified:** causal parameter effects; frequentist significance of nonlinear component coefficients; an independently validated improvement from weights chosen on these 138 labels; generalization to unseen municipalities; calibrated uncertainty for the 59 hidden yields.

## 2. Existing real results

| Metric | Reference26 mean | Outcome-selected blend |
|---|---:|---:|
| RMSE (t/ha) | 0.468138 | 0.467811 |
| MAE (t/ha) | 0.333786 | 0.332489 |
| Predictive R-squared | 0.695292 | 0.695717 |
| Shapiro-Wilk p (exploratory) | 1.401988e-7 | 1.306209e-7 |
| Moran I, signed residuals, k=8 | -0.076861 | -0.074302 |
| Two-sided permutation p, k=8 | 0.0630 | 0.0720 |

For Reference26, signed-error Moran I at k=4,8,12,24 respectively:
-0.055864 (p=.3455), -0.076861 (p=.0630), -0.062976 (p=.0640), -0.041802 (p=.0645). None of these unadjusted permutation tests rejects at .05. **This does not establish independence or rule out clusters in absolute/squared error.** The already implemented jury report computes those additional Moran/Geary rows, but their numerical values were not available in the transcript.

The Shapiro tests reject the marginal normal reference and the residual distribution is positively skewed. That does not invalidate predictive RMSE. Koenker/Brown-Forsythe did not detect heteroscedasticity under their chosen auxiliary specifications, but this is not proof of homogeneous error variance.

The historical +0.6797 Moran I for **observed yields** at k=5, reported in Checkpoint 04A, concerns a different variable, weight matrix and protocol from the current residual tests. Do not subtract the statistics to claim a quantified amount of spatial structure removed.

The blend gain over Reference26 is only about 0.07% RMSE and disappears when the most favorable observation is deleted in the earlier paired experiment. Its mixing weights were chosen using the same 138 outcomes; subsequent tests on those errors are not independent validation.

## 3. New analysis: geographic group bootstrap

Let G denote a geographic grouping of the 138 parcels. In each replicate draw the existing G clusters *with replacement*, then count how many times the group containing parcel i is sampled (m_i). For each fixed model:

\[
\mathrm{MSE}^{*} = \frac{\sum_i m_i e_i^2}{\sum_i m_i},\quad
\mathrm{MAE}^{*} = \frac{\sum_i m_i|e_i|}{\sum_i m_i},\quad
\mathrm{bias}^{*} = \frac{\sum_i m_i e_i}{\sum_i m_i}.
\]

RMSE is the square root of the resampled MSE. Compare squared loss, RMSE and absolute loss against Reference26 with **identical resampled groups** for every model. Report empirical percentile intervals and grouping sensitivity. Do not call them exact 95% inferential coverage, due to overlap between LOOCV fits, neighboring-block dependence and prior model selection.

The script tries state-plus-municipality clusters (to disambiguate repeated municipality names), and approximate latitude/longitude equirectangular square grids of 15, 30 and 50 km with both unshifted and half-cell shifted origins. A partition with fewer than eight populated groups is **skipped**, never quietly replaced by iid parcel bootstrapping. Grid size/offset dependence is part of the result.

## 4. New analysis: influential observations

For each model report (a) largest-one and largest-two shares of total squared error and (b) RMSE after deleting those individual prediction errors **without fitting again**. In paired comparisons remove the most favorable one, two, five and ten observations for each candidate and recompute its mean-squared-loss difference from Reference26. These are deliberately adversarial sensitivity checks, not new LOOCV scores or formal hypothesis tests.

Private per-parcel diagnostics include each ID and residual so the data owner can audit units, harvest records, geometry and covariates. A large model error **does not demonstrate** that the observed label is wrong. Only independently documented source errors justify corrections.

## 5. Execution on the private workstation

The runner checks source hashes, original manifest, frozen ensemble weights, 138 unique completed heldouts, ID/geography alignment, and finite coordinates. It never refits models, reads hidden yields or publishes data.

Use PowerShell in GeoC with the geocebada environment:

    git fetch origin
    git switch analysis/reference26-inferential-sensitivity-20261019
    git pull --ff-only
    python -m pytest -q tests/test_inferential_sensitivity.py
    python tools/report_reference26_compact14_final.py --summary
    python tools/report_reference26_inferential_sensitivity.py --run
    python tools/report_reference26_inferential_sensitivity.py --summary

Defaults are 3,000 spatial group resamples and seed 20261019. All files stay in the Git-ignored local folder:

    reports/reference26_compact14_fresh138/inferential_sensitivity/

Files:

- grouped_bootstrap_intervals.csv: per-model conditional ranges and paired contrasts;
- spatial_partition_audit.csv: explicit computed/skipped cluster configurations;
- leave_one_out_deletion_summary.csv: aggregate concentration of squared errors;
- paired_advantage_fragility.csv: paired stress-test trajectories;
- PRIVATE_parcel_influence.csv: **private IDs and per-parcel residuals; NEVER commit**;
- provenance.json: immutable input provenance and limits.

No publish option exists. Do not copy parcel IDs, yields, predictions or coordinates to the public website.

## 6. Honest inferential conclusion and outstanding work

The existing LOOCV supports describing Reference26's **observed reconstruction accuracy for individual holdouts under current neighborhood support**. Existing signed-error Moran tests do not detect a spatial signal after their multiplicity correction; the full absolute/squared-error Moran/Geary and empirical semivariogram tables still need inspection before stronger statements about geography are possible.

The new analysis assesses sensitivity to spatial partition, error concentration and the blend's most favorable parcels. It does **not** restore independence to the 138 overlapping fits. More ambitious claims about generalization to entirely new regions require spatially blocked cross-validation with refitting, ideally designed before inspecting outcomes; formal selection-corrected comparison of the ensemble requires nested weight/model selection or genuinely new labels.

There is no defensible universal adjusted R-squared parameter count for the nested, regularized mixture. The fixed algorithm settings k and alpha are hyperparameters, not regression coefficients with conventional t-tests. AIC/BIC likewise require a specified likelihood and cannot be inferred from point predictions alone.

**Outstanding:** execute the new script locally, inspect all grouped intervals (including skipped partitions), reconcile private Moran I of absolute/squared errors and Geary C, audit the strongest influential observations using source data, and decide whether to publish Reference26 alone. No invented results, no retraining and no changes to the 59 targets.
