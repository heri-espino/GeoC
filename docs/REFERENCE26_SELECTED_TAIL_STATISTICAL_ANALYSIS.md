# Final 26-feature analog correction: statistical analysis

Status: IMPLEMENTED, **not yet executed** against private VM data (2026-10-19).
This is a new report; earlier Reference26/Compact14 reports are preserved.

## Three distinct predictors

The final 59-target selection manifest records the winner. The user's latest
workstation log selected analog__robust26__k3__blend0.25__supported.
The new runner reads the actual manifest rather than hardcoding that value.

- Reference26_mean: genuine native 138-fold leave-one-out predictions, with
  137 observed yields per outer fold.
- SelectedFinalRule: apply the **globally selected fixed correction** to each
  original 137/1 baseline prediction. Its recipe has been chosen using these
  138 observed yields: error estimates are post-selection exploratory.
- NestedSelection: one correction chosen separately per outer heldout using
  four training-only inner folds. This is NOT exactly the same estimator
  as the final global rule. The five candidates had already been screened on
  these 138 labels, so it remains conditional on historical model selection.

The model minimizes RMSE and therefore maximizes the predictive R² on the
same y vector; these are equivalent rankings, since R² = 1 - SSE/TSS.

## Mathematical mechanism

Let b_i be native Reference26's heldout estimate, a_i be the weighted yield
of the k=3 nearest labeled analogs in robustly standardized 26-feature space,
and g_i the geometric support multiplier:

    corrected_i = b_i + clip( alpha*g_i*(a_i-b_i), -c, +c )
    alpha = 0.25
    c = 0.5 t/ha by the default selected experiment
    g_i = exp(-0.5*(nearest_distance/median_train_spacing)^2)
          / (1+(local_y_dispersion/train_y_sd)^2)

Neighbors receive inverse-feature-distance weights with power 1.
At every 137/1 heldout, heldout y is excluded from all neighbor labels.
X-only standardization may use all 197 known feature vectors.

Alpha, k, geometry, support gate and cap are algorithmic hyperparameters;
there is **no ordinary vector of regression coefficients** for which OLS
coefficient t-tests or ordinary adjusted R² would be mathematically valid.
The runner separately estimates observed-on-predicted calibration slope
and intercept with conditional HC3 descriptive uncertainty. These are
not coefficients or confidence intervals of the production predictor.

## Diagnostics reproduced and expanded

- Prediction metrics: MSE, RMSE, MAE, median absolute error, L0, L1,
  L2, Linfinity, SSE, predictive R², explained variance, Pearson,
  Spearman, bias, residual SD, percentiles and worst errors.
- Robust and relative losses: Huber, log-cosh, quantile losses, concordance,
  WAPE, MAPE/sMAPE, RMSLE and calibrated regression.
- Residual distribution: Shapiro-Wilk, Anderson-Darling, D'Agostino,
  skewness/kurtosis, MAD, sign and mean-error diagnostics.
- Conditional variance: Koenker/Breusch-Pagan and Brown-Forsythe.
- Geographic autocorrelation: global Moran's I and Geary's C for signed,
  absolute and squared errors, as well as the observed-y signal,
  predictions and correction delta; geographic k=4,8,12,24.
  Default 1,999 permutation replicates, Holm and BH across ALL reported
  spatial tests, residual semivariogram.
- Matched losses: sign/Wilcoxon contrasts, Holm, fraction of parcels improved.
- Leave-out influence and how fixed-rule advantage changes after the 1, 2,
  5 or 10 most favorable cases are deleted.
- Spatial-grouped bootstrap over municipalities and grids of 15/30/50 km,
  including shifted grid origins and 3,000 default repetitions.
- Bias and RMSE for lower 10%, middle 80%, upper 10% of observed y and,
  separately, original predicted yield. Observed-y groups are diagnostic
  ONLY: hidden target yields cannot be used to define tail correction.
- Private histograms, Q-Q, yield-vs-prediction, residual-vs-prediction,
  Moran-by-k, ECDF, semivariogram and residual coordinate maps.

The global Moran statistic uses row-standardized geographic kNN weights;
it tests covariance of a geographic signal, not the presence of useful
information in 26-dimensional feature space. Spatial p-values assume a
valid exchangeability null, which may fail for correlated LOOCV forecasts.

## Run in PowerShell with the existing geocebada environment

    git fetch origin
    git switch --track origin/analysis/selected-analog-spatial-inference-20261019
    python -m pytest -q tests/test_reference26_top5_statistics.py tests/test_inferential_sensitivity.py
    python tools/report_reference26_top5_statistics.py --check
    python tools/report_reference26_top5_statistics.py --run
    python tools/report_reference26_top5_statistics.py --summary

For improved Monte Carlo p-value resolution use these optional arguments:

    python tools/report_reference26_top5_statistics.py --run --permutations 9999 --replicates 10000

The runner requires these existing *private* local experiment folders:

    reports/reference26_compact14_fresh138/
    reports/reference26_analog_tail_screen/
    reports/reference26_top5_nested/
    reports/reference26_top5_final59/

The original peer bundle and feature files must also be available at the
same paths as the original modeling scripts (or supply CLI path overrides).
Every fold and the final-selection manifest are verified before analysis.

## Outputs and privacy

All files are in ignored reports/reference26_top5_statistics/:

- STATISTICAL_REPORT.md, selected_algorithm_parameters.json, provenance.json
- primary_metrics.csv, specialized_metrics.csv
- global_spatial_autocorrelation.csv, residual_semivariogram.csv
- normality_and_distribution.csv, heteroscedasticity_and_calibration.csv
- calibration_parameters_conditional.csv, paired_loss_tests.csv
- tail_conditional_metrics.csv, private_regional_errors.csv
- grouped_bootstrap_intervals.csv, spatial_partition_audit.csv
- leave_one_out_deletion_summary.csv, paired_advantage_fragility.csv
- PRIVATE_oof_138.csv, PRIVATE_selected_gate_by_fold.csv
- PRIVATE_parcel_influence.csv and figures/

**Never commit raw outputs without a privacy audit.**
The 59 unknown yields remain unavailable; their actual RMSE, Moran's I of
errors, and error intervals cannot be calculated.

### Limitations

The fixed selected rule was historically chosen using the same 138 outcomes,
and the nested policy's candidate family was historically preselected.
Diagnostics on saved overlapping LOOCV predictions do not account for
selection uncertainty or retraining uncertainty. The grouped bootstrap is
conditional on frozen forecasts and grouping choices. Normality and
autocorrelation tests cannot prove iid residuals, unbiasedness, or causal
geographic effects. Conventional adjusted R² and model coefficient p-values
are not defined here without additional model assumptions.
