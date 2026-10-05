# Checkpoint 06A/06B — Tail-aware diagnostic report

Generated: 2026-10-05T21:37:51.462688+00:00
Git commit: ca5fcc4a202ee04c091389e6d3a69e32f524cd43

## Contract

- active track: competition only
- hidden 59 FIRA y used or scored: no
- base predictions: frozen honest target-matched Checkpoint 05 predictions
- clustering/specialists/calibration: not run in this stage

## Diagnostic decision

**TAIL_HYPOTHESIS = SUPPORTED**

This is a continuation gate, not promotion of a competition model.

Decision payload:
{
  "TAIL_HYPOTHESIS": "SUPPORTED",
  "best_error_risk_model": {
    "family": "soil_profile",
    "risk_spearman": 0.2190783410138249
  },
  "best_tail_classifier": {
    "family": "nonlinear_basis",
    "mean_split_pr_lift": 1.5851493068879936,
    "mean_split_roc_auc": 0.7100026372437961
  },
  "catboost_high_tail": {
    "expected_sign_fraction": 1.0,
    "mean_residual": 0.523902235862669
  },
  "catboost_low_tail": {
    "expected_sign_fraction": 0.9032258064516129,
    "mean_residual": -0.42709961738773156
  },
  "catboost_shrinkage_corroboration": true,
  "concentration_supported": true,
  "error_risk_supported": true,
  "local04d_high_tail": {
    "expected_sign_fraction": 1.0,
    "mean_residual": 0.47257951770967727
  },
  "local04d_low_tail": {
    "expected_sign_fraction": 0.8064516129032258,
    "mean_residual": -0.3576638799361657
  },
  "local04d_top10pct_sse_share": 0.6217760775160602,
  "shrinkage_supported_local04d": true,
  "tail_classifier_supported": true,
  "thresholds": {
    "min_abs_tail_bias": 0.1,
    "min_error_risk_spearman": 0.2,
    "min_expected_sign_fraction": 0.6,
    "min_pr_lift": 1.2,
    "min_tail_auc": 0.65,
    "min_top10pct_sse_share": 0.3
  },
  "x_predictability_supported": true
}

## Local04D SSE concentration

  method             aggregation     selection  k  fraction  sse_share
Local04D equal_weight_parcel_mse top_5_parcels  5      0.04   0.445723
Local04D equal_weight_parcel_mse     top_10pct 13      0.10   0.621776

## Local04D tail bias

  method tail_regime  n_parcels  mean_observed  mean_residual  median_residual     rmse      mae  mean_residual_ci_low  mean_residual_ci_high  expected_shrinkage_sign_fraction
Local04D        high         21       5.010952       0.472580         0.469320 0.547073 0.475199              0.354829               0.592410                          1.000000
Local04D         low         31       2.783548      -0.357664        -0.138465 0.658097 0.381597             -0.574373              -0.177747                          0.806452

## CatBoost tail bias

     method tail_regime  n_parcels  mean_observed  mean_residual  median_residual     rmse      mae  mean_residual_ci_low  mean_residual_ci_high  expected_shrinkage_sign_fraction
CatBoost_C1        high         21       5.010952       0.523902         0.488908 0.600355 0.524084              0.404151               0.641520                          1.000000
CatBoost_C1         low         31       2.783548      -0.427100        -0.240710 0.712908 0.441935             -0.636264              -0.243904                          0.903226

## Local04D vs CatBoost overlap

                 metric    value    k
       residual_pearson 0.962214  NaN
      residual_spearman 0.928344  NaN
 squared_error_spearman 0.824393  NaN
residual_sign_agreement 0.864000  NaN
    top_5_error_jaccard 0.666667  5.0
   top_10_error_jaccard 0.666667 10.0

## Best agronomic-family tail classifiers

            family           task  n_splits  n_features  mean_split_roc_auc  median_split_roc_auc  mean_split_average_precision  mean_split_prevalence  mean_split_pr_lift  mean_split_brier  pooled_row_roc_auc  pooled_row_average_precision  pooled_row_pr_lift
   nonlinear_basis tail_vs_center        16           7            0.710003              0.692826                      0.577746               0.371951            1.585149          0.231929            0.693379                      0.516026            1.387348
      water_timing tail_vs_center        16          10            0.678752              0.670273                      0.547589               0.371951            1.502134          0.229780            0.663387                      0.501018            1.347000
water_productivity tail_vs_center        16          40            0.665909              0.659545                      0.573153               0.371951            1.581360          0.254145            0.655777                      0.517279            1.390717
           thermal tail_vs_center        16          36            0.659507              0.630576                      0.537726               0.371951            1.474616          0.236823            0.647640                      0.486992            1.309289
     all_agronomic tail_vs_center        16         350            0.613711              0.608085                      0.491228               0.371951            1.333617          0.324480            0.614754                      0.447186            1.202270
      cross_domain tail_vs_center        16          19            0.608799              0.607090                      0.516155               0.371951            1.399880          0.258143            0.593526                      0.444622            1.195378
  soil_interaction tail_vs_center        16           8            0.602525              0.589927                      0.516821               0.371951            1.423722          0.248972            0.584633                      0.438438            1.178752
  sensor_agreement tail_vs_center        16           9            0.568197              0.561983                      0.419640               0.371951            1.133964          0.260564            0.564251                      0.381400            1.025402
         phenology tail_vs_center        16         162            0.549163              0.566905                      0.470087               0.371951            1.273265          0.336942            0.545639                      0.415579            1.117294
      soil_profile tail_vs_center        16          18            0.535724              0.519492                      0.450864               0.371951            1.236171          0.276077            0.520939                      0.392470            1.055164

## Best Local04D error-risk family models

  method            family  n_parcels  n_features  risk_spearman  risk_r2_log  risk_rmse_log
Local04D      soil_profile        125          18       0.219078    -0.043011       0.276294
Local04D      cross_domain        125          19       0.125125    -0.039064       0.275771
Local04D         phenology        125         162       0.123525    -0.396559       0.319711
Local04D           thermal        125          36       0.106402    -0.019014       0.273098
Local04D     all_agronomic        125         350       0.103930    -0.607235       0.342979
Local04D phenology_anomaly        125          41       0.051269    -0.122407       0.286618
Local04D  sensor_agreement        125           9       0.045601    -0.036360       0.275412
Local04D      water_timing        125          10       0.036933    -0.028491       0.274365
Local04D   nonlinear_basis        125           7      -0.002937    -0.066272       0.279358
Local04D  soil_interaction        125           8      -0.025579    -0.038196       0.275656

## Next step

The diagnostic gate supports continuing to Checkpoint 06C.
Implement low-capacity calibration first; do not jump directly to clustering.

Local04D and reports/checkpoint_05/final_predictions.csv remain canonical.
