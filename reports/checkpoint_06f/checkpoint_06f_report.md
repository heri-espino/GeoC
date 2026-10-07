# Checkpoint 06F — selective center expert / abstention gate

Generated: 2026-10-07T05:48:26.991187+00:00
Git commit: 3a4ee5408502f619924f192bdc246675045f5f0f

## Architecture

If both supervised tail scores are below a frozen cutoff:
    prediction = Local04D + frozen G1-PCA Ridge center residual correction
Else:
    prediction = Local04D

No low-tail or high-tail correction is applied.

## Frozen center expert

representation: all_agronomic_pca
PCA retained variance: 0.8
Ridge alpha: 1000.0

## Development ranking

              method  n_splits  n_predictions  rmse_mean  rmse_median  rmse_worst  pooled_rmse  center_pooled_rmse  tail_pooled_rmse  low_pooled_rmse  high_pooled_rmse  center_bias  low_bias  high_bias
SelectiveCenter_t0p7        16            656   0.487108     0.464622    0.676572     0.494704            0.354471          0.667687         0.724256          0.553372     0.025703 -0.440558   0.492752
SelectiveCenter_t0p6        16            656   0.487692     0.467445    0.680430     0.495321            0.357771          0.665942         0.723038          0.550357     0.020568 -0.436189   0.486956
            Local04D        16            656   0.487845     0.468742    0.694700     0.495741            0.360770          0.664049         0.727121          0.534243     0.013806 -0.431968   0.473166
SelectiveCenter_t0p5        16            656   0.487889     0.466792    0.680314     0.495567            0.360089          0.664323         0.721483          0.548547     0.021065 -0.430119   0.482264
SelectiveCenter_t0p8        16            656   0.488503     0.467027    0.670659     0.495765            0.351048          0.672836         0.721973          0.575511     0.033696 -0.450677   0.514606
SelectiveCenter_t0p4        16            656   0.489099     0.464870    0.694700     0.497018            0.361360          0.666071         0.725024          0.546141     0.017076 -0.431275   0.478868

## Routing diagnostics

              method  n_predictions  n_center_expert_used  expert_use_rate  true_center_use_rate  true_tail_use_rate  routed_center_precision
            Local04D            656                     0         0.000000              0.000000            0.000000                      NaN
SelectiveCenter_t0p4            656                   138         0.210366              0.274272            0.102459                 0.818841
SelectiveCenter_t0p5            656                   217         0.330793              0.436893            0.151639                 0.829493
SelectiveCenter_t0p6            656                   286         0.435976              0.558252            0.229508                 0.804196
SelectiveCenter_t0p7            656                   334         0.509146              0.648058            0.274590                 0.799401
SelectiveCenter_t0p8            656                   409         0.623476              0.757282            0.397541                 0.762836

## LOSO

{
  "center_rmse_improvement": -0.0006374560158893261,
  "incumbent_center_rmse": 0.36076982220030784,
  "incumbent_pooled_rmse": 0.49574142871073495,
  "incumbent_rmse_mean": 0.48784531674343823,
  "incumbent_tail_rmse": 0.6640489003338566,
  "n_splits": 16,
  "pooled_rmse_improvement": -0.002894428279336947,
  "rmse_mean_improvement": -0.003002120526943375,
  "selected_center_rmse": 0.36140727821619717,
  "selected_pooled_rmse": 0.4986358569900719,
  "selected_rmse_mean": 0.4908474372703816,
  "selected_tail_rmse": 0.669269457296827,
  "selection_counts": {
    "Local04D": 6,
    "SelectiveCenter_t0p6": 5,
    "SelectiveCenter_t0p7": 5
  },
  "tail_rmse_change": 0.005220556962970413
}

## Decision

**06F_LOSO_GATE = FAIL**

Selective hard routing did not survive the predeclared LOSO gate.
Retain Local04D and close center/tail correction work on these development splits.
