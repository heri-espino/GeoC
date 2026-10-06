# Checkpoint 06C — low-capacity tail calibration

Generated: 2026-10-06T01:05:31.869929+00:00
Git commit: 2a8372e1b92262611afd621be4e3dd90a349b02e

## Contract

- prerequisite 06A/06B: SUPPORTED
- calibration fit rows are cross-fitted pseudo-train predictions
- score rows are untouched pseudo-target predictions
- hidden 59 FIRA y used/scored: no
- clustering and specialists: not run

## Development ranking

                      method  n_splits  n_predictions  rmse_mean  rmse_median  rmse_worst  pooled_rmse  tail_pooled_rmse  center_pooled_rmse  low_pooled_rmse  high_pooled_rmse  low_bias  high_bias
  LocalPiecewiseRidge_a100p0        16            656   0.485375     0.479426    0.682580     0.492747          0.669400            0.348161         0.718358          0.572413 -0.436578   0.500305
   LocalResidualRidge_a100p0        16            656   0.487033     0.464592    0.694585     0.494945          0.655506            0.368205         0.713833          0.536752 -0.404533   0.477492
    LocalResidualRidge_a10p0        16            656   0.487739     0.464956    0.699140     0.495808          0.651438            0.374279         0.710684          0.530388 -0.392117   0.470803
                    Local04D        16            656   0.487845     0.468742    0.694700     0.495741          0.664049            0.360770         0.727121          0.534243 -0.431968   0.473166
     LocalResidualRidge_a1p0        16            656   0.487913     0.465041    0.700084     0.496014          0.650649            0.375524         0.710090          0.529114 -0.389621   0.469459
          LocalStretch_s1p05        16            656   0.490063     0.471078    0.707885     0.498411          0.649511            0.381694         0.716119          0.510505 -0.389789   0.445631
LocalCatResidualRidge_a100p0        16            656   0.490137     0.463569    0.694476     0.497606          0.656284            0.373062         0.711374          0.545109 -0.399820   0.484772
   LocalPiecewiseRidge_a10p0        16            656   0.493411     0.491291    0.682716     0.500837          0.676874            0.357848         0.716486          0.600263 -0.433843   0.492822
           LocalStretch_s1p1        16            656   0.494862     0.475909    0.722995     0.503587          0.637228            0.404113         0.707948          0.487257 -0.347611   0.418096
 LocalCatResidualRidge_a10p0        16            656   0.495467     0.464517    0.701372     0.502683          0.656874            0.383150         0.709999          0.550232 -0.390140   0.487527
  LocalCatResidualRidge_a1p0        16            656   0.496943     0.465084    0.703376     0.504113          0.657627            0.385369         0.710307          0.552020 -0.388923   0.488568
    LocalPiecewiseRidge_a1p0        16            656   0.497757     0.493232    0.684089     0.505401          0.679607            0.364927         0.716986          0.607719 -0.430946   0.487263

## Best development candidate

{
  "method": "LocalPiecewiseRidge_a100p0",
  "n_splits": 16,
  "n_predictions": 656,
  "rmse_mean": 0.48537481073683075,
  "rmse_median": 0.47942606370922286,
  "rmse_worst": 0.6825798061170116,
  "pooled_rmse": 0.49274703915094653,
  "tail_pooled_rmse": 0.6694004443168138,
  "center_pooled_rmse": 0.34816059606440014,
  "low_pooled_rmse": 0.7183577615174492,
  "high_pooled_rmse": 0.5724126443152051,
  "low_bias": -0.4365784104215861,
  "high_bias": 0.50030511708687
}

## LOSO selection

{
  "center_rmse_change": 0.0,
  "incumbent_center_rmse": 0.36076982220030784,
  "incumbent_pooled_rmse": 0.49574142871073495,
  "incumbent_rmse_mean": 0.48784531674343823,
  "incumbent_tail_rmse": 0.6640489003338566,
  "n_splits": 16,
  "pooled_rmse_improvement": 0.0,
  "rmse_mean_improvement": 0.0,
  "selected_center_rmse": 0.36076982220030784,
  "selected_pooled_rmse": 0.49574142871073495,
  "selected_rmse_mean": 0.48784531674343823,
  "selected_tail_rmse": 0.6640489003338566,
  "selection_counts": {
    "Local04D": 16
  },
  "tail_rmse_improvement": 0.0
}

## Decision

**06C_LOSO_GATE = FAIL**

Simple calibration did not survive split-excluded selection.
Proceed to 06D X-only regime discovery rather than adding more calibration flexibility.
