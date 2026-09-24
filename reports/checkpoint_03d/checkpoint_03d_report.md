# Checkpoint 03D — Large-compute global benchmark

Generated: 2026-09-24T06:54:23.935078+00:00
Git commit: 9213dc275cdc284951a9f812924e010d7c2be022

## Contract

- active track: competition only
- hidden FIRA y scored: no
- compute mode: GPU
- outer fits: 230
- ONNX verified: True

## Global finalists

| rank | representation | model | state RMSE | grouped RMSE | worst RMSE |
|---:|---|---|---:|---:|---:|
| 1 | G1_agronomic | HistGBLarge | 0.549924 | 0.715339 | 0.715339 |
| 2 | G1_agronomic | XGBoostLarge | 0.525847 | 0.733080 | 0.733080 |
| 3 | G1_agronomic | LightGBMLarge | 0.540812 | 0.748478 | 0.748478 |

The top finalist is refit on all 138 labeled parcels after hyperparameter
selection across the union of state-stratified and municipality-grouped
inner folds. Its model-only ONNX graph consumes the median-imputed
float32 feature matrix described by the adjacent JSON manifest.

03D does not replace Local04D. These finalists are inputs to the
conditional Checkpoint 05B target-matched remix.
