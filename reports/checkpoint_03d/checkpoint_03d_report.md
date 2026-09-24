# Checkpoint 03D — Balanced-compute global benchmark

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

All scientific finalists are refit on all 138 labeled parcels after
hyperparameter selection across the union of state-stratified and
municipality-grouped inner folds. The ONNX artifact was exported from the
highest-ranked finalist with verified converter support, XGBoostLarge G1
(scientific rank 2), because HistGBLarge G1 (scientific rank 1) was not
converter-eligible under the pinned stable stack. The final ONNX numerical
verification passed.

03D does not replace Local04D. Any competition use of these globals requires
the separate Checkpoint 05B target-matched gate.
