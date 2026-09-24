# Checkpoint 03D — Balanced-compute global modeling

**Status:** CLOSED / COMPLETED  
**Opened:** 2026-09-23  
**Completed:** 2026-09-24  
**Canonical interpretation:** `docs/CHECKPOINT_03D_FINDINGS.md`

Checkpoint 03D is the final additive extension of the frozen Checkpoint 03
global line. It tested whether the small 03C grids were the main reason global
models lagged under harder validation.

## Contract

Competition-only deterministic representations G0/G1/G2/G3 were evaluated with
CatBoost, XGBoost, LightGBM, ExtraTrees, HistGB and Ridge/PLS controls. Hidden
FIRA y was never scored. Both frozen five-fold outer protocols and three-fold
inner CV were preserved.

The initial wide grid was stopped because CatBoost outer fits took ~15 minutes
each and projected to multiple days. The balanced config reduced only redundant
tuning: three candidates per heavy nonlinear family, boosting capped at 2,500
iterations/estimators and ExtraTrees at 1,500 trees. Wide-grid partials were not
reused. Resume now checks a run fingerprint.

## Completed run

```text
elapsed         248.085 min (~4 h 08 min)
eligible pairs  23
outer fits      230
OOF rows        6348
compute         GPU
ONNX verified   yes
```

| rank | representation | model | state RMSE | grouped RMSE |
|---:|---|---|---:|---:|
| 1 | G1 agronomic | HistGBLarge | 0.549924 | **0.715339** |
| 2 | G1 agronomic | XGBoostLarge | **0.525847** | 0.733080 |
| 3 | G1 agronomic | LightGBMLarge | 0.540812 | 0.748478 |

All three robust finalists use the compact agronomic layer. 03D improves the
historical grouped global frontier but does not uniformly dominate state and
grouped performance.

## Deployment

Scientific ranking is independent of converter support. HistGB rank 1 was not
ONNX-convertible under the stable stack. XGBoost G1 rank 2 was the
highest-ranked deployable finalist. Its final ONNX round trip passed with
maximum absolute difference `3.814697265625e-06`.

Local artifacts remain gitignored:

```text
models/final/checkpoint03d_global.joblib
models/final/checkpoint03d_global.onnx
models/final/checkpoint03d_global.onnx.json
```

The global ONNX model is a generic deployment estimator, not Local04D.

## Decision

No further broad global sweep is planned. Local04D remains the canonical
competition method after Checkpoint 05. A last blend test, if wanted, must use
the narrow target-matched 05B plan. State/grouped 03D scores must not be
compared directly with Local04D's target-matched RMSE as if they were the same
protocol.
