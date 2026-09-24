# Checkpoint 03D findings — balanced-compute global closure

**Completed:** 2026-09-24  
**Status:** CLOSED / COMPLETED  
**Active track:** competition-only  
**Training rows:** 138  
**Fixed hidden targets:** 59  
**Hidden FIRA target y used or scored:** no

Checkpoint 03D answered the last broad global-model question left open by the
small search grids of Checkpoint 03C: whether substantially larger nonlinear
models and a materially larger tuning budget could move the global frontier.

## Runtime decision and final budget

The first wide configuration was intentionally stopped after seven completed
CatBoost outer fits (the eighth had started). Each outer fit was taking roughly
15 minutes because it contained nine hyperparameter candidates, three inner
folds and a refit. Extrapolating that design to all 230 outer fits implied a
multi-day workstation run for only 138 labeled parcels.

The final 03D run preserved both frozen five-fold outer protocols and three-fold
inner CV, while reducing each heavy nonlinear family to three representative
candidates, capping boosting at 2,500 estimators/iterations and reducing
ExtraTrees from 3,000 to 1,500 trees. Stale wide-grid partials were not reused.
The runner now fingerprints config/model/representation/compute state and
refuses an incompatible `--resume`.

This is a compute-budget change, not a validation-protocol change.

## Completed execution

```text
elapsed                         248.085 min (~4 h 08 min)
eligible representation/model pairs     23
outer fits                              230
OOF prediction rows                    6348
training parcels                        138
hidden target parcels                    59
compute                                 GPU
ONNX verification                      PASS
```

Representations were deterministic and target-free: G0 base, G1 agronomic,
G2 base+agronomic and G3 all deterministic. No target-aware expression
discovery was reopened.

## Robust global finalists

| rank | representation | model | state RMSE | municipality-grouped RMSE | worst RMSE |
|---:|---|---|---:|---:|---:|
| 1 | G1 agronomic | HistGBLarge | 0.549924 | **0.715339** | **0.715339** |
| 2 | G1 agronomic | XGBoostLarge | **0.525847** | 0.733080 | 0.733080 |
| 3 | G1 agronomic | LightGBMLarge | 0.540812 | 0.748478 | 0.748478 |

All three robust finalists use the compact agronomic representation. Larger
concatenations often looked strong under state CV but degraded under grouped CV:

```text
G2 + CatBoost       state 0.507368   grouped 0.930677
G3 + CatBoost       state 0.514583   grouped 0.930504
G2 + ExtraTrees     state 0.498841   grouped 0.857749
G3 + ExtraTrees     state 0.503888   grouped 0.881164
```

## Relation to frozen 03C

Historical E123/E13 were approximately 0.5121/0.7481 and 0.5082/0.7488 in
(state, grouped). HistGB G1 moves the grouped global frontier to 0.7153 but
worsens state RMSE; XGBoost G1 is 0.5258/0.7331. Thus larger compute found
stronger grouped global signal but no model that dominates both protocols.

The hypothesis closed by 03D is not that large models are useless. It is that
the small 03C grid was not the sole explanation for global limitations.

## Relation to Local04D

Do **not** compare 03D state/grouped scores directly with Local04D's 0.487845
target-matched RMSE as though they came from the same validation distribution.
They are different protocols.

03D cannot by itself promote a competition predictor. Local04D remains canonical
because Checkpoint 05 already failed its promotion gate. Any last attempt to use
03D globals must place them inside the same target-matched pseudo-competition
protocol through Checkpoint 05B.

## ONNX deployment

Scientific ranking and deployability are separate. HistGBLarge G1 is scientific
rank 1 but is not convertible with the pinned stable stack. XGBoostLarge G1,
scientific rank 2, is the highest-ranked deployable finalist.

The final ONNX round trip passed over 138 rows with maximum absolute difference
`3.814697265625e-06`; the post-imputation deployment object reproduced the
Python pipeline exactly before conversion.

Local/gitignored artifacts:

```text
models/final/checkpoint03d_global.joblib
models/final/checkpoint03d_global.onnx
models/final/checkpoint03d_global.onnx.json
```

Versioned verification evidence is in
`reports/checkpoint_03d/checkpoint_03d_report.json`. The global ONNX estimator
is not an ONNX encoding of Local04D.

## Final decision

Checkpoint 03D is closed. No further broad global hyperparameter sweep is
justified. The canonical competition output remains
`reports/checkpoint_05/final_predictions.csv`, which retains Local04D.

The only remaining modeling branch, if explicitly desired, is the narrow
Checkpoint 05B plan. Otherwise the next work is submission/app/report
integration.
