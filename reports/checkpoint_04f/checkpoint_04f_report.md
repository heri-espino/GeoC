# Checkpoint 04F — Final transductive reconstruction

Generated: 2026-09-21T21:19:39.386565+00:00
Git commit: 77aac47d9e4a08ec50c7f12a50aba5b736182261

## Frozen final rule

- selected method: **Baseline_Local04D**
- final rows: **59**
- hidden 59 yields used or scored: **no**
- frozen target-matched mean RMSE: **0.487845**

The final rule is intentionally simpler than the full development suite. SIAP and CatBoost did not add reproducible target-matched improvement, and support-tier routing was not promoted after constrained checks.

## Local/Graph sensitivity

| method | graph_weight | rmse_mean | pooled_rmse | pooled_mae | rmse_worst |
|---|---|---|---|---|---|
| LocalGraph_g0p25 | 0.250000 | 0.486768 | 0.494892 | 0.340479 | 0.693090 |
| LocalGraph_g0p30 | 0.300000 | 0.486780 | 0.494943 | 0.340151 | 0.692886 |
| LocalGraph_g0p20 | 0.200000 | 0.486832 | 0.494914 | 0.340816 | 0.693334 |
| LocalGraph_g0p40 | 0.400000 | 0.487032 | 0.495268 | 0.339582 | 0.692595 |
| LocalGraph_g0p10 | 0.100000 | 0.487188 | 0.495180 | 0.341582 | 0.693939 |
| LocalGraph_g0p50 | 0.500000 | 0.487588 | 0.495887 | 0.339077 | 0.692461 |
| LocalGraph_g0p00 | 0.000000 | 0.487845 | 0.495741 | 0.342501 | 0.694700 |
| LocalGraph_g1p00 | 1.000000 | 0.494881 | 0.503346 | 0.340288 | 0.694149 |

The Local/Graph blend table is a development sensitivity diagnostic, not an automatic final selector.

## Constrained LOSO blend-weight selection

Mean holdout RMSE: **0.487941**; pooled RMSE: **0.495987**.

| selected method | holdouts |
|---|---:|
| LocalGraph_g0p20 | 3 |
| LocalGraph_g0p25 | 5 |
| LocalGraph_g0p30 | 7 |
| LocalGraph_g0p40 | 1 |

## Actual-target disagreement diagnostics

- mean |Local-Graph| disagreement: **0.1056 t/ha**
- median |Local-Graph| disagreement: **0.0769 t/ha**
- maximum |Local-Graph| disagreement: **0.4980 t/ha**

These diagnostics do not modify any parcel prediction. They are retained for uncertainty interpretation and final reporting.

## Output

final_predictions.csv contains exactly two columns: ID_POLIGONO, RENDIMIENTO_T_HA.

final_prediction_diagnostics.csv contains the frozen Local04D value, Graph04D reference, X-support information and disagreement diagnostics.
