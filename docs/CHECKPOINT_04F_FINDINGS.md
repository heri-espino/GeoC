# Checkpoint 04F — Final transductive reconstruction findings

Checkpoint 04F closes the competition-modeling sequence by freezing one
reproducible 59-value prediction table after the target-matched, robustness,
external-evidence and focused global-anchor experiments.

The hidden FIRA yields were never loaded or scored.

## Final rule

The frozen final predictor is:

`Baseline_Local04D`

which corresponds to:

`LocalRidge_C4_all_deterministic_Geo_k24_a30_p1`

Its frozen target-matched mean split RMSE is **0.487845** with pooled RMSE
**0.495741**.

The final artifact contains exactly 59 rows and exactly two columns:

`ID_POLIGONO, RENDIMIENTO_T_HA`

at:

`reports/checkpoint_04f/final_predictions.csv`

## Provenance checks

Checkpoint 04F verified that both final reference methods are numerically
identical across the frozen 04D.1 and 04E.1 actual-target artifacts:

- Local04D maximum absolute difference: **0.0**
- Graph04D maximum absolute difference: **0.0**

This removes ambiguity about which fitted candidate was frozen for the final
59 parcels.

## Why the final rule is not the nominal best fixed Local/Graph blend

A post-development Local/Graph sensitivity table shows a shallow optimum near
25–30% Graph weight:

| rule | mean split RMSE | pooled RMSE | pooled MAE |
|---|---:|---:|---:|
| Local75 / Graph25 | 0.486768 | 0.494892 | 0.340479 |
| Local70 / Graph30 | 0.486780 | 0.494943 | 0.340151 |
| Local80 / Graph20 | 0.486832 | 0.494914 | 0.340816 |
| Local100 | 0.487845 | 0.495741 | 0.342501 |
| Graph100 | 0.494881 | 0.503346 | 0.340288 |

The absolute gain over Local04D is small. More importantly, constrained
leave-one-target-matched-split-out weight selection does not improve the frozen
local rule: the LOSO blend check reaches mean RMSE **0.487941** and pooled RMSE
**0.495987**.

The LOSO selector chose only graph weights 0.20–0.40, primarily 0.25–0.30, but
its held-out performance remains slightly worse than the fixed Local04D
baseline. Therefore the 0.486768 full-development-table value is treated as a
sensitivity result rather than sufficient evidence to change the final rule.

## Actual-target disagreement

For the 59 hidden-target parcels, Local04D and Graph04D differ by:

- mean absolute disagreement: **0.1056 t/ha**
- median absolute disagreement: **0.0769 t/ha**
- maximum absolute disagreement: **0.4980 t/ha**

The largest disagreement occurs for AGC_003. These values are retained only as
uncertainty/disagreement diagnostics; they do not alter the final frozen
prediction.

## Final model-selection interpretation

The full Checkpoint 04 sequence supports a deliberately simple final choice:

- Local04D is the strongest stable target-matched predictor.
- Graph04D is the main robustness and disagreement reference.
- SIAP has complete coverage but does not improve the primary target-matched
  validation.
- CatBoost does not add reproducible improvement to Local04D.
- Support-tier routing is not promoted because its small apparent gain was not
  established under a sufficiently constrained candidate universe.
- Local/Graph blending shows a small development-table gain but no convincing
  split-excluded improvement.

Accordingly, the final submission-facing prediction table is frozen to
Local04D without parcel-specific manual changes.
