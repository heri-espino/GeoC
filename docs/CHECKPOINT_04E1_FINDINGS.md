# Checkpoint 04E.1 — SIAP external localization findings

Checkpoint 04E.1 tested whether the public 2025 SIAP municipal closure adds useful
competition-mode information beyond the spatial/local models frozen in Checkpoint
04D.1. The experiment preserved the committed Checkpoint 04B pseudo-competition
splits and never loaded or scored the 59 hidden parcel yields.

## Information boundary

The external prior was defined at municipality level using the exact scope

`Cebada grano + Primavera-Verano + Temporal + CVEGEO + 2025`.

Broader SIAP scopes were kept only as explicit fallbacks. In every pseudo split,
target-aware calibration or residual modeling used only the currently visible
pseudo-training labels. Public SIAP values and all X-only information remained
available for pseudo-target parcels, matching the competition information boundary.

## Coverage

The exact 2025 scope covered **197/197** parcels. The selected prior also covered
**197/197** before any median fallback, including **59/59** actual competition
targets. Therefore the result is not explained by missing SIAP coverage.

The exact-scope audit found 83 municipalities in 2025 and 83 exact
Primavera-Verano + Temporal rows for those municipalities. The broader
all-grain scope contained 94 rows in 2025.

## Primary target-matched result

| method | mean split RMSE | pooled RMSE | pooled MAE |
|---|---:|---:|---:|
| Baseline Local04D | **0.4878** | **0.4957** | 0.3425 |
| Local + SIAP-local residual, 25% | 0.4922 | 0.4995 | 0.3528 |
| Baseline Graph04D | 0.4949 | 0.5033 | **0.3403** |
| Graph + SIAP-graph residual, 25% | 0.5014 | 0.5095 | 0.3500 |
| SIAP local residual | 0.5456 | 0.5519 | 0.4151 |
| SIAP graph residual | 0.5558 | 0.5636 | 0.3976 |
| SIAP affine calibration | about 0.837 | about 0.838 | about 0.717 |
| SIAP direct | 1.4368 | 1.4379 | 1.3095 |

The Checkpoint 04D.1 controls were reproduced essentially exactly:
Local04D returned 0.487845 mean target-matched RMSE and Graph04D returned
0.494881. This is an important integrity check: the new external-data layer did
not change the frozen pseudo-competition evaluation.

## Leave-one-split-out selection

The controlled leave-one-target-matched-split-out selector chose
**Baseline_Local04D on all 16/16 holdout splits**. Its nested mean RMSE was
0.4878, with pooled RMSE 0.4957.

This is stronger evidence than selecting the best row from the full development
table. SIAP was available to every selector training split, yet no SIAP-based
candidate displaced the local baseline.

## Direct proxy diagnostic

Across the 138 observed parcels, the selected SIAP municipal prior had:

- direct RMSE: **1.4926 t/ha**;
- Spearman correlation with observed parcel yield: **-0.2181**;
- legacy aggregated SIAP 2025 direct RMSE: **1.4894 t/ha**.

Filtering SIAP correctly by crop, cycle and modality therefore fixes the semantic
definition but does not turn the municipality-level statistic into a strong
parcel-level yield proxy.

## Stress protocols

SIAP was not uniformly useless. Under municipality-grouped validation, the
25% graph/SIAP blend improved mean split RMSE from **0.5279** to **0.5096**.
That suggests municipal public statistics can help when the evaluation problem
forces extrapolation across municipality groups.

However, the challenge target geometry is represented by the committed
target-matched pseudo-splits, where the same blend worsened performance from
0.4949 to 0.5014. The grouped result is therefore retained as a diagnostic, not
used to override the primary competition-matched evidence.

## Decision

**Do not include SIAP as a required component of the current final predictor.**

Keep SIAP as:

1. a documented public external evidence source;
2. a possible diversity feature for later sensitivity analysis;
3. a useful stress-test signal for municipality extrapolation.

Do not assign it positive ensemble weight merely because coverage is complete or
because it is contemporaneous 2025 information.

The next focused experiment is the deferred **Checkpoint 04C.1 global CatBoost
anchor**. Its purpose is not another broad model search. It tests whether the
historically complementary C1 agronomic CatBoost signal adds validated diversity
to the much stronger Local04D / Graph04D transductive predictors on the exact
same frozen pseudo-competition splits.
