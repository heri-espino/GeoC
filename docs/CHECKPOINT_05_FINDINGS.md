# Checkpoint 05 findings — final global/local mixture

**Completed:** 2026-09-23  
**Status:** CLOSED / NO PROMOTION  
**Canonical final method:** `Local04D`  
**Hidden FIRA target y used or scored:** no

Checkpoint 05 tested whether the strongest global line from Checkpoint 03 could
improve the frozen Checkpoint 04 local/transductive incumbent when combined
through fully cross-fitted stacking and a support-conditioned mixture-of-experts.

## Final decision

The best full-development challenger was:

```text
LocalE13_w0p10 = 90% Local04D + 10% E13 (PLS + CatBoost)
```

Target-matched development:

```text
                              mean RMSE   pooled RMSE
LocalE13_w0p10                 0.487243     0.495261
Local04D                       0.487845     0.495741
nominal improvement            0.000602     0.000481
```

This is approximately a 0.12% reduction in mean RMSE and 0.10% reduction in
pooled RMSE. The gain is therefore small even before robustness selection.

The predeclared leave-one-development-split-out method-selection gate produced:

```text
LOSO selector mean RMSE        0.487934
LOSO selector pooled RMSE      0.495933
Local04D mean RMSE             0.487845
Local04D pooled RMSE           0.495741
```

The selector is slightly worse than Local04D on both required metrics.
Therefore the development gate failed and the fresh confirmation bank was
correctly **not scored**. The promotion rule retained `Local04D`.

## What the experiment established

1. **There is weak complementary global signal.** A 10% E13 blend improves the
   full development table by a very small amount.
2. **That gain is not stable enough for promotion.** Split-excluded method
   selection reverses the advantage.
3. **More sophisticated stacking did not solve this.** On target-matched
   validation the principal learned meta-models were materially worse than
   Local04D:
   - ExtraTreesStack mean RMSE 0.495479;
   - MoE_Support 0.495574;
   - ConvexStack 0.496403.
4. **The local structure remains dominant.** Standalone global experts were
   weaker on the target-matched protocol:
   - ExtraTrees C1 0.510906;
   - CatBoost C1 0.515880;
   - Ridge C3 0.529490;
   - PLS C0 0.550157.
5. **The conservative gate was useful.** Choosing only from the complete
   development ranking would have promoted a tiny improvement that did not
   survive split-excluded selection.

The meta-model internal CV scores must not be interpreted as outer performance.
For example, ExtraTreesStack obtained a low internal meta-CV RMSE but did not
beat the incumbent on outer target-matched pseudo-competitions. This is direct
evidence that the full nested outer protocol is necessary.

## Final model

The project should now treat the following as frozen:

```text
method: Local04D
source: LocalRidge_C4_all_deterministic_Geo_k24_a30_p1
target-matched mean RMSE: 0.4878453167
target-matched pooled RMSE: 0.4957414287
prediction rows: 59
```

Checkpoint 05 reproduced both Local04D and Graph04D to floating-point precision
(max absolute difference approximately 8.9e-16) and verified the 59-row
app/model bundle round trip with max absolute difference 0.

The canonical competition output is:

```text
reports/checkpoint_05/final_predictions.csv
```

It contains the same promoted method, Local04D, but is the final post-Checkpoint
05 artifact and should be preferred over the older Checkpoint 04F path for
submission/app wiring.

## Interpretation

Checkpoint 05 closes the model-search question that motivated merging
Checkpoints 03 and 04. The global models do contain some residual complementary
signal, but the available 138 labels do not support learning a reliably better
global/local weighting rule under the target-matched protocol.

The correct conclusion is not that global models are useless. It is that their
incremental benefit is too small and unstable to justify replacing the simpler
local reconstruction rule for these fixed 59 competition targets.

## Evidence

Generated evidence is in `reports/checkpoint_05/`, especially:

- `checkpoint_05_report.md`;
- `protocol_summary.csv`;
- `loso_selection.csv`;
- `loso_summary.csv`;
- `final_predictions.csv`;
- `final_prediction_diagnostics.csv`;
- `model_manifest.json`.

The confirmation CSVs are intentionally empty because the development/LOSO gate
failed before fresh confirmation scoring.
