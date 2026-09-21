# Checkpoint 04B Findings — Transductive pseudo-competition validation

**Status:** COMPLETE  
**Run commit:** `f9ad005d51bcbe488cfad96c6a7ea8434ab6709d`

The canonical generated report is
`reports/checkpoint_04b/checkpoint_04b_report.md`. This note records the interpretation that
should guide the next modeling stage.

## Main result

The fixed-target validation supports the central transductive hypothesis: **local and graph
methods are materially more useful than the original global-only modeling path**.

Primary target-matched pseudo-competition results:

| Method | Mean RMSE | Median RMSE | Worst split RMSE |
|---|---:|---:|---:|
| LocalRidge_k20_a100 | 0.4946 | 0.4778 | 0.6927 |
| Graph_k8_lam2 | 0.4974 | 0.4878 | 0.6995 |
| LocalRidge_k30_a100 | 0.4976 | 0.4847 | 0.6950 |
| GeoKNN_10 | 0.5021 | 0.4794 | 0.7137 |
| Blend_PLS8_Graph8 | 0.5093 | 0.5012 | 0.6968 |
| GlobalPLS_C0_4 | 0.5379 | — | — |

The best local model improves mean target-matched RMSE by about 8% relative to
`GlobalPLS_C0_4` in the same 04B validation framework.

The margin between LocalRidge_k20 and Graph_k8 is small. Across the 16 target-matched splits,
LocalRidge_k20 beats Graph_k8 on 11 splits, but the mean RMSE difference is only 0.0028. This
should be treated as a near-tie rather than proof that one method dominates.

## Graph robustness is the strongest signal

Across all four validation families, `Graph_k8_lam2` is the most robust method in the tested
04B suite:

| Family | Graph_k8_lam2 RMSE |
|---|---:|
| target_matched | 0.4974 |
| state_random | 0.4655 |
| fold_state_stratified | 0.4779 |
| fold_municipality_grouped | 0.5466 |

Its worst-family RMSE is 0.5466, substantially below every other tested method.

This is especially important for the municipality-grouped stress test. The frozen First
Modeling Delivery had E123 grouped RMSE around 0.7481. The 04B graph model reaches 0.5466 under
the same broad hold-out logic, roughly a 27% reduction in RMSE. The exact pipelines differ
because 04B allows transductive X-only topology, so this is evidence of a better strategy rather
than a pure apples-to-apples estimator comparison.

## Geographic locality remains valuable

`GeoKNN_10` is the fourth-best target-matched method at RMSE 0.5021 and is competitive with
the graph/local methods. This independently confirms the 04A finding that geographic proximity
is a strong yield-transfer signal.

Adding agronomic or temporal distance to kNN did not improve the simple geographic baseline in
this first grid. The tested mixed-distance KNN variants are around 0.523–0.527 target-matched
RMSE, while GeoKNN_10 reaches 0.5021.

This does not mean agronomic or temporal information is useless. It means the current fixed
distance mixtures are not better than pure geography and should not be carried forward without
re-tuning or a learned weighting scheme.

## Target-matched split construction worked

The primary pseudo-target masks are materially closer to the actual 59-target X geometry than
the state-random masks:

| Match diagnostic | target-matched | state-random |
|---|---:|---:|
| municipality TV | 0.0547 | 0.0889 |
| profile mean distance | 0.5369 | 0.9227 |

This is about a 38% reduction in municipality-distribution discrepancy and a 42% reduction in
the standardized X-profile distance.

Therefore target-matched pseudo-competition should remain the primary model-selection protocol.
State-random and frozen state/municipality folds remain useful robustness checks.

## Support-tier behavior

The X-only support tiers for the real 59 targets are:

```text
low   19
mid   32
high   8
```

The exploratory 04B tier-wise winners are:

| X-only support tier | Best pseudo RMSE method | RMSE |
|---|---|---:|
| low | LocalRidge_k30_a100 | 0.4174 |
| mid | GeoKNN_10 | 0.5645 |
| high | LocalRidge_k30_a100 | 0.3198 |

This produces an exploratory routing proposal of 27 actual targets to
`LocalRidge_k30_a100` and 32 to `GeoKNN_10`.

### Support score is not monotone prediction difficulty

The pseudo-target results do **not** show a monotone relationship of
`low support -> larger RMSE`. In this run, the mid-support stratum is harder than both low
and high support. This means the current X-only support score is useful for describing the
target geometry and as a candidate routing variable, but it is not yet a calibrated
uncertainty score. Geography/municipality composition can make a nominally low-support parcel
locally predictable.

### Important methodological boundary

The tier routing is **not yet a validated final mixture-of-experts rule**. The same 16
target-matched splits were used both to compare candidate methods and to select the best method
within each tier. That creates model-selection optimism.

Before using routing in the final 59 reconstruction, the next local/graph stage should validate
the routing rule out of sample, for example with leave-one-split-out or nested split-level
selection. The routing CSV is useful as a hypothesis, not yet as a final decision rule.

## What should survive into the next stage

Strong survivors:

```text
Graph_k8_lam2
LocalRidge_k20_a100
LocalRidge_k30_a100
GeoKNN_10
Blend_PLS8_Graph8
GlobalPLS_C0_4 as a global anchor
```

Lower priority in their current form:

```text
fixed Geo+Agro(+Temporal) KNN mixtures
high-component PLS
strongly regularized C1 Ridge
global/state means
municipality shrinkage as a standalone model
```

Municipality information remains useful as context, but simple municipality means are fragile
when a municipality is absent from pseudo-training.

## Next modeling stage

The next stage should combine what 04A and 04B established rather than restart a generic model
search.

Recommended sequence:

1. **04D.1 local/graph refinement**
   - tune graph k and Laplacian regularization more densely;
   - tune local Ridge neighborhood size, alpha and distance weighting;
   - test graph topology based on geography alone versus geography + agronomic similarity;
   - test residual graph correction on top of a strong global model;
   - validate all hyperparameter choices with repeated target-matched pseudo-competition.

2. **04C focused global anchor**
   - run a larger CatBoost/PLS/Ridge search only as a global component;
   - do not treat 04C as the primary strategy after 04B;
   - evaluate whether global residuals improve the graph/local layer.

3. **04E external localization**
   - prioritize SIAP 2025 municipality-level barley yield evidence;
   - test it as a local prior or graph/node feature;
   - retain only external signals that improve target-matched pseudo-RMSE.

4. **Nested routing**
   - learn whether support tier or a continuous support score should control global/local mixing;
   - evaluate the routing rule on splits not used to choose the rule.

## Current working hypothesis

```text
global model = stable low-variance anchor
geographic/local model = query-specific interpolation
graph model = robust spatial propagation across the fixed 197-node universe
external municipality evidence = localized prior
support = gate controlling how much local information to trust
```

The final 59 reconstruction should be built from these components only after their mixture is
validated under target-matched pseudo-competition.
