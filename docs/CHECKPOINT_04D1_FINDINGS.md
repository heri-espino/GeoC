# Checkpoint 04D.1 Findings — Local/graph refinement

**Status:** COMPLETE  
**Run commit:** `a71a8f3a3bea2215f950759662a6d74841a24120`

Canonical generated evidence lives under `reports/checkpoint_04d1/`.

## Main result

The exhaustive 626-method target-matched search confirmed that the useful local
region is broad rather than a single lucky hyperparameter setting.

Best primary model:

```text
LocalRidge_C4_all_deterministic_Geo_k24_a30_p1
mean RMSE   0.48785
median RMSE 0.46874
worst RMSE  0.69470
pooled R²   0.65434
```

Nearby C1/C4 geographic Local Ridge variants with k around 20–30 and alpha 30
remain within a few thousandths of RMSE, which supports parameter stability.

## Nested validation

Leave-one-target-matched-split-out selection gives:

```text
LOSO method selection       mean RMSE 0.48837
LOSO support-tier routing   mean RMSE 0.48581
```

This removes direct same-split model-selection optimism. Repeated pseudo-splits
still reuse parcels, so the 16 holdouts are not statistically independent.

## Graph robustness

The strongest robust method is:

```text
GraphDirect_GeoAgro25_k6_lam8
```

with RMSE:

```text
target_matched             0.49488
state_random               0.46794
fold_state_stratified      0.47916
fold_municipality_grouped  0.52788
worst-family               0.52788
```

This materially improves municipality-grouped robustness relative to the earlier
04B graph and strongly outperforms the local models under that stress test.

The useful graph topology is `GeoAgro25`: 25% normalized geographic distance
and 75% normalized agronomic distance. Mixed distance did not help simple kNN in
04B, but it does help graph topology.

## Negative result

PLS4 residual graph correction did not improve the direct graph family. The
best residual-PLS graph variants remain around target-matched RMSE 0.537, so the
global PLS anchor should not be forced into the spatial model.

## Actual-target routing remains provisional

The all-primary-split route maps 51 actual targets to the C4 geographic Local
Ridge and 8 high-support targets to the C1 geographic Local Ridge. These are
candidate predictions only and must not be treated as the final 59-value
submission.

## Next action

Checkpoint 04E.1 should test whether a correctly scoped public SIAP 2025
municipal prior adds incremental information to the validated Local Ridge and
direct graph survivors. The primary SIAP scope is:

```text
Cebada grano
Primavera-Verano
Temporal
exact municipality / CVEGEO
2025
```
