# Checkpoint 03C.1 results

Workstation run completed successfully on 2026-09-18.

```text
training parcels       138
tracks                   2
representations/track    7
model families           2
protocols                2
outer fits             280
OOF prediction rows    7728
discovery rows          800
```

No hidden prediction target was used and no final prediction for the 59 challenge parcels was generated.

## Main representation findings

### ExtraTrees

The most useful representation effects appear under ExtraTrees.

State-stratified OOF RMSE:

```text
clean
  B7 all + fold-local discovery     0.4948
  B3 base + agronomic               0.5021
  B5 all deterministic              0.5053
  B0 base                            0.5119
  B1 agronomic only                  0.5143
  B2 empirical only                  0.6567

competition
  B7 all + fold-local discovery     0.4992
  B0 base                            0.5003
  B1 agronomic only                  0.5041
  B5 all deterministic              0.5045
  B2 empirical only                  0.6470
```

Municipality-grouped OOF RMSE:

```text
clean
  B7 all + fold-local discovery     0.8488
  B1 agronomic only                  0.8762
  B3 base + agronomic               0.9106
  B5 all deterministic              0.9267
  B0 base                            0.9431
  B2 empirical only                  1.0265

competition
  B1 agronomic only                  0.8590
  B7 all + fold-local discovery     0.8675
  B0 base                            0.8974
  B3 base + agronomic               0.8982
  B5 all deterministic              0.9216
  B2 empirical only                  0.9962
```

For clean ExtraTrees, B7 improves OOF RMSE relative to B0 by 0.0171 under state-stratified CV
and by 0.0942 under municipality-grouped CV. The state-fold improvement occurs in all 5/5
folds. Under municipality grouping B7 improves 3/5 folds, including the 73-parcel fold, but
degrades the 16- and 10-parcel folds.

For competition ExtraTrees, B7 is essentially tied with B0 under state-stratified CV
(0.4992 vs 0.5003). Under municipality grouping, B1 agronomic-only is strongest among the
tested ExtraTrees representations (0.8590), while B7 is second (0.8675).

### Ridge10

Ridge is much more sensitive to representation size and redundancy.

Notable OOF RMSE:

```text
state-stratified clean
  B1 agronomic only                  0.5888
  B0 base                            0.6718

municipality-grouped competition
  B0 base                            0.8059
  B4 base + empirical                0.8464
  B5 all deterministic              0.8976
  B7 all + discovery                 0.9142
  B1 agronomic only                  1.2403
```

This instability is consistent with a fixed-alpha linear model being sensitive to high
dimensionality, collinearity and representation scaling. It is a reason to tune regularization
inside folds in 03C.2, not evidence that the agronomic variables are intrinsically poor.

## What 03C.1 says about the feature layers

1. **Agronomic Features v1 carry substantial signal.** They preserve or improve ExtraTrees
   performance with far fewer variables than the source table, especially under municipality
   grouping.
2. **Empirical Features v1 should not be used alone.** B2 is clearly weaker across both models
   and protocols. Their main value is as diagnostics / ingredients for controlled discovery,
   not as a standalone representation.
3. **Blindly concatenating every deterministic feature is not consistently beneficial.** B5
   often underperforms B0 or B1, especially under grouped validation.
4. **Fold-local expression discovery is promising for tree models.** Clean B7 is the strongest
   ExtraTrees representation in both protocols. The gain is not universal across models, so
   the discovered formulas should remain an optional fold-fitted augmentation rather than a
   globally promoted index.
5. **Competition-mode covariates improve the base representation for Ridge and generally help
   the base ExtraTrees model under grouped validation, but they do not automatically improve
   every expanded representation.

## 03C.2 candidate representations

Carry a deliberately small set forward:

```text
C0 = B0 base                         control
C1 = B1 agronomic only               compact process-informed representation
C2 = B7 all + fold-local discovery   current strongest nonlinear augmentation
C3 = base + agronomic + discovery    new ablation for 03C.2; omit deterministic empirical-only block
```

`C3` is important because 03C.1 suggests the deterministic empirical block is often redundant
or noisy, while fold-local discovered expressions can help ExtraTrees. C3 isolates that effect
more cleanly than B7.

Run these separately for clean and competition tracks. Do not carry B2 empirical-only into broad
model tuning unless a later targeted hypothesis specifically requires it.

## Validation caveat

03C.1 used the frozen development folds to compare representations. Therefore subsequent
performance estimates on the same folds are development evidence, not a fresh independent
estimate after representation selection. Hyperparameter selection in 03C.2 must still be
nested inside each outer training fold, and claims should emphasize robustness across both
protocols rather than a single minimum CV score.

## Next

Checkpoint 03C.2 should compare controlled model families on C0-C3:

- tuned Ridge / ElasticNet;
- CatBoost;
- tuned ExtraTrees;
- HistGradientBoosting or one justified gradient-boosting implementation;
- polynomial and RBF kernel models on compact / fold-reduced inputs;
- PCA + Ridge and PLS as explicit dimension-reduction comparisons.

Do not generate final predictions yet.
