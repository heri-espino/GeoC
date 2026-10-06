# Checkpoint 06 findings — Tail and center refinement

**Status:** CLOSED / NO PROMOTION  
**Date closed:** 2026-10-06  
**Track:** competition-only  
**Incumbent retained:** Local04D  
**Canonical 59 predictions:** `reports/checkpoint_05/final_predictions.csv`  
**Hidden FIRA target y used or scored:** NO

## Executive conclusion

Checkpoint 06 verified that the team's residual observation was real, but none
of the tested corrections generalized strongly enough to replace Local04D.

The final conclusion is:

1. Local04D errors are highly concentrated in a small number of parcels.
2. Very low-yield parcels tend to be overpredicted and very high-yield parcels
   tend to be underpredicted: the model compresses the tails toward the mean.
3. Low/high yield membership is partly predictable from observable X.
4. A single global calibration is too unstable.
5. Natural X-only agronomic clusters exist, but they do not explain Local04D
   residuals.
6. A supervised center/tail mixture can improve the center on the full
   development table, but the improvement does not survive leave-one-split-out
   selection.
7. Therefore Local04D remains the competition method and no fresh confirmation
   is warranted.

## 06A/06B — tail hypothesis supported

The diagnostic stage returned:

```text
TAIL_HYPOTHESIS = SUPPORTED
```

Key evidence:

- the top 10% largest-error parcels account for about 62.2% of equal-weight
  Local04D parcel SSE;
- Local04D low-tail mean residual is negative, consistent with overprediction;
- Local04D high-tail mean residual is positive, consistent with
  underprediction;
- CatBoost shows the same compression direction;
- Local04D and CatBoost residuals are highly correlated, so they often fail on
  similar parcels.

Representative Local04D tail biases:

```text
low-tail bias    approximately -0.36 to -0.43 t/ha
high-tail bias   approximately +0.47 t/ha
```

The exact value changes slightly with the diagnostic aggregation and fold-valid
tail definition.

Tail membership is not random with respect to X. The strongest family-level
held-out classifiers included:

```text
low tail   water_productivity   mean split ROC-AUC ~ 0.889
high tail  thermal              mean split ROC-AUC ~ 0.805
```

This justified testing corrections, but tail predictability alone did not imply
that a correction would reduce RMSE.

## 06C — global calibration failed LOSO

Checkpoint 06C tested small global corrections before any clustering.

The best full-development calibrator was a strongly regularized piecewise Ridge
variant and reduced development RMSE slightly, but the gain did not survive
split-excluded selection.

```text
best development mean RMSE   ~0.4854
Local04D mean RMSE            0.487845
LOSO selected Local04D        16 / 16 holdouts
06C_LOSO_GATE                 FAIL
```

Interpretation: the shrinkage pattern is real, but one fixed global calibration
is not stable enough across pseudo-competition realizations.

## 06D — natural X-only regimes rejected as residual regimes

Checkpoint 06D selected clusters without using y.

The selected X-only solution was:

```text
representation          G1 agronomic
PCA retained variance   0.80
algorithm               KMeans
K                       4
silhouette              0.1966
seed ARI mean           0.8809
subsample ARI mean      0.8204
```

So the agronomic clusters themselves were legitimate and reasonably stable.

However, after freezing the clusters and only then examining honest residuals:

```text
Local04D residual eta^2       ~0.0040
permutation p                 ~0.935
Local04D MSE eta^2            ~0.0119
MSE permutation p             ~0.762
residual mean range           ~0.088 t/ha
```

Therefore:

```text
REGIME_HYPOTHESIS = NOT_SUPPORTED
```

The clusters have different agronomic/yield profiles, but they are not useful
routing groups for correcting Local04D. Independent models by cluster are
therefore explicitly rejected.

## 06E — center/tail supervised mixture failed LOSO

06E tested the remaining supported mechanism:

```text
Local04D
+ w_center(X) * center residual expert
+ w_low(X)    * low-tail correction
+ w_high(X)   * high-tail correction
```

The low and high gates used the family-level signals identified in 06B:

- low gate: water_productivity;
- high gate: thermal.

The center expert learned honest cross-fitted Local04D residuals from only
fold-valid center parcels.

### Important positive diagnostic

The center **can be improved on the aggregate development table**.

Local04D:

```text
center pooled RMSE   0.360770
overall mean RMSE    0.487845
overall pooled RMSE  0.495741
tail pooled RMSE     0.664049
```

A G1-PCA center residual expert reached center RMSE around 0.3506–0.3514,
roughly 0.009–0.010 t/ha lower than Local04D.

For example:

```text
Center_G1PCA_a1000
center pooled RMSE   0.351396
overall mean RMSE    0.487933
overall pooled RMSE  0.495477
tail pooled RMSE     0.671959
```

This is scientifically useful: ordinary/non-tail residuals contain some
learnable structure in the aggregate.

However, the center improvement comes with slightly worse tail behavior and is
not stable enough under split-excluded model selection.

### LOSO result

```text
selected methods:
  Local04D                 13 / 16
  Center_Cross_a1000        3 / 16

LOSO selected mean RMSE     0.488632
Local04D mean RMSE          0.487845

LOSO selected pooled RMSE   0.496568
Local04D pooled RMSE        0.495741

LOSO selected center RMSE   0.360888
Local04D center RMSE        0.360770

LOSO selected tail RMSE     0.665598
Local04D tail RMSE          0.664049
```

Thus every aggregate LOSO metric is slightly worse.

Final decision:

```text
06E_LOSO_GATE = FAIL
```

## What this means scientifically

The remaining Local04D error is not simply a problem that can be solved by:

- expanding all predictions away from the mean;
- splitting parcels into natural agronomic clusters;
- adding one global tail correction;
- training a small center residual model and routing by low/high probabilities.

The center signal seen on the complete development table appears real but too
weak or unstable for reliable target-matched selection with only 138 labels.

This is a useful negative result. It suggests that the current performance
limit is driven more by limited/stochastic information than by an obvious
unmodeled regime that can be recovered from the existing deterministic feature
set.

## Final model status

No Checkpoint 06 candidate is promoted.

The frozen competition method remains:

```text
Local04D = LocalRidge_C4_all_deterministic_Geo_k24_a30_p1
```

Primary validation reference:

```text
target-matched mean RMSE  0.487845
pooled RMSE               0.495741
center pooled RMSE        0.360770
tail pooled RMSE          0.664049
```

The canonical 59-row prediction file remains:

```text
reports/checkpoint_05/final_predictions.csv
```

No fresh confirmation bank is run because 06E did not pass LOSO.

## Closed hypotheses

Do not reopen these without genuinely new information:

- generic Local04D/CatBoost stacking;
- global tail-stretch calibration;
- piecewise calibration on the same development splits;
- KMeans/GMM cluster specialists;
- low/high soft gating with the existing 06B feature families;
- Ridge center residual corrections using the current G1/cross-domain
  representations.

A future modeling attempt should introduce a new source of information,
measurement, or validation hypothesis rather than another small variation on
these same mechanisms.
