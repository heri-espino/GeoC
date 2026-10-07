# Checkpoint 06F findings — Selective Center Expert / Abstention Gate

**Status:** CLOSED / NO PROMOTION  
**Date closed:** 2026-10-07  
**Track:** competition-only  
**Incumbent retained:** Local04D  
**Canonical 59 predictions:** `reports/checkpoint_05/final_predictions.csv`  
**Hidden FIRA target y used or scored:** NO

## Executive result

Checkpoint 06F tested the final narrow center-routing hypothesis:

```text
if p_low_raw < tau and p_high_raw < tau:
    prediction = Local04D + frozen G1-PCA Ridge center residual correction
else:
    prediction = Local04D
```

The best full-development cutoff was `tau = 0.70`. It improved the complete
development table slightly:

```text
                         Local04D     SelectiveCenter_t0p7
mean RMSE                0.487845     0.487108
pooled RMSE              0.495741     0.494704
center pooled RMSE       0.360770     0.354471
tail pooled RMSE         0.664049     0.667687
worst split RMSE         0.694700     0.676572
```

So hard abstention recovered part of the aggregate center gain and even lowered
global development RMSE. However, it still changed too many true tail cases and
the gain was not stable under leave-one-split-out selection.

## Routing behavior

For `tau=0.70`:

```text
center expert used on all pseudo-target rows      50.9%
true center rows routed to center expert          64.8%
true tail rows routed to center expert            27.5%
precision of routed set being truly center        79.9%
```

The gate therefore had useful discriminatory power, but not enough precision to
protect the tails while retaining the center gain.

The more conservative cutoffs reduced tail intrusion but also removed too much
center coverage. The broader cutoffs increased center improvement while
increasing tail damage.

## LOSO result

Across the 16 leave-one-split-out selections:

```text
Local04D                  6 / 16
SelectiveCenter_t0p6      5 / 16
SelectiveCenter_t0p7      5 / 16
```

The aggregate selected rule was worse than Local04D:

```text
                         Local04D     LOSO selected
mean RMSE                0.487845     0.490847
pooled RMSE              0.495741     0.498636
center pooled RMSE       0.360770     0.361407
tail pooled RMSE         0.664049     0.669269
```

Changes relative to Local04D:

```text
mean RMSE improvement      -0.003002
pooled RMSE improvement    -0.002894
center RMSE improvement    -0.000637
tail RMSE change           +0.005221
```

Therefore:

```text
06F_LOSO_GATE = FAIL
```

## Interpretation

06F strengthens the conclusion from 06E rather than overturning it.

There is real aggregate structure in the center residuals. When the gate is
allowed to use roughly half of the pseudo-target rows, the full development
table improves. But the observable-X gate cannot identify the safe subset with
enough stability across pseudo-competition realizations.

This means the remaining problem is not simply the lack of a hard routing rule.
The center correction is conditionally useful, but the project does not have a
reliable enough selector to know where to deploy it on unseen target parcels.

Because 06F was already a post-hoc follow-up and fails LOSO, no fresh
confirmation is warranted.

## Final Checkpoint 06 conclusion

Checkpoint 06A–06F is closed.

Do not continue tuning:

- additional center-score cutoffs;
- asymmetric low/high cutoffs;
- another hard/soft combination of the same gates;
- the same G1-PCA Ridge center residual expert;
- cluster specialists or tail offsets already rejected by 06C–06E.

Reopening this line would require genuinely new information or a materially new
validation hypothesis.

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
