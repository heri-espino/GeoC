# Frozen LOOCV benchmark — 2026-10-18

These numbers were recovered from completed local runs and user-supplied terminal output. This file preserves the results independently of ignored `reports/` folders. It is **not** a new model fit.

| Model | Held-out parcels | Training labels per fold | RMSE (t/ha) | MAE (t/ha) | Mean error (t/ha) | R² |
|---|---:|---:|---:|---:|---:|---:|
| Reference26 | 138 | 137 | 0.4681855146560597 | 0.3336730043055529 | 0.002374634259229654 | approximately 0.69523 (terminal truncated) |
| Compact14 | 138 | 137 | 0.4696729969475559 | 0.333044183790906 | -0.005593570410635428 | approximately 0.69329 (terminal truncated) |
| Original Local04D (GeoPCA20_GeographicWeightedRidge) | 138 | 137 | 0.485133 | not supplied | not supplied | 0.672766 |

## Provenance and comparability

- Reference26 and Compact14: completed full 138-fold leave-one-out refits using the teammate's original selected-model comparison, with 137 labels available in each training fold. Numbers copied from the user's saved terminal table.
- Local04D: **separate earlier** 138-fold original GeoC benchmark. Original frozen configuration: PCA 20 components, 24 nearest geographic neighbors, Ridge alpha 30, inverse-distance power 1. The user reported its RMSE/R² from an earlier LOOCV table. Do not claim an independently verified bitwise-identical pipeline across both runs without matching per-parcel prediction files.
- Local04D climate extensions from that earlier evaluation: unweighted climate RMSE 0.486455, R² 0.670981; area-weighted climate RMSE 0.486682, R² 0.670674; area-weighted temporal climate RMSE 0.487624, R² 0.669398.
- No individual parcel identifiers, yields or predictions are published here.
- **Next experiment:** full 138-fold Reference26 with inner-tuned geological DBSCAN gate, retaining the same original held-out parcel order. Compare against these frozen numbers; when available, verify the per-parcel labels and predictions align before claiming a strictly paired four-model comparison.
- The 46-fold exploratory gate search gave DBSCAN RMSE 0.545343 vs original Reference26 0.547536 and Compact14 0.552954 **on only the first 46 folds**. These are not directly comparable to the full 138-fold scores.
