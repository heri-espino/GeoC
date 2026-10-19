# GeoC · Historical LOOCV results (recorded 2026-10-18)

These figures were supplied from previously completed local runs. They are
preserved **as historical reported results**, not reconstructed or
independently verified from private per-parcel predictions in GitHub.

## Original teammate models — complete 138-fold LOOCV

Every held-out parcel was predicted from the other 137 labeled parcels.

| Model | Holdouts | RMSE (t/ha) | MAE (t/ha) | Mean error (t/ha) | R² |
|---|---:|---:|---:|---:|---:|
| Reference26 | 138 | 0.4681855146560597 | 0.3336730043055529 | 0.002374634259229654 | approximately 0.69523 (terminal display truncated) |
| Compact14 | 138 | 0.4696729969475559 | 0.333044183790906 | -0.005593570410635428 | approximately 0.69329 (terminal display truncated) |

Source on the user's computer:
`reports/selected_champions_loo/loo_summary.csv`. These two models were
evaluated in the same experiment on the same parcel IDs. Their full
`loo_predictions.csv` is private and must not be committed.

## Earlier Local04D and climate augmentation LOOCV

A **separate** 138-fold frozen model comparison was reported previously,
using the original Local04D with k=24, alpha=30, p=1 and 20 PCA components.

| Model | RMSE (t/ha) | R² |
|---|---:|---:|
| Original Local04D (GeoPCA20_GeographicWeightedRidge) | 0.485133 | 0.672766 |
| Local04D + unweighted climate | 0.486455 | 0.670981 |
| Local04D + area-weighted climate | 0.486682 | 0.670674 |
| Local04D + area-weighted temporal climate | 0.487624 | 0.669398 |

Source on the user's computer:
`reports/area_weighted_climate_loo/loo_summary.csv` and the
`loo_predictions.csv` containing four model rows per labeled parcel.
The old Local04D performance must be verified against the **same exact 138
parcel IDs/yields** before calculating paired differences with the teammate's
LOOCV. Do not assume matching experiments merely from equal sample sizes.

## Original fixed first-46 geological gate comparisons

| Model | Holdouts | RMSE (t/ha) | MAE (t/ha) | R² |
|---|---:|---:|---:|---:|
| Reference26 DBSCAN geological gate | 46 | 0.545343 | 0.412532 | 0.626351 |
| Reference26 global weight | 46 | 0.546217 | 0.410352 | 0.625152 |
| Reference26 original KMeans | 46 | 0.547536 | 0.408309 | 0.623339 |
| Reference26 expanded KMeans | 46 | 0.548442 | 0.409405 | 0.622092 |
| Reference26 geological support | 46 | 0.548607 | 0.411310 | 0.621865 |
| Original Compact14 | 46 | 0.552954 | 0.416481 | 0.615848 |

Source: `reports/peer_gate_grid_loo46/loo46_scores.csv`. The first 46 are
the first 46 training *bundle rows*, and are not a random or independent
held-out third. Nested gate selection uses the 137 outer-training parcels.

## Pending

- Complete 138-fold LOOCV for geological DBSCAN (not yet reported).
- Match each model by parcel ID and observed yield, compare Reference26,
  Compact14, original Local04D, and DBSCAN on all 138 parcels.
- Run `tools/publish_peer_results.py --export` to copy **exact aggregate
  CSV metrics** from private workstation outputs to tracked
  `results/peer_models/`. The truncated R² shown above will then be
  replaced in the machine-readable records with the true stored value.
- Never push `data_n/`, per-fold CSVs, individual yields, or model caches.

The DBSCAN 46-fold RMSE 0.545343 is not directly comparable with the
138-fold Reference26 RMSE 0.468186: the held-out sets differ.
