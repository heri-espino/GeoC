# Checkpoint 05 — Final global-local mixture

Generated: 2026-09-23T08:29:26.547987+00:00
Git commit: d0638f589228ad93f8724a9ec89462ac478f4724

## Decision

- incumbent: **Local04D**
- best development candidate: **LocalE13_w0p10**
- promoted final method: **Local04D**
- promotion passed: **False**
- hidden FIRA y used/scored: **no**

## Target-matched ranking

| method | rmse_mean | pooled_rmse | pooled_mae | rmse_worst |
|---|---:|---:|---:|---:|
| LocalE13_w0p10 | 0.487243 | 0.495261 | 0.340353 | 0.692825 |
| LocalE123_w0p10 | 0.487274 | 0.495257 | 0.340399 | 0.692819 |
| LocalE13_w0p20 | 0.487439 | 0.495547 | 0.339678 | 0.691354 |
| LocalE123_w0p20 | 0.487497 | 0.495538 | 0.339854 | 0.691352 |
| Local04D | 0.487845 | 0.495741 | 0.342501 | 0.694700 |
| LocalE13_w0p30 | 0.488432 | 0.496600 | 0.340204 | 0.690289 |
| LocalE123_w0p30 | 0.488515 | 0.496584 | 0.340583 | 0.690301 |
| Graph04D | 0.494881 | 0.503346 | 0.340288 | 0.694149 |
| ExtraTreesStack | 0.495479 | 0.503922 | 0.332886 | 0.682405 |
| MoE_Support | 0.495574 | 0.503354 | 0.344212 | 0.687022 |
| ConvexStack | 0.496403 | 0.504554 | 0.341884 | 0.689340 |
| ElasticNetStack | 0.500242 | 0.507676 | 0.345578 | 0.696047 |
| RidgeStack | 0.500347 | 0.507934 | 0.346399 | 0.698111 |
| HuberStack | 0.500813 | 0.508945 | 0.345086 | 0.684581 |
| ExtraTrees_C1 | 0.510906 | 0.519528 | 0.362805 | 0.702862 |
| CatBoost_C1 | 0.515880 | 0.524476 | 0.370050 | 0.684850 |
| E13_PLS_CatBoost | 0.516941 | 0.524617 | 0.370301 | 0.694288 |
| E123_PLS_Ridge_CatBoost | 0.517028 | 0.524507 | 0.369411 | 0.694654 |
| Ridge_C3 | 0.529490 | 0.536121 | 0.378111 | 0.700746 |
| HistGBStack | 0.534655 | 0.541271 | 0.387673 | 0.699896 |
| PLS_C0 | 0.550157 | 0.556296 | 0.399307 | 0.726098 |

## Split-excluded promotion gate

- LOSO selector mean RMSE: **0.487934**
- LOSO selector pooled RMSE: **0.495933**
- incumbent mean RMSE: **0.487845**
- incumbent pooled RMSE: **0.495741**

The full target-matched table may identify the lowest development score,
but Checkpoint 05 first requires the leave-one-development-split-out
selector to improve both mean and pooled RMSE relative to Local04D.

## Fresh target-matched confirmation

Fresh confirmation was not executed because the development/LOSO gate did not produce a justified challenger.

A candidate replaces Local04D only if the preselected fixed method
also improves both mean and pooled RMSE on this fresh X-only split bank.

## Export

- canonical predictions: reports/checkpoint_05/final_predictions.csv
- model manifest: reports/checkpoint_05/model_manifest.json
- local model bundle: models/final/checkpoint05_model.joblib

The joblib bundle is intended for reproducibility and the later Python app.
It is ignored by Git and is not automatically committed.
