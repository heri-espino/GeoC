# Checkpoint 04C.1 — Focused global CatBoost anchor

Generated: 2026-09-21T19:58:38.624345+00:00
Git commit: b8d878df267f0782ece043ddc8263bd50a6259d9

## Scope

- active track: **competition only**
- hidden 59 yields used: **no**
- agronomic features: **350**
- pseudo-split families: **fold_municipality_grouped, fold_state_stratified, state_random, target_matched**
- CatBoost task type: **GPU**
- CatBoost seeds per candidate: **3**
- CatBoost candidates: **3**

This checkpoint is intentionally narrow. The three CatBoost hyperparameter candidates come from the frozen 03C.2 grid; it is not a new broad search.

## Primary target-matched ranking

| method | rmse_mean | rmse_median | rmse_worst | pooled_rmse | pooled_mae |
|---|---:|---:|---:|---:|---:|
| Blend_Local_CatBoost_w0p10 | 0.4878 | 0.4711 | 0.6932 | 0.4959 | 0.3423 |
| Baseline_Local04D | 0.4878 | 0.4687 | 0.6947 | 0.4957 | 0.3425 |
| Blend_Local_CatBoost_w0p20 | 0.4885 | 0.4741 | 0.6919 | 0.4967 | 0.3430 |
| Blend_Local_CatBoost_w0p30 | 0.4898 | 0.4777 | 0.6908 | 0.4982 | 0.3444 |
| Blend_Graph_CatBoost_w0p30 | 0.4914 | 0.4864 | 0.6894 | 0.5004 | 0.3398 |
| Blend_Graph_CatBoost_w0p20 | 0.4916 | 0.4837 | 0.6907 | 0.5005 | 0.3382 |
| Blend_Graph_CatBoost_w0p10 | 0.4928 | 0.4821 | 0.6923 | 0.5014 | 0.3381 |
| Baseline_Graph04D | 0.4949 | 0.4817 | 0.6941 | 0.5033 | 0.3403 |
| CatBoost_C1_d5_lr006_l2_3 | 0.5153 | 0.5232 | 0.6848 | 0.5239 | 0.3709 |
| CatBoost_C1_d4_lr003_l2_3 | 0.5168 | 0.5203 | 0.6877 | 0.5254 | 0.3712 |
| CatBoost_C1_d4_lr006_l2_6 | 0.5206 | 0.5256 | 0.6849 | 0.5288 | 0.3730 |

## Controlled LOSO anchor gate

The gate can choose only the frozen Local04D baseline, the stable CatBoost anchor, or the three low-weight Local+CatBoost blends.

- Baseline_Local04D: **8/16** holdouts
- Blend_Local_CatBoost_w0p10: **8/16** holdouts

## Residual diversity on target-matched pseudo-targets

| method A | method B | n | residual correlation |
|---|---|---:|---:|
| Baseline_Local04D | Baseline_Graph04D | 656 | 0.9707 |
| Baseline_Local04D | CatBoost_C1_d4_lr003_l2_3 | 656 | 0.9423 |
| Baseline_Graph04D | CatBoost_C1_d4_lr003_l2_3 | 656 | 0.9143 |

## Decision rule for 04F

CatBoost should enter 04F only if the frozen target-matched evidence shows either a reproducible low-weight blend improvement or useful residual diversity without a material robustness penalty. A weaker standalone CatBoost score is not sufficient by itself.

Actual 59 predictions in actual_anchor_candidates.csv are candidates only. No hidden target is loaded or scored.
