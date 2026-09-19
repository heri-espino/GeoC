# Checkpoint 03C.2 — Competition-only nested model benchmark

Generated: 2026-09-19T07:39:04.238077+00:00
Git commit: e441e882683d6f376dbe95283ee09e8e02839012

## Scope

- active feature track: **competition only**
- clean track evaluated: **no**
- training parcels: **138**
- prediction parcels scored: **0**
- outer protocols: **fold_state_stratified, fold_municipality_grouped**
- inner folds: **3**
- representations: **C0_base, C1_agronomic, C2_all_plus_discovery, C3_base_agro_plus_discovery**
- model families: **Ridge, ElasticNet, ExtraTrees, CatBoost, PLS, PCA_RBF, PCA_Poly2**
- CatBoost execution: **GPU**

Hyperparameters are selected only from each outer training split using inner CV. The frozen Checkpoint 02 folds remain the outer evaluation.

CHIRPS remains excluded pending QC.

## Representation sizes

| representation | n_features | n_primitives | discovery |
|---|---|---|---|
| C0_base | 1400 | 0 | False |
| C1_agronomic | 350 | 0 | False |
| C2_all_plus_discovery | 2085 | 24 | True |
| C3_base_agro_plus_discovery | 1754 | 24 | True |

## Cross-protocol robustness ranking

The table is sorted by worst-protocol OOF RMSE, then mean OOF RMSE. This is development evidence, not a fresh independent test after 03C.1.

| representation | model | state_oof_rmse | grouped_oof_rmse | mean_oof_rmse | worst_protocol_rmse | protocol_gap |
|---|---|---|---|---|---|---|
| C0_base | PLS | 0.5339 | 0.7621 | 0.6480 | 0.7621 | 0.2282 |
| C3_base_agro_plus_discovery | Ridge | 0.5331 | 0.7650 | 0.6490 | 0.7650 | 0.2319 |
| C1_agronomic | CatBoost | 0.5235 | 0.7709 | 0.6472 | 0.7709 | 0.2474 |
| C0_base | Ridge | 0.5281 | 0.7872 | 0.6576 | 0.7872 | 0.2591 |
| C3_base_agro_plus_discovery | PLS | 0.5260 | 0.7884 | 0.6572 | 0.7884 | 0.2624 |
| C2_all_plus_discovery | PLS | 0.5352 | 0.7998 | 0.6675 | 0.7998 | 0.2646 |
| C1_agronomic | PLS | 0.5369 | 0.8345 | 0.6857 | 0.8345 | 0.2975 |
| C1_agronomic | ExtraTrees | 0.5024 | 0.8469 | 0.6746 | 0.8469 | 0.3446 |
| C3_base_agro_plus_discovery | ExtraTrees | 0.4972 | 0.8698 | 0.6835 | 0.8698 | 0.3726 |
| C2_all_plus_discovery | CatBoost | 0.5313 | 0.8706 | 0.7010 | 0.8706 | 0.3393 |
| C0_base | ExtraTrees | 0.5057 | 0.8771 | 0.6914 | 0.8771 | 0.3713 |
| C3_base_agro_plus_discovery | CatBoost | 0.5298 | 0.8883 | 0.7091 | 0.8883 | 0.3585 |
| C2_all_plus_discovery | Ridge | 0.5431 | 0.8956 | 0.7193 | 0.8956 | 0.3525 |
| C0_base | CatBoost | 0.5326 | 0.9044 | 0.7185 | 0.9044 | 0.3718 |
| C2_all_plus_discovery | ExtraTrees | 0.4996 | 0.9052 | 0.7024 | 0.9052 | 0.4056 |
| C1_agronomic | Ridge | 0.5618 | 1.0053 | 0.7836 | 1.0053 | 0.4435 |
| C1_agronomic | ElasticNet | 0.5250 | 1.0351 | 0.7801 | 1.0351 | 0.5101 |
| C0_base | ElasticNet | 0.5198 | 1.0569 | 0.7884 | 1.0569 | 0.5371 |
| C2_all_plus_discovery | ElasticNet | 0.5246 | 1.0646 | 0.7946 | 1.0646 | 0.5400 |
| C3_base_agro_plus_discovery | ElasticNet | 0.5218 | 1.0655 | 0.7937 | 1.0655 | 0.5437 |
| C1_agronomic | PCA_Poly2 | 0.7248 | 1.6296 | 1.1772 | 1.6296 | 0.9049 |
| C0_base | PCA_Poly2 | 0.7704 | 1.9253 | 1.3479 | 1.9253 | 1.1548 |
| C2_all_plus_discovery | PCA_Poly2 | 0.7568 | 2.1613 | 1.4591 | 2.1613 | 1.4046 |
| C3_base_agro_plus_discovery | PCA_Poly2 | 0.7460 | 2.2170 | 1.4815 | 2.2170 | 1.4710 |
| C1_agronomic | PCA_RBF | 3.2431 | 3.9712 | 3.6071 | 3.9712 | 0.7281 |
| C0_base | PCA_RBF | 4.0216 | 4.0452 | 4.0334 | 4.0452 | 0.0236 |
| C3_base_agro_plus_discovery | PCA_RBF | 4.0287 | 4.0452 | 4.0370 | 4.0452 | 0.0166 |
| C2_all_plus_discovery | PCA_RBF | 4.0349 | 4.0453 | 4.0401 | 4.0453 | 0.0104 |

## fold_state_stratified

| representation | model | n_features | n_discovered | oof_rmse | oof_mae | oof_r2 | fold_rmse_mean | fold_rmse_std | inner_best_rmse_mean | elapsed_seconds |
|---|---|---|---|---|---|---|---|---|---|---|
| C3_base_agro_plus_discovery | ExtraTrees | 1754 | 20 | 0.4972 | 0.3575 | 0.6563 | 0.4922 | 0.0801 | 0.4977 | 30.5084 |
| C2_all_plus_discovery | ExtraTrees | 2085 | 20 | 0.4996 | 0.3595 | 0.6530 | 0.4946 | 0.0806 | 0.4998 | 31.5792 |
| C1_agronomic | ExtraTrees | 350 | 0 | 0.5024 | 0.3597 | 0.6491 | 0.4989 | 0.0714 | 0.5123 | 17.6413 |
| C0_base | ExtraTrees | 1400 | 0 | 0.5057 | 0.3646 | 0.6444 | 0.5017 | 0.0745 | 0.5063 | 19.7075 |
| C0_base | ElasticNet | 1400 | 0 | 0.5198 | 0.3784 | 0.6243 | 0.5164 | 0.0723 | 0.5243 | 9.8933 |
| C3_base_agro_plus_discovery | ElasticNet | 1754 | 20 | 0.5218 | 0.3776 | 0.6214 | 0.5164 | 0.0879 | 0.5279 | 22.8843 |
| C1_agronomic | CatBoost | 350 | 0 | 0.5235 | 0.3795 | 0.6190 | 0.5202 | 0.0689 | 0.5170 | 131.3613 |
| C2_all_plus_discovery | ElasticNet | 2085 | 20 | 0.5246 | 0.3794 | 0.6174 | 0.5194 | 0.0873 | 0.5357 | 24.4015 |
| C1_agronomic | ElasticNet | 350 | 0 | 0.5250 | 0.3903 | 0.6167 | 0.5211 | 0.0760 | 0.5311 | 9.6860 |
| C3_base_agro_plus_discovery | PLS | 1754 | 20 | 0.5260 | 0.3833 | 0.6154 | 0.5208 | 0.0871 | 0.5358 | 11.5013 |
| C0_base | Ridge | 1400 | 0 | 0.5281 | 0.3980 | 0.6122 | 0.5270 | 0.0444 | 0.5237 | 2.1190 |
| C3_base_agro_plus_discovery | CatBoost | 1754 | 20 | 0.5298 | 0.3937 | 0.6097 | 0.5273 | 0.0597 | 0.5200 | 239.9317 |
| C2_all_plus_discovery | CatBoost | 2085 | 20 | 0.5313 | 0.3847 | 0.6075 | 0.5271 | 0.0775 | 0.5176 | 257.7667 |
| C0_base | CatBoost | 1400 | 0 | 0.5326 | 0.3851 | 0.6057 | 0.5302 | 0.0592 | 0.5189 | 183.9501 |
| C3_base_agro_plus_discovery | Ridge | 1754 | 20 | 0.5331 | 0.3989 | 0.6049 | 0.5318 | 0.0480 | 0.5225 | 12.2497 |
| C0_base | PLS | 1400 | 0 | 0.5339 | 0.4086 | 0.6037 | 0.5303 | 0.0740 | 0.5385 | 2.2760 |
| C2_all_plus_discovery | PLS | 2085 | 20 | 0.5352 | 0.3937 | 0.6018 | 0.5309 | 0.0815 | 0.5452 | 11.9251 |
| C1_agronomic | PLS | 350 | 0 | 0.5369 | 0.3986 | 0.5991 | 0.5314 | 0.0913 | 0.5482 | 0.8673 |
| C2_all_plus_discovery | Ridge | 2085 | 20 | 0.5431 | 0.4064 | 0.5899 | 0.5418 | 0.0493 | 0.5318 | 12.6393 |
| C1_agronomic | Ridge | 350 | 0 | 0.5618 | 0.4207 | 0.5612 | 0.5587 | 0.0696 | 0.5576 | 0.8319 |
| C1_agronomic | PCA_Poly2 | 350 | 0 | 0.7248 | 0.5300 | 0.2697 | 0.7141 | 0.1320 | 1.0726 | 2.2019 |
| C3_base_agro_plus_discovery | PCA_Poly2 | 1754 | 20 | 0.7460 | 0.5506 | 0.2262 | 0.7385 | 0.1122 | 0.9545 | 14.6895 |
| C2_all_plus_discovery | PCA_Poly2 | 2085 | 20 | 0.7568 | 0.5756 | 0.2037 | 0.7513 | 0.1006 | 0.9626 | 15.5046 |
| C0_base | PCA_Poly2 | 1400 | 0 | 0.7704 | 0.5747 | 0.1747 | 0.7614 | 0.1190 | 0.8879 | 3.9234 |
| C1_agronomic | PCA_RBF | 350 | 0 | 3.2431 | 3.0723 | -13.6237 | 3.2400 | 0.1752 | 3.4351 | 2.3796 |
| C0_base | PCA_RBF | 1400 | 0 | 4.0216 | 3.9317 | -21.4867 | 4.0206 | 0.0923 | 4.0310 | 3.9699 |
| C3_base_agro_plus_discovery | PCA_RBF | 1754 | 20 | 4.0287 | 3.9392 | -21.5661 | 4.0277 | 0.0927 | 4.0345 | 14.5683 |
| C2_all_plus_discovery | PCA_RBF | 2085 | 20 | 4.0349 | 3.9455 | -21.6356 | 4.0338 | 0.0936 | 4.0378 | 15.8087 |

## fold_municipality_grouped

| representation | model | n_features | n_discovered | oof_rmse | oof_mae | oof_r2 | fold_rmse_mean | fold_rmse_std | inner_best_rmse_mean | elapsed_seconds |
|---|---|---|---|---|---|---|---|---|---|---|
| C0_base | PLS | 1400 | 0 | 0.7621 | 0.6416 | 0.1925 | 0.6462 | 0.2181 | 0.8018 | 2.4722 |
| C3_base_agro_plus_discovery | Ridge | 1754 | 20 | 0.7650 | 0.6390 | 0.1864 | 0.6962 | 0.2328 | 0.8229 | 12.1167 |
| C1_agronomic | CatBoost | 350 | 0 | 0.7709 | 0.6340 | 0.1737 | 0.7728 | 0.3099 | 0.8472 | 131.7704 |
| C0_base | Ridge | 1400 | 0 | 0.7872 | 0.6607 | 0.1385 | 0.7042 | 0.3123 | 0.8260 | 2.1867 |
| C3_base_agro_plus_discovery | PLS | 1754 | 20 | 0.7884 | 0.6656 | 0.1358 | 0.6899 | 0.2074 | 0.7873 | 11.0741 |
| C2_all_plus_discovery | PLS | 2085 | 20 | 0.7998 | 0.6746 | 0.1106 | 0.6818 | 0.2237 | 0.8130 | 11.7392 |
| C1_agronomic | PLS | 350 | 0 | 0.8345 | 0.6892 | 0.0318 | 0.7582 | 0.2469 | 0.7906 | 0.7975 |
| C1_agronomic | ExtraTrees | 350 | 0 | 0.8469 | 0.7115 | 0.0027 | 0.8675 | 0.3773 | 0.8402 | 17.9574 |
| C3_base_agro_plus_discovery | ExtraTrees | 1754 | 20 | 0.8698 | 0.7388 | -0.0519 | 0.8770 | 0.3821 | 0.7896 | 30.8618 |
| C2_all_plus_discovery | CatBoost | 2085 | 20 | 0.8706 | 0.7543 | -0.0538 | 0.8123 | 0.2332 | 0.8255 | 256.7665 |
| C0_base | ExtraTrees | 1400 | 0 | 0.8771 | 0.7704 | -0.0696 | 0.9128 | 0.2316 | 0.8209 | 20.3299 |
| C3_base_agro_plus_discovery | CatBoost | 1754 | 20 | 0.8883 | 0.7628 | -0.0972 | 0.8044 | 0.2715 | 0.8163 | 229.7175 |
| C2_all_plus_discovery | Ridge | 2085 | 20 | 0.8956 | 0.7718 | -0.1152 | 0.7595 | 0.2787 | 0.8450 | 12.4858 |
| C0_base | CatBoost | 1400 | 0 | 0.9044 | 0.7736 | -0.1372 | 0.8433 | 0.2577 | 0.8277 | 201.4311 |
| C2_all_plus_discovery | ExtraTrees | 2085 | 20 | 0.9052 | 0.7750 | -0.1393 | 0.8978 | 0.3991 | 0.7873 | 31.9875 |
| C1_agronomic | Ridge | 350 | 0 | 1.0053 | 0.8598 | -0.4052 | 0.8345 | 0.4208 | 0.8319 | 0.8222 |
| C1_agronomic | ElasticNet | 350 | 0 | 1.0351 | 0.8941 | -0.4897 | 0.8061 | 0.3978 | 0.8283 | 10.0234 |
| C0_base | ElasticNet | 1400 | 0 | 1.0569 | 0.9255 | -0.5532 | 0.9296 | 0.3456 | 0.7920 | 12.3831 |
| C2_all_plus_discovery | ElasticNet | 2085 | 20 | 1.0646 | 0.9377 | -0.5757 | 0.9401 | 0.3256 | 0.8035 | 25.2264 |
| C3_base_agro_plus_discovery | ElasticNet | 1754 | 20 | 1.0655 | 0.9371 | -0.5786 | 0.9332 | 0.3318 | 0.7594 | 25.7836 |
| C1_agronomic | PCA_Poly2 | 350 | 0 | 1.6296 | 1.3356 | -2.6925 | 1.4289 | 0.4021 | 1.7441 | 2.1954 |
| C0_base | PCA_Poly2 | 1400 | 0 | 1.9253 | 1.5254 | -4.1538 | 1.1967 | 0.7781 | 1.7056 | 4.1964 |
| C2_all_plus_discovery | PCA_Poly2 | 2085 | 20 | 2.1613 | 1.6950 | -5.4951 | 1.6304 | 1.1426 | 1.8981 | 15.4506 |
| C3_base_agro_plus_discovery | PCA_Poly2 | 1754 | 20 | 2.2170 | 1.6680 | -5.8339 | 1.7967 | 1.5852 | 1.8691 | 15.0848 |
| C1_agronomic | PCA_RBF | 350 | 0 | 3.9712 | 3.8792 | -20.9268 | 3.7931 | 0.6807 | 3.7327 | 2.2433 |
| C0_base | PCA_RBF | 1400 | 0 | 4.0452 | 3.9553 | -21.7516 | 3.9041 | 0.7481 | 3.7797 | 4.1821 |
| C3_base_agro_plus_discovery | PCA_RBF | 1754 | 20 | 4.0452 | 3.9554 | -21.7523 | 3.9042 | 0.7482 | 3.7798 | 14.6396 |
| C2_all_plus_discovery | PCA_RBF | 2085 | 20 | 4.0453 | 3.9554 | -21.7524 | 3.9042 | 0.7482 | 3.7798 | 15.1017 |

## Interpretation rules

- Do not compare or revive the clean track in 03C.2.
- Do not use the 59 hidden targets for selection.
- B/C discovery remains fitted inside inner/outer training folds only.
- PCA and PLS are fitted inside the nested pipeline.
- Prefer models that remain competitive under municipality-grouped CV.
- Do not treat the minimum development score as an unbiased final error estimate.
- Final challenge prediction is a later frozen-model step.
