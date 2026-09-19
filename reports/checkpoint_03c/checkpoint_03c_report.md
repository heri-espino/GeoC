# Checkpoint 03C.1 — Representation benchmark

Generated: 2026-09-19T01:45:22.834261+00:00
Git commit: 82be98e82150536bc70bb712c6851876b5d21128

## Contract

- training parcels: **138**
- prediction parcels scored: **0**
- tracks: **clean, competition**
- representations per track: **7**
- fixed model families: **Ridge10, ExtraTrees**
- frozen protocols: **fold_state_stratified, fold_municipality_grouped**

This stage isolates representation effects. Ridge10 and ExtraTrees use the same fixed settings across representations; there is no broad hyperparameter tuning.

CHIRPS is explicitly excluded pending QC.

## Representation sizes

| track | representation | n_features | discovery | n_primitives |
|---|---|---|---|---|
| clean | B0_base | 692 | False | 0 |
| clean | B1_agronomic | 123 | False | 0 |
| clean | B2_empirical | 91 | False | 0 |
| clean | B3_base_plus_agronomic | 815 | False | 0 |
| clean | B4_base_plus_empirical | 783 | False | 0 |
| clean | B5_all | 906 | False | 0 |
| clean | B7_all_plus_discovery | 906 | True | 7 |
| competition | B0_base | 1400 | False | 0 |
| competition | B1_agronomic | 350 | False | 0 |
| competition | B2_empirical | 335 | False | 0 |
| competition | B3_base_plus_agronomic | 1750 | False | 0 |
| competition | B4_base_plus_empirical | 1735 | False | 0 |
| competition | B5_all | 2085 | False | 0 |
| competition | B7_all_plus_discovery | 2085 | True | 24 |

## OOF results

OOF metrics pool all 138 held-out predictions and therefore weight parcels rather than folds. Mean-fold metrics are also retained because the grouped folds are intentionally unequal in size.

A negative delta_oof_rmse_vs_B0 means the representation improved over Feature Table v1 for the same track/model/protocol.

### fold_state_stratified — clean

| model | representation | n_features | n_discovered | oof_rmse | oof_mae | oof_r2 | fold_rmse_mean | fold_rmse_std | delta_oof_rmse_vs_B0 | delta_oof_rmse_vs_B5 |
|---|---|---|---|---|---|---|---|---|---|---|
| ExtraTrees | B7_all_plus_discovery | 906 | 20 | 0.4948 | 0.3571 | 0.6596 | 0.4915 | 0.0663 | -0.0171 | -0.0104 |
| ExtraTrees | B3_base_plus_agronomic | 815 | 0 | 0.5021 | 0.3623 | 0.6495 | 0.4996 | 0.0587 | -0.0099 | -0.0032 |
| ExtraTrees | B5_all | 906 | 0 | 0.5053 | 0.3662 | 0.6451 | 0.5024 | 0.0627 | -0.0066 | 0.0000 |
| ExtraTrees | B4_base_plus_empirical | 783 | 0 | 0.5110 | 0.3727 | 0.6369 | 0.5080 | 0.0647 | -0.0009 | 0.0058 |
| ExtraTrees | B0_base | 692 | 0 | 0.5119 | 0.3747 | 0.6357 | 0.5093 | 0.0586 | 0.0000 | 0.0066 |
| ExtraTrees | B1_agronomic | 123 | 0 | 0.5143 | 0.3702 | 0.6323 | 0.5126 | 0.0506 | 0.0023 | 0.0090 |
| ExtraTrees | B2_empirical | 91 | 0 | 0.6567 | 0.5315 | 0.4004 | 0.6561 | 0.0272 | 0.1448 | 0.1514 |
| Ridge10 | B1_agronomic | 123 | 0 | 0.5888 | 0.4352 | 0.5179 | 0.5860 | 0.0703 | -0.0830 | -0.0923 |
| Ridge10 | B7_all_plus_discovery | 906 | 20 | 0.6710 | 0.5228 | 0.3739 | 0.6703 | 0.0394 | -0.0008 | -0.0101 |
| Ridge10 | B2_empirical | 91 | 0 | 0.6718 | 0.5080 | 0.3726 | 0.6688 | 0.0740 | -0.0000 | -0.0093 |
| Ridge10 | B0_base | 692 | 0 | 0.6718 | 0.5198 | 0.3725 | 0.6707 | 0.0429 | 0.0000 | -0.0093 |
| Ridge10 | B3_base_plus_agronomic | 815 | 0 | 0.6722 | 0.5217 | 0.3718 | 0.6713 | 0.0382 | 0.0004 | -0.0089 |
| Ridge10 | B4_base_plus_empirical | 783 | 0 | 0.6783 | 0.5296 | 0.3602 | 0.6776 | 0.0382 | 0.0065 | -0.0028 |
| Ridge10 | B5_all | 906 | 0 | 0.6811 | 0.5309 | 0.3550 | 0.6803 | 0.0389 | 0.0093 | 0.0000 |

### fold_state_stratified — competition

| model | representation | n_features | n_discovered | oof_rmse | oof_mae | oof_r2 | fold_rmse_mean | fold_rmse_std | delta_oof_rmse_vs_B0 | delta_oof_rmse_vs_B5 |
|---|---|---|---|---|---|---|---|---|---|---|
| ExtraTrees | B7_all_plus_discovery | 2085 | 20 | 0.4992 | 0.3574 | 0.6535 | 0.4956 | 0.0698 | -0.0011 | -0.0053 |
| ExtraTrees | B0_base | 1400 | 0 | 0.5003 | 0.3595 | 0.6519 | 0.4975 | 0.0631 | 0.0000 | -0.0041 |
| ExtraTrees | B1_agronomic | 350 | 0 | 0.5041 | 0.3605 | 0.6467 | 0.5002 | 0.0742 | 0.0038 | -0.0004 |
| ExtraTrees | B5_all | 2085 | 0 | 0.5045 | 0.3587 | 0.6462 | 0.5012 | 0.0675 | 0.0041 | 0.0000 |
| ExtraTrees | B3_base_plus_agronomic | 1750 | 0 | 0.5050 | 0.3600 | 0.6455 | 0.5019 | 0.0656 | 0.0046 | 0.0005 |
| ExtraTrees | B4_base_plus_empirical | 1735 | 0 | 0.5086 | 0.3635 | 0.6403 | 0.5058 | 0.0628 | 0.0083 | 0.0041 |
| ExtraTrees | B2_empirical | 335 | 0 | 0.6470 | 0.5188 | 0.4180 | 0.6462 | 0.0359 | 0.1466 | 0.1425 |
| Ridge10 | B0_base | 1400 | 0 | 0.6534 | 0.4991 | 0.4063 | 0.6516 | 0.0465 | 0.0000 | -0.0164 |
| Ridge10 | B4_base_plus_empirical | 1735 | 0 | 0.6587 | 0.5112 | 0.3967 | 0.6583 | 0.0236 | 0.0053 | -0.0112 |
| Ridge10 | B3_base_plus_agronomic | 1750 | 0 | 0.6644 | 0.5013 | 0.3863 | 0.6633 | 0.0361 | 0.0109 | -0.0055 |
| Ridge10 | B7_all_plus_discovery | 2085 | 20 | 0.6679 | 0.5081 | 0.3798 | 0.6677 | 0.0168 | 0.0145 | -0.0020 |
| Ridge10 | B5_all | 2085 | 0 | 0.6699 | 0.5110 | 0.3761 | 0.6697 | 0.0166 | 0.0164 | 0.0000 |
| Ridge10 | B1_agronomic | 350 | 0 | 0.6809 | 0.5063 | 0.3555 | 0.6784 | 0.0708 | 0.0274 | 0.0110 |
| Ridge10 | B2_empirical | 335 | 0 | 0.8269 | 0.6615 | 0.0493 | 0.8226 | 0.1036 | 0.1735 | 0.1570 |

### fold_municipality_grouped — clean

| model | representation | n_features | n_discovered | oof_rmse | oof_mae | oof_r2 | fold_rmse_mean | fold_rmse_std | delta_oof_rmse_vs_B0 | delta_oof_rmse_vs_B5 |
|---|---|---|---|---|---|---|---|---|---|---|
| ExtraTrees | B7_all_plus_discovery | 906 | 20 | 0.8488 | 0.7423 | -0.0018 | 0.8875 | 0.2683 | -0.0942 | -0.0779 |
| ExtraTrees | B1_agronomic | 123 | 0 | 0.8762 | 0.7542 | -0.0675 | 0.8841 | 0.3660 | -0.0669 | -0.0505 |
| ExtraTrees | B3_base_plus_agronomic | 815 | 0 | 0.9106 | 0.8167 | -0.1528 | 0.9068 | 0.2223 | -0.0325 | -0.0162 |
| ExtraTrees | B5_all | 906 | 0 | 0.9267 | 0.8357 | -0.1941 | 0.9132 | 0.2114 | -0.0164 | 0.0000 |
| ExtraTrees | B0_base | 692 | 0 | 0.9431 | 0.8585 | -0.2367 | 0.9073 | 0.1692 | 0.0000 | 0.0164 |
| ExtraTrees | B4_base_plus_empirical | 783 | 0 | 0.9451 | 0.8603 | -0.2420 | 0.8994 | 0.1603 | 0.0020 | 0.0184 |
| ExtraTrees | B2_empirical | 91 | 0 | 1.0265 | 0.9284 | -0.4652 | 0.8812 | 0.2812 | 0.0834 | 0.0998 |
| Ridge10 | B0_base | 692 | 0 | 0.8741 | 0.7213 | -0.0623 | 0.9117 | 0.2657 | 0.0000 | -0.0614 |
| Ridge10 | B4_base_plus_empirical | 783 | 0 | 0.8932 | 0.7350 | -0.1094 | 0.9534 | 0.3038 | 0.0192 | -0.0423 |
| Ridge10 | B7_all_plus_discovery | 906 | 20 | 0.9112 | 0.7471 | -0.1545 | 1.0279 | 0.2898 | 0.0372 | -0.0243 |
| Ridge10 | B3_base_plus_agronomic | 815 | 0 | 0.9257 | 0.7710 | -0.1914 | 0.9691 | 0.2624 | 0.0516 | -0.0098 |
| Ridge10 | B5_all | 906 | 0 | 0.9355 | 0.7760 | -0.2168 | 1.0177 | 0.2836 | 0.0614 | 0.0000 |
| Ridge10 | B1_agronomic | 123 | 0 | 0.9681 | 0.7783 | -0.3032 | 0.9302 | 0.3278 | 0.0941 | 0.0326 |
| Ridge10 | B2_empirical | 91 | 0 | 1.0002 | 0.8509 | -0.3910 | 0.9455 | 0.1809 | 0.1261 | 0.0647 |

### fold_municipality_grouped — competition

| model | representation | n_features | n_discovered | oof_rmse | oof_mae | oof_r2 | fold_rmse_mean | fold_rmse_std | delta_oof_rmse_vs_B0 | delta_oof_rmse_vs_B5 |
|---|---|---|---|---|---|---|---|---|---|---|
| ExtraTrees | B1_agronomic | 350 | 0 | 0.8590 | 0.7416 | -0.0259 | 0.8539 | 0.3158 | -0.0385 | -0.0626 |
| ExtraTrees | B7_all_plus_discovery | 2085 | 20 | 0.8675 | 0.7656 | -0.0464 | 0.8585 | 0.2362 | -0.0299 | -0.0541 |
| ExtraTrees | B0_base | 1400 | 0 | 0.8974 | 0.8032 | -0.1198 | 0.8748 | 0.2383 | 0.0000 | -0.0241 |
| ExtraTrees | B3_base_plus_agronomic | 1750 | 0 | 0.8982 | 0.7941 | -0.1216 | 0.8862 | 0.2651 | 0.0007 | -0.0234 |
| ExtraTrees | B5_all | 2085 | 0 | 0.9216 | 0.8210 | -0.1809 | 0.8900 | 0.2584 | 0.0241 | 0.0000 |
| ExtraTrees | B4_base_plus_empirical | 1735 | 0 | 0.9309 | 0.8350 | -0.2048 | 0.9082 | 0.2300 | 0.0334 | 0.0093 |
| ExtraTrees | B2_empirical | 335 | 0 | 0.9962 | 0.8822 | -0.3799 | 0.8659 | 0.2344 | 0.0988 | 0.0746 |
| Ridge10 | B0_base | 1400 | 0 | 0.8059 | 0.6428 | 0.0970 | 0.8326 | 0.3085 | 0.0000 | -0.0917 |
| Ridge10 | B4_base_plus_empirical | 1735 | 0 | 0.8464 | 0.6816 | 0.0039 | 0.8574 | 0.3115 | 0.0405 | -0.0512 |
| Ridge10 | B3_base_plus_agronomic | 1750 | 0 | 0.8676 | 0.7036 | -0.0465 | 0.9174 | 0.2910 | 0.0617 | -0.0300 |
| Ridge10 | B5_all | 2085 | 0 | 0.8976 | 0.7204 | -0.1202 | 0.9282 | 0.3152 | 0.0917 | 0.0000 |
| Ridge10 | B7_all_plus_discovery | 2085 | 20 | 0.9142 | 0.7380 | -0.1620 | 0.9533 | 0.2853 | 0.1083 | 0.0166 |
| Ridge10 | B2_empirical | 335 | 0 | 0.9485 | 0.7541 | -0.2507 | 0.9386 | 0.2051 | 0.1426 | 0.0509 |
| Ridge10 | B1_agronomic | 350 | 0 | 1.2403 | 0.9932 | -1.1388 | 1.1501 | 0.6005 | 0.4344 | 0.3427 |

## Interpretation rules

- This is not final model selection.
- Compare the same model across representations before comparing model families.
- Treat municipality-grouped results as a first-class robustness signal.
- B7 discovers expressions inside each outer training fold only.
- Do not promote a discovered expression because its training association is large.
- Use these results to choose a small representation set for 03C.2.
- 03C.2 may then tune CatBoost/ElasticNet/kernels/PCA/PLS inside folds.
