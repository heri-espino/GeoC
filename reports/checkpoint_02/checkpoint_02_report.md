# Checkpoint 02 run report

Generated: 2026-09-18T16:47:00.335593+00:00
Git commit: 601d725e57c6b3d256d7b7e646a7bebe2f9ebc1c
Feature-table SHA256: bf27e28f01f1765319f5c9d60af3786039ebf0c93ba550d31f7e5ad917b03984

## Frozen feature table

- rows: **197**
- training parcels: **138**
- prediction parcels: **59**
- clean features: **694**
- competition-only features: **709**
- total features: **1403**

No hidden prediction target is used anywhere in this checkpoint.

## Feature inventory

| family | mode | source | feature_count | numeric_features | non_numeric_features |
|---|---|---|---|---|---|
| admin | clean | official_split+official_parcels+admin | 8 | 6 | 2 |
| basic | clean | sat_basic | 539 | 539 | 0 |
| cem | clean | external_inegi_cem | 5 | 5 | 0 |
| climate_official | clean | official_climate | 24 | 24 | 0 |
| geometry | clean | official_split+official_parcels+admin | 6 | 6 | 0 |
| siap | clean | external_siap | 14 | 14 | 0 |
| soilgrids | clean | external_soilgrids | 90 | 90 | 0 |
| topography | clean | official_topography | 8 | 8 | 0 |
| basic | competition | sat_basic | 539 | 539 | 0 |
| chirps | competition | external_chirps_daily | 1 | 1 | 0 |
| climate_official | competition | official_climate | 45 | 45 | 0 |
| pro | competition | sat_pro | 97 | 97 | 0 |
| siap | competition | external_siap | 5 | 5 | 0 |
| wapor | competition | external_wapor | 22 | 22 | 0 |

## Train vs prediction diagnostics

Numeric features screened: **1401**; features with at least one shift/support flag: **6**.

Flags are descriptive diagnostics only. They are not used to select features or tune models in the clean track.

| feature | family | mode | abs_smd | p_adjusted | missing_gap_abs | prediction_outside_train_range_fraction |
|---|---|---|---|---|---|---|
| sat_basic_hist_monthly__sentinel_2__enddi_promedio__m08 | basic | clean | 0.5343 | 1.0000 | 0.0000 | 0.0000 |
| sat_basic_hist_monthly__sentinel_2__dswi2_promedio__m08 | basic | clean | 0.5303 | 0.6892 | 0.0000 | 0.0000 |
| sat_basic_hist_monthly__sentinel_2__mcrc_promedio__m08 | basic | clean | 0.5296 | 0.6892 | 0.0000 | 0.0169 |
| sat_basic_hist_monthly__sentinel_2__crc_promedio__m08 | basic | clean | 0.5035 | 1.0000 | 0.0000 | 0.0339 |
| soilgrids__bdod__30_60cm__std | soilgrids | clean | 0.3218 | 0.8939 | 0.0000 | 0.1017 |
| sat_basic_2025__sentinel_2__evi_min__std | basic | competition | 0.2212 | 1.0000 | 0.0000 | 0.1017 |

## Frozen validation

Primary protocol: fold_state_stratified; robustness protocol: fold_municipality_grouped; folds: **5**.

The primary folds preserve state composition. The robustness folds keep municipalities together to expose spatial/generalization sensitivity.

## Ablations

| ablation | n_features | rationale |
|---|---:|---|
| A0_geometry_admin | 12 | Parcel geometry plus administrative numeric fields only. |
| A1_static_environment | 115 | Add static soil and terrain context. |
| A2_static_plus_siap_history | 129 | Add historical municipal barley production context through 2024. |
| A3_clean_plus_climate | 153 | Add historical official climate climatology. |
| A4_clean_plus_basic_history | 692 | Add historical BASIC satellite summaries through 2024. |
| A5_clean_full | 692 | Every clean numeric feature. |
| A6_competition_remote_weather | 1396 | Clean full set plus 2025 BASIC/PRO, official climate, CHIRPS and WaPOR; deliberately excludes contemporaneous SIAP 2025 outcome proxies.
 |
| A7_competition_full | 1401 | Every numeric feature, including SIAP 2025 municipal outcome proxies. This is a challenge/competition representation, not the clean prospective track.
 |

## Initial models

Only untuned baselines are run here: DummyMean, fixed-alpha Ridge and regularized ExtraTrees. Median imputation is fitted inside each fold. PCA, supervised feature selection, Optuna, CatBoost and final hyperparameter search are intentionally deferred.

| protocol | ablation | model | n_features | rmse_mean | rmse_std | mae_mean | r2_mean |
|---|---|---|---|---|---|---|---|
| fold_municipality_grouped | A0_geometry_admin | Ridge | 12 | 0.9676 | 0.5393 | 0.9008 | -5.1669 |
| fold_municipality_grouped | A0_geometry_admin | ExtraTrees | 12 | 0.9809 | 0.3903 | 0.9319 | -5.6206 |
| fold_municipality_grouped | A0_geometry_admin | DummyMean | 12 | 1.0066 | 0.2618 | 0.9442 | -6.1150 |
| fold_municipality_grouped | A1_static_environment | ExtraTrees | 115 | 0.9721 | 0.2938 | 0.9146 | -6.1811 |
| fold_municipality_grouped | A1_static_environment | Ridge | 115 | 0.9937 | 0.2762 | 0.8758 | -9.2350 |
| fold_municipality_grouped | A1_static_environment | DummyMean | 115 | 1.0066 | 0.2618 | 0.9442 | -6.1150 |
| fold_municipality_grouped | A2_static_plus_siap_history | ExtraTrees | 129 | 0.9523 | 0.1936 | 0.8924 | -7.0961 |
| fold_municipality_grouped | A2_static_plus_siap_history | Ridge | 129 | 0.9582 | 0.3190 | 0.8134 | -10.0149 |
| fold_municipality_grouped | A2_static_plus_siap_history | DummyMean | 129 | 1.0066 | 0.2618 | 0.9442 | -6.1150 |
| fold_municipality_grouped | A3_clean_plus_climate | Ridge | 153 | 0.7426 | 0.2558 | 0.6459 | -5.1998 |
| fold_municipality_grouped | A3_clean_plus_climate | ExtraTrees | 153 | 0.9599 | 0.2118 | 0.9023 | -8.0782 |
| fold_municipality_grouped | A3_clean_plus_climate | DummyMean | 153 | 1.0066 | 0.2618 | 0.9442 | -6.1150 |
| fold_municipality_grouped | A4_clean_plus_basic_history | ExtraTrees | 692 | 0.9073 | 0.1692 | 0.8498 | -7.0745 |
| fold_municipality_grouped | A4_clean_plus_basic_history | Ridge | 692 | 0.9117 | 0.2657 | 0.7851 | -6.3317 |
| fold_municipality_grouped | A4_clean_plus_basic_history | DummyMean | 692 | 1.0066 | 0.2618 | 0.9442 | -6.1150 |
| fold_municipality_grouped | A5_clean_full | ExtraTrees | 692 | 0.9073 | 0.1692 | 0.8498 | -7.0745 |
| fold_municipality_grouped | A5_clean_full | Ridge | 692 | 0.9117 | 0.2657 | 0.7851 | -6.3317 |
| fold_municipality_grouped | A5_clean_full | DummyMean | 692 | 1.0066 | 0.2618 | 0.9442 | -6.1150 |
| fold_municipality_grouped | A6_competition_remote_weather | Ridge | 1396 | 0.8178 | 0.2993 | 0.6991 | -4.0401 |
| fold_municipality_grouped | A6_competition_remote_weather | ExtraTrees | 1396 | 0.8835 | 0.2233 | 0.8181 | -6.4972 |
| fold_municipality_grouped | A6_competition_remote_weather | DummyMean | 1396 | 1.0066 | 0.2618 | 0.9442 | -6.1150 |
| fold_municipality_grouped | A7_competition_full | Ridge | 1401 | 0.8373 | 0.3174 | 0.7177 | -4.2785 |
| fold_municipality_grouped | A7_competition_full | ExtraTrees | 1401 | 0.8916 | 0.2080 | 0.8300 | -6.4127 |
| fold_municipality_grouped | A7_competition_full | DummyMean | 1401 | 1.0066 | 0.2618 | 0.9442 | -6.1150 |
| fold_state_stratified | A0_geometry_admin | ExtraTrees | 12 | 0.5095 | 0.0845 | 0.3659 | 0.6272 |
| fold_state_stratified | A0_geometry_admin | Ridge | 12 | 0.5548 | 0.0675 | 0.4016 | 0.5603 |
| fold_state_stratified | A0_geometry_admin | DummyMean | 12 | 0.8506 | 0.0254 | 0.7511 | -0.0179 |
| fold_state_stratified | A1_static_environment | ExtraTrees | 115 | 0.5262 | 0.0672 | 0.3957 | 0.6069 |
| fold_state_stratified | A1_static_environment | Ridge | 115 | 0.5652 | 0.0680 | 0.4461 | 0.5460 |
| fold_state_stratified | A1_static_environment | DummyMean | 115 | 0.8506 | 0.0254 | 0.7511 | -0.0179 |
| fold_state_stratified | A2_static_plus_siap_history | ExtraTrees | 129 | 0.5027 | 0.0867 | 0.3708 | 0.6378 |
| fold_state_stratified | A2_static_plus_siap_history | Ridge | 129 | 0.5621 | 0.0745 | 0.4412 | 0.5528 |
| fold_state_stratified | A2_static_plus_siap_history | DummyMean | 129 | 0.8506 | 0.0254 | 0.7511 | -0.0179 |
| fold_state_stratified | A3_clean_plus_climate | ExtraTrees | 153 | 0.5035 | 0.0901 | 0.3726 | 0.6360 |
| fold_state_stratified | A3_clean_plus_climate | Ridge | 153 | 0.5298 | 0.0688 | 0.4152 | 0.6025 |
| fold_state_stratified | A3_clean_plus_climate | DummyMean | 153 | 0.8506 | 0.0254 | 0.7511 | -0.0179 |
| fold_state_stratified | A4_clean_plus_basic_history | ExtraTrees | 692 | 0.5093 | 0.0586 | 0.3746 | 0.6324 |
| fold_state_stratified | A4_clean_plus_basic_history | Ridge | 692 | 0.6707 | 0.0429 | 0.5196 | 0.3670 |
| fold_state_stratified | A4_clean_plus_basic_history | DummyMean | 692 | 0.8506 | 0.0254 | 0.7511 | -0.0179 |
| fold_state_stratified | A5_clean_full | ExtraTrees | 692 | 0.5093 | 0.0586 | 0.3746 | 0.6324 |
| fold_state_stratified | A5_clean_full | Ridge | 692 | 0.6707 | 0.0429 | 0.5196 | 0.3670 |
| fold_state_stratified | A5_clean_full | DummyMean | 692 | 0.8506 | 0.0254 | 0.7511 | -0.0179 |
| fold_state_stratified | A6_competition_remote_weather | ExtraTrees | 1396 | 0.5001 | 0.0634 | 0.3610 | 0.6447 |
| fold_state_stratified | A6_competition_remote_weather | Ridge | 1396 | 0.6522 | 0.0481 | 0.4993 | 0.3947 |
| fold_state_stratified | A6_competition_remote_weather | DummyMean | 1396 | 0.8506 | 0.0254 | 0.7511 | -0.0179 |
| fold_state_stratified | A7_competition_full | ExtraTrees | 1401 | 0.4966 | 0.0637 | 0.3574 | 0.6498 |
| fold_state_stratified | A7_competition_full | Ridge | 1401 | 0.6515 | 0.0464 | 0.4989 | 0.3964 |
| fold_state_stratified | A7_competition_full | DummyMean | 1401 | 0.8506 | 0.0254 | 0.7511 | -0.0179 |

## Interpretation policy

- Do not select a final model from this report alone.
- Compare ablation deltas under both frozen protocols.
- Treat large disagreement between protocols as evidence of spatial sensitivity.
- Keep clean and competition tracks separate in all subsequent reporting.
- Do not use prediction-set X diagnostics as a hidden-label substitute.
