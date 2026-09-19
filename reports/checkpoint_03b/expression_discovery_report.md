# Checkpoint 03B — Expression Discovery Stability

This report is a training-fold recurrence audit, not a model benchmark.
Validation targets are never used to select, rank or report expressions.

- training parcels: 138
- primitives configured: 24
- top primitives per fit: 12
- top expressions per fit: 20
- unique selected expressions: 88

An expression recurring across folds is only a candidate for later testing. Training Spearman values are not held-out performance estimates.

## fold_state_stratified

| Selected folds | Mean rank | Mean abs(rho) train | Direction consistency | Formula |
|---:|---:|---:|---:|---|
| 5/5 | 5.60 | 0.692 | 1.00 | (siap_hist__yield_mean_recent) / (abs(siap_hist__yield_trend_recent) + eps) |
| 5/5 | 5.60 | 0.692 | 1.00 | (agro_thermal__2025__gdd_proxy_base5p0c) / (abs(siap_hist__yield_trend_recent) + eps) |
| 5/5 | 6.00 | 0.691 | 1.00 | (clim_official_2025__prec__season_sum) / (abs(siap_hist__yield_trend_recent) + eps) |
| 5/5 | 7.80 | 0.684 | 1.00 | 2*((siap_hist__yield_trend_recent)-(clim_official_2025__prec__season_sum)) / (abs(siap_hist__yield_trend_recent)+abs(clim_official_2025__prec__season_sum)+eps) |
| 5/5 | 11.20 | 0.661 | 1.00 | (agro_soil__soc_x_cec) / (abs(clim_official_2025__prec__season_sum) + eps) |
| 5/5 | 12.20 | 0.661 | 1.00 | (clim_official_2025__prec__season_sum) / (abs(agro_soil__soc_x_cec) + eps) |
| 5/5 | 12.60 | 0.661 | 1.00 | 2*((siap_hist__yield_trend_recent)-(siap_hist__yield_mean_recent)) / (abs(siap_hist__yield_trend_recent)+abs(siap_hist__yield_mean_recent)+eps) |
| 5/5 | 13.20 | 0.661 | 1.00 | 2*((clim_official_2025__prec__season_sum)-(agro_soil__soc_x_cec)) / (abs(clim_official_2025__prec__season_sum)+abs(agro_soil__soc_x_cec)+eps) |
| 5/5 | 13.40 | 0.658 | 1.00 | (wapor_2025__aeti__season_total) / (abs(siap_hist__yield_trend_recent) + eps) |
| 4/5 | 2.00 | 0.697 | 1.00 | (clim_official_2025__prec__season_sum) / (abs(soilgrids__soc__depth_weighted_0_30cm) + eps) |
| 4/5 | 3.00 | 0.697 | 1.00 | (soilgrids__soc__depth_weighted_0_30cm) / (abs(clim_official_2025__prec__season_sum) + eps) |
| 4/5 | 4.00 | 0.697 | 1.00 | 2*((soilgrids__soc__depth_weighted_0_30cm)-(clim_official_2025__prec__season_sum)) / (abs(soilgrids__soc__depth_weighted_0_30cm)+abs(clim_official_2025__prec__season_sum)+eps) |
| 4/5 | 6.75 | 0.689 | 1.00 | 2*((siap_hist__yield_trend_recent)-(agro_water__2025__precip_minus_aeti_season)) / (abs(siap_hist__yield_trend_recent)+abs(agro_water__2025__precip_minus_aeti_season)+eps) |
| 4/5 | 7.50 | 0.686 | 1.00 | (agro_water__2025__precip_minus_aeti_season) / (abs(siap_hist__yield_trend_recent) + eps) |
| 4/5 | 16.75 | 0.638 | 1.00 | (soilgrids__nitrogen__depth_weighted_0_30cm) / (abs(clim_official_2025__prec__season_sum) + eps) |

## fold_municipality_grouped

| Selected folds | Mean rank | Mean abs(rho) train | Direction consistency | Formula |
|---:|---:|---:|---:|---|
| 4/5 | 11.50 | 0.628 | 1.00 | (agro_thermal__2025__gdd_proxy_base5p0c) / (abs(siap_hist__yield_trend_recent) + eps) |
| 3/5 | 3.00 | 0.703 | 1.00 | (clim_official_2025__prec__season_sum) / (abs(soilgrids__soc__depth_weighted_0_30cm) + eps) |
| 3/5 | 4.00 | 0.703 | 1.00 | (soilgrids__soc__depth_weighted_0_30cm) / (abs(clim_official_2025__prec__season_sum) + eps) |
| 3/5 | 5.00 | 0.703 | 1.00 | 2*((soilgrids__soc__depth_weighted_0_30cm)-(clim_official_2025__prec__season_sum)) / (abs(soilgrids__soc__depth_weighted_0_30cm)+abs(clim_official_2025__prec__season_sum)+eps) |
| 3/5 | 6.67 | 0.693 | 1.00 | 2*((siap_hist__yield_trend_recent)-(siap_hist__yield_mean_recent)) / (abs(siap_hist__yield_trend_recent)+abs(siap_hist__yield_mean_recent)+eps) |
| 3/5 | 8.33 | 0.684 | 1.00 | (clim_official_2025__prec__season_sum) / (abs(siap_hist__yield_trend_recent) + eps) |
| 3/5 | 8.33 | 0.682 | 1.00 | 2*((siap_hist__yield_trend_recent)-(clim_official_2025__prec__season_sum)) / (abs(siap_hist__yield_trend_recent)+abs(clim_official_2025__prec__season_sum)+eps) |
| 2/5 | 3.50 | 0.719 | 1.00 | (siap_hist__yield_mean_recent) / (abs(siap_hist__yield_trend_recent) + eps) |
| 2/5 | 6.50 | 0.712 | 1.00 | 2*((siap_hist__yield_trend_recent)-(agro_water__2025__precip_minus_aeti_season)) / (abs(siap_hist__yield_trend_recent)+abs(agro_water__2025__precip_minus_aeti_season)+eps) |
| 2/5 | 13.50 | 0.574 | 1.00 | (soilgrids__nitrogen__depth_weighted_0_30cm) / (abs(clim_official_2025__prec__season_sum) + eps) |
| 2/5 | 14.50 | 0.574 | 1.00 | (clim_official_2025__prec__season_sum) / (abs(soilgrids__nitrogen__depth_weighted_0_30cm) + eps) |
| 2/5 | 15.50 | 0.700 | 1.00 | (agro_water__2025__precip_minus_aeti_season) / (abs(siap_hist__yield_trend_recent) + eps) |
| 2/5 | 15.50 | 0.574 | 1.00 | 2*((clim_official_2025__prec__season_sum)-(soilgrids__nitrogen__depth_weighted_0_30cm)) / (abs(clim_official_2025__prec__season_sum)+abs(soilgrids__nitrogen__depth_weighted_0_30cm)+eps) |
| 2/5 | 16.00 | 0.698 | 1.00 | (soilgrids__cec__depth_weighted_0_30cm) / (abs(siap_hist__yield_trend_recent) + eps) |
| 1/5 | 1.00 | 0.566 | 1.00 | (soilgrids__cec__depth_weighted_0_30cm) / (abs(agro_pheno__y2025_s2_ndvi__auc) + eps) |

## Cross-protocol recurrence

Expressions selected in at least 3/5 fits under both protocols:

- reverse_safe_ratio::siap_hist__yield_trend_recent::agro_thermal__2025__gdd_proxy_base5p0c
- reverse_safe_ratio::siap_hist__yield_trend_recent::clim_official_2025__prec__season_sum
- reverse_safe_ratio::soilgrids__soc__depth_weighted_0_30cm::clim_official_2025__prec__season_sum
- safe_ratio::soilgrids__soc__depth_weighted_0_30cm::clim_official_2025__prec__season_sum
- symmetric_change::siap_hist__yield_trend_recent::clim_official_2025__prec__season_sum
- symmetric_change::siap_hist__yield_trend_recent::siap_hist__yield_mean_recent
- symmetric_change::soilgrids__soc__depth_weighted_0_30cm::clim_official_2025__prec__season_sum

Any predictive evaluation belongs in Checkpoint 03C, where the miner is refit inside each outer training fold.
