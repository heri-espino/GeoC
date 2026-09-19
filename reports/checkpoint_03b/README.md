# Checkpoint 03B results

The workstation run completed successfully on 2026-09-18.

## Deterministic empirical layer

```text
rows                         197
derived features             335
clean                         91
competition                  244
mean/max missing fraction    0.0
infinite values              0
dropped all-null             0
dropped constant             1
```

The only constant candidate removed was:

```text
emp_histpos__landsat_vi6t__above_history_fraction
```

The materialized layer does not contain `RENDIMIENTO_T_HA` or `CONJUNTO` and was not
selected using yield.

## Fold-local expression discovery

The recurrence audit used the frozen Checkpoint 02 folds and 138 labeled parcels.

```text
configured primitives                 24
top primitives retained per fit       12
top expressions retained per fit      20
protocols                              2
fits                                  10
selected rows                         200
unique expressions                     88
raw expressions recurring >=3/5
  under both protocols                  7
validation targets used                no
```

The seven recurrent expression strings collapse to **four conceptual primitive-pair motifs**,
because several formulas are monotone transforms of the same positive-variable ratio and
therefore have identical Spearman rankings.

### Motif 1 — SOC versus precipitation

Primitive pair:

```text
soilgrids__soc__depth_weighted_0_30cm
clim_official_2025__prec__season_sum
```

Three equivalent ranking forms recur:

```text
precip / SOC
SOC / precip
2*(SOC - precip) / (|SOC| + |precip| + eps)
```

Observed recurrence:

```text
state-stratified        4/5 folds
municipality-grouped    3/5 folds
mean |Spearman| train   ~0.697 / ~0.703
direction consistency   1.00
```

Because SOC and seasonal precipitation are positive here, reciprocal ratios reverse the rank
and the symmetric-change transform is monotone in their ratio. These three rows therefore
represent one signal, not three independent discoveries.

### Motif 2 — historical municipal yield trend versus precipitation

Stable forms:

```text
precip / |SIAP recent yield trend|
symmetric_change(SIAP recent yield trend, precip)
```

Observed recurrence:

```text
precip / |trend|
  state-stratified        5/5
  municipality-grouped    3/5
  mean |Spearman| train   ~0.691 / ~0.684

symmetric change
  state-stratified        5/5
  municipality-grouped    3/5
  mean |Spearman| train   ~0.684 / ~0.682
```

This motif should be treated cautiously because the SIAP trend is a municipal feature.

### Motif 3 — historical municipal yield trend versus thermal time

Formula:

```text
GDD_proxy_base5C / |SIAP recent yield trend|
```

Observed recurrence:

```text
state-stratified        5/5
municipality-grouped    4/5
mean |Spearman| train   ~0.692 / ~0.628
direction consistency   1.00
```

This is the most recurrent cross-protocol raw expression by fold count, but its denominator is
still the municipality-level SIAP trend and it remains a candidate only.

### Motif 4 — recent SIAP trend versus recent SIAP mean

Formula:

```text
2*(SIAP recent yield trend - SIAP recent yield mean)
  / (|trend| + |mean| + eps)
```

Observed recurrence:

```text
state-stratified        5/5
municipality-grouped    3/5
mean |Spearman| train   ~0.661 / ~0.693
direction consistency   1.00
```

This is effectively an administrative historical-productivity signal and must not be
interpreted as a parcel-level physiological index.

## Primitive stability

The most consistently retained primitives across the ten training-fold fits were:

```text
selected 10/10:
  siap_hist__yield_mean_recent
  clim_official_2025__prec__season_sum
  emp_vci_like_short__s2_ndvi__season_mean

selected 9/10:
  siap_hist__yield_trend_recent
  wapor_2025__aeti__season_total
  agro_thermal__2025__gdd_proxy_base5p0c
  soilgrids__soc__depth_weighted_0_30cm
  soilgrids__nitrogen__depth_weighted_0_30cm
  wapor_2025__npp__season_total
```

The short-baseline VCI-like NDVI feature is notable because it was retained as a top-12
primitive in all ten fits, even though no pairwise expression involving it met the cross-
protocol recurrence threshold. This makes it worth testing as a standalone predictor in 03C,
but it is not evidence that the project has discovered a new VCI.

## Geographic / administrative caution

The SIAP historical mean and trend each have only **9 unique values across the 138 training
parcels** because they are municipality-level features.

The largest groups are:

```text
Chignahuapan       71 parcels
Calpulalpan        30
Singuilucan        16
Almoloya           10
others             11 total
```

For `siap_hist__yield_trend_recent`, 50/138 training parcels have absolute trend below 0.05.
The configured safe ratios do not hit a singularity, but dividing by a small municipality-level
trend can amplify administrative/geographic separation.

Therefore the recurrent SIAP-denominator expressions may partly encode municipality structure.
They must be judged by held-out predictive performance under municipality-grouped CV before
being considered useful beyond the competition setting.

The frozen grouped folds are also strongly imbalanced because whole municipalities remain
together:

```text
held-out rows by grouped fold:
33, 16, 10, 73, 6
```

Recurrence counts should therefore not be interpreted as five equally informative
replications.

## Interpretation

Checkpoint 03B discovered **candidate motifs**, not a validated new barley-yield metric.

The strongest empirical themes are:

```text
soil carbon × precipitation balance
historical municipal productivity × current precipitation
historical municipal productivity × accumulated thermal time
short-baseline NDVI condition as a stable primitive
```

The next scientific test belongs in Checkpoint 03C:

1. refit the expression miner separately inside every outer training fold;
2. compare models with and without the discovered-expression augmentation;
3. report held-out RMSE, MAE and R² under both frozen protocols;
4. test the VCI-like primitive separately;
5. inspect whether gains survive municipality-grouped validation.

Only a formula that repeatedly contributes **held-out** signal should be considered for a
named empirical index or deeper symbolic-regression search.
