# Transductive Competition Objective

**Status:** canonical active modeling doctrine  
**Effective:** 2026-09-19

## 1. Problem definition

GeoCebada has a fixed universe of 197 known barley parcels. Let `L` denote the 138 parcels with observed yield and `U` the 59 fixed target parcels whose yields are hidden by FIRA.

```text
|L| = 138
|U| = 59
L ∪ U = 197 known parcels
X_L and X_U are observable
y_L is observable
y_U is hidden
```

The active goal is to estimate the vector `y_U` with minimum error against FIRA's hidden reference values. The project is therefore treated as **transductive fixed-target regression** rather than as a generic future-parcel forecasting problem.

## 2. Reconstruction interpretation

Operationally, treat the 59 target yields as latent, missing or potentially untrusted reported values that must be reconstructed from independent observable evidence. FIRA's hidden values are the scoring reference that will be used to judge how credible the reconstruction is.

This framing does **not** imply that the hidden reference is directly accessible or that it is error-free physical truth. It means that the optimization target of the competition is the distance between our 59 reconstructed values and FIRA's 59 reserved reference values.

## 3. Evidence policy

The project should aggressively exploit every rule-permitted observable source that can reduce uncertainty about the fixed 59 targets.

Allowed evidence includes:

- complete covariates for all 197 parcels;
- full-season 2025 `competition` features;
- original BASIC/PRO longitudinal trajectories;
- spatial geometry and neighborhood structure;
- state/municipality and administrative context;
- historical and contemporaneous public production information;
- SIAP 2025 and historical SIAP;
- CHIRPS, WaPOR, SoilGrids, CEM/topography and official climate;
- new public sources with documented provenance and a testable hypothesis;
- X-only representations learned jointly over labeled and target parcels.

The only hard target boundary is that the 59 hidden FIRA yields themselves are unavailable. Do not attempt to obtain them directly.

## 4. Why transduction matters

The exact 59 X vectors to be predicted are already known. Their geometry relative to the 138 labeled parcels is therefore useful information.

A method may use `X_L ∪ X_U` to learn scaling or normalization, PCA/manifold coordinates, clusters, density/support diagnostics, nearest-neighbor structure, graph edges, unlabeled target-set topology, covariate-shift weights and target-specific local neighborhoods.

Any operation that uses y remains restricted to the labeled part of the corresponding validation/training problem.

## 5. Validation must be transductive

Conventional CV from Checkpoint 03 remains useful historical evidence, but active method selection should use pseudo-competition experiments.

For each pseudo-competition split:

1. choose a subset `U*` from the 138 labeled parcels;
2. hide `y_U*`;
3. expose `X_U*` to all X-only transductive steps;
4. use only the remaining labeled y for supervised fitting;
5. predict `U*`;
6. reveal `y_U*` only for RMSE/MAE/R².

Pseudo-target selection should be designed to resemble the real 59 target parcels rather than only using arbitrary random partitions. Candidate matching dimensions include state, municipality, feature-space density, geography, phenology, soil, climate and target-set coverage.

Legacy `fold_state_stratified` and `fold_municipality_grouped` remain stress tests, not the single optimization objective.

## 6. Target-set topology and similarity

Before expensive model tuning, characterize the relationship between each target parcel and the labeled set.

Required analyses:

- nearest labeled neighbors in geographic space;
- nearest labeled neighbors in agronomic/tabular feature spaces;
- correlation of 2025 vegetation-index trajectories;
- cross-correlation and possible phenological lag;
- dynamic-time-warping or related curve distance where justified;
- soil/climate/topography similarity;
- same-municipality/state relationships;
- train-vs-target adversarial classification;
- support/extrapolation scores;
- spatial autocorrelation of observed yield;
- spatial autocorrelation of baseline residuals;
- empirical relationship between similarity distance and absolute yield difference.

The purpose is to determine which target parcels are interpolation-like and which are true extrapolation cases.

## 7. Local and graph models

A single global function is not required. Checkpoint 04 may fit target-specific local estimators or mixtures of experts.

For a target parcel, candidate experts include global PLS/Ridge/CatBoost/tree ensembles, geographic k-nearest neighbors, phenology-trajectory neighbors, climate/soil/topography neighbors, municipality/regional priors, local regressions, graph-based semi-supervised regression and covariate-shift-weighted global models.

Expert weights may depend on measurable target properties such as local density, nearest-neighbor distance, target support and uncertainty, provided the weighting rule is validated through pseudo-competition RMSE.

## 8. Longitudinal data are not exhausted by Feature Table v1

BASIC has 107,666 rows × 96 columns over 197 parcels and 2022–2025 Sentinel-2/Landsat observations. PRO has 47,804 rows × 20 columns over the same 197 parcels with Planet in 2025.

Feature Table v1 compressed these repeated parcel/date/sensor observations into one parcel row for conventional tabular modeling. That representation remains valuable but is not the only allowed representation.

Checkpoint 04 should also retain direct access to the time-series trajectories so that temporal shape, correlation, lag and local-curve similarity can be modeled without losing information through global summary statistics.

## 9. External-data priorities

Existing external data should be treated as reconstruction evidence rather than optional decoration.

Highest-priority audit:

- SIAP 2025 municipal yield/production information aligned specifically to `Cebada grano`, the relevant cycle and `Temporal` modality where the source supports that granularity.

Then:

- CHIRPS daily precipitation after current QC is resolved;
- WaPOR AETI/NPP;
- SoilGrids;
- INEGI CEM and terrain;
- official 2025 climate;
- richer daily weather/stress variables such as ERA5-Land or other public sources if they add non-redundant information.

Every external feature must retain source, time, spatial granularity, units and join semantics.

## 10. First Modeling Delivery baseline

Checkpoint 03 is frozen as the first conventional modeling delivery. It used leakage-safe nested CV and identified:

```text
C0_base + PLS                         state 0.5339   grouped 0.7621
C3_base_agro_plus_discovery + Ridge  state 0.5331   grouped 0.7650
C1_agronomic + CatBoost              state 0.5235   grouped 0.7709
E123 equal top 3                      state 0.5121   grouped 0.7481
E13 PLS + CatBoost                    state 0.5082   grouped 0.7488
```

These values are the baseline to beat. They are not unbiased estimates of the final hidden-test RMSE and they do not constrain Checkpoint 04 to those model families or equal weights.

## 11. Checkpoint 04 work plan

```text
04A  Target topology, autocorrelation, similarity and extrapolation diagnostics
04B  Repeated transductive pseudo-competition validation
04C  Higher-compute global models and richer hyperparameter searches
04D  Local-per-target, graph, semi-supervised and covariate-shift models
04E  External-evidence enrichment, especially SIAP Temporal/cycle alignment
04F  Transductive ensemble, uncertainty audit and final 59-value reconstruction
```

Do not begin 04F until 04B can score every proposed strategy with reproducible pseudo-test predictions.

## 12. Success criterion

Every new modeling idea must answer one question:

> Does this use observable evidence from the fixed target problem to reduce pseudo-competition error in a way likely to transfer to the actual 59 parcels?

Complexity, elegance and generic deployability are secondary to that objective.
