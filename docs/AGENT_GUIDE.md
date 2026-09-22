# GeoCebada agent guide

**Current phase:** Checkpoint 05 — global/local cross-fitted mixture implemented; workstation run pending  
**Canonical objective:** `docs/TRANSDUCTIVE_OBJECTIVE.md`

This is the operational entry point for any AI agent, collaborator or new contributor.

## 1. Read order

Before changing modeling or data logic, read:

1. `docs/TRANSDUCTIVE_OBJECTIVE.md`.
2. `.ai_handoff`.
3. this file.
4. `docs/CHECKPOINT_05_PLAN.md` and `checkpoints/05_global_local_mixture/README.md`.
5. `checkpoints/04_transductive_competition/README.md`.
6. `docs/PROJECT_HISTORY.md`.
7. `data/.ai_handoff` and `docs/DATA_SOURCES.md`.
8. `reports/checkpoint_03c2/checkpoint_03c2_report.md` and `checkpoints/03c3_finalist_ensembles/README.md` only as frozen First Modeling Delivery evidence.
9. `docs/FUNCTION_INDEX.md` before adding reusable functions.
10. `app/AGENTS.md` before app/deployment work.

Do not infer the active objective from old Checkpoint 03 text. Checkpoint 03 is historical.

## 2. Fixed-target problem

```text
ID:              ID_POLIGONO
target:          RENDIMIENTO_T_HA
split:           CONJUNTO
total parcels:   197
labeled:         138 ENTRENAMIENTO
targets:          59 PREDICCION
cycle:           April–October 2025
states:          Hidalgo, Puebla, Tlaxcala
active track:    competition
```

The exact 59 target parcels are already known through their covariates. The project therefore
optimizes reconstruction of those fixed 59 hidden reference yields rather than universal
future-parcel generalization.

All 197 X vectors may be used in X-only transductive learning. The 59 hidden y values are never
available and must not be acquired directly.

## 3. Reconstruction interpretation

Use the following mental model: the 59 parcel yields are missing or potentially untrusted
reported values. We want to reconstruct the latent/reference value for each parcel from all
independent evidence available about that specific parcel and its relationship to the 138
labeled parcels.

FIRA's reserved values are the external scoring reference. This is an optimization/evaluation
statement, not a claim that an agricultural measurement is metaphysically error-free.

## 4. Data structure

BASIC and PRO are longitudinal, not independent labeled samples.

```text
BASIC  107,666 × 96
       197 parcels
       2022–2025
       Sentinel-2 + Landsat
       23 index families × promedio/std/max/min

PRO     47,804 × 20
       197 parcels
       2025
       Planet
       NDVI/EVI/LAI/MSAVI × promedio/std/max/min
```

Feature Table v1 compresses this history to one row per parcel:

```text
parcel_features_competition.csv  197 × 1408
manifest total features          1403
competition-only features         709
```

Additional deterministic layers:

```text
Agronomic Features v1  197 × 351 = ID + 350 features
Empirical Features v1  197 × 336 = ID + 335 features
```

These tabular representations remain canonical inputs, but Checkpoint 04 may return to the
original longitudinal trajectories for time-series similarity, lag, graph and local-neighbor
methods.

## 5. What Checkpoint 03 accomplished

Checkpoint 03 is closed as the **First Modeling Delivery**.

It built/validated agronomic and empirical feature layers, compared representations, ran
competition-only nested model-family benchmarking and evaluated conservative equal-weight
ensembles.

Frozen reference evidence:

```text
C0_base + PLS                         state 0.5339   municipality 0.7621
C3_base_agro_plus_discovery + Ridge  state 0.5331   municipality 0.7650
C1_agronomic + CatBoost              state 0.5235   municipality 0.7709
E123 equal top 3                      state 0.5121   municipality 0.7481
E13 PLS + CatBoost                    state 0.5082   municipality 0.7488
```

Do not edit historical Checkpoint 03 configs/reports to improve those scores.

Checkpoint 03 used conservative inductive leakage rules. Those rules remain correct for
interpreting those historical scores, but they do not define active Checkpoint 04 transductive
validation.

## 6. Active validation doctrine

A Checkpoint 04 validation split must mimic the real task:

1. choose pseudo-target parcels from the 138 labels;
2. hide their y;
3. keep their X visible;
4. allow X-only transformations to use labeled + pseudo-target X;
5. ensure every y-aware operation sees only the remaining labeled y;
6. predict the pseudo-targets;
7. reveal y only for scoring.

Primary evidence should come from repeated pseudo-competition splits designed to resemble the
actual 59 target X distribution. State-stratified and municipality-grouped folds remain useful
stress tests.

This distinction is essential:

- X-only transductive use of target covariates is allowed and desirable;
- target-aware selection using hidden/pseudo-hidden y is not.

## 7. Checkpoint 04 plan

### 04A — target topology and support — COMPLETE

Canonical interpretation: `docs/CHECKPOINT_04A_FINDINGS.md`.

Key evidence: adversarial AUC 0.4892; geographic distance versus absolute yield difference
Spearman rho 0.4882; observed-y Moran's I 0.6797 (p=0.001); municipality-grouped E123 residual
Moran's I 0.5483 (p=0.001). Geography is the strongest 04A similarity-transfer signal and
monthly temporal similarity alone is weak.

### 04A — temporal similarity

Use original BASIC/PRO trajectories to test:

- Pearson/Spearman curve similarity;
- cross-correlation and best lag;
- phenological phase alignment;
- DTW or another justified sequence distance;
- whether greater trajectory similarity actually predicts smaller absolute yield difference.

Do not assume a time-series similarity metric transfers yield until this is measured on the 138
labeled parcels.

### 04B — pseudo-competition validation — COMPLETE

Canonical interpretation: `docs/CHECKPOINT_04B_FINDINGS.md`.

The primary 16 target-matched splits show LocalRidge k20 (RMSE 0.4946), Graph k8/lambda2
(0.4974), LocalRidge k30 (0.4976) and GeoKNN k10 (0.5021) as the strongest methods. Graph
k8/lambda2 is the most robust across validation families and reaches 0.5466 RMSE on the frozen
municipality-grouped stress protocol.

Do not treat the current support-tier routing CSV as final: tier winners were selected on the
same pseudo-test evidence and require out-of-sample routing validation.

### 04D.1 — local/graph refinement — COMPLETE

Canonical findings: `docs/CHECKPOINT_04D1_FINDINGS.md`.

```text
best Local Ridge target-matched   0.4878
LOSO method selection             0.4884
LOSO support routing              0.4858
robust graph grouped stress       0.5279
```

This stage reuses the exact 04B pseudo-target memberships and searches only the method families
supported by 04B. It adds a leakage-safe residual-graph model: cross-fitted PLS residuals from
visible labels are propagated over an X-only graph and added back to the full PLS anchor.

Selection quality is measured leave-one-target-matched-split-out. Support-tier routing is also
fitted on the other pseudo-splits and scored on the held-out split. Repeated splits share
parcels, so this removes direct same-split tuning optimism but is not equivalent to 16
independent datasets.

Do not promote the 59 candidate predictions from 04D.1 to final output until nested evidence,
stress robustness and later external/global-anchor experiments have been reviewed.

### 04C.1 — focused global anchor — COMPLETE

Canonical findings: `docs/CHECKPOINT_04C1_FINDINGS.md`.

CatBoost remained weaker than Local04D on the target-matched pseudo-competitions
(standalone RMSE about 0.515–0.521). A 10% Local+CatBoost blend was numerically
indistinguishable from Local04D (0.487842 vs 0.487845 mean split RMSE) and had
slightly worse pooled RMSE. The controlled LOSO gate reached 0.488884 and did
not improve on the frozen local rule. Residual correlation Local-vs-CatBoost
was still high at 0.9423.

CatBoost remains a sensitivity/diversity diagnostic only and is not a required
final component.

### 04D — target-specific/local/transductive models

Evaluate:

- spatial kNN;
- phenology/time-series kNN;
- learned mixed distances;
- local regression;
- graph/semi-supervised regression over all 197 parcels;
- covariate-shift / importance weighting;
- global-local mixtures of experts;
- parcel-specific blending based on support/uncertainty.

### 04E — external evidence

#### 04E.1 — SIAP localization — COMPLETE

Canonical findings: `docs/CHECKPOINT_04E1_FINDINGS.md`.

The exact `Cebada grano + Primavera-Verano + Temporal + CVEGEO + 2025` prior
covered all 197 parcels and all 59 targets, but it did not improve the primary
target-matched protocol. Local04D remained best at 0.4878 mean split RMSE;
Local+SIAP 25% worsened to 0.4922, and the controlled LOSO selector chose
Local04D in 16/16 holdouts. Direct SIAP RMSE on the 138 observed parcels was
1.4926 with Spearman correlation -0.2181.

The 25% graph/SIAP blend did help the municipality-grouped stress protocol
(0.5279 -> 0.5096), so SIAP remains documented external evidence and a stress
signal, but it is not a required current final-predictor component.



External public evidence is desirable.

Priority audit: SIAP 2025 at the most specific valid combination of:

```text
Cebada grano
relevant cycle
Temporal modality
correct municipality / CVEGEO
```

Do not mix `Cebada grano` with forage barley.

Also retain/evaluate CHIRPS, WaPOR, SoilGrids, CEM/topography, official climate and additional
daily weather/stress data when there is a concrete hypothesis.

### 04F — final reconstruction — COMPLETE

Canonical findings: `docs/CHECKPOINT_04F_FINDINGS.md`.

The final rule is frozen to `Baseline_Local04D`, corresponding to
`LocalRidge_C4_all_deterministic_Geo_k24_a30_p1`, with target-matched mean
RMSE 0.487845 and pooled RMSE 0.495741.

Cross-checkpoint provenance is exact: Local04D and Graph04D actual-target
predictions agree between 04D.1 and 04E.1 to maximum absolute difference 0.0.

A fixed 75/25 Local/Graph sensitivity blend reaches 0.486768 on the full
development table, but constrained LOSO weight selection reaches 0.487941 and
does not improve the frozen local rule. The blend remains diagnostic only.

The canonical competition output is
`reports/checkpoint_04f/final_predictions.csv`, containing exactly 59
`ID_POLIGONO,RENDIMIENTO_T_HA` rows. Do not modify parcel predictions
manually based on support or disagreement diagnostics.

## 8. Data and engineering invariants

- preserve `ID_POLIGONO`;
- preserve sensor provenance;
- no accidental many-to-many joins;
- explicit CRS handling before metric spatial operations;
- source files under `data/source/`, `docs/official/`, `docs/reference/` are immutable;
- stable reusable logic belongs in `src/geocebada/`;
- notebooks are exploratory/orchestration only;
- use `geocebada.paths`, not machine-specific absolute paths;
- add schema, ID, row-count and split assertions;
- prefer GPU for supported expensive model searches;
- update tests for stable implementation.

## 9. Known data caveats

- CHIRPS: only one daily competition feature survived Feature Table v1; investigate extraction/QC before judging signal.
- WaPOR: seasonal totals must remain duration weighted.
- SoilGrids: preserve frozen unit/scale conversion and CRS rules.
- Same-named indices across Sentinel-2/Landsat/Planet must not be silently merged.
- The old prospective `clean` cutoff question remains scientifically interesting but is not the active competition-reconstruction objective.

## 10. Documentation obligations

After substantive work:

- update `.ai_handoff`;
- update `data/.ai_handoff` for data discoveries;
- update `docs/PROJECT_HISTORY.md`;
- update the active checkpoint README (`checkpoints/05_global_local_mixture/README.md` while 05 is active);
- record material AI assistance in `docs/AI_USAGE.md`;
- regenerate `docs/FUNCTION_INDEX.md` after public API changes;
- follow `app/AGENTS.md` for Streamlit/deployment changes.

If historical documentation conflicts with the current doctrine, preserve the historical
artifact and add a clear historical/current distinction instead of silently rewriting old
results.


## 11. Checkpoint 05 — active final combination

Checkpoint 05 is intentionally separate from Checkpoints 03 and 04 because it
combines their surviving experts under one validation contract.

Global experts:
- PLS C0;
- Ridge C3 with fold-local discovery;
- CatBoost C1 with a three-seed prediction average after inner tuning;
- ExtraTrees C1.

Transductive experts:
- Local04D = C4 / Geo / k24 / alpha30 / p1;
- Graph04D = GeoAgro25 / k6 / lambda8.

Meta-training is based only on repeated cross-fitted base predictions. The new
primary hypothesis is a support-conditioned softmax mixture-of-experts, with
simpler stacking models retained as controls.

The incumbent remains Checkpoint 04F until the predeclared 05 promotion rule
passes. A lower all-split development score alone is insufficient.

Promotion has two validation layers after development ranking. First, the
leave-one-development-split-out method-selection procedure must improve both
mean and pooled RMSE relative to fixed Local04D. Only then is a fresh bank of
16 X-only target-matched masks generated with an unused seed. The challenger is
frozen before those pseudo-target yields are scored, and the fresh bank compares
only that challenger against Local04D. Promotion requires improvement in both
mean and pooled RMSE on the fresh bank as well.

The workstation runner supports `--preflight` and split-level `--resume`.
The final local bundle is
`models/final/checkpoint05_model.joblib`; the runner performs a serialization
round-trip verification before completion.
