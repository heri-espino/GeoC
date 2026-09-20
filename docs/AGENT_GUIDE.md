# GeoCebada agent guide

**Current phase:** Checkpoint 04B — transductive pseudo-competition validation (implemented; workstation run pending)  
**Canonical objective:** `docs/TRANSDUCTIVE_OBJECTIVE.md`

This is the operational entry point for any AI agent, collaborator or new contributor.

## 1. Read order

Before changing modeling or data logic, read:

1. `docs/TRANSDUCTIVE_OBJECTIVE.md`.
2. `.ai_handoff`.
3. this file.
4. `checkpoints/04_transductive_competition/README.md`.
5. `docs/PROJECT_HISTORY.md`.
6. `data/.ai_handoff` and `docs/DATA_SOURCES.md`.
7. `reports/checkpoint_03c2/checkpoint_03c2_report.md` and `checkpoints/03c3_finalist_ensembles/README.md` only as frozen First Modeling Delivery evidence.
8. `docs/FUNCTION_INDEX.md` before adding reusable functions.
9. `app/AGENTS.md` before app/deployment work.

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

### 04B — pseudo-competition validation — IMPLEMENTED, RUN NEXT

Run:

```powershell
python tools\run_checkpoint_04b.py
```

04B creates repeated 41-parcel target-matched pseudo-target masks using only X-derived profile
information, plus state-random and frozen state/municipality stress protocols. It compares
global means, municipality shrinkage, geographic/agronomic/mixed kNN, Ridge, PLS, local Ridge,
graph-Laplacian regression and fixed blends.

The dynamic support variable used for method routing is X-only. Do not reuse the 04A composite
support score for validation routing because 04A also included consistency of labeled-neighbor y.

### 04C — stronger global models

The 03C.2 search was deliberately small. Revisit CatBoost with a larger GPU budget, more
iterations, early stopping, stronger regularization/search and multiple seeds. Keep PLS/Ridge
because they were genuinely competitive with only 138 labels.

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

### 04F — final reconstruction

Only after pseudo-competition validation is stable:

- compare surviving global/local/graph/domain-adaptation methods;
- blend/stack with transductive OOF evidence;
- fit using all 138 labels and all 197 X;
- output exactly 59 `ID_POLIGONO, RENDIMIENTO_T_HA` predictions;
- freeze provenance and uncertainty diagnostics.

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
- update `checkpoints/04_transductive_competition/README.md`;
- record material AI assistance in `docs/AI_USAGE.md`;
- regenerate `docs/FUNCTION_INDEX.md` after public API changes;
- follow `app/AGENTS.md` for Streamlit/deployment changes.

If historical documentation conflicts with the current doctrine, preserve the historical
artifact and add a clear historical/current distinction instead of silently rewriting old
results.
