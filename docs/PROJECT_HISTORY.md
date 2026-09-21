# GeoCebada project history

**Purpose:** preserve a chronological record of what was done, why it was done, what was
validated, and what remains open so future collaborators and AI agents do not repeat work or
silently overwrite earlier decisions.

**Last updated:** 2026-09-19  
**Current phase:** Checkpoint 04 — Transductive Competition Modeling. Checkpoint 03 is closed as the First Modeling Delivery.

This file is historical context. For current operating rules, read `AGENTS.md`,
`.ai_handoff` and `docs/AGENT_GUIDE.md`. If this history conflicts with immutable official
sources or directly inspected metadata, the source evidence wins.

---

## 2026-09-14 — Repository architecture and dataset understanding

The repository was reorganized around a reusable Python package, reproducible data contracts,
notebooks, documentation and a Streamlit application.

Key decisions established at this point:

- supervised unit = parcel, not satellite observation;
- canonical parcel key = `ID_POLIGONO`;
- canonical target = `RENDIMIENTO_T_HA`;
- official split = 197 parcels = 138 `ENTRENAMIENTO` + 59 `PREDICCION`;
- target cycle = April–October 2025;
- BASIC/PRO rows are longitudinal parcel/date/sensor observations, not independent supervised
  samples;
- sensor provenance must be preserved;
- hidden prediction targets must never be reconstructed or used as pseudo-ground-truth;
- learned preprocessing must ultimately be fitted inside validation folds;
- spatial autocorrelation must be treated as a validation risk.

The reusable library contract was established under `src/geocebada/`, with notebooks and
Streamlit expected to import shared behavior rather than duplicate it.

Documentation infrastructure was added:

- Sphinx + Furo;
- `docs/FUNCTION_INDEX.md` generated from Python AST;
- `AGENTS.md`, `.ai_handoff`, `data/.ai_handoff`;
- source-backed variable documentation;
- app-specific maintenance instructions.

The Streamlit application grew into a laboratory for data exploration, prediction-set
inspection, statistics, feature engineering and early model comparison.

---

## 2026-09-14 — Temporal alignment and exploratory tooling

Reusable temporal utilities were added to compare and align irregular satellite observations
using nearest-date weighting, cloud quality and optional GPU array operations.

These tools were intentionally exploratory. Parameters such as grid spacing, temporal
bandwidth and neighbor counts were not frozen as final modeling choices.

The key methodological point was preserved: repeated satellite rows cannot be randomly split
as if they were independent labels.

---

## 2026-09-16 — Streamlit dependency correction

A Streamlit Community Cloud failure caused by missing `plotly.express` was traced to dependency
file precedence. The project separated local Conda configuration into
`environment.dev.yml` and kept Cloud bootstrapping through `requirements.txt` and
`pyproject.toml`.

`app/AGENTS.md` became the app-specific maintenance contract so later changes do not
reintroduce the same deployment problem.

---

## 2026-09-16 to 2026-09-17 — External data acquisition

The project expanded beyond the official challenge files with public external sources relevant
to barley yield in Hidalgo, Puebla and Tlaxcala.

The prioritized local external set became:

- INEGI municipal boundaries;
- CHIRPS daily precipitation for the 2025 target season;
- SoilGrids;
- WaPOR AETI/NPP;
- higher-resolution INEGI CEM products;
- SIAP municipal agricultural history.

Large external raw files remain local and gitignored. A versionable manifest records their
presence, hashes and selected metadata.

Important source policies established:

- official parcel Estado/Municipio remains primary for administrative/SIAP joins;
- INEGI overlap is QA/fallback rather than silent replacement;
- SoilGrids source-specific CRS and unit conversion rules are explicit;
- WaPOR seasonal totals are duration-weighted;
- SIAP joins use full CVEGEO, never municipality ID alone;
- `Cebada grano` is kept distinct from forage barley;
- SIAP 2025 is a competition-mode contemporaneous proxy, not a clean prospective feature.

ERA5-Land remained optional and was not made part of the completeness gate.

---

## 2026-09-17 — Data Contract v2

A machine-checkable data contract was created in:

```text
configs/data_contract_v2.yaml
tools/audit_data_contract_v2.py
docs/DATA_CONTRACT_V2.md
```

The audit verifies IDs, split, schemas, dates, sensors, CRS, geometry validity, municipality
mapping, external-source inventories, SIAP semantics, SoilGrids scaling, CHIRPS continuity,
WaPOR metadata and CEM coverage.

After resolving SIAP aliases/duplicates and municipality policy, the full local audit closed
with:

```text
PASS = 46
WARN = 0
FAIL = 0
SKIP = 1
```

The single SKIP is intentional ERA5-Land policy.

This became the gate before full feature extraction.

---

## 2026-09-18 — Integration fixtures

A deterministic 12-parcel integration dataset was generated so the same join/extraction logic
could be tested end-to-end without the entire raw workstation dataset.

The fixture build passed and included BASIC, PRO, SIAP, climate, CHIRPS, SoilGrids, WaPOR and
CEM artifacts.

Checkpoint 01 — Data foundation was then recorded under:

```text
checkpoints/01_data/README.md
```

That checkpoint closes the acquisition/audit foundation.

---

## 2026-09-18 — Feature Table v1

A deterministic one-row-per-parcel feature builder was implemented through:

```text
configs/features_v1.yaml
src/geocebada/features/parcel.py
tools/build_parcel_feature_table.py
docs/FEATURE_TABLE_V1.md
```

The builder separates two representations:

**clean**
: static/historical features intended to support a defensible historical/prospective track.

**competition**
: clean features plus rule-permitted 2025/full-season/contemporaneous public covariates.

The full workstation build passed:

```text
197 parcels
138 training
59 prediction
1,408 total columns
694 clean features
709 competition-only features
1,403 manifest features total
WaPOR included
```

Independent CSV validation then confirmed:

```text
parcel_features_clean.csv       -> (197, 699)  PASS
parcel_features_competition.csv -> (197, 1408) PASS
parcel_features_all.csv         -> (197, 1408) PASS
```

All 138 training targets are populated and all 59 prediction targets remain missing.

Two non-fatal implementation/environment warnings were observed:

- pandas fragmentation warnings while constructing the daily CHIRPS matrix;
- h5py/HDF5 runtime/build-version mismatch warning.

Neither warning caused the validated build to fail, but both remain legitimate maintenance
items.

A large number of generated candidate columns were all-null and therefore dropped. Future
changes must inspect source/provenance before assuming every dropped sensor/index combination
is physically meaningful.

---

## 2026-09-18 — Checkpoint 02: diagnostics and fixed validation

Checkpoint 02 was implemented through:

```text
configs/checkpoint02.yaml
src/geocebada/evaluation/parcel_modeling.py
tools/build_checkpoint_02.py
tests/test_checkpoint02.py
checkpoints/02_features/README.md
```

The pipeline froze four modeling prerequisites before serious tuning:

1. feature-family inventory;
2. train-versus-prediction covariate diagnostics;
3. fixed validation folds;
4. interpretable feature-family ablations.

A fifth requirement was added: every result and design choice must be historically documented.

The local test suite passed:

```text
53 passed
```

The full Checkpoint 02 pipeline passed and generated versioned artifacts under
`reports/checkpoint_02/`.

### Feature inventory observed

```text
clean:
  admin                8  (6 numeric, 2 non-numeric)
  basic              539
  cem                  5
  climate_official    24
  geometry             6
  siap                14
  soilgrids           90
  topography           8

competition-only:
  basic              539
  chirps               1
  climate_official    45
  pro                  97
  siap                  5
  wapor                22
```

There are 1,401 numeric features out of 1,403 manifest features.

A notable QC point: only **one CHIRPS daily competition feature survived** into the final
feature table. Most expected CHIRPS daily aggregate columns were dropped as all-null during the
build. Do not claim that the final table contains a rich CHIRPS block until this extraction
behavior is explicitly investigated.

### Train versus prediction shift

Only 6 of 1,401 numeric features crossed at least one configured descriptive support/shift
threshold.

Four historical Sentinel-2 August variables had absolute SMD slightly above 0.5. One SoilGrids
feature and one 2025 BASIC feature had just over 10% of prediction values outside the observed
training range.

No reported FDR-adjusted KS p-value among those six was below 0.05.

These diagnostics are descriptive only. They are not labels and must not become a covert
prediction-set feature-selection mechanism.

### Frozen validation protocols

Two five-fold protocols were frozen with random seed 42:

`fold_state_stratified`
: primary protocol preserving state composition.

`fold_municipality_grouped`
: robustness protocol keeping each state/municipality group entirely within one fold.

Later model comparisons must reuse `reports/checkpoint_02/cv_folds.csv` rather than regenerate
folds to improve a result.

### Frozen ablations

```text
A0  geometry + administrative numeric fields                 12
A1  A0 + soil/topography/CEM                                115
A2  A1 + historical SIAP                                    129
A3  A2 + historical official climate                        153
A4  A3 + historical BASIC                                   692
A5  all clean numeric                                       692
A6  A5 + 2025 remote/weather, excluding SIAP 2025          1396
A7  all numeric competition, including SIAP 2025           1401
```

A4 and A5 have the same numeric feature count because A4 already contains all clean numeric
families; the remaining clean manifest fields are the two non-numeric administrative columns.

### Initial baseline evidence

Checkpoint 02 deliberately used only untuned DummyMean, fixed-alpha Ridge and regularized
ExtraTrees.

State-stratified validation was much easier than municipality-grouped validation.

Representative results:

```text
state-stratified:
  ExtraTrees A7  RMSE 0.4966, MAE 0.3574, mean R² 0.6498
  ExtraTrees A6  RMSE 0.5001
  ExtraTrees A2  RMSE 0.5027
  DummyMean       RMSE 0.8506

municipality-grouped:
  Ridge A3        RMSE 0.7426, MAE 0.6459
  Ridge A6        RMSE 0.8178
  ExtraTrees A6   RMSE 0.8835
  ExtraTrees A5   RMSE 0.9073
  DummyMean       RMSE 1.0066
```

The large gap between the protocols is evidence of meaningful spatial/municipal generalization
sensitivity. Future work must not optimize only the easier state-stratified metric.

Adding SIAP 2025 from A6 to A7 produced only small changes in the baseline runs and did not show
a large consistent robustness gain.

Checkpoint 02 is therefore closed.

---

## 2026-09-18 — Feature Table v1 promoted to Git

The project deliberately changed its earlier blanket policy that all processed data remain
gitignored.

The canonical model-ready artifacts are now versioned:

```text
data/processed/features_v1/
├── parcel_features_clean.csv
├── parcel_features_competition.csv
├── parcel_features_all.csv
├── feature_manifest.json
├── build_report.json
└── README.md
```

This means a collaborator or AI agent can clone the private repository and run modeling from
the exact 197-row tables without downloading all large external raw sources.

Large `data/raw/`, `data/interim/` and non-canonical processed artifacts remain ignored.

Commit `30136fc` added the five canonical generated artifacts to `main`.

---

## 2026-09-18 — Checkpoint 03A: Agronomic Features v1

Checkpoint 03 opened with a deliberate feature-engineering phase before serious model tuning.

The design decision was to leave Feature Table v1 frozen and create a separate additive layer:

```text
data/processed/agronomic_features_v1/
├── parcel_agronomic_features.csv
├── feature_manifest.json
├── build_report.json
└── README.md
```

The layer was designed before looking at target performance. It encodes compact nonlinear
agronomic hypotheses instead of generating all pairwise products of the ~1,400 base numeric
features.

Implemented families include phenology/seasonal curve shape, 2025-versus-history satellite
anomalies, Sentinel-2/Planet agreement, thermal-time proxies, rainfall timing, WaPOR
water-productivity proxies, soil vertical contrasts, soil interactions, cross-domain
environment/crop interactions and a small set of log basis functions.

Scientific rationale and formulas are documented in `docs/AGRONOMIC_FEATURES_V1.md`; the
retrieved literature is versioned in `docs/references/agronomic_features_v1.bib`.

Evidence is explicitly classified as `literature_backed`, `mechanistic_proxy` or
`experimental`. Experimental formulas are predictive hypotheses, not causal/agronomic laws.

The canonical build passed with:

```text
197 parcels
350 derived features
123 clean
227 competition
0.0 mean/max missing fraction
0 infinite values
target used for construction: false
CHIRPS used: false
```

Family inventory:

```text
phenology              162
phenology_anomaly       41
water_productivity      40
thermal                 36
cross_domain            19
soil_profile            18
water_timing            10
sensor_agreement         9
soil_interaction         8
nonlinear_basis          7
```

Three candidate features based on a 25 C monthly-mean heat hinge were constant across all 197
parcels and were dropped automatically. This means only that the current coarse monthly climate
representation has no cross-parcel variation for that basis; it is not evidence that barley
heat stress is irrelevant.

CHIRPS was intentionally excluded because Checkpoint 02 retained only one CHIRPS feature and
that source still needs QC.

No yield value was used to create or select these 350 features. Any future target-guided
interaction search or feature selection must occur inside training folds.

Phase 03A is therefore closed. Its output feeds the separate 03B empirical-discovery phase before model comparison.

---

## 2026-09-18 — Checkpoint 03B: Empirical Feature Discovery

Before model comparison, the project inserted a second feature-discovery phase to distinguish
agronomic hypotheses from mathematical structure learned from the observed covariates.

A third canonical processed layer was added:

```text
data/processed/empirical_features_v1/
├── parcel_empirical_features.csv
├── feature_manifest.json
├── build_report.json
└── README.md
```

The materialized layer is entirely X-only and contains temporal-shape geometry,
short-baseline historical condition, symmetric 2025-versus-history change,
Sentinel-2/Planet shape similarity and scale-normalized distribution geometry.

Canonical build:

```text
197 parcels
335 derived features
91 clean
244 competition
0.0 mean/max missing fraction
0 infinite values
target used for materialized features: false
supervised formulas materialized globally: false
short VCI-like is standard VCI: false
CHIRPS used: false
```

Family inventory:

```text
temporal_shape              162
aggregate_geometry           72
symmetric_change             42
short_baseline_condition     41
sensor_shape_similarity      18
```

One `emp_histpos__landsat_vi6t__above_history_fraction` candidate was constant across all
197 parcels and was dropped. This is a property of that representation, not a biological
conclusion.

The NDVI historical-range normalization is deliberately named
`emp_vci_like_short__*`. Literature on VCI/TCI/VHI motivated the historical-range idea,
but the project has only a 2022–2024 parcel-level remote-sensing baseline, so it is not
reported as standard climatological VCI. Likewise, CWSI was not reconstructed because the
available temperature fields are air-temperature products rather than the canopy-temperature
wet/dry reference system required by CWSI.

The target-aware part of 03B is implemented as `FoldLocalExpressionMiner`. It starts from 24
configured source/agronomic/empirical primitives, ranks primitives and simple pairwise formulas
using only training-fold yield, then applies the selected formulas to validation rows without
reading validation targets. Candidate operations are products, sums, oriented differences,
safe oriented ratios and symmetric relative change.

This controlled search is a first step toward symbolic-regression-style index discovery. A
candidate formula is not considered a new barley-yield index merely because it correlates with
training yield; recurrence across independent fold fits and held-out contribution, especially
under municipality-grouped validation, are required before promoting any formula.

PCA, PLS, clustering and kernels remain fold-fitted model-layer transforms and were not
materialized globally.

Scientific/methodological context is documented in `docs/EMPIRICAL_FEATURES_V1.md`, with
references versioned in `docs/references/empirical_features_v1.bib`.

Checkpoint 03B closed after the workstation run passed. The fold-local audit selected 88 unique expressions; seven raw expressions recurred in at least 3/5 folds under both protocols, but they collapse to four conceptual motifs because several are monotone ratio reparameterizations. Interpretation is recorded in `reports/checkpoint_03b/README.md`.

---

## 2026-09-18 — Checkpoint 03C.1: Representation benchmark

03C.1 compared seven feature representations in both clean and competition tracks under the
exact frozen Checkpoint 02 state-stratified and municipality-grouped folds. To isolate
representation effects, only fixed Ridge10 and ExtraTrees baselines were used.

The workstation run completed 280 outer fits and produced 7,728 out-of-fold predictions.
CHIRPS remained excluded pending QC; no hidden prediction target was used.

Key OOF RMSE findings:

```text
ExtraTrees clean:
  state:   B7 discovery 0.4948 vs B0 base 0.5119
  grouped: B7 discovery 0.8488 vs B0 base 0.9431

ExtraTrees competition:
  state:   B7 discovery 0.4992 vs B0 base 0.5003
  grouped: B1 agronomic 0.8590 vs B0 base 0.8974

Ridge10 competition grouped:
  B0 base 0.8059
```

Interpretation:

- agronomic-only features preserve substantial signal with much lower dimensionality;
- empirical-only features are consistently weak as a standalone representation;
- concatenating every deterministic feature is not reliably beneficial;
- fold-local expression discovery can improve tree models, especially in the clean track;
- fixed Ridge is highly representation-sensitive, motivating nested regularization tuning.

03C.2 should therefore use a compact candidate set rather than carry all seven representations:
B0 base, B1 agronomic-only, B7 all+discovery, and a new base+agronomic+discovery ablation that
removes the deterministic empirical-only block.

Full interpretation is in `reports/checkpoint_03c/README.md`.

---
## 2026-09-18 — Checkpoint 03C.2 implementation: competition-only nested modeling

After 03C.1, the modeling contract was simplified: **only the competition track matters for
current development and final challenge prediction**. The clean label remains in the repository
for provenance/history but is no longer an active model-selection track unless explicitly
reopened.

03C.2 was implemented as a nested model-family benchmark over four competition-only
representations:

```text
C0 = base
C1 = agronomic-only
C2 = base + agronomic + empirical + fold-local discovery
C3 = base + agronomic + four empirical discovery-support primitives
     + fold-local discovery
```

C3 was added specifically to test whether the large deterministic empirical block is unnecessary
once the fold-local discovery miner can still access the four empirical support primitives.

The frozen Checkpoint 02 folds remain the outer evaluation:

```text
fold_state_stratified
fold_municipality_grouped
```

Within each outer training fold, 3-fold inner CV selects hyperparameters. Inner splitting mirrors
the outer protocol: state-stratified inner CV for the state protocol and municipality-grouped
inner CV for the municipality protocol. Imputation, scaling, PCA, PLS, expression discovery and
hyperparameter selection all remain inside the training folds.

Implemented model families:

```text
Ridge
ElasticNet
ExtraTrees
CatBoost
PLS
PCA + RBF Kernel Ridge
PCA + degree-2 Polynomial Kernel Ridge
```

CatBoost defaults to GPU on the university workstation and has a guarded CPU fallback for
GPU/CUDA availability failures.

The implementation is versioned in:

```text
configs/checkpoint03c2.yaml
src/geocebada/evaluation/checkpoint03c2.py
tools/run_checkpoint_03c2.py
checkpoints/03c2_competition_modeling/README.md
```

No final model has been selected. No final 59-parcel predictions have been generated.
The workstation benchmark and interpretation of its generated reports are the next milestone.

---

## Current state at Checkpoint 03C.2

Completed:

- official data understanding and source-backed variable dictionary;
- reusable library/repository architecture;
- Streamlit exploratory application and deployment contract;
- external-source acquisition workflow;
- Data Contract v2;
- 12-parcel integration fixtures;
- full Feature Table v1;
- clean versus competition provenance;
- fixed CV folds;
- train/prediction shift diagnostics;
- feature-family ablations;
- initial untuned baseline models;
- versioned Feature Table v1;
- versioned Agronomic Features v1 with 350 target-free derived variables and scientific provenance;
- versioned Empirical Features v1 with 335 target-free derived variables;
- fold-local target-aware expression-discovery infrastructure with 24 configured primitives;
- closed Checkpoint 03C.1 representation benchmark with 280 outer fits and 7,728 OOF predictions;
- competition-only modeling directive from 03C.2 onward;
- implemented 03C.2 nested-CV model benchmark over C0/C1/C2/C3;
- implemented Ridge, ElasticNet, ExtraTrees, CatBoost, PLS, PCA+RBF KRR and PCA+Poly2 KRR candidate families;
- repository CI green for the current 03C.2 implementation.

Not completed:

- no final model is selected;
- the 03C.2 workstation benchmark has not yet been executed;
- no final hyperparameters are frozen;
- no final preprocessing/feature-selection pipeline is frozen;
- no final 59 predictions have been produced;
- the exact operational pre-harvest prediction cutoff remains unresolved;
- CHIRPS daily extraction contribution needs QC because only one feature survives;
- the spatial validation gap needs to drive Checkpoint 03 decisions.

The next milestone is the **Checkpoint 03C.2 workstation run** using `python tools/run_checkpoint_03c2.py`, followed by interpretation of `reports/checkpoint_03c2/` and selection of a small finalist set. Final 59-parcel prediction comes later.


---

## 2026-09-19 — Checkpoint 03C.2 workstation benchmark completed

The full competition-only nested benchmark completed and the generated artifacts were committed
under `reports/checkpoint_03c2/`.

Run facts:

```text
training rows          138
outer fits             280
OOF prediction rows    7,728
inner candidate rows   1,640
discovery rows         2,800
CatBoost task type     GPU
hidden targets used    false
```

The strongest individual candidates by worst-protocol OOF RMSE were:

```text
C0_base + PLS                         state 0.5339   grouped 0.7621
C3_base_agro_plus_discovery + Ridge  state 0.5331   grouped 0.7650
C1_agronomic + CatBoost              state 0.5235   grouped 0.7709
```

ExtraTrees remained strongest on the easier state-stratified protocol but was materially weaker
under municipality-grouped validation, so it was not selected for the robust finalist set.

## 2026-09-19 — Checkpoint 03C.3 finalist equal-weight ensembles

To avoid another high-variance tuning layer with only 138 labels, 03C.3 uses only the three
reviewed individual finalists and fixed equal-weight combinations of their committed OOF
predictions. No continuous ensemble-weight optimization is allowed.

The four pre-specified ensembles are E12 = PLS+Ridge, E13 = PLS+CatBoost,
E23 = Ridge+CatBoost and E123 = PLS+Ridge+CatBoost.

Reviewed OOF results:

```text
E123_equal_top3      state 0.5121   grouped 0.7481
E13_PLS_CatBoost     state 0.5082   grouped 0.7488
E23_Ridge_CatBoost   state 0.5146   grouped 0.7544
E12_PLS_Ridge        state 0.5270   grouped 0.7546
```

E13 and E123 are both retained. Their municipality-grouped RMSE differs by less than 0.001, so
the project does not interpret that development-score difference as evidence for a unique
winner. The next step is to freeze a full-data fitting rule for the retained finalist(s) before
producing the 59 challenge predictions.


---

## 2026-09-19 — First Modeling Delivery closed; project reframed as transductive reconstruction

Checkpoint 03C.2/03C.3 were reviewed as a deliberately controlled first modeling delivery rather
than a final solution. The conventional nested-CV evidence retained three diverse individual
models — C0+PLS, C3+Ridge and C1+CatBoost — and two equal-weight ensembles, with E123 at
0.5121 state-stratified / 0.7481 municipality-grouped RMSE and E13 at 0.5082 / 0.7488.

The project then changed its active modeling doctrine. The actual competition exposes the
covariates of the exact 59 parcels that must be scored. Therefore the central task is now
treated as fixed-target transductive regression: use X for all 197 parcels and y for the
138 labeled parcels to reconstruct the 59 hidden FIRA reference yields.

The working interpretation is analogous to receiving 59 missing or potentially untrusted yield
reports and attempting to reconstruct their latent values from independent evidence. FIRA's
reserved yield values are the external scoring reference. They remain unavailable and are not
to be obtained directly.

Checkpoint 04 was opened with six workstreams: target-set topology/similarity; transductive
pseudo-competition validation; higher-compute global models; local/graph/domain-adaptation
models; external-data enrichment; and final transductive ensemble/reconstruction.

Important methodological consequences:

- pseudo-validation hides y but keeps pseudo-target X visible;
- the 59 real target X vectors may inform X-only representations;
- spatial/temporal/phenological similarity and autocorrelation become first-class modeling questions;
- Feature Table v1 remains useful but original BASIC/PRO longitudinal trajectories are reopened as direct similarity/modeling inputs;
- SIAP 2025 is elevated as a priority source, especially the most specific valid combination of `Cebada grano`, relevant cycle, `Temporal` modality and municipality;
- the older state/municipality folds remain stress tests, while target-matched repeated pseudo-competition RMSE becomes the primary evidence for new methods.

Canonical specification: `docs/TRANSDUCTIVE_OBJECTIVE.md`.  
Active checkpoint: `checkpoints/04_transductive_competition/README.md`.


---

## 2026-09-20 — Checkpoint 04A implementation: fixed-target topology and support

The first transductive diagnostic implementation was added before any higher-compute model
search. 04A is intentionally diagnostic rather than predictive.

Implemented:

```text
configs/checkpoint04a.yaml
src/geocebada/evaluation/checkpoint04a.py
src/geocebada/visualization/checkpoint04a.py
tools/run_checkpoint_04a.py
tests/test_checkpoint04a.py
```

The runner studies the 59 target parcels against the 138 labeled parcels using geographic
nearest neighbors, three deterministic transductive PCA spaces, monthly 2025 temporal
similarity across Sentinel-2/Landsat/Planet signals, best lag correlation, normalized DTW,
adversarial train-vs-target classification, per-feature shift, train-pair similarity versus
absolute yield difference, Moran's I for observed yield and frozen Checkpoint 03 ensemble
residuals, and a composite target-support taxonomy.

The temporal v1 analysis deliberately reuses the canonical April-October monthly trajectories
already extracted into Feature Table v1. This provides reproducible sensor-aware temporal
topology from a clean clone. Capture-level BASIC/PRO alignment can be added later if the
monthly results justify greater temporal resolution.

The workstation command is:

```powershell
python tools\run_checkpoint_04a.py
```

No numerical 04A conclusions are recorded yet. The generated report and figures must be
reviewed after the workstation run before 04B pseudo-competition validation is designed.


---

## 2026-09-20 — Checkpoint 04A completed: target topology and support

The first transductive workstation run was committed in `77f5b90a3288dc7365902ed918baa6b132d3ec52`.
Generated artifacts live in `reports/checkpoint_04a/`; canonical interpretation is
`docs/CHECKPOINT_04A_FINDINGS.md`.

The strongest empirical finding is spatial structure. Across all 9,453 labeled-labeled pairs,
geographic distance has Spearman rho 0.4882 with absolute yield difference, stronger than
agronomic PCA distance (0.2200), all-deterministic PCA distance (0.1833) or the monthly
temporal metrics (roughly 0.05-0.08 in magnitude). Observed yield has Moran's I 0.6797 with
permutation p=0.001.

Checkpoint 03 state-stratified ensemble residuals show no significant spatial autocorrelation,
but municipality-grouped E13/E123 residuals remain strongly spatially autocorrelated
(Moran's I 0.5589/0.5483, both p=0.001). This means local context matters most when an
administrative/spatial neighborhood is absent from model training.

The adversarial train-vs-target classifier has AUC 0.4892, so the 59 targets do not form a
globally distinguishable covariate population in the tested representation. Support is still
heterogeneous: 5 high-support, 2 feature-supported, 5 geographically supported, 13
extrapolation and 34 mixed-support targets.

04A therefore closes with a target-specific doctrine: use local/spatial evidence aggressively
where pseudo-competition RMSE validates it, but do not apply the same local trust to every
target. 04B becomes the active phase.


---

## 2026-09-20 — Checkpoint 04B implemented: target-matched pseudo-competition validation

After 04A established strong spatial yield structure, weak global train-target shift and
heterogeneous parcel-level support, the repository added the first validation layer that
matches the actual information structure of the fixed 59 targets.

04B hides only y for pseudo-target parcels. All 197 X rows remain available to X-only PCA,
distance, support and graph construction. Target-aware estimators receive only the pseudo-train
labels.

The primary split family contains 16 repeated 41-parcel pseudo-target masks. Forty-one is the
59/197 competition target fraction scaled to the 138-label universe. Mask construction uses
only state/municipality plus X-derived geographic, agronomic, temporal and adversarial profile
descriptors and explicitly excludes neighbor-y consistency.

A secondary state-random family and the frozen Checkpoint 02 state-stratified and
municipality-grouped folds are retained as stress tests.

The lightweight 04B suite compares group means, geographic/agronomic/mixed kNN, transductive
Ridge, PLS, query-specific local Ridge, graph-Laplacian regression and fixed global/local
blends. The run also recomputes dynamic X-only support inside every pseudo split and learns
support-tier method routing for the actual 59 targets without producing their yields.

Workstation command:

```powershell
python tools\run_checkpoint_04b.py
```

No 04B numerical conclusions are recorded until that run completes.


---

## 2026-09-20 — Checkpoint 04B completed: local/graph methods survive pseudo-competition

The 04B workstation run was committed as `f9ad005d51bcbe488cfad96c6a7ea8434ab6709d`.
Generated artifacts live in `reports/checkpoint_04b/`; canonical interpretation is
`docs/CHECKPOINT_04B_FINDINGS.md`.

The 16 target-matched 41-parcel pseudo-competitions identify LocalRidge k20/alpha100
(RMSE 0.4946), Graph k8/lambda2 (0.4974), LocalRidge k30/alpha100 (0.4976) and GeoKNN k10
(0.5021) as the strongest tested methods. GlobalPLS C0 with four components reaches 0.5379 in
the same primary protocol.

Graph k8/lambda2 is the most robust candidate across all validation families and reaches
0.5466 RMSE on the frozen municipality-grouped stress protocol. This is consistent with the
strong spatial autocorrelation discovered in 04A and makes graph/local refinement the main
active direction.

The target-matched split generator also worked as intended: its mean municipality-TV distance
to the real target set is 0.0547 versus 0.0889 for state-random masks, and its standardized
profile-mean distance is 0.5369 versus 0.9227.

The initial support-tier routing is retained only as an exploratory hypothesis. It must be
validated with out-of-split method selection before it can influence final reconstruction.


---

## 2026-09-20 — Checkpoint 04D.1 implemented: local/graph refinement and nested routing

After 04B showed LocalRidge, graph-Laplacian and geographic kNN outperforming the tested global
anchors, the repository implemented a focused refinement stage rather than a generic model
sweep.

04D.1 reuses the exact committed 04B pseudo-competition memberships. The primary target-matched
grid contains 626 predefined methods: 300 graph variants, 324 local-Ridge variants and two
fixed anchors. Graph search varies geography/agronomy topology, k and Laplacian
regularization. Local Ridge varies C1/C4 representation, distance topology, neighborhood size,
Ridge alpha and inverse-distance power.

A new residual-graph strategy fits PLS4 on visible pseudo-training labels, obtains residuals by
cross-fitting inside those visible labels, propagates that residual field over the 197-node
X-only graph, and adds the correction to the full global anchor prediction.

To reduce the routing optimism identified in 04B, 04D.1 evaluates both global method selection
and support-tier routing leave-one-target-matched-split-out. Finalists only are then scored on
state-random and frozen state/municipality stress protocols.

The runner also materializes candidate predictions from finalists for the actual 59 targets.
These candidates are diagnostics for later blending and are not a final submission.


## 2026-09-20 — Checkpoint 04D.1 completed; 04E.1 SIAP localization implemented

The workstation completed 04D.1 and committed the generated evidence. The
626-method refinement found a stable local optimum around geographic Local Ridge
(k 20–30, alpha 30), with best target-matched mean RMSE 0.48785. LOSO method
selection remained at 0.48837 and LOSO support-tier routing at 0.48581. The
direct GeoAgro25 graph (k=6, lambda=8) was the strongest robustness hedge, with
municipality-grouped RMSE 0.52788.

The next stage was narrowed to a direct test of public SIAP 2025 localization.
04E.1 now loads detailed SIAP rows without collapsing production cycle or
modality, audits `Cebada grano + Primavera-Verano + Temporal + CVEGEO`, keeps
broader grain scopes as explicit fallbacks, and evaluates direct/calibrated,
local-residual, graph-residual and fixed-blend uses on the exact frozen 04B
pseudo-competition splits. Actual-target outputs remain candidates until 04F.


---

## 2026-09-21 — Checkpoint 04E.1 completed; focused Checkpoint 04C.1 implemented

The workstation completed the SIAP localization experiment with the full local SIAP
archive available. Exact `Cebada grano + Primavera-Verano + Temporal + CVEGEO + 2025`
coverage reached 197/197 parcels and all 59 actual competition targets.

The external municipal proxy did not improve the primary target-matched protocol.
Local04D reproduced its 0.48785 mean split RMSE, while the 25% Local+SIAP blend
reached 0.49218. The direct SIAP prior reached 1.43684 target-matched RMSE; on the
138 observed parcels its direct RMSE was 1.4926 and Spearman correlation with parcel
yield was -0.2181. The controlled leave-one-target-matched-split-out selector chose
Local04D in all 16 holdouts. A 25% graph/SIAP blend did improve the
municipality-grouped stress score from 0.52788 to 0.50965, so SIAP is retained as
external-evidence and stress-test context rather than a required final component.
Canonical interpretation is in `docs/CHECKPOINT_04E1_FINDINGS.md`.

The deferred global-model step was then narrowed to Checkpoint 04C.1. Instead of
reopening a broad model-family search, 04C.1 tests whether the historically relevant
C1 agronomic CatBoost signal adds complementary information to the stronger 04D.1
local and graph predictors. Three CatBoost configurations are inherited from the
frozen 03C.2 grid and averaged across seeds 42, 314 and 2718. The stable shallow
candidate is blended at fixed 10/20/30% weights with Local04D and Graph04D.

04C.1 reuses the exact frozen 04B pseudo-competition memberships, computes residual
correlation, and applies a deliberately small LOSO gate whose candidate universe is
fixed in advance. GPU is the default and automatic CPU fallback is disabled. Actual
59-target predictions are generated only as candidates for 04F; hidden FIRA y remains
unavailable and is never scored.
