# GeoCebada project history

**Purpose:** preserve a chronological record of what was done, why it was done, what was
validated, and what remains open so future collaborators and AI agents do not repeat work or
silently overwrite earlier decisions.

**Last updated:** 2026-09-18  
**Current phase:** Checkpoint 02 closed; Checkpoint 03 — Modeling is the next phase.

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
351 derived features
123 clean
228 competition
0.0 mean/max missing fraction
0 infinite values
target used for construction: false
CHIRPS used: false
```

Family inventory:

```text
phenology              162
phenology_anomaly       42
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

No yield value was used to create or select these 351 features. Any future target-guided
interaction search or feature selection must occur inside training folds.

Phase 03A is therefore closed. Checkpoint 03B will compare base-only, agronomic-only,
base+agronomic and controlled reduced/selected representations under the frozen
state-stratified and municipality-grouped folds.

---

## Current state after Checkpoint 03A

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
- versioned Agronomic Features v1 with 351 target-free derived variables and scientific provenance.

Not completed:

- no final model is selected;
- no final hyperparameters are frozen;
- no final preprocessing/feature-selection pipeline is frozen;
- no final 59 predictions have been produced;
- the exact operational pre-harvest prediction cutoff remains unresolved;
- CHIRPS daily extraction contribution needs QC because only one feature survives;
- the spatial validation gap needs to drive Checkpoint 03 decisions.

The next milestone is **Checkpoint 03B — model/representation comparison** using the frozen Checkpoint 02 folds.
