# GeoCebada agent guide

This is the operational entry point for any AI agent, collaborator or new contributor taking
over GeoCebada work.

Read this file before changing modeling/data logic. It tells you what is already done, which
files are authoritative, what can be changed, what must not be redone, and where the project
currently stands.

**Current phase:** Checkpoint 03 is open. Phases 03A (Agronomic Features v1) and 03B (Empirical Features v1) are closed; start from 03C — model/representation comparison.

---

## 1. Read order

Read in this order before substantial work:

1. `AGENTS.md` — repository-wide rules.
2. `.ai_handoff` — compact current-state handoff.
3. `docs/PROJECT_HISTORY.md` — what happened and why.
4. `checkpoints/01_data/README.md` — closed data-foundation milestone.
5. `checkpoints/02_features/README.md` — closed feature/validation milestone.
6. `checkpoints/03_modeling/README.md` — 03A/03B closed; 03C current modeling milestone.
7. `checkpoints/03b_empirical_feature_discovery/README.md` — closed empirical-discovery contract.
8. `data/processed/features_v1/README.md`, `data/processed/agronomic_features_v1/README.md` and `data/processed/empirical_features_v1/README.md` — canonical representations.
9. `reports/checkpoint_02/checkpoint_02_report.md` — frozen diagnostics/baselines/folds.
10. `docs/AGRONOMIC_FEATURES_V1.md` and `docs/EMPIRICAL_FEATURES_V1.md` before changing derived variables.
11. `docs/DATA_SOURCES.md`, `docs/DATA_CONTRACT_V2.md`, `docs/FEATURE_TABLE_V1.md`.
12. `docs/FUNCTION_INDEX.md` before adding reusable code.
13. `app/AGENTS.md` before touching Streamlit/deployment.

Do not start by re-deriving facts from scratch unless a source changed or a validation check
fails.

---

## 2. Core challenge facts

The prediction unit is one barley parcel.

```text
ID column:       ID_POLIGONO
target:          RENDIMIENTO_T_HA
split column:    CONJUNTO
total parcels:   197
training:        138
prediction:       59
target cycle:    April–October 2025
states:          Hidalgo, Puebla, Tlaxcala
```

The 59 prediction targets are hidden by FIRA and remain missing in every canonical table.

Never reconstruct, impute, infer or treat hidden targets as pseudo-ground-truth.

BASIC and PRO are longitudinal remote-sensing tables. Their rows are not independent target
samples.

---

## 3. Repository map

```text
GeoCebada/
├── AGENTS.md
├── .ai_handoff
├── checkpoints/
│   ├── 01_data/
│   ├── 02_features/
│   ├── 03_modeling/
│   └── 03b_empirical_feature_discovery/
├── configs/
│   ├── base.yaml
│   ├── data_contract_v2.yaml
│   ├── features_v1.yaml
│   ├── checkpoint02.yaml
│   ├── agronomic_features_v1.yaml
│   └── empirical_features_v1.yaml
├── data/
│   ├── source/                       # immutable official challenge sources
│   ├── raw/                          # local/generated, mostly gitignored
│   ├── interim/                      # local/generated, gitignored
│   ├── samples/integration/          # committed integration fixtures
│   ├── processed/features_v1/        # committed canonical source-derived tables
│   ├── processed/agronomic_features_v1/ # committed target-free nonlinear layer
│   └── processed/empirical_features_v1/ # committed target-free empirical layer
├── docs/
│   ├── AGENT_GUIDE.md
│   ├── PROJECT_HISTORY.md
│   ├── DATA_SOURCES.md
│   ├── DATA_CONTRACT_V2.md
│   ├── FEATURE_TABLE_V1.md
│   ├── VARIABLES.md
│   ├── FUNCTION_INDEX.md
│   ├── AI_USAGE.md
│   └── ...
├── reports/
│   └── checkpoint_02/                # committed fixed folds + diagnostics + baselines
├── src/geocebada/                    # reusable implementation
├── tools/                            # reproducible CLI workflows
├── tests/
├── notebooks/
├── app/
└── models/
```

The repository is private.

---

## 4. Component map

### Reusable Python package

`src/geocebada/` is the shared implementation layer.

```text
config.py          YAML/config loading
paths.py           repository/data path resolution
data/              file discovery, targets/split, raster/data loading
geo/               CRS and geospatial safeguards
features/          interactive recipes, temporal alignment, parcel feature construction
statistics/        statistical inference and diagnostics
evaluation/        regression baselines + Checkpoint 02 fixed-fold evaluation
models/            reserved for frozen/final modeling interfaces
visualization/     reusable plotting/exploration helpers
```

Search `docs/FUNCTION_INDEX.md` before adding public helpers.

### Main CLI tools

```text
tools/audit_official_tabular.py
    quick CI/schema check for official tabular files

tools/audit_data_contract_v2.py
    complete source/external data contract audit

tools/build_data_catalog.py
    structural data catalog generation

tools/build_external_data_manifest.py
    versionable manifest of local external/raw files

tools/download_external_data.py
    reproducible external-source downloader

tools/build_integration_fixtures.py
    deterministic 12-parcel end-to-end fixtures

tools/build_parcel_feature_table.py
    canonical 197-row Feature Table v1 builder

tools/build_checkpoint_02.py
    feature inventory, shift diagnostics, frozen folds, ablations and baselines

tools/build_agronomic_features_v1.py
    197-row target-free phenology/thermal/water/soil/nonlinear layer for Checkpoint 03A

tools/build_empirical_features_v1.py
    197-row target-free temporal-geometry/relative-condition layer for Checkpoint 03B

tools/generate_function_index.py
    regenerate docs/FUNCTION_INDEX.md from public package symbols

tools/inspect_reference_docx.py
    inspect source/reference DOCX material without manually rewriting it

tools/_netcdf_catalog_worker.py
    internal helper for NetCDF catalog inspection; not a user-facing entry point
```

### Notebooks

`notebooks/01_visualizacion_datos.ipynb`
: first-look/exploration notebook for teammates.

`notebooks/02_cobertura_alineacion_temporal.ipynb`
: temporal coverage/alignment exploration; its temporal parameters are not frozen final model
  choices.

Notebooks must not become alternate sources of truth for reusable logic.

### Streamlit

`app/main.py` and `app/pages/` provide the interactive laboratory.

`app/AGENTS.md` is mandatory reading before app/deployment changes.

The application must call shared `geocebada` functions rather than fork modeling/data logic.

### Reports and models

`reports/` stores experiment metrics, figures, predictions and checkpoint outputs.

`models/` is reserved for promoted/frozen model artifacts. Do not place exploratory metrics or
raw data there.

---

## 5. Sources of truth

Use this precedence order when facts conflict:

1. immutable official files under `data/source/` and `docs/official/`;
2. directly inspected metadata;
3. executable contracts/configs;
4. canonical generated manifests/reports;
5. checkpoints/handoffs/documentation;
6. notebook commentary.

Never silently alter source evidence to make documentation consistent.

---

## 6. Data policy

### Versioned

The following are intentionally in Git:

- official challenge source files;
- documentation/contracts;
- integration fixtures;
- `data/processed/features_v1/`;
- `data/processed/agronomic_features_v1/`;
- `data/processed/empirical_features_v1/`;
- Checkpoint 02 reports/folds.

### Local or ignored

Large reproducible external/raw/interim files stay local:

- downloaded CHIRPS daily rasters;
- SoilGrids rasters;
- WaPOR NetCDF;
- full SIAP raw downloads;
- large INEGI external products;
- transient intermediate data;
- trained model binaries unless explicitly promoted later.

Do not commit secrets, credentials, `.cdsapirc`, arbitrary caches or machine-specific paths.

---

## 7. Canonical model-ready input

Use:

```text
data/processed/features_v1/parcel_features_all.csv
data/processed/features_v1/feature_manifest.json
```

for general Checkpoint 03 experiments.

Use `parcel_features_clean.csv` when running the clean/historical track.

Use `parcel_features_competition.csv` when running the challenge/competition track.

Validated shapes:

```text
clean:        197 × 699
competition:  197 × 1408
all:          197 × 1408

manifest:
  clean features:             694
  competition-only features: 709
  total features:            1403
  numeric features:          1401
```

Metadata/target columns are not model features just because they appear in the CSV.

Use the manifest instead of guessing from prefixes whenever possible.

---

## 8. Agronomic Features v1

Checkpoint 03A is complete. The additive table is:

```text
data/processed/agronomic_features_v1/parcel_agronomic_features.csv
197 rows × 351 columns
= ID_POLIGONO + 350 derived features
```

Manifest counts:

```text
clean          123
competition    227
total          350
missingness    0.0
```

Families:

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

This table is **target-free**: it contains neither `RENDIMIENTO_T_HA` nor `CONJUNTO`, and
no interaction was chosen by examining yield. It is joined to Feature Table v1 one-to-one by
`ID_POLIGONO`.

CHIRPS is intentionally absent from Agronomic Features v1 pending QC. Three 25 C monthly-mean
heat-derived candidates were constant across the current 197 parcels and were dropped; do not
interpret that as evidence that heat stress is biologically irrelevant.

Each manifest record contains source inputs, formula, meaning, rationale, evidence class,
bibliography keys and caveats. Treat `literature_backed`, `mechanistic_proxy` and
`experimental` as different evidence levels.

Read `docs/AGRONOMIC_FEATURES_V1.md` for the formulas and references.

---

## 9. Empirical Features v1

Checkpoint 03B is complete. The additive X-only table is:

```text
data/processed/empirical_features_v1/parcel_empirical_features.csv
197 rows × 336 columns
= ID_POLIGONO + 335 derived features
```

Manifest counts:

```text
clean           91
competition    244
total          335
missingness    0.0
infinities       0
```

Families:

```text
temporal_shape              162
aggregate_geometry           72
symmetric_change             42
short_baseline_condition     41
sensor_shape_similarity      18
```

The materialized table is target-free. The short-baseline NDVI normalization is named
`emp_vci_like_short__*` and must not be described as standard VCI because its parcel-level
historical reference spans only 2022–2024. One Landsat VI6T exceedance candidate was constant
and dropped.

Target-aware discovery is separate. `FoldLocalExpressionMiner` uses a configured pool of 24
primitives and may inspect yield only inside `fit(X_train, y_train)` for one training fold.
Never pre-discover expressions on all 138 labels before scoring frozen folds.

PCA, clustering, PLS and kernels also learn cross-row/model parameters and belong inside
Checkpoint 03C fold pipelines rather than canonical processed tables.

Read `docs/EMPIRICAL_FEATURES_V1.md` for formulas, literature context and leakage rules.

---

## 10. Feature families

Observed Feature Table v1 inventory:

```text
clean
  admin                8
  basic              539
  cem                  5
  climate_official    24
  geometry             6
  siap                14
  soilgrids           90
  topography           8

competition-only
  basic              539
  chirps               1
  climate_official    45
  pro                  97
  siap                  5
  wapor                22
```

Two admin features are non-numeric.

### Important CHIRPS caveat

Only one CHIRPS daily feature survived the final all-null drop. Do not describe Feature Table v1
as containing a full CHIRPS daily feature block. Investigate extraction/coverage before relying
on CHIRPS as a modeled family.

---

## 11. Clean vs competition semantics

`clean` is the defensible historical/static track. It avoids full-season target-year
information and contemporaneous outcome proxies.

`competition` adds rule-permitted 2025/full-season/transductive public information, including
PRO, target-year climate/WaPOR and SIAP 2025 proxies.

Keep both tracks visible in results. Never quietly merge their interpretation.

The exact operational prediction cutoff is still unresolved. Therefore competition features
must not be described as a strict prospective forecast.

---

## 12. Frozen Checkpoint 02 validation

Always reuse:

```text
reports/checkpoint_02/cv_folds.csv
```

Do not regenerate folds merely because a model performs poorly.

Protocols:

### Primary — `fold_state_stratified`

Five deterministic folds, random seed 42, preserving state composition.

### Robustness — `fold_municipality_grouped`

Five folds that keep state/municipality groups together.

The robustness protocol is intentionally harder and is critical because spatial/municipal
generalization is a major project risk.

---

## 13. Frozen ablations

Use the same names in future reports:

```text
A0_geometry_admin               12
A1_static_environment          115
A2_static_plus_siap_history    129
A3_clean_plus_climate          153
A4_clean_plus_basic_history    692
A5_clean_full                  692
A6_competition_remote_weather 1396
A7_competition_full           1401
```

A4 and A5 have the same numeric count because A4 already contains all clean numeric families;
the remaining clean manifest fields are non-numeric admin metadata.

Do not redefine these labels halfway through a modeling study.

---

## 14. Checkpoint 02 baseline evidence

These are historical baseline results, not a final-model declaration.

State-stratified CV:

```text
ExtraTrees A7  RMSE 0.4966
ExtraTrees A6  RMSE 0.5001
ExtraTrees A2  RMSE 0.5027
DummyMean      RMSE 0.8506
```

Municipality-grouped CV:

```text
Ridge A3       RMSE 0.7426
Ridge A6       RMSE 0.8178
ExtraTrees A6  RMSE 0.8835
ExtraTrees A5  RMSE 0.9073
DummyMean      RMSE 1.0066
```

The important fact is not one single minimum. The important fact is the large gap between
protocols.

Any Checkpoint 03 model should report both protocols or explicitly justify why a different
validated protocol is being added.

---

## 15. Train-versus-prediction diagnostics

Checkpoint 02 screened 1,401 numeric features.

Six crossed at least one configured descriptive threshold:

- four historical Sentinel-2 August features with absolute SMD around 0.50–0.53;
- `soilgrids__bdod__30_60cm__std` with ~10.17% prediction observations outside train range;
- `sat_basic_2025__sentinel_2__evi_min__std` with ~10.17% outside train range.

These are not target signals.

Do not use the 59 prediction covariates as a supervised tuning set or choose a model because its
final predictions look plausible.

---

## 16. What Checkpoint 03C should do

Checkpoints 03A and 03B are closed. Do not recreate or target-select their canonical materialized
features outside CV.

03C should compare the following representations under both frozen validation protocols before
heavy tuning:

```text
B0 = Feature Table v1
B1 = Agronomic Features v1 only
B2 = Empirical Features v1 only
B3 = base + agronomic
B4 = base + empirical
B5 = base + agronomic + empirical
B6 = reduced/selected representation
B7 = fold-local discovered-expression augmentation
```

Good candidate model families:

- CatBoost;
- Ridge/ElasticNet with tuned regularization;
- ExtraTrees/RandomForest refinements;
- HistGradientBoosting;
- LightGBM/XGBoost if justified;
- polynomial/RBF kernels;
- PCA/PLS variants as comparisons, never as assumptions.

Because n=138 and p is large, keep hyperparameter search disciplined. Prefer small,
interpretable search spaces and repeated held-out evidence over huge optimizer sweeps.

Any learned operation must be inside the CV pipeline:

- imputation;
- scaling;
- PCA/PLS;
- clustering;
- supervised feature selection;
- encoding;
- target-aware expression discovery;
- kernel/model hyperparameters.

---

## 17. Modeling implementation rules

Stable shared logic belongs under `src/geocebada/`, not inside a notebook.

Use `docs/FUNCTION_INDEX.md` before creating helpers.

Notebooks may orchestrate experiments and visualization, but should call the shared package.

For every experiment record:

```text
feature track / ablation
fold protocol
model
hyperparameters
preprocessing
random seed
RMSE
MAE
R²
fold-level scores
software/code commit
```

Prefer machine-readable CSV/JSON outputs plus a human-readable Markdown summary.

Do not overwrite historical Checkpoint 02 outputs with Checkpoint 03 experiments.

---

## 18. Spatial caution

The municipality-grouped degradation is a first-class project finding.

Possible causes include genuine spatial autocorrelation, municipal proxies, regional management
patterns and difficult extrapolation to unseen municipalities. Do not assert a causal reason
without evidence.

Future modeling should test whether gains survive grouped validation rather than optimizing
only state-stratified CV.

Coordinates and administrative features may be predictive, but they can also make random/state
CV look easier than geographic generalization really is.

---

## 19. Temporal caution

The target corresponds to April–October 2025, but the exact operational forecasting time is not
frozen.

Therefore:

- clean and competition tracks must remain separate;
- full-season 2025 features are challenge-specific unless a valid forecast cutoff supports them;
- post-cutoff observations must be excluded from any future strict prospective variant.

Do not retroactively label competition-mode features as leakage-free prospective features.

---

## 20. Known technical maintenance items

The feature-table build passed, but two non-fatal warnings were observed:

### pandas fragmentation

`build_daily_chirps_features` inserts many day columns iteratively. This is a performance issue,
not a correctness failure. Optimize only with output-equivalence tests.

### h5py/HDF5 mismatch

The workstation reported an h5py runtime/build HDF5 version mismatch. WaPOR extraction still
completed. Environment cleanup is advisable before relying on long-running NetCDF/HDF5 jobs.

Do not confuse either warning with failed validation.

---

## 20. Commands that define the current pipeline

From the repository root:

```powershell
conda activate geocebada

python tools\audit_data_contract_v2.py
python tools\build_integration_fixtures.py
python tools\build_parcel_feature_table.py
python tools\build_checkpoint_02.py
python tools\build_agronomic_features_v1.py

python -m pytest -q -p no:cacheprovider
python tools\generate_function_index.py
sphinx-build -b html docs docs\_build\html
```

Do not rerun expensive raw-data workflows unless necessary. Feature Table v1 and Checkpoint 02
artifacts are already versioned.

---

## 21. CI and code quality

CI checks:

- official tabular schema audit;
- Ruff;
- pytest;
- generated function-index synchronization;
- Sphinx build with warnings treated as errors.

Before merging substantive changes, ensure CI is green.

If CI fails, inspect the failing step rather than bypassing it.

---

## 22. Streamlit rules

Before any app change, read `app/AGENTS.md`.

Important historical deployment rule:

- local Conda environment: `environment.dev.yml`;
- Cloud install path: root `requirements.txt` + package extras;
- do not reintroduce root `environment.yml` without checking Streamlit dependency precedence.

The app is an interface over `geocebada`, not a second implementation.

---

## 23. AI-use documentation

Material AI contributions are logged in `docs/AI_USAGE.md`.

The challenge requires disclosure of generative-AI use, so substantive AI-assisted code,
analysis, modeling decisions, figures and report drafting should be recorded there.

Do not put secrets or personal credentials in the AI log.

---

## 24. Things an agent should not redo

Do not redo the following unless source data/code changed or a validation fails:

- basic understanding of 197/138/59 split;
- BASIC/PRO schema discovery;
- Data Contract v2 design;
- SIAP municipality key policy;
- SoilGrids scaling/CRS policy;
- WaPOR duration weighting policy;
- integration-fixture selection;
- Feature Table v1 generation design;
- clean/competition separation;
- Checkpoint 02 fold generation;
- Checkpoint 02 ablation definitions;
- initial baseline run.

Build on these artifacts instead.

---

## 25. Things still open

These are legitimate next questions:

- which model family improves both frozen protocols;
- whether feature selection or PCA/PLS helps high-p/small-n generalization;
- whether CatBoost can exploit the feature table without overfitting;
- whether a smaller family subset beats the full 1,401 numeric feature set;
- why CHIRPS daily contributes only one surviving feature;
- how much municipal/spatial structure explains the CV gap;
- whether additional spatial grouping/distance validation is useful;
- what exact operational forecast cutoff should define a strict prospective model;
- how to create a final stable ensemble and uncertainty summary;
- how to explain feature-family contributions in the FIRA report.

---

## 26. Definition of a good agent handoff

Before ending a substantial agent session:

1. update code/tests/docs together;
2. preserve historical reports instead of overwriting them;
3. record new facts in the appropriate checkpoint/handoff;
4. update `docs/AI_USAGE.md` for material AI contributions;
5. run relevant tests/CI;
6. state exactly what was verified versus what remains inferred;
7. leave the next concrete action explicit.

The project should always remain understandable from a fresh clone without relying on chat
history.
