# AGENTS.md — GeoCebada

These instructions apply repository-wide unless a more specific handoff adds constraints.

## Read before changing code

1. `docs/AGENT_GUIDE.md` — operational project entry point.
2. `.ai_handoff` — compact current-state handoff.
3. `docs/PROJECT_HISTORY.md` — chronological record of what happened and why.
4. `README.md`.
5. `checkpoints/01_data/README.md` and `checkpoints/02_features/README.md` — completed milestones.
6. `checkpoints/03_modeling/README.md` and `checkpoints/03b_empirical_feature_discovery/README.md` — 03A closed; 03B current phase.
7. `data/processed/features_v1/README.md`, `data/processed/agronomic_features_v1/README.md`, `data/processed/empirical_features_v1/README.md`, `docs/AGRONOMIC_FEATURES_V1.md`, `docs/EMPIRICAL_FEATURES_V1.md` and `reports/checkpoint_02/checkpoint_02_report.md` before modeling.
8. `docs/DATA_SOURCES.md`, `docs/DATA_CONTRACT_V2.md` and `configs/data_contract_v2.yaml` before data/feature changes.
9. `data/.ai_handoff` and `data/README.md` for data/geospatial work.
10. `docs/VARIABLES.md` for the source-backed variable dictionary.
11. `src/geocebada/README.md` and `docs/FUNCTION_INDEX.md` before creating reusable helpers.
12. relevant official/reference material under `docs/`.
13. **`app/AGENTS.md` before any Streamlit/app/deployment change**.

Do not replace confirmed project facts with guesses or generic ML assumptions. Checkpoint 01, Checkpoint 02 and Checkpoint 03A are closed; do not redo them unless source/code changes invalidate a check. The current phase is Checkpoint 03B — empirical/data-driven feature discovery. Model comparison is Checkpoint 03C.

## Core invariants

- Canonical parcel identifier: `ID_POLIGONO`.
- Canonical target: `RENDIMIENTO_T_HA`.
- Official split: 197 parcels = 138 `ENTRENAMIENTO` + 59 `PREDICCION`.
- The target corresponds to the **April–October 2025 production cycle**.
- Hidden prediction targets are never pseudo-ground-truth.
- `data/source/` and `docs/official/` are immutable source evidence.
- Generated data belongs in `data/raw/`, `data/interim/` or `data/processed/`; the canonical `data/processed/features_v1/`, `data/processed/agronomic_features_v1/` and `data/processed/empirical_features_v1/` artifacts are intentionally versioned, while other raw/interim/processed outputs remain ignored unless explicitly promoted.
- Models belong in `models/`; metrics/figures/predictions in `reports/`.
- Stable production/reusable logic belongs in `src/geocebada/`; notebooks are exploratory.

## Confirmed remote-sensing sources

- BASIC: 107,666 rows × 96 columns, 197 parcels, Sentinel-2 + Landsat, 2022–2025.
- BASIC has 23 index families with parcel-level `promedio`, `std`, `max`, `min`; see `docs/VARIABLES.md`.
- PRO: 47,804 rows × 20 columns, 197 parcels, Planet, 2025.
- PRO contains NDVI, EVI, LAI and MSAVI with `promedio`, `std`, `max`, `min`.
- BASIC/PRO are longitudinal: rows are repeated parcel/date/sensor observations, not independent yield samples.
- Same-named indices from different sensors must not be silently treated as equivalent.

## Reusable-library rule

Team notebooks and the web app must import common behavior from the `geocebada` package. Do not duplicate loaders, path helpers, feature transforms, model utilities or evaluation code across notebooks.

Before writing a public function:

1. search `docs/FUNCTION_INDEX.md`;
2. inspect the relevant package module;
3. reuse/extend existing behavior if responsibility overlaps.

After changing the public API, run:

```bash
python tools/generate_function_index.py
```

Public reusable functions should have docstrings, type hints and tests when stable. Export commonly used symbols through the relevant `__init__.py`.

## Checkpoint 03B discovery rule

The materialized Empirical Features v1 layer must remain target-free. Deterministic row-wise
X transforms may be built once for all 197 parcels.

Any target-aware formula search must be fitted inside training folds. Use
`FoldLocalExpressionMiner` or an equivalently leakage-safe pipeline; never discover formulas
from all 138 labels and then report the frozen CV as if those formulas had been pre-specified.

PCA, clustering, kernels and other learned representations are not global processed features;
fit them inside folds during Checkpoint 03C.

## Streamlit maintenance rule

`app/AGENTS.md` is the authoritative app-specific maintenance contract. It documents:

- current pages/capabilities;
- thin-interface architecture;
- prediction-set and leakage guardrails;
- CRS/map safety;
- Streamlit Community Cloud dependency precedence;
- local and Cloud smoke tests;
- caching/performance conventions;
- the checklist that must be followed when dependencies, data interpretation, model artifacts or pages change.

Important deployment invariant: local Conda configuration lives in `environment.dev.yml`; do **not** reintroduce a root `environment.yml` without checking Streamlit dependency-file precedence. Cloud bootstrapping uses root `requirements.txt`, which installs the package/extras defined in `pyproject.toml`.

## Documentation rule

Sphinx API docs live under `docs/api/` and derive from source docstrings. Build with:

```bash
pip install -e ".[docs]"
sphinx-build -b html docs docs/_build/html
```

`checkpoints/01_data/README.md` and `checkpoints/02_features/README.md` are completed checkpoints. `docs/DATA_SOURCES.md` is the operational inventory. `docs/DATA_CONTRACT_V2.md` and `configs/data_contract_v2.yaml` define the executable feature-engineering gate. `docs/VARIABLES.md` is the human-facing variable dictionary; source/reference documents remain authoritative when they conflict with derived documentation.

## Known spatial constraints

- CHIRPS precipitation: EPSG:4326.
- CHIRTS-ERA5 temperature: EPSG:4326.
- INEGI CEM topography: EPSG:6372.
- Official parcel CRS has been directly inspected as EPSG:4326.

Never perform overlays, metric distances or area calculations without explicit CRS checks/reprojection.

## Leakage constraints

- Do not use hidden targets for model selection/evaluation.
- The target cycle is April–October 2025, but the **operational prediction cutoff is not yet frozen**. Do not use satellite/climate observations after the chosen forecast horizon.
- Fit learned preprocessing inside CV folds.
- Never row-random split BASIC/PRO for supervised validation; group by parcel at minimum.
- Treat spatial autocorrelation as a validation risk; random parcel CV alone may be optimistic.
- Do not tune against the final 59 predictions by subjective inspection.
- Preserve sensor provenance when combining satellite sources.

## Current unresolved questions

Do not silently assume:

- exact pre-harvest prediction cutoff/window within the April–October 2025 cycle;
- sensor-harmonization strategy between BASIC and PRO;
- official split stratification mechanism;
- final model-selection decision after comparing the frozen state-stratified and municipality-grouped protocols;

## Implementation style

- Python 3.11+.
- Prefer small, testable functions.
- Preserve reproducibility and `random_seed: 42` unless documented otherwise.
- Add assertions for schema, unique parcel IDs, row counts, CRS and joins.
- Avoid absolute local paths; use `geocebada.paths`.
- Keep app logic thin; inference must reuse the frozen library pipeline.
- Add/update tests for stable behavior.

## Before finishing a substantive task

- run relevant tests/lint/docs when available;
- report what was actually verified;
- regenerate `docs/FUNCTION_INDEX.md` after public API changes;
- update `.ai_handoff` if project facts/status changed;
- update `data/.ai_handoff` for data-specific discoveries;
- update `docs/AI_USAGE.md` when AI materially contributed;
- for Streamlit/deployment changes, follow `app/AGENTS.md` and update `docs/DEPLOYMENT.md` when the deployment contract changes.

If source evidence contradicts repository documentation, preserve source evidence and update the derived documentation.
