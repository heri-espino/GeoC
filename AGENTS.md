# AGENTS.md — GeoCebada

These instructions apply repository-wide unless a more specific handoff adds constraints.

## Read before changing code

1. `.ai_handoff`
2. `README.md`
3. `data/.ai_handoff` and `data/README.md` for data/model/geospatial work
4. `src/geocebada/README.md` for reusable-code conventions
5. `docs/FUNCTION_INDEX.md` before creating a new helper
6. relevant official/reference material under `docs/`

Do not replace confirmed project facts with guesses or generic ML assumptions.

## Core invariants

- Canonical parcel identifier: `ID_POLIGONO`.
- Canonical target: `RENDIMIENTO_T_HA`.
- Official split: 197 parcels = 138 `ENTRENAMIENTO` + 59 `PREDICCION`.
- Hidden prediction targets are never pseudo-ground-truth.
- `data/source/` and `docs/official/` are immutable source evidence.
- Generated data belongs in `data/raw/`, `data/interim/` or `data/processed/`.
- Models belong in `models/`; metrics/figures/predictions in `reports/`.
- Stable production/reusable logic belongs in `src/geocebada/`; notebooks are exploratory.

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

## Documentation rule

Sphinx API docs live under `docs/api/` and derive from source docstrings. Build with:

```bash
pip install -e ".[docs]"
sphinx-build -b html docs docs/_build/html
```

Do not maintain divergent manual descriptions of the same function. Keep the code docstring authoritative.

## Known spatial constraints

- CHIRPS precipitation: EPSG:4326.
- CHIRTS-ERA5 temperature: EPSG:4326.
- INEGI CEM topography: EPSG:6372.
- Parcel CRS must be inspected before use.

Never perform overlays, metric distances or area calculations without explicit CRS checks/reprojection.

## Leakage constraints

- Do not use hidden targets for model selection/evaluation.
- Do not use climate/satellite dates after the target harvest period; target cycle remains unresolved.
- Fit learned preprocessing inside CV folds.
- Treat spatial autocorrelation as a validation risk; random CV alone may be optimistic.
- Do not tune against the final 59 predictions by subjective inspection.

## Current unresolved questions

Do not silently assume:

- target agricultural year/cycle;
- BASIC/PRO exact schema and semantics;
- official split stratification mechanism;
- valid temporal feature windows;
- final CV strategy;
- parcel source CRS until verified.

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
- update `docs/AI_USAGE.md` when AI materially contributed.

If source evidence contradicts repository documentation, preserve source evidence and update the derived documentation.
