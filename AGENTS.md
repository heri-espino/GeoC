# AGENTS.md — GeoCebada

These instructions apply repository-wide unless a more specific handoff in a subdirectory adds constraints.

## Read before changing code

1. `.ai_handoff`
2. `README.md`
3. `data/.ai_handoff` and `data/README.md` for any data, feature, modeling or geospatial task
4. relevant official/reference material under `docs/`

Do not replace confirmed project facts with guesses or generic ML assumptions.

## Core invariants

- Canonical parcel identifier: `ID_POLIGONO`.
- Canonical target: `RENDIMIENTO_T_HA`.
- Official split: 197 parcels = 138 `ENTRENAMIENTO` + 59 `PREDICCION`.
- The 59 prediction targets are hidden. Never create pseudo-ground-truth labels for them.
- `data/source/` is immutable source evidence.
- `docs/official/` is immutable official documentation evidence.
- Generated data belongs in `data/raw/`, `data/interim/` or `data/processed/`.
- Trained models belong in `models/`; metrics/figures/predictions belong in `reports/`.
- Production transformations belong in `src/geocebada/`; notebooks are exploratory.

## Known spatial constraints

- CHIRPS precipitation: EPSG:4326.
- CHIRTS-ERA5 temperature: EPSG:4326.
- INEGI CEM topography: EPSG:6372.
- Parcel CRS must be inspected before use.

Never perform overlays, metric distances or area computations without explicit CRS checks/reprojection.

## Leakage constraints

- Do not use hidden targets for model selection or evaluation.
- Do not use climate/satellite dates after the target harvest period; the exact target cycle is still unresolved.
- Fit learned preprocessing inside CV folds.
- Treat spatial autocorrelation as a validation risk; random CV alone may be optimistic.
- Do not tune against the final 59 predictions by subjective inspection.

## Current unresolved questions

Do not silently assume answers to these:

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
- Avoid absolute local paths.
- Keep app logic thin; inference must reuse the frozen pipeline rather than duplicate feature engineering.
- Add/update tests for stable behavior.

## Before finishing a substantive task

- run relevant tests/lint when available;
- report what was actually verified;
- update `.ai_handoff` if project facts/status changed;
- update `data/.ai_handoff` for data-specific discoveries;
- update `docs/AI_USAGE.md` when AI materially contributed to code, methodology, analysis, writing or design.

If source evidence contradicts repository documentation, preserve source evidence and update the derived documentation.
