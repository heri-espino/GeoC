# AGENTS.md — GeoCebada

These instructions apply repository-wide unless a more specific handoff adds constraints.

## Read before changing modeling/data logic

1. `docs/TRANSDUCTIVE_OBJECTIVE.md` — current mathematical and methodological objective.
2. `.ai_handoff` — compact current-state handoff.
3. `docs/AGENT_GUIDE.md` — operational entry point.
4. `docs/PROJECT_HISTORY.md` — chronology and frozen historical decisions.
5. `checkpoints/04_transductive_competition/README.md` — active checkpoint.
6. `checkpoints/03c3_finalist_ensembles/README.md` and `reports/checkpoint_03c2/checkpoint_03c2_report.md` — First Modeling Delivery baseline only.
7. `data/.ai_handoff`, `docs/DATA_SOURCES.md`, `docs/FEATURE_TABLE_V1.md` — data semantics.
8. `docs/FUNCTION_INDEX.md` before adding reusable helpers.
9. `app/AGENTS.md` before Streamlit/deployment changes.

## Active objective

Checkpoint 04 is **Transductive Competition Modeling**. There are 59 fixed target parcels whose
reference yields are hidden by FIRA. The current purpose is to reconstruct those 59 values as
accurately as possible, not to optimize a generic model intended to generalize to arbitrary
future parcels.

Use all rule-permitted observable evidence. All X for all 197 parcels may inform X-only
representations and the final transductive inference. The hidden y values may not.

Treat FIRA's hidden values as the external scoring reference, not as available data and not
necessarily as error-free physical truth. A useful operational analogy is missing/untrusted
reported parcel yields that must be reconstructed from independent evidence.

## Core invariants

- canonical ID: `ID_POLIGONO`;
- target: `RENDIMIENTO_T_HA`;
- split: 197 = 138 `ENTRENAMIENTO` + 59 `PREDICCION`;
- target cycle: April–October 2025;
- states: Hidalgo, Puebla, Tlaxcala;
- BASIC/PRO rows are repeated parcel/date/sensor observations, not independent yield samples;
- current modeling track: `competition`; `clean` is historical/provenance only;
- direct use or acquisition of the 59 hidden y values is forbidden.

## Transductive validation rule

From Checkpoint 04 onward, an evaluation fold should mimic the real competition:

- pseudo-target y is hidden;
- pseudo-target X remains visible to X-only transductive steps;
- any target-aware step sees only the remaining labeled y;
- RMSE is computed only after predictions are frozen.

This permits joint X-only PCA, clustering, graph construction, density estimation,
target-set-aware normalization and similar operations, provided the same information structure
is reproduced during validation.

Target-aware feature search, supervised feature selection and any operation using y must still
be restricted to the labeled portion of each pseudo-competition split.

Legacy state-stratified and municipality-grouped folds are retained as stress-test evidence;
they are not the only selection criterion in Checkpoint 04.

## Similarity and dependence are first-class signals

Checkpoint 04 should explicitly test:

- spatial autocorrelation of observed yields;
- spatial autocorrelation of baseline residuals;
- target-to-train geographic neighborhoods;
- correlation/cross-correlation of longitudinal vegetation curves;
- lagged phenological similarity;
- dynamic-time-warping or other time-series distances when justified;
- climate/soil/topography similarity;
- joint graph neighborhoods over the 197 parcels;
- whether closer/similar labeled parcels actually have smaller yield differences.

Do not assume similarity transfers yield; measure it on the 138 labels.

## External-data directive

Public external data are desirable when they can reduce uncertainty for the fixed 59 targets.
Priority sources already present include SIAP 2003–2025, CHIRPS, WaPOR, SoilGrids, INEGI CEM
and official climate/topography.

SIAP 2025 deserves special attention. Audit the most specific valid proxy for `Cebada grano`,
the relevant production cycle and `Temporal` modality. Never mix grain barley with forage
barley. Administrative joins use full CVEGEO / state+municipality semantics, not municipality
ID alone.

Additional public sources may be added when there is a concrete hypothesis and provenance is
documented.

## First Modeling Delivery is frozen history

Checkpoint 03 is closed and preserved as the first conventional modeling delivery.

```text
F1 C0+PLS:      state 0.5339, municipality 0.7621
F2 C3+Ridge:    state 0.5331, municipality 0.7650
F3 C1+CatBoost: state 0.5235, municipality 0.7709
E123:           state 0.5121, municipality 0.7481
E13:            state 0.5082, municipality 0.7488
```

Do not mutate Checkpoint 03 configs/reports to improve these historical scores. Checkpoint 04
may reuse, retune, supersede or ensemble these models under the new validation doctrine.

## Data and implementation rules

- source evidence in `data/source/`, `docs/official/`, `docs/reference/` is immutable;
- preserve sensor provenance across Sentinel-2, Landsat and Planet;
- no many-to-many accidental joins;
- explicit CRS checks/reprojection before spatial distances/areas;
- stable reusable logic belongs in `src/geocebada/`; notebooks orchestrate exploration;
- use `geocebada.paths`; avoid machine-specific absolute paths;
- add schema/ID/row-count assertions;
- Python 3.11+;
- use GPU for expensive supported searches when available;
- add/update tests for stable behavior.

## Documentation and provenance

After substantive work:

- update `.ai_handoff` if current state changed;
- update `data/.ai_handoff` for data-specific discoveries;
- update `docs/PROJECT_HISTORY.md` for milestones;
- update the active Checkpoint 04 README;
- record material AI contribution in `docs/AI_USAGE.md`;
- regenerate `docs/FUNCTION_INDEX.md` after public API changes;
- follow `app/AGENTS.md` for app work.

Historical documentation may describe older leakage-safe inductive rules. Those remain valid
for interpreting Checkpoint 03 but are superseded for active Checkpoint 04 by
`docs/TRANSDUCTIVE_OBJECTIVE.md`.
