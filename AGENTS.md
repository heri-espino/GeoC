# AGENTS.md — GeoCebada

These instructions apply repository-wide unless a more specific handoff adds constraints.

## Read before changing modeling/data logic

1. `docs/TRANSDUCTIVE_OBJECTIVE.md` — current mathematical and methodological objective.
2. `.ai_handoff` — compact current-state handoff.
3. `docs/AGENT_GUIDE.md` — operational entry point.
4. `docs/PROJECT_HISTORY.md` — chronology and frozen historical decisions.
5. `checkpoints/04_transductive_competition/README.md` and `checkpoints/05_global_local_mixture/README.md` — frozen competition-method evidence.
6. `checkpoints/03c3_finalist_ensembles/README.md` and `reports/checkpoint_03c2/checkpoint_03c2_report.md` — First Modeling Delivery baseline only.
7. `data/.ai_handoff`, `docs/DATA_SOURCES.md`, `docs/FEATURE_TABLE_V1.md` — data semantics.
8. `docs/FUNCTION_INDEX.md` before adding reusable helpers.
9. `app/AGENTS.md` before Streamlit/deployment changes.

## Current objective and frozen competition method

The project remains a **Transductive Competition Modeling** problem. Broad model search is now closed. There are 59 fixed target parcels whose
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
they are not interchangeable with the target-matched pseudo-competition protocol.

## Similarity and dependence are first-class signals

Checkpoint 04 tested:

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

Do not mutate frozen Checkpoint 03C configs/reports to improve these historical scores. Checkpoint 03D is also closed and additive; its new state/grouped scores do not directly replace target-matched Local04D evidence.

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
- update the relevant checkpoint README and canonical findings document;
- record material AI contribution in `docs/AI_USAGE.md`;
- regenerate `docs/FUNCTION_INDEX.md` after public API changes;
- follow `app/AGENTS.md` for app work.

Historical documentation may describe older leakage-safe inductive rules. Those remain valid
for interpreting Checkpoint 03 but are superseded for active Checkpoint 04 by
`docs/TRANSDUCTIVE_OBJECTIVE.md`.


## Checkpoint 07 — only active modeling experiment

Checkpoint 06A–06F is closed without promotion. Do not continue tuning its
center/tail thresholds, gates, cluster specialists or residual corrections.

Checkpoint 07 is allowed to reopen modeling because it introduces genuinely new
information and a pretrained EO representation. Canonical design:
`docs/CHECKPOINT_07_PLAN.md`.

Active sources:

- NASA HLS v2 (HLSL30 + HLSS30) for actual raster chips and Prithvi input;
- NASA SMAP Enhanced L3 9 km daily soil moisture as a regional water-state signal;
- Sentinel-1 GRD VV/VH for SAR structure/moisture signal;
- AgERA5 v2 daily weather/stress;
- Prithvi-EO-2.0-300M-TL and 600M-TL.

Primary model lines:

1. Local07: Local04D/local-Ridge family augmented with new physical/spatial features;
2. Prithvi07: frozen Prithvi embeddings first, compact Ridge/PLS/GP heads;
3. Local07 + Prithvi07 fusion from honest OOF predictions.

Do not fine-tune hundreds of millions of parameters before frozen Prithvi
embeddings show useful honest OOF signal. Do not use hidden 59-target y.

For Checkpoint 07, LOSO is a stability diagnostic rather than an absolute veto.
Development uses the existing 16 target-matched splits. Final candidate
selection requires a fresh X-only target-matched confirmation bank constructed
before finalist scores are inspected. The primary goal is lowest expected RMSE
on the fixed 59 targets.

Large Checkpoint 07 acquisitions and model weights are local only:
`data/raw/checkpoint_07/` and `models/checkpoint_07/`. Never commit them.

Local04D and `reports/checkpoint_05/final_predictions.csv` remain canonical
until fresh confirmation supports a replacement.

## Checkpoint 07A2 streaming processing boundary (2026-10-09)

The source download is complete for HLS, SMAP, AgERA5, Prithvi; Sentinel-1
was marked running at the user's last status check. The verified inventory is
180.140 GiB raw, dominated by SMAP (135.492 GiB), not HLS (41.846 GiB).
HLS has 197 scenes with the six required Prithvi bands and Fmask.

Active code:
- src/geocebada/data/checkpoint07_hls.py
- tools/run_checkpoint_07_processing.py
- src/geocebada/data/checkpoint07_smap.py
- tools/run_checkpoint_07_smap.py

Inspect docs/CHECKPOINT_07A_PROCESSING.md and the generated processing reports.
Do not invent an extraction PASS before workstation execution.

Do not load full satellite scenes into RAM. Read scene/window and unique
SMAP regional grid cells; keep full temporal acquisition date, sensor and
QA provenance. Prithvi chips must use fixed georeferenced 30-m 224x224
grids with masks and exactly four X-only selected dates. Missing windows
must be reported, never filled silently. Keep packed HLS int16 DN for
the future model-config normalization; avoid double scaling.

Do not commit data/processed/checkpoint_07/, models/checkpoint_07/,
data/raw/checkpoint_07/ or any credentials. Never auto-delete raw.

Local04D remains canonical; 07A2 does no supervised modeling or hidden
FIRA-y access.

## Checkpoint 07B high-dimensional and neural ablations (2026-10-09)

Implemented `src/geocebada/features/checkpoint07_highdim.py`,
`src/geocebada/evaluation/checkpoint07_pca.py`,
`tools/run_checkpoint_07_highdim.py` and
`tests/test_checkpoint07_highdim.py`. Scientific readme:
`docs/CHECKPOINT_07B_HIGH_DIMENSIONAL.md`.

The user wants many high-order candidate features, PCA based on X variance,
and a neural comparison. This is explicitly a modeling research project,
not a contest. Do not describe it as a challenge or competition in new prose.

All feature construction is X-only and 197-row aligned. Imputation, scaling,
dual Gram PCA, Ridge and regularized MLP fitting are performed separately
inside each 97-label pseudo-training fold. Only frozen held-out labels may
be used to score the fold. Do not pool 16 overlapping splits as independent
observations or promote the lowest development RMSE without fresh confirmation.

A big-feature matrix is not a big-label dataset: only 138 observed y.
The large pretrained neural model Prithvi remains a distinct branch.
Concurrently reported 2025 SIAP municipal outcome proxies are excluded
by default from 07B due to potential outcome leakage; enabling them is
explicit and must be disclosed.

No raw raster re-download; respect prior 07A processing QA.
Local-only outputs in `reports/checkpoint_07/highdim/`, gitignored.

## Checkpoint 07C.1 resumed nested search (2026-10-10)

The new HLS/SMAP 07B full feature matrix contains ~21,445 X-only columns
and 12,000 generated interaction products. 07B's 16-split PCA16+MLP16
development RMSE was 0.552642; do not compare unmatched protocols.

The user wants the partially written *heavy* 07C search finished, not a
fresh modeling stack. Continue the already committed
`src/geocebada/evaluation/checkpoint07c.py` and
`tools/run_checkpoint_07c.py`; documented at
`docs/CHECKPOINT_07C_NESTED_SEARCH.md`.

The 07C.1 heavy nested tuning families are CatBoost, XGBoost, LightGBM,
RBF Kernel Ridge, PLS; representations include fold-local no-PCA and PCA.
Model/hyperparameter selection is inside 97-label pseudo-train subsets,
with 41 held-out labels never used during selection. Use frozen Local04D
paired predictions only for comparison and fixed blends. True supervised
residual fitting is NOT implemented and must not be faked with OOF
predictions that may contain current pseudo-target labels.

Results are resumable via separate Optuna SQLite files with a JSON run
fingerprint locking code/data versions, inner CV, seed and compute settings.
Do not mix past partial studies with changed search configurations. Run
heavy training locally, GPU-first. Optuna/ML tests are lightweight CI only.
Use `reports/checkpoint_07c/` locally; this directory is ignored by Git.
Original 07C Prithvi encoder is now 07C.2, still pending. Local04D remains
canonical, the 59 hidden yields are NEVER available.

## Frozen final-state summary

As of 2026-09-24:

- Checkpoint 05 completed without promotion;
- Local04D remains the competition-facing method;
- canonical fixed-59 CSV: `reports/checkpoint_05/final_predictions.csv`;
- Checkpoint 03D completed the balanced global-capacity closure;
- 03D scientific rank 1: HistGB G1;
- 03D highest-ranked deployable finalist: XGBoost G1, verified ONNX;
- Checkpoint 05B is optional/narrow only; do not reopen another broad search.

Do not compare 03D state/grouped RMSE directly with Local04D target-matched RMSE
as if they estimated the same validation distribution.
