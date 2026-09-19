# Checkpoint 03C.2 — Competition-only model-family benchmark

**Status:** implementation ready; workstation run pending  
**Opened:** 2026-09-18

03C.2 follows the closed 03C.1 representation benchmark.

## Active modeling track

From this checkpoint onward the active track is **competition only**.

```text
competition = clean + competition-mode predictors
clean       = historical/diagnostic only; not evaluated in 03C.2
```

CHIRPS remains excluded pending QC.

## Representations

03C.1 narrowed the search to four competition representations:

```text
C0 = base
C1 = agronomic-only
C2 = base + agronomic + empirical + fold-local discovery
C3 = base + agronomic + four empirical discovery-support primitives
     + fold-local discovery
```

C3 intentionally omits the bulk 335-feature empirical block while retaining the four
configured empirical primitives needed to let the discovery miner use VCI-like/shape signals.

Canonical expected dimensions:

```text
C0  1400 features
C1   350 features
C2  2085 features + 20 discovered expressions per fitted fold
C3  1754 features + 20 discovered expressions per fitted fold
```

## Nested validation

The frozen Checkpoint 02 folds remain the outer evaluation:

```text
fold_state_stratified
fold_municipality_grouped
```

Inside each outer training fold, 3-fold inner CV selects model hyperparameters.

For state-stratified outer CV, the inner split is stratified by state. For
municipality-grouped outer CV, the inner split groups by Estado|Municipio and never splits a
municipality across inner train/validation.

Target-aware expression discovery is inside the estimator pipeline, so it is refitted inside
every inner training split and then refitted on the full outer training fold before scoring the
outer holdout.

## Model families

03C.2 compares disciplined candidate grids rather than a large optimizer sweep:

- tuned Ridge;
- ElasticNet;
- ExtraTrees;
- CatBoost;
- PLS;
- PCA + RBF Kernel Ridge;
- PCA + degree-2 polynomial Kernel Ridge.

PCA/PLS, scaling, imputation, hyperparameter selection and discovery all remain inside nested
training folds.

CatBoost defaults to GPU (`devices=0`) and falls back to CPU only for a detected GPU/CUDA
availability failure.

## Workstation command

Install model extras once:

```powershell
python -m pip install -e ".[models]"
```

Then run:

```powershell
python tools\run_checkpoint_03c2.py
```

Optional recovery/debug subsets:

```powershell
python tools\run_checkpoint_03c2.py --models Ridge,ExtraTrees
python tools\run_checkpoint_03c2.py --representations C0_base,C1_agronomic
python tools\run_checkpoint_03c2.py --catboost-task-type CPU
```

## Outputs

```text
reports/checkpoint_03c2/
├── representation_manifest.json
├── outer_fold_metrics.csv
├── oof_predictions.csv
├── inner_search_results.csv
├── discovery_by_outer_fold.csv
├── protocol_summary.csv
├── robustness_summary.csv
├── checkpoint_03c2_report.json
└── checkpoint_03c2_report.md
```

`robustness_summary.csv` ranks each representation/model pair by worst-protocol OOF RMSE and
then mean OOF RMSE. It is a development comparison, not an independent final test.

No prediction for the 59 challenge parcels is generated in 03C.2.
