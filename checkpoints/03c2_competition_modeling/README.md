# Checkpoint 03C.2 — Competition-only model-family benchmark

**Status:** implementation ready; workstation run pending  
**Opened:** 2026-09-18

03C.2 follows the closed 03C.1 representation benchmark.

## Snapshot at this checkpoint

Completed before the workstation run:

- 03C.1 is frozen and interpreted in `reports/checkpoint_03c/README.md`;
- `competition` is the only active modeling track from this point onward;
- the four 03C.2 representations are fixed as C0/C1/C2/C3;
- the two outer Checkpoint 02 validation protocols remain frozen;
- nested 3-fold inner CV is implemented for hyperparameter selection;
- the inner splitter mirrors the outer validation intent:
  - state-stratified outer folds use state-stratified inner CV;
  - municipality-grouped outer folds use municipality-grouped inner CV;
- target-aware expression discovery is inside the nested estimator pipeline;
- imputation, scaling, PCA, PLS and model tuning are also fold-local;
- CatBoost is configured for GPU on the university workstation with CPU fallback;
- Ridge, ElasticNet, ExtraTrees, CatBoost, PLS, PCA+RBF Kernel Ridge and
  PCA+degree-2 Polynomial Kernel Ridge are implemented;
- reproducible CSV/JSON/Markdown outputs are defined under `reports/checkpoint_03c2/`;
- no final model has been selected and no prediction has been generated for the 59 hidden rows.

Repository implementation:

```text
configs/checkpoint03c2.yaml
src/geocebada/evaluation/checkpoint03c2.py
tools/run_checkpoint_03c2.py
checkpoints/03c2_competition_modeling/README.md
```

The branch has passed repository CI. The remaining milestone for this checkpoint is the
workstation benchmark itself and interpretation of its outputs.

## Why this checkpoint exists

03C.1 answered **which representations are worth carrying forward** under fixed models.

03C.2 now asks a different question:

> Given the competition-only representations that survived 03C.1, which model families and
> fold-local hyperparameters remain robust under both frozen validation protocols?

This phase is still development/model selection. It is not the final challenge fit.

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

## Planned next steps

1. Run the full workstation benchmark:

   ```powershell
   python -m pip install -e ".[models]"
   python tools\run_checkpoint_03c2.py
   ```

2. Commit/push the generated `reports/checkpoint_03c2/` artifacts.

3. Review `robustness_summary.csv` first. The preferred development candidates should remain
   competitive under both state-stratified and municipality-grouped outer validation rather
   than minimizing only one protocol.

4. Inspect fold-level failures, selected hyperparameters and whether CatBoost actually used GPU
   or fell back to CPU.

5. Compare C3 against C2 to answer whether the bulk deterministic empirical block adds value
   beyond the small empirical support set used by fold-local discovery.

6. Freeze a small finalist set. Do not keep every model family merely because it was evaluated.

7. Only after 03C.2 is interpreted should the project define the final fit/ensemble,
   uncertainty analysis and the 59 competition predictions.

## Completion criteria

03C.2 can be closed when:

- the complete competition-only nested run finishes successfully;
- every protocol/representation/model has exactly one OOF prediction per training parcel;
- all search/preprocessing steps remain inside the nested folds;
- generated reports are committed;
- results under both frozen protocols are interpreted;
- a small finalist set is documented without using hidden prediction targets;
- CI remains green and handoffs/history are synchronized.

