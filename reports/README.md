# Reportes y entregables

> **Checkpoint 05 is active.** Checkpoint 04F is the frozen incumbent, while 05 combines the
> surviving Checkpoint 03 global experts with Checkpoint 04 local/transductive experts under
> cross-fitted pseudo-competition validation and a fresh confirmation gate. Historical
> Checkpoint 02/03/04 reports remain frozen reference evidence.

This directory stores reproducible analysis/model outputs and final-deliverable material. It is
not a source-data directory.

## Current committed reports

```text
reports/
├── data_audit_v2.json
├── data_audit_v2.md
├── data_schema_official.md
└── checkpoint_02/
    ├── feature_catalog.csv
    ├── feature_inventory.csv
    ├── train_prediction_diagnostics.csv
    ├── cv_folds.csv
    ├── ablation_features.json
    ├── baseline_fold_scores.csv
    ├── baseline_summary.csv
    ├── checkpoint_02_report.json
    └── checkpoint_02_report.md
```

`reports/checkpoint_02/cv_folds.csv` remains a frozen historical artifact. Checkpoint 04 adds a new documented transductive pseudo-competition protocol; legacy state/municipality folds remain stress tests.

Do not overwrite Checkpoint 02/03/04 artifacts. New modeling outputs belong under
`reports/checkpoint_05/` while Checkpoint 05 is active.

Suggested final-deliverable areas remain:

```text
reports/
├── figures/
├── tables/
├── predictions/
├── technical_report/
└── references/
```

Original/source data do not belong here. Tables, figures, metrics and predictions should be
regenerable from code and should record the relevant commit/config/model context.

See `docs/AGENT_GUIDE.md` for current project state and reporting rules.


## Checkpoint 04B

`reports/checkpoint_04b/` is the frozen pseudo-competition namespace. Its runner hides
pseudo-target y while keeping pseudo-target X visible and compares global/local/spatial/graph
methods with repeated RMSE.

Checkpoint 04B is complete and its generated artifacts are frozen reference evidence.


## Checkpoint 04D.1

`reports/checkpoint_04d1/` is the frozen local/graph refinement namespace.

Generated numerical results must come from:

```powershell
python tools\run_checkpoint_04d1.py
```

The runner reuses frozen 04B memberships, performs exhaustive target-matched refinement,
leave-one-split-out selection/routing validation, finalist stress testing and candidate
prediction disagreement analysis for the actual 59 targets.


- `checkpoint_04e1/`: exact SIAP 2025 grain-barley/Primavera-Verano/Temporal
  audit and external-prior pseudo-competition validation.


## Checkpoint 04C.1

`reports/checkpoint_04c1/` is the active focused global-anchor namespace. The
runner compares three inherited C1 agronomic CatBoost candidates, fixed low-weight
Local04D/Graph04D blends, target-matched residual correlation and a controlled
LOSO anchor gate. Actual 59 predictions are candidates only.

## Checkpoint 04E.1

`reports/checkpoint_04e1/` is complete. Exact SIAP coverage was 197/197, but
SIAP did not improve target-matched pseudo-competition RMSE. Canonical
interpretation: `docs/CHECKPOINT_04E1_FINDINGS.md`.


## Checkpoint 04F

`reports/checkpoint_04f/` is the finalization namespace. The configured final
rule is Local04D; the runner verifies provenance, audits a constrained
Local/Graph blend family and writes exactly 59 `ID_POLIGONO,RENDIMIENTO_T_HA`
rows plus separate uncertainty/disagreement diagnostics.

The finalization run is complete. The canonical submission-facing file is
`reports/checkpoint_04f/final_predictions.csv`; interpretation is frozen in
`docs/CHECKPOINT_04F_FINDINGS.md`.


## Checkpoint 05

`reports/checkpoint_05/` is the active final global/local ensemble namespace.
It will contain cross-fitted base predictions, all pseudo-competition stack
predictions, LOSO promotion evidence, actual 59 candidates, the final selected
table and a model manifest. Until its promotion gate passes, Checkpoint 04F
remains the canonical submission-facing prediction file.
