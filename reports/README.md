# Reportes y entregables

> **Checkpoint 04:** new development reports should evaluate fixed-target transductive
> reconstruction. The primary new evidence is repeated pseudo-competition RMSE with
> pseudo-target X visible and pseudo-target y hidden. Historical Checkpoint 02/03 reports remain
> frozen reference evidence.

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

Do not overwrite Checkpoint 02/03 artifacts. New outputs belong under a Checkpoint 04 report namespace.

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

`reports/checkpoint_04b/` is the active pseudo-competition namespace. Its runner hides
pseudo-target y while keeping pseudo-target X visible and compares global/local/spatial/graph
methods with repeated RMSE.

Checkpoint 04B is complete and its generated artifacts are frozen reference evidence.


## Checkpoint 04D.1

`reports/checkpoint_04d1/` is the active local/graph refinement namespace. Until the
workstation run finishes, its README is the only expected artifact.

Generated numerical results must come from:

```powershell
python tools\run_checkpoint_04d1.py
```

The runner reuses frozen 04B memberships, performs exhaustive target-matched refinement,
leave-one-split-out selection/routing validation, finalist stress testing and candidate
prediction disagreement analysis for the actual 59 targets.
