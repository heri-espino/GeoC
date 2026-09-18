# Reportes y entregables

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

`reports/checkpoint_02/cv_folds.csv` is a frozen modeling artifact. Checkpoint 03 experiments
must reuse it unless a new validation protocol is explicitly added and documented.

Do not overwrite Checkpoint 02 artifacts with later modeling results. Create a separate
Checkpoint 03 report namespace.

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
