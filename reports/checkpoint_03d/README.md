# Checkpoint 03D generated artifacts

Checkpoint 03D completed on 2026-09-24 with the balanced-compute configuration.
Canonical interpretation: `docs/CHECKPOINT_03D_FINDINGS.md`.

The run produced 230 outer fits and 6,348 OOF rows across 23 eligible pairs in
248.085 minutes. The robust finalists were HistGBLarge, XGBoostLarge and
LightGBMLarge, all on `G1_agronomic`.

Versioned outputs include `outer_fold_metrics.csv`, `oof_predictions.csv`,
`inner_search_results.csv`, `protocol_summary.csv`,
`robustness_summary.csv`, `finalists.csv`,
`actual_global_predictions.csv`, `representation_manifest.json` and the
JSON/Markdown reports.

Fitted bundles and ONNX binaries remain under `models/final/` and are
gitignored. Their numerical verification is recorded in the versioned JSON
report.
