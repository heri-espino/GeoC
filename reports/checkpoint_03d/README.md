# Checkpoint 03D generated artifacts

This directory is populated by:

```powershell
python tools\run_checkpoint_03d.py
```

Expected versionable scientific outputs:

- `outer_fold_metrics.csv`
- `oof_predictions.csv`
- `inner_search_results.csv`
- `protocol_summary.csv`
- `robustness_summary.csv`
- `finalists.csv`
- `actual_global_predictions.csv`
- `representation_manifest.json`
- `checkpoint_03d_report.json`
- `checkpoint_03d_report.md`

Large fitted model artifacts and ONNX binaries are stored under
`models/final/` and remain gitignored by default.
