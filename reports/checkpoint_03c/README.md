# Checkpoint 03C reports

Checkpoint 03C.1 is a representation benchmark. The canonical run should be produced on the
workstation with:

```powershell
python tools\run_checkpoint_03c.py
```

Expected generated artifacts:

```text
representation_manifest.json
representation_fold_metrics.csv
representation_summary.csv
representation_oof_predictions.csv
discovery_by_outer_fold.csv
checkpoint_03c_report.json
checkpoint_03c_report.md
```

The first run uses fixed Ridge10 and ExtraTrees baselines to isolate representation effects.
It does not select a final model and does not score the 59 prediction parcels.

After the run is reviewed, the outputs should be committed so 03C.2 model-family tuning can
be based on a frozen representation benchmark.
