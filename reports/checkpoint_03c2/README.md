# Checkpoint 03C.2 reports

03C.2 is the competition-only nested model-family benchmark.

Run on the workstation:

```powershell
python -m pip install -e ".[models]"
python tools\run_checkpoint_03c2.py
```

Generated files are expected to include outer-fold metrics, pooled OOF predictions, every
inner candidate result, fold-local discovery formulas, protocol summaries and a cross-protocol
robustness table.

The run does not use hidden targets and does not generate the 59 final challenge predictions.
