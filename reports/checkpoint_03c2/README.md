# Checkpoint 03C.2 reports

03C.2 is the competition-only nested model-family benchmark.

The workstation run completed on 2026-09-19 with CatBoost using GPU. Reproduce with:

```powershell
python -m pip install -e ".[models]"
python tools\run_checkpoint_03c2.py
```

Generated files are expected to include outer-fold metrics, pooled OOF predictions, every
inner candidate result, fold-local discovery formulas, protocol summaries and a cross-protocol
robustness table.

The run does not use hidden targets and does not generate the 59 final challenge predictions.
\nThe run produced 280 outer fits, 7,728 OOF predictions and the cross-protocol robustness table used by Checkpoint 03C.3.\n