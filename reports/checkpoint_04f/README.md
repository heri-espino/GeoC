# Checkpoint 04F — Final transductive reconstruction

Run from the repository root:

```powershell
python tools\run_checkpoint_04f.py
```

04F does not reopen model search. It verifies that the frozen Local04D and
Graph04D actual-target predictions are identical across Checkpoints 04D.1 and
04E.1, reconstructs a small Local/Graph blend sensitivity table, performs a
constrained leave-one-target-matched-split-out blend-weight check, and then
freezes the configured final rule.

The configured final rule is currently **Baseline_Local04D**, corresponding to:

```text
LocalRidge_C4_all_deterministic_Geo_k24_a30_p1
```

Expected outputs:

```text
blend_validation_summary.csv
loso_blend_selection.csv
loso_blend_predictions.csv
final_predictions.csv
final_prediction_diagnostics.csv
checkpoint_04f_report.json
checkpoint_04f_report.md
```

`final_predictions.csv` contains exactly:

```text
ID_POLIGONO,RENDIMIENTO_T_HA
```

for the 59 official `PREDICCION` parcels. The hidden FIRA target values are
never loaded or scored.

The separate diagnostics file retains Graph04D, X-support and Local-vs-Graph
disagreement for uncertainty interpretation; those diagnostics do not alter
the frozen final prediction per parcel.
