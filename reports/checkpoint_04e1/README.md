# Checkpoint 04E.1 reports

Generated outputs for the SIAP external-localization stage belong here.

Run from the repository root:

```powershell
python tools\run_checkpoint_04e1.py
```

The stage audits the public 2025 SIAP municipal closure at the explicit
`Cebada grano + Primavera-Verano + Temporal + CVEGEO` scope, records broader
fallback scopes separately, and evaluates direct/calibrated/residual uses of
that public prior on the exact frozen Checkpoint 04B pseudo-competition splits.

Expected generated files:

- `siap_scope_audit.csv`
- `siap_municipal_panel.csv`
- `siap_parcel_panel.csv`
- `pseudo_predictions.csv`
- `split_metrics.csv`
- `protocol_summary.csv`
- `loso_method_selection.csv`
- `loso_selected_predictions.csv`
- `actual_external_candidates.csv`
- `checkpoint_04e1_report.json`
- `checkpoint_04e1_report.md`
- `figures/siap_scope_coverage.png`
- `figures/external_method_ranking.png`
- `figures/siap_proxy_vs_yield.png`

The 59 actual-target predictions are candidate evidence only. They are not the
final Checkpoint 04F submission.
