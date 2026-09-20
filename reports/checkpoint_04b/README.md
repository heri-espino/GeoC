# Checkpoint 04B reports

**Status:** runner implemented; workstation run pending.

Run:

```powershell
python tools\run_checkpoint_04b.py
```

Expected outputs:

```text
pseudo_competition_splits.csv
split_quality.csv
actual_target_x_support.csv
pseudo_test_predictions.csv
split_metrics.csv
protocol_summary.csv
support_tier_summary.csv
robustness_summary.csv
support_tier_method_routing.csv
actual_target_method_routing.csv
checkpoint_04b_report.json
checkpoint_04b_report.md

figures/
  primary_method_ranking.png
  top_method_rmse_boxplot.png
  split_match_quality.png
  support_tier_method_rmse.png
  actual_vs_pseudo_support.png
  actual_method_routing.png
  actual_target_support_routing.png
```

The primary protocol uses repeated target-matched pseudo-target masks selected only from X-derived
information. Pseudo-target y is revealed only after predictions are frozen.

The actual 59 targets receive X-only support and method-routing metadata, never scored yields.
