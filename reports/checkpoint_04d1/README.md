# Checkpoint 04D.1 reports

**Status:** COMPLETE. Generated workstation evidence is committed.

Canonical interpretation: `docs/CHECKPOINT_04D1_FINDINGS.md`.

Reproduce with:

```powershell
python tools\run_checkpoint_04d1.py
```

Expected outputs include:

```text
method_manifest.csv
primary_predictions.csv
primary_summary.csv
primary_split_metrics.csv
finalist_manifest.csv
stress_predictions.csv
stress_summary.csv
finalist_robustness.csv
loso_method_selection.csv
loso_method_predictions.csv
loso_tier_selection.csv
loso_tier_predictions.csv
nested_validation_summary.csv
actual_candidate_predictions.csv
actual_routing_proposal.csv
checkpoint_04d1_report.json
checkpoint_04d1_report.md

figures/
  refinement_ranking.png
  graph_k_profiles.png
  local_k_profile.png
  nested_validation_comparison.png
  loso_selection_frequency.png
  actual_candidate_spread.png
```

04D.1 reuses the exact committed Checkpoint 04B pseudo-target memberships. It does not create a
new split design and it never inspects hidden FIRA target yields.

The exhaustive search is restricted to the method families that survived 04B:

- local Ridge with query-specific neighborhoods;
- direct graph-Laplacian regression;
- graph correction of cross-fitted PLS residuals;
- fixed GeoKNN10 and PLS4 anchors.

Primary tuning uses the 16 target-matched pseudo-competition splits. Hyperparameter-selection
performance is then measured leave-one-pseudo-split-out. Only a small finalist set is evaluated
on state-random and frozen state/municipality stress protocols.

The 59 real targets receive candidate predictions from finalists and an exploratory routing
proposal. These are not the final 04F submission.
