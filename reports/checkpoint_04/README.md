# Checkpoint 04 reports

**Status:** placeholder for active transductive modeling outputs.

Expected report families:

```text
04A_target_topology/
  target_support.csv
  nearest_neighbors.csv
  spatial_autocorrelation.json
  temporal_similarity_summary.csv
  adversarial_shift.json

04B_transductive_validation/
  pseudo_competition_splits.csv
  pseudo_test_predictions.csv
  protocol_summary.csv

04C_global_models/
04D_local_graph_shift/
04E_external_enrichment/
04F_final_ensemble/
```

Every scored method must preserve pseudo-target y until scoring time while allowing pseudo-target
X to participate in X-only transductive operations.

Historical Checkpoint 02/03 reports remain immutable baseline evidence.
