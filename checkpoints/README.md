# GeoCebada checkpoints

This directory records reproducible project milestones. A checkpoint is
documentation of a stable state, not a second source of truth.

| Checkpoint | Status | State captured |
|---|---|---|
| [01_data](01_data/README.md) | Closed | Data acquisition, semantics, audit and integration fixtures. |
| [02_features](02_features/README.md) | Closed | 197-row parcel table, diagnostics, frozen folds, ablations and baselines. |
| [03_modeling](03_modeling/README.md) | Closed | First global modeling delivery and frozen 03A–03C evidence. |
| [03b_empirical_feature_discovery](03b_empirical_feature_discovery/README.md) | Closed | X-only empirical layer plus fold-local expression audit. |
| [03c_representation_benchmark](03c_representation_benchmark/README.md) | Closed | Frozen representation benchmark. |
| [03c2_competition_modeling](03c2_competition_modeling/README.md) | Closed | Competition-only nested family tuning. |
| [03c3_finalist_ensembles](03c3_finalist_ensembles/README.md) | Closed | E13/E123 historical ensemble evidence. |
| [03d_large_global_modeling](03d_large_global_modeling/README.md) | **Closed — completed** | Balanced high-compute global closure and verified ONNX deployment path. |
| [04_transductive_competition](04_transductive_competition/README.md) | Closed / incumbent line | Target topology, pseudo-competition, local/graph/external modeling and Local04D. |
| [05_global_local_mixture](05_global_local_mixture/README.md) | **Closed — no promotion** | Cross-fitted global-local stacking/MoE; Local04D retained. |

Broad model search is closed. The canonical competition file is
`reports/checkpoint_05/final_predictions.csv`.

03D was executed after 05 as an additive closure of the global line. Its
state/grouped metrics are not directly interchangeable with Local04D's
target-matched metric. If one last modeling test is explicitly chosen,
`docs/CHECKPOINT_05B_PLAN.md` defines the narrow target-matched remix.
