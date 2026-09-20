# GeoCebada checkpoints

This directory records stable project milestones so a human or AI agent can understand
what had already been completed before starting the next phase.

A checkpoint is **documentation of a reproducible state**, not a second source of truth.
When a checkpoint conflicts with immutable official data or directly inspected source
metadata, the source evidence wins and the checkpoint must be corrected.

## Checkpoints

| Checkpoint | Status | State captured |
|---|---|---|
| [01_data](01_data/README.md) | Closed | Official/external data acquisition, Data Contract v2 audit, normalization decisions and validated shared integration fixtures. |
| [02_features](02_features/README.md) | Closed | Validated/versioned 197-row feature table, feature-family inventory, train-vs-prediction diagnostics, frozen CV folds, ablations and baseline results. |
| [03_modeling](03_modeling/README.md) | Closed — First Modeling Delivery | Conventional feature discovery, nested model-family benchmark and fixed equal-weight ensemble baseline. |
| [03b_empirical_feature_discovery](03b_empirical_feature_discovery/README.md) | Closed | 335-feature X-only empirical layer plus leakage-safe fold-local expression recurrence audit. |
| [03c_representation_benchmark](03c_representation_benchmark/README.md) | Closed | Clean/competition B0/B1/B2/B3/B4/B5/B7 representation benchmark using frozen folds and fixed Ridge10/ExtraTrees. |
| [03c2_competition_modeling](03c2_competition_modeling/README.md) | Closed | Competition-only nested model-family tuning over C0/C1/C2/C3; workstation run completed with CatBoost on GPU. |
| [03c3_finalist_ensembles](03c3_finalist_ensembles/README.md) | Closed | Fixed equal-weight OOF ensemble review; retained E13/E123 as historical baseline evidence. |
| [04_transductive_competition](04_transductive_competition/README.md) | **Open / current** | Fixed-target transductive reconstruction: target topology, pseudo-competition validation, local/graph/domain adaptation, external enrichment and final 59-value reconstruction. |

Checkpoint **03_modeling** is frozen as the First Modeling Delivery. Checkpoint **04_transductive_competition** is now the active phase:

```text
checkpoints/
  01_data/
  02_features/
  03_modeling/
  04_transductive_competition/
```
