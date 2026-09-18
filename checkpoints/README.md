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
| [03_modeling](03_modeling/README.md) | In progress | 03A closed; 03B is empirical/data-driven feature discovery; 03C will compare representations and model families on the frozen folds. |
| [03b_empirical_feature_discovery](03b_empirical_feature_discovery/README.md) | In progress | Deterministic X-only empirical layer plus fold-local target-aware expression discovery. |

Checkpoint **03_modeling** is now open. Future checkpoints should continue the numeric sequence, for example:

```text
checkpoints/
  01_data/
  02_features/
  03_modeling/
  04_submission/
```
