# GeoCebada checkpoints

This directory records stable project milestones so a human or AI agent can understand
what had already been completed before starting the next phase.

A checkpoint is **documentation of a reproducible state**, not a second source of truth.
When a checkpoint conflicts with immutable official data or directly inspected source
metadata, the source evidence wins and the checkpoint must be corrected.

## Checkpoints

| Checkpoint | State captured |
|---|---|
| [01_data](01_data/README.md) | Official/external data acquisition, Data Contract v2 audit, normalization decisions and validated shared integration fixtures. |

Future checkpoints should continue the numeric sequence, for example:

```text
checkpoints/
  01_data/
  02_features/
  03_modeling/
  04_submission/
```
