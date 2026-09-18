# configs/

Machine-readable project contracts and reproducibility settings.

| Config | Purpose | Status |
|---|---|---|
| `base.yaml` | Core paths, challenge invariants, metrics and unresolved validation/cutoff fields | active |
| `data_contract_v2.yaml` | Full source/external audit contract | frozen for Checkpoint 01 unless evidence changes |
| `features_v1.yaml` | Feature Table v1 temporal/source/output contract | frozen for current Feature Table v1 |
| `checkpoint02.yaml` | Frozen Checkpoint 02 diagnostics, folds, ablations and baseline settings | closed/frozen |

Do not silently edit a frozen config to improve historical results. If a later phase changes a
contract, create/update the appropriate new-phase config and document the rationale in the
relevant checkpoint/history.

Current next phase: create a Checkpoint 03 modeling config rather than mutating
`checkpoint02.yaml`.

See `docs/AGENT_GUIDE.md` and `docs/PROJECT_HISTORY.md`.
