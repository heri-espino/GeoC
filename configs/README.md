# configs/

Machine-readable project contracts and reproducibility settings.

| Config | Purpose | Status |
|---|---|---|
| `base.yaml` | Core paths, challenge invariants, metrics and unresolved validation/cutoff fields | active |
| `data_contract_v2.yaml` | Full source/external audit contract | frozen for Checkpoint 01 unless evidence changes |
| `features_v1.yaml` | Feature Table v1 temporal/source/output contract | frozen for current Feature Table v1 |
| `checkpoint02.yaml` | Frozen Checkpoint 02 diagnostics, folds, ablations and baseline settings | closed/frozen |
| `agronomic_features_v1.yaml` | Explicit target-free nonlinear/agronomic transformations over Feature Table v1 | Checkpoint 03A closed |
| `empirical_features_v1.yaml` | Deterministic empirical X-only transforms plus fold-local expression-discovery contract | Checkpoint 03B |

Do not silently edit a frozen config to improve historical results. If a later phase changes a
contract, create/update the appropriate new-phase config and document the rationale in the
relevant checkpoint/history.

Checkpoint 03A has its own `agronomic_features_v1.yaml`; do not mutate the frozen Checkpoint 02 config. Checkpoint 03B uses `empirical_features_v1.yaml`. Model comparison moves to 03C and should receive its own experiment config.

See `docs/AGENT_GUIDE.md` and `docs/PROJECT_HISTORY.md`.
