# configs/

Machine-readable project contracts and reproducibility settings.

| Config | Purpose | Status |
|---|---|---|
| `base.yaml` | Core project invariants and active fixed-target transductive objective | active |
| `data_contract_v2.yaml` | Full source/external audit contract | frozen for Checkpoint 01 unless evidence changes |
| `features_v1.yaml` | Feature Table v1 temporal/source/output contract | frozen |
| `checkpoint02.yaml` | Frozen diagnostics, folds, ablations and baseline settings | closed/frozen |
| `agronomic_features_v1.yaml` | Target-free agronomic/nonlinear transformations | Checkpoint 03A closed |
| `empirical_features_v1.yaml` | Deterministic empirical X-only transforms and historical discovery contract | Checkpoint 03B closed |
| `checkpoint03c.yaml` | Representation benchmark | Checkpoint 03C.1 closed/frozen |
| `checkpoint03c2.yaml` | Competition-only nested model-family benchmark | Checkpoint 03C.2 closed/frozen |
| `checkpoint03c3.yaml` | Fixed finalist/equal-weight OOF ensemble review | Checkpoint 03C.3 closed/frozen |
| `checkpoint04.yaml` | Fixed-target transductive objective, validation and phase contract | **Checkpoint 04 active** |
| `checkpoint04a.yaml` | Target topology, temporal similarity, shift, autocorrelation and support diagnostics | 04A complete |
| `checkpoint04b.yaml` | Repeated transductive pseudo-competition validation with global/local/graph methods and X-only routing | **04B implemented; workstation run next** |

Do not silently edit frozen Checkpoint 01–03 configs to improve historical results. Checkpoint
03 is preserved as the First Modeling Delivery.

Active doctrine is defined in `docs/TRANSDUCTIVE_OBJECTIVE.md`; active implementation should
create Checkpoint 04-specific configuration rather than mutating historical contracts.

See `docs/AGENT_GUIDE.md` and `docs/PROJECT_HISTORY.md`.
