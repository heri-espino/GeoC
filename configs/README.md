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
| `checkpoint04b.yaml` | Repeated transductive pseudo-competition validation with global/local/graph methods and X-only routing | 04B complete |
| `checkpoint04d1.yaml` | Exhaustive local/graph refinement, residual-graph correction and LOSO routing validation | 04D.1 complete |
| `checkpoint04c1.yaml` | Focused C1 agronomic CatBoost anchor and low-weight blend validation | **04C.1 implemented; workstation run next** |
| `checkpoint04e1.yaml` | Exact SIAP 2025 localization and external-prior validation | 04E.1 complete |
| `checkpoint04f.yaml` | Final Local04D freeze, Local/Graph sensitivity and 59-row output contract | 04F complete/frozen |
| `checkpoint05.yaml` | Cross-fitted global/local stacking, support-conditioned mixture-of-experts and conservative promotion gate | **05 implemented; workstation run next** |

Do not silently edit frozen Checkpoint 01–03 configs to improve historical results. Checkpoint
03 is preserved as the First Modeling Delivery.

Active doctrine is defined in `docs/TRANSDUCTIVE_OBJECTIVE.md`; active implementation should
create Checkpoint 04-specific configuration rather than mutating historical contracts.

See `docs/AGENT_GUIDE.md` and `docs/PROJECT_HISTORY.md`.


### `checkpoint04e1.yaml`

Checkpoint 04E.1 external-localization configuration. Freezes the exact SIAP
scope (`Cebada grano + Primavera-Verano + Temporal + CVEGEO`), historical
window, 04D.1 local/graph anchors, affine calibration strengths and fixed blend
weights.


### `checkpoint04c1.yaml`

Checkpoint 04C.1 focused global-anchor configuration. It freezes three CatBoost
candidates inherited from 03C.2, seeds 42/314/2718, GPU-first execution, fixed
10/20/30% blend weights and the controlled target-matched LOSO candidate universe.
Automatic CPU fallback is disabled.


### `checkpoint04f.yaml`

Checkpoint 04F freezes `Baseline_Local04D` as the final rule, verifies exact
04D.1/04E.1 prediction provenance, limits Local/Graph blending to a diagnostic
sensitivity grid and specifies the exact 59-row final-output contract.


### `checkpoint05.yaml`

Checkpoint 05 combines the surviving Checkpoint 03 global experts with the
Checkpoint 04 Local04D/Graph04D experts. It freezes the nested cross-fitting
contract, support descriptors, meta-model candidate family, GPU CatBoost seeds,
LOSO promotion rule and final model-bundle path.
