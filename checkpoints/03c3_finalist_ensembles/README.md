# Checkpoint 03C.3 — Finalist equal-weight ensembles

**Status:** closed  
**Opened/closed:** 2026-09-19

03C.3 consumes only the frozen out-of-fold predictions from Checkpoint 03C.2. It
does not refit any model, inspect the 59 hidden targets or optimize continuous
ensemble weights.

## Frozen individual finalists

```text
F1 = C0_base + PLS
F2 = C3_base_agro_plus_discovery + Ridge
F3 = C1_agronomic + CatBoost
```

These were the three strongest individual 03C.2 candidates under the
cross-protocol worst-RMSE criterion.

## Ensembles evaluated

Only fixed equal-weight combinations were evaluated:

```text
E12  = 0.5 F1 + 0.5 F2
E13  = 0.5 F1 + 0.5 F3
E23  = 0.5 F2 + 0.5 F3
E123 = (F1 + F2 + F3) / 3
```

No continuous weight search was performed. This avoids adding another
high-variance tuning layer with only 138 labeled parcels.

## Results

| Candidate | State RMSE | Municipality RMSE | Worst RMSE |
|---|---:|---:|---:|
| E123 equal top 3 | 0.5121 | **0.7481** | **0.7481** |
| E13 PLS + CatBoost | **0.5082** | 0.7488 | 0.7488 |
| E23 Ridge + CatBoost | 0.5146 | 0.7544 | 0.7544 |
| E12 PLS + Ridge | 0.5270 | 0.7546 | 0.7546 |
| F1 PLS C0 | 0.5339 | 0.7621 | 0.7621 |
| F2 Ridge C3 | 0.5331 | 0.7650 | 0.7650 |
| F3 CatBoost C1 | 0.5235 | 0.7709 | 0.7709 |

Both retained ensembles improve municipality-grouped RMSE relative to every
individual finalist. Their grouped RMSE values differ by only about 0.0007, so
03C.3 does not claim that one is meaningfully superior.

## Retained finalists

```text
E13_PLS_CatBoost
E123_equal_top3
```

They are retained as a two-member final decision set for the final fitting
stage. The project should not add optimized weights or revive discarded model
families without a new explicit methodological reason.

ExtraTrees remains useful evidence from 03C.2, especially under the
state-stratified protocol, but it was not carried forward because its
municipality-grouped RMSE was materially weaker.

## Reproduce

```powershell
python tools\run_checkpoint_03c3.py
```

This is a lightweight deterministic run; no GPU is required.

## Next step

Define the final full-data fitting rule for the retained finalist(s), including
how nested-CV hyperparameter evidence is converted into a single full-training
configuration. Only after that rule is frozen should the project generate the
59 challenge predictions.

The 03C.3 scores are development evidence from reused frozen OOF folds, not an
independent final-test estimate.
