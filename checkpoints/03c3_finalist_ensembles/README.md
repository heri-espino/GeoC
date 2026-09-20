# Checkpoint 03C.3 — Finalist equal-weight ensembles

> **First Modeling Delivery remark (2026-09-19):** E13 and E123 are frozen as baseline
> evidence, not as mandatory final models. The project subsequently adopted fixed-target
> transductive reconstruction in Checkpoint 04, which explicitly reopens model search,
> target-specific models, external enrichment and ensemble design.


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

Within the historical Checkpoint 03 doctrine, they were retained as a two-member final
decision set. That restriction was superseded when Checkpoint 04 adopted transductive
fixed-target reconstruction; E13/E123 now serve as baseline evidence rather than a limit on
new model families or ensemble weights.

ExtraTrees remains useful evidence from 03C.2, especially under the
state-stratified protocol, but it was not carried forward because its
municipality-grouped RMSE was materially weaker.

## Reproduce

```powershell
python tools\run_checkpoint_03c3.py
```

This is a lightweight deterministic run; no GPU is required.

## Historical handoff

The next active stage is `checkpoints/04_transductive_competition/README.md`. Do not proceed directly from this checkpoint to a final E13/E123 submission without evaluating the new transductive strategies.

The 03C.3 scores are development evidence from reused frozen OOF folds, not an independent final-test estimate.
