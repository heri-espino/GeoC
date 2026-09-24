# Checkpoint 05 report namespace

**Status:** COMPLETED — NO PROMOTION  
**Completed:** 2026-09-23  
**Canonical interpretation:** `docs/CHECKPOINT_05_FINDINGS.md`

Checkpoint 05 tested whether fully cross-fitted global/local stacking and a
support-conditioned mixture-of-experts could improve the frozen Local04D
incumbent under target-matched pseudo-competition validation.

The completed run retained Local04D. The best full-development candidate was
`LocalE13_w0p10` (mean RMSE 0.487243, pooled 0.495261), but the predeclared
LOSO selector reached 0.487934 mean and 0.495933 pooled, slightly worse than
Local04D (0.487845 / 0.495741). The fresh confirmation bank was therefore not
scored.

Canonical output:

```text
reports/checkpoint_05/final_predictions.csv
```

It contains the same Local04D rule for exactly 59 target parcels.

The workstation runner is:

```text
python tools/run_checkpoint_05.py
```

Partial files beginning with `_partial_` exist only during interrupted runs
and are deleted after successful completion. Local joblib bundles under
`models/final/` remain gitignored.

Checkpoint 03D was executed later as an additive global-capacity closure. Its
state/grouped scores do not retroactively alter Checkpoint 05 because they use
different validation protocols. Any new Local04D + Global03D mixture requires
the separate narrow 05B gate.
