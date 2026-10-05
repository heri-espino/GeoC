# Checkpoint 06 — Tail-Aware Regime Refinement

**Status:** 06A/06B SUPPORTED; 06C IMPLEMENTED / RUN PENDING  
**Opened:** 2026-10-03  
**Canonical plan:** docs/CHECKPOINT_06_PLAN.md

Checkpoint 06 is motivated by an unverified team observation that a small
number of very low- and high-yield parcels may account for a disproportionate
share of Local04D/CatBoost squared error.

The checkpoint must first establish, using honest target-matched predictions:

1. whether squared error is actually concentrated in yield tails;
2. whether low/high residuals show systematic shrinkage toward the mean;
3. whether Local04D and CatBoost fail on the same parcels/directions;
4. whether tail membership or large residual risk is predictable from X;
5. which feature families and variables carry that information.

Only if those questions are supported may 06 test simple calibration, X-only
clustering, partially pooled regime corrections or a soft tail-risk gate.

This is not a broad model-search checkpoint.

Current incumbent:

    Local04D = LocalRidge_C4_all_deterministic_Geo_k24_a30_p1
    reports/checkpoint_05/final_predictions.csv

The first diagnostic runner is now implemented:

```powershell
git pull --ff-only
git lfs pull
conda activate geocebada
python -m pip install -e ".[dev,geo]"
python tools\run_checkpoint_06.py --preflight
python tools\run_checkpoint_06.py
```

This first run executes only 06A/06B. It does **not** cluster parcels, calibrate
the incumbent, train specialists or change the 59 final predictions.

Its terminal decision is:

```text
TAIL_HYPOTHESIS = SUPPORTED / NOT_SUPPORTED / AMBIGUOUS
```

Review `reports/checkpoint_06/checkpoint_06_report.md` before 06C or any
regime model is implemented.


## Checkpoint 06C — calibration before clustering

06A/06B returned `TAIL_HYPOTHESIS = SUPPORTED`, including strong low/high
shrinkage and X-predictable tail structure. Therefore the next implemented
stage is deliberately low capacity:

```powershell
python tools\run_checkpoint_06c.py --preflight
python tools\run_checkpoint_06c.py
```

06C fits calibrators only on the 97 pseudo-train rows whose base predictions
were themselves cross-fitted, then scores the untouched 41 pseudo-target rows.
Candidates are frozen stretch factors, Ridge residual corrections, piecewise
Ridge tail corrections, and a three-variable Local04D/CatBoost residual
correction. LOSO selection is mandatory.

Do not start X-only clustering unless 06C fails its split-excluded gate.
