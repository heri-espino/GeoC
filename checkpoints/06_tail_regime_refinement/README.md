# Checkpoint 06 — Tail-Aware Regime Refinement

**Status:** PLANNED / NOT IMPLEMENTED / NOT RUN  
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

No Checkpoint 06 scientific result exists yet.
