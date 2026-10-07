# Checkpoint 06 — Tail-Aware Regime Refinement

**Status:** CLOSED / NO PROMOTION  
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


## Checkpoint 06D — X-only regime discovery

06C improved the same development splits slightly but failed LOSO: Local04D was
selected on all 16 holdouts. The next stage therefore tests whether the verified
tail bias depends on stable X-only regimes rather than one global calibration.

Run:

```powershell
python tools\run_checkpoint_06d.py --preflight
python tools\run_checkpoint_06d.py
```

Cluster selection uses **only X** from all 197 parcels. Candidate space is
G1 agronomic -> RobustScaler -> PCA (80% or 90%) -> KMeans/GMM with K=2,3,4.
Algorithm/K are chosen only from silhouette, cluster balance and stability
across seeds/subsamples. Yield and residuals are inspected only after one
clustering is frozen.


## Checkpoint 06E — supervised center/tail mixture

06D found stable agronomic clusters, but they do not explain Local04D residual
structure (`eta^2 ~= 0.004`, permutation `p ~= 0.935`). Therefore no
cluster-specific models are allowed.

The remaining supported mechanism is supervised tail probability. 06B showed
that low and high tails are distinguishable from X, especially with
`water_productivity` and `thermal` features.

06E keeps Local04D as the anchor and adds three soft corrections:

```text
Local04D
+ w_center(X) * center residual expert
+ w_low(X)    * shrunken low-tail offset
+ w_high(X)   * shrunken high-tail offset
```

The center expert is explicitly designed to improve non-outlier parcels. It
learns only honest Local04D residuals from fold-valid center rows.

Canonical design: `docs/CHECKPOINT_06E_PLAN.md`.

Run:

```powershell
python tools\run_checkpoint_06e.py --preflight
python tools\run_checkpoint_06e.py
```

Expected decision:

```text
06E_LOSO_GATE = PASS / FAIL
```


## Final Checkpoint 06 decision

06E was run and failed its LOSO promotion gate.

The aggregate development table showed a real center improvement for some
G1-PCA Ridge residual experts (center RMSE roughly 0.351 vs 0.361 for
Local04D), but split-excluded selection did not reproduce the gain. The LOSO
selector retained Local04D on 13/16 holdouts and the aggregate LOSO mean,
pooled, center and tail RMSE were all slightly worse than Local04D.

```text
06E_LOSO_GATE = FAIL
```

Checkpoint 06 is therefore closed without promotion. No fresh confirmation is
run and the canonical 59 predictions remain
`reports/checkpoint_05/final_predictions.csv`.

Canonical interpretation: `docs/CHECKPOINT_06_FINDINGS.md`.


## Checkpoint 06F — selective center expert / abstention gate

After 06E closed without promotion, one narrower post-hoc follow-up was proposed:
do not blend the center expert softly into every parcel. Instead, apply the
frozen best 06E center residual expert only when **both** supervised tail scores
are below a conservative cutoff; otherwise return Local04D unchanged.

06F freezes the G1-PCA(80%) Ridge(alpha=1000) center expert and the existing
water-productivity/thermal tail classifiers. It varies only five predeclared
tail-score cutoffs: 0.40, 0.50, 0.60, 0.70 and 0.80. There are no tail
corrections and no new model families.

Because this hypothesis was proposed after seeing 06E, LOSO on the same 16
splits is only a screening gate. Even a PASS cannot promote the model: exactly
one cutoff must then be frozen and evaluated on a fresh confirmation bank
against Local04D.

Canonical design: `docs/CHECKPOINT_06F_PLAN.md`.

Run:

```powershell
python tools\run_checkpoint_06f.py --preflight
python tools\run_checkpoint_06f.py
```

Expected decision:

```text
06F_LOSO_GATE = PASS / FAIL
```

Until a fresh confirmation passes, Local04D and
`reports/checkpoint_05/final_predictions.csv` remain canonical.


## Checkpoint 06F completed — no promotion

06F was run after the 06E soft-mixture result. The best full-development hard
gate was `SelectiveCenter_t0p7`:

```text
                         Local04D     SelectiveCenter_t0p7
mean RMSE                0.487845     0.487108
pooled RMSE              0.495741     0.494704
center pooled RMSE       0.360770     0.354471
tail pooled RMSE         0.664049     0.667687
```

However, the split-excluded selection rule again failed:

```text
LOSO selected mean RMSE    0.490847
LOSO selected pooled RMSE  0.498636
LOSO selected center RMSE  0.361407
LOSO selected tail RMSE    0.669269
06F_LOSO_GATE              FAIL
```

The selected methods across the 16 holdouts were Local04D 6/16,
SelectiveCenter_t0p6 5/16 and SelectiveCenter_t0p7 5/16.

Checkpoint 06A–06F is now closed. Do not continue threshold/gate tuning on the
same information. Canonical 06F interpretation:
`docs/CHECKPOINT_06F_FINDINGS.md`.

Checkpoint 07 is the new active line because it adds genuinely new data and a
new pretrained representation.
