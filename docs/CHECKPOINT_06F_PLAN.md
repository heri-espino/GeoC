# Checkpoint 06F — Selective Center Expert / Abstention Gate

**Status:** COMPLETED / LOSO FAIL / NO PROMOTION  
**Track:** competition-only  
**Incumbent:** Local04D  
**Hidden 59 FIRA y may be read/scored:** NO  
**Post-hoc follow-up to 06E:** YES

## Motivation

Checkpoint 06E found one useful but unstable signal. The strongest center-only
candidate, G1 agronomic -> PCA 80% -> Ridge alpha=1000, reduced aggregate center
RMSE from about 0.3608 to 0.3514 t/ha, but its soft center weight also changed
some true tail predictions and the gain did not survive leave-one-split-out
selection.

06F tests one narrower hypothesis proposed after reviewing that result:

> use the center residual expert only when the observable-X tail classifiers
> are both sufficiently confident that the parcel is not in a tail; otherwise
> abstain and return Local04D unchanged.

No tail correction is attempted.

## Model

For each parcel define two fold-valid supervised scores:

```text
p_low(X)  = low-tail logistic score from water_productivity features
p_high(X) = high-tail logistic score from thermal features
```

For a frozen cutoff `tau`, the hard center gate is

```text
G_center(X; tau) = 1[p_low(X) < tau and p_high(X) < tau].
```

The prediction is

```text
Local04D + G_center(X; tau) * delta_center(X)
```

where `delta_center(X)` is the frozen center residual expert:

```text
all_agronomic -> StandardScaler -> PCA(80% variance) -> Ridge(alpha=1000)
```

The residual expert is trained only on fold-valid center rows and targets the
honest cross-fitted Local04D residual `y - Local04D_oof`.

If `G_center=0`, the prediction is **exactly Local04D**.

## What is frozen from 06E

06F does not reopen model-family selection. It freezes:

```text
low gate family       water_productivity
high gate family      thermal
logistic C            1.0
tail definitions      q15 / q85 within visible pseudo-train y
center representation all_agronomic_pca
PCA variance          0.80
Ridge alpha           1000
```

Only the abstention cutoff varies.

## Cutoff grid

The fixed grid is:

```text
0.40, 0.50, 0.60, 0.70, 0.80
```

This range was chosen from the already-generated 06E **X-derived gate-score
distribution only**, not from y-performance. Across the 656 development
pseudo-target rows it spans approximately 21% to 62% routing coverage.

Because the logistic classifiers use class balancing, these values are treated
as tail-score cutoffs rather than calibrated posterior probabilities.

## Leakage boundary

For each of the 16 target-matched pseudo-competition splits:

1. the 97 pseudo-train labels are visible;
2. the 41 pseudo-target labels are hidden;
3. q15/q85 tail thresholds are computed only from the 97 visible y values;
4. low/high classifiers are fit only on those 97 labels;
5. the center residual expert is fit only on visible center rows, using honest
   cross-fitted Local04D residuals;
6. both tail scores and the center residual correction are generated for the 41
   pseudo-target X rows;
7. the hard gate is applied without reading pseudo-target y;
8. only after all predictions are frozen are the 41 pseudo-target y values used
   for scoring.

The real 59 FIRA y values are never accessed.

## Diagnostics

For every cutoff 06F records:

- total fraction routed to the center expert;
- fraction of true center rows routed to the expert;
- fraction of true tail rows accidentally routed to the expert;
- precision of the routed set with respect to the scored center definition;
- overall, center, low-tail, high-tail and pooled RMSE.

These diagnostics are descriptive after scoring and are not used to redefine
the candidate grid.

## LOSO selection

On each leave-one-split-out iteration, a cutoff is eligible on the other 15
splits only if it:

- improves mean RMSE versus Local04D;
- improves pooled RMSE versus Local04D;
- improves center RMSE by at least 0.002 t/ha;
- keeps tail RMSE within
  `max(0.003 t/ha, 0.5% of incumbent tail RMSE)`.

Among eligible cutoffs, selection minimizes mean RMSE, then center RMSE, then
pooled RMSE. If none is eligible, the selector returns Local04D.

The aggregate 06F LOSO gate passes only if the split-excluded selected rule:

- lowers mean RMSE;
- lowers pooled RMSE;
- improves center RMSE by at least 0.002 t/ha;
- keeps tail RMSE inside the frozen tolerance.

## Important post-hoc status

06F was proposed **after observing the 06E results**. Therefore a LOSO PASS on
the same 16 development splits is not sufficient for promotion.

If 06F passes:

1. freeze exactly one cutoff;
2. do not tune the gate again;
3. create/use a fresh target-matched confirmation bank;
4. score only Local04D and the frozen 06F challenger;
5. change the canonical 59 predictions only if fresh confirmation passes.

If 06F fails, retain Local04D and close center/tail correction work on these
development splits.

## Run

```powershell
git pull --ff-only
git lfs pull
conda activate geocebada
python -m pip install -e ".[dev,geo]"

python tools\run_checkpoint_06f.py --preflight
```

If preflight passes:

```powershell
python tools\run_checkpoint_06f.py
```

The terminal decision is:

```text
06F_LOSO_GATE = PASS
```

or

```text
06F_LOSO_GATE = FAIL
```

Inspect:

```text
reports/checkpoint_06f/selective_summary.csv
reports/checkpoint_06f/routing_summary.csv
reports/checkpoint_06f/loso_selection.csv
reports/checkpoint_06f/loso_summary.json
reports/checkpoint_06f/checkpoint_06f_report.md
```

No 59-target prediction is generated by 06F.


## Completed result

06F was executed on 2026-10-07. The best full-development candidate was
`SelectiveCenter_t0p7` with mean RMSE 0.487108 and pooled RMSE 0.494704,
compared with Local04D 0.487845 / 0.495741. Center RMSE improved to 0.354471,
but tail RMSE worsened to 0.667687.

LOSO selection failed:

```text
selected mean RMSE    0.490847
selected pooled RMSE  0.498636
selected center RMSE  0.361407
selected tail RMSE    0.669269
06F_LOSO_GATE          FAIL
```

No fresh confirmation was run for 06F. Checkpoint 06 is closed and Checkpoint
07 is the active new-information modeling line.

Canonical interpretation: `docs/CHECKPOINT_06F_FINDINGS.md`.
