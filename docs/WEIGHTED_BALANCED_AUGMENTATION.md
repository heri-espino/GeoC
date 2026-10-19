# Weighted and balanced augmentation: real 137/1 LOOCV

**2026-10-19. Draft experiment; real run not yet executed.** Branch:
`experiment/weighted-balanced-augmentation` off the completed negative
global-augmentation study PR #31. No changes to the official
Reference26 mean joblib, deployed website, or 59 forecasts.

## Motivation

Previously tested global Ridge/ExtraTrees models trained on 137 real rows
and 256/1024 synthetic rows around only the difficult cases did not improve
the true 138-fold Reference26 mean baseline (RMSE 0.468138 t/ha), nor their
own **no augmentation** controls. The best synthetic candidate had
RMSE 0.506404. The most aggressive rare-case clouding shifted estimates
toward two extreme cases at the cost of the other 136.

The next experiments distinguish:

1. **Sample importance**, not feature importance: weigh real hard-to-predict
   *parcels* more strongly. If the user means weighing feature columns
   instead, that is a separate analysis (rescaling X / feature learning).
2. **Uniform augmentation of every fold-training parcel**: make the same
   number of synthetic neighbors around each of the 137 real training
   observations, instead of only the two rare anchors. Synthetic rows
   inherit the real center's y label.
3. **Larger but controlled augmented datasets**: 2 and 10 replicas
   per real observation by default; 50 is available as a separate
   large stress-test. Test fixed total synthetic sample mass against
   giving every pseudo-observation full weight.

Two **global learner** families (Ridge and ExtraTrees) remain the
targets of this experiment. The native multifamily Reference26 is not
retrained with synthetic rows and remains an authentic saved 138-LOOCV
control. Do not claim augmentation of native Reference26.

## Weighted observations selected safely

For each outer fold, there are 137 original observed yields in train
and 1 original yield held out.

Two ways to pick hard training rows, both **inside the current 137 labels**:

- `press`: largest 8 inner LOOCV absolute residuals of an
  inexpensive auxiliary Ridge model using the **same 137**
  allowed y labels. Obtained by exact Ridge PRESS identities.
- `sparse`: largest nearest-neighbor distances in observable X
  among the **137** real training feature vectors. This is a
  feature-space rarity score; no y required.

Only the selected training rows receive an importance multiplier
(2× or 5× by default). We normalize all weights so their total
always sums to 137; this avoids inadvertently changing the effective
Ridge regularization by multiplying every sample weight.

**Never** use the previously computed globally selected Reference26
LOOCV errors to assign weights inside an outer fold: such OOF
predictions for other parcels may have been obtained with the
current heldout parcel's y in training, causing fold contamination.

The fold-safe hard observation sets are *not necessarily* the same
two historical anomalies. Thus the experiment tests a reproducible
selection mechanism, not magic recognition of those two holdouts.

## Uniform Gaussian augmentation of all real training parcels

For every real training parcel i and j=1,...,r, create

```math
X_(i,j) = X_i + sigma * L_i * eps_(i,j),
eps_(i,j) ~ N(0,I),
Y_(i,j) = Y_i.
```

Each `L_i` is a shrunk X-only local covariance root computed using
20 neighbors in the 197 observable feature vectors; normalized X
positions are clipped to [-9,9]. There is no added yield noise.
This is a *local constancy of response* modeling assumption, not
new independent agricultural information.

Training/test row counts:
- `r=2`: 137 real + 274 artificial = 411 fit rows; 1 real test.
- `r=10`: 137 real + 1,370 artificial = 1,507 fit rows; 1 real test.
- `r=50` (optional heavy): 137 real + 6,850 artificial =
  6,987 fit rows; 1 real test.

Only 138 genuine heldout yields can assess those fits.
All pseudo-y are calculated from allowed training labels, never from
the heldout's y.

Mass controls:
- `equal`: synthetic **block** has same total fitting weight
  as the real block. If r increases, synthetic rows each get
  smaller individual weights. After normalization, the entire
  sample-weight vector sums to 137.
- `full`: artificial row starts with the same fitting weight as
  a real row. For r=10, synthetic block has 10 times the
  combined real-block mass. Normalize whole vector to 137 while
  preserving relative weights.

With sigma=0 and `equal`, weighted Gaussian replication of
every real X,y should reproduce the same Ridge fit as without
augmentation (test checks numerical equality). This is a crucial
implementation sanity check. For ExtraTrees duplication can still
affect split/leaf behavior.

## Evaluation

Each candidate fits **one global learner on the genuine 137 labels**
(optionally + synthetics) and predicts **one excluded original
parcel**. Reference26 mean LOOCV predictions are loaded from the
existing verified 138-outer-fold bundle, not retrained or recomputed.

Default grid: 2 global learners × [1 plain + (2 hardness methods ×
2 importance multipliers) + (2 replication counts × 2 sigmas ×
2 synthetic masses)] = **26 predictions per fold**, 138 real
heldouts. The weight multiplier grid and Gaussian settings are
predeclared. Scores include full 138 RMSE and MAE, historical
hard-two RMSE (descriptive post-selection), other-136 RMSE,
count of other-136 worsening, paired squared-error difference versus
plain SAME learner and original Reference26 mean, plus maximum
predicted-value shift.

The historical two hard observations were discovered by prior
full-138 OOF error review: those subgroup metrics are descriptive
and post-selection, not an independent accuracy estimate.
Full 138 RMSE grid comparison is exploratory because the work
was motivated by inspecting these same folds.

## Workstation execution

```powershell
cd C:\Users\175199\GeoC
conda activate geocebada
git fetch origin
git switch experiment/weighted-balanced-augmentation
git pull --ff-only

python -m pytest tests/test_weighted_balanced_augmentation.py -q
python -m py_compile tools/run_weighted_balanced_augmentation.py

python tools/run_weighted_balanced_augmentation.py --check
python tools/run_weighted_balanced_augmentation.py --smoke
python tools/run_weighted_balanced_augmentation.py --run --jobs 4
python tools/run_weighted_balanced_augmentation.py --summary
```

The output folder is private and ignored by git:

```text
reports/weighted_balanced_augmentation/
  run_spec.json
  fold_001.csv ... fold_138.csv
  comparison_aggregate.csv
  private_fold_predictions.csv
```

`--smoke` uses 4 real fold tests only and is **not** a
full LOOCV RMSE.

An optional stress test is available **after** the default grid
is checked. Use a distinct folder and a smaller model list to
control ExtraTrees computational cost:

```powershell
python tools/run_weighted_balanced_augmentation.py --run --jobs 4 `
  --models ridge --replicas 50 --sigmas 0.1 `
  --output reports/weighted_balanced_augmentation_large
```

This reruns all 138 real heldouts but produces 6,850 synthetic
positions *per heldout fit* (6,987 training positions total).
The original 59 unknown y remain unknown and untouched.

## Decision

A convincing augmentation should improve the SAME learner versus
its unaugmented control, and preferably improve the Reference26 mean
baseline without sacrificing the ordinary 136 cases. If all these
variants fail, the evidence favors retaining Reference26 and
investigating model bias/feature support rather than increasing
synthetic counts without bound.
