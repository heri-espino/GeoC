# Experiment: direct analog and tail corrections to Reference26 mean

**Date 2026-10-19 — experimental code implemented; private VM screening pending.**
Branch: `experiment/reference26-analog-tail-correction`. No official
prediction promotion; do not automatically merge or change Todoseboso.

## Motivation

Reference26 mean with arithmetic training-only imputation remains the
strongest interpretable model found in this competition, with authentic
138 × (137 training, 1 heldout) parcel LOOCV RMSE **0.468138 t/ha**.
A long chain of efforts with two synthetic Ridge experts, a supervised
three-class router, global Ridge and ExtraTrees trained with 256/1024
synthetic samples, 2×/5× weights on 8 hard or X-isolated observations
and balanced augmentation on ALL 137 training parcels did not improve
this benchmark.

The last VM full CV (PR #32) gave best augmented ExtraTrees 0.487998,
plain Ridge 0.492732; thus neither a larger fake dataset nor a
different global regressor should be presumed better.

A low-cost next test should work **directly on the existing frozen
Reference26 predictions**, not replace the winner.

## Candidate A: real agronomic feature analogs

For a real heldout parcel i, let `I_minus_i` denote the original
137 real **training** parcels. Standardize the 26 original champion
features using X from all 197 observable parcels, permitting
X-only transductive PCA (8 components) and comparison with original
26 robust-scaled variables.

Find k nearest real training X feature vectors to the heldout parcel.
The neighbor response estimate is the inverse-distance weighted
mean of **real allowed labels**:

```math
a_i = sum_{j in kNN(X_i; I_minus_i)}
          (w_ij y_j) / sum_j w_ij,
w_ij = 1 / max(d(X_i,X_j), 1e-5).
```

The corrected forecast is

```math
yhat_i = yhat_Reference26_LOO_i
      + clip(lambda * g_i *
         (a_i - yhat_Reference26_LOO_i), -delta, delta).
```

Two gates:
- `none`: g_i=1.
- `supported`: g_i is a product of X-only nearest-distance
  support and training-only neighbor-response agreement. Its distance
  scale is the median training X-space nearest-neighbor distance.

Candidate grid: feature representations original robust26 / PCA8,
k=3/8/16, lambda=0.1/0.25, gate=none/supported and
maximum correction delta=0.5 t/ha. This yields 24 analog candidates.
No synthetic data, no fitted classifier and no extra true labels.

Crucial: all neighbor Y and train-Y variability come exclusively from
the current 137 training parcel labels. If a hard heldout parcel is
X-dissimilar to its historical analogs, the support gate can suppress
the correction. This is deliberate and guards already-good forecasts.

## Candidate B: conservative tail expansion

The Checkpoint 06 findings showed empirical tail compression: small
observed yields often overpredicted and high yields often underpredicted.
Apply a **fixed, limited expansion** of the already-saved
Reference26 prediction away from the *other 137 training-label mean*:

```math
yhat_i = yhat_ref_i + clip(
 gamma * sign(yhat_ref_i - mean(y_train)) *
 max(|yhat_ref_i - mean(y_train)| - t*sd(y_train), 0),
 -delta, delta).
```

Try gamma=0.1/0.2/0.4 and t=0.5/1.0 SD. This is six
predeclared transformations, with the exact unmodified Reference26
prediction as a mandatory 31st row (1+24+6=31).

This is a screen, NOT a claim this form must improve anything. Earlier
tail-correction attempts were unsuccessful under different protocols.

## Leakage and selection

- Never use the current heldout y in a training neighbor average,
  training y mean, threshold or gate.
- Start from original *saved genuine* Reference26 mean 138-fold
  predictions, which were obtained without that heldout y.
- Using observable X from all 197 is permitted in this competition
  transductive setting. The 59 hidden Y values are never read.
- Do NOT fit the correction using other heldout OOF residuals as targets:
  their Reference26 models were fitted with the current heldout y,
  so a correction trained on those full 138 OOF residuals could leak.
- The user has repeatedly examined the same 138 labels and observed
  model errors. Selecting the minimum of 31 observed RMSE scores is
  **post-hoc exploratory model selection** and not independent
  confirmation of generalization.
- Improvement should be meaningful relative to 0.468138 t/ha on the
  same 138, not just the two historically hard parcels. Report
  changed/worsened cases on all 138 and on other 136.
- A score no better than Reference26 means *keep Reference26*. No
  candidate59 changes without explicit follow-up.

## Commands

```powershell
cd C:\Users\175199\GeoC
conda activate geocebada
git fetch origin
git switch experiment/reference26-analog-tail-correction
git pull --ff-only
python -m py_compile tools/run_reference26_analog_tail_screen.py
python -m pytest tests/test_reference26_analog_tail_screen.py -q
python tools/run_reference26_analog_tail_screen.py --check
python tools/run_reference26_analog_tail_screen.py --smoke
python tools/run_reference26_analog_tail_screen.py --run
python tools/run_reference26_analog_tail_screen.py --summary
```

Local private outputs (gitignored):

- `reports/reference26_analog_tail_screen/comparison_aggregate.csv`
- `reports/reference26_analog_tail_screen/private_heldout_predictions.csv`
- `reports/reference26_analog_tail_screen/run_spec.json`

A potential explicitly chosen candidate can be applied to the existing
138-fit 59-target **native Reference26 mean** predictions, but only
after evaluating this screen and only via the separate
`--predict59` flag. It produces private
`candidate59_private.csv`, never changes the original file, model
artifacts or deployment.

Example — only a demonstration, not a selected model:

```powershell
python tools/run_reference26_analog_tail_screen.py --predict59 `
  --method analog --geometry robust26 --k 8 --blend 0.1 `
  --gate supported
```

## Interpretation

This is a direct correction of the Reference26 predictions, not a
refitting of its native CT3/local Ridge/residual architecture. Its
attraction is that the original model stays fixed and the additional
information comes from genuine neighbor yields in the fold rather than
artificial pseudo-labels. Its limitation is that nearest X analogs may
not imply similar agronomic yield and Reference26 may already have
captured their useful information. Record a negative result if it fails.
