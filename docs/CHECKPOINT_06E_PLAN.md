# Checkpoint 06E — Supervised Center/Tail Mixture

**Status:** IMPLEMENTED / WORKSTATION RUN PENDING  
**Track:** competition-only  
**Incumbent:** Local04D  
**Hidden 59 FIRA y may be read/scored:** NO

## Why 06E exists

Checkpoint 06 established three different facts:

1. **The remaining error is strongly concentrated.** The largest-error 10% of
   parcels account for about 62% of Local04D parcel SSE.
2. **The tails are systematically compressed.** Low-yield parcels are
   overpredicted and high-yield parcels are underpredicted.
3. **Unsupervised X-only clusters do not explain Local04D residuals.**
   Checkpoint 06D found stable agronomic clusters, but their Local04D residual
   effect was essentially zero (`eta^2 ~= 0.004`, permutation `p ~= 0.935`).

Checkpoint 06C also showed that one global calibration is too blunt: it can
improve the full development table slightly, but it failed leave-one-split-out
selection in all 16 holdouts.

The remaining supported hypothesis is therefore **supervised regime
probability**, not clustering.

## Scientific question

Can we improve the already-well-predicted center while protecting the tails by
using observable X to estimate whether a parcel is likely to be low, central or
high yield?

The proposed model is

```text
prediction =
    Local04D
    + w_center(X) * delta_center(X)
    + w_low(X)    * delta_low
    + w_high(X)   * delta_high
```

Local04D remains the anchor. The extra terms are corrections, not independent
replacement models.

## Tail gates

Checkpoint 06B found different X families for the two tails:

```text
low yield:   water_productivity   mean split ROC-AUC ~= 0.889
high yield:  thermal              mean split ROC-AUC ~= 0.805
```

For every target-matched pseudo-competition split, 06E fits two independent
class-weighted logistic models using only the 97 visible pseudo-train labels:

```text
p_low(X)  = P(low tail | X_water_productivity)
p_high(X) = P(high tail | X_thermal)
```

The thresholds are fold-valid q15/q85 values computed from the visible
pseudo-train y only.

The separate probabilities are converted into normalized soft weights:

```text
raw_center = (1 - p_low) * (1 - p_high)

w_low    = p_low / denominator
w_center = raw_center / denominator
w_high   = p_high / denominator
```

No hard routing is used.

## Center specialist

The center specialist is the part added specifically to answer the team's new
question: can the non-outlier parcels be predicted even better?

For the visible pseudo-train rows only:

1. define center using the fold-valid q15/q85 thresholds;
2. use the **honest cross-fitted Local04D residual**
   `y - Local04D_oof` as the target;
3. fit a strongly regularized Ridge residual model using only center parcels.

Two deliberately small representations are tested:

- `cross_domain`: compact agronomic interaction features;
- `all_agronomic_pca`: all G1 agronomic features compressed to 80% PCA variance.

Ridge alphas are frozen to:

```text
10, 100, 1000
```

The center correction is multiplied by `w_center(X)`, so it is automatically
suppressed when the gate thinks the parcel is likely to be a tail case.

## Tail specialists

Because the tail samples are small, 06E does **not** train separate CatBoost or
other high-capacity regressors inside the tails.

Instead, it estimates the mean honest Local04D residual separately in the low
and high pseudo-train tails and shrinks those offsets toward zero.

Frozen shrink factors:

```text
0.25, 0.50, 0.75
```

This creates conservative, direction-aware corrections:

```text
low tail:   typically negative correction
high tail:  typically positive correction
```

Each is multiplied by its corresponding soft probability.

## Candidate universe

06E evaluates:

1. Local04D unchanged;
2. tail-gate-only corrections;
3. center-only residual experts;
4. center + asymmetric low/high soft corrections.

There is no arbitrary model zoo and no clustering search in 06E.

## Leakage boundary

For each of the 16 target-matched pseudo-competition splits:

- 97 pseudo-train rows are visible;
- their Local04D/CatBoost base predictions are cross-fitted;
- q15/q85 thresholds are derived only from those 97 y values;
- tail classifiers are fit only on those 97 labels;
- center residual experts are fit only on center rows among those 97;
- tail offsets are computed only from those 97;
- the 41 pseudo-target y values are untouched until scoring.

The real 59 FIRA target y values are never available or used.

## What we expect to find

There are three plausible outcomes.

### A. Center specialist improves reliably

If the center residual contains learnable structure, a gated center specialist
should reduce center RMSE below the Local04D center benchmark while leaving the
tails approximately unchanged.

This is the main new hypothesis.

### B. Tail gates help but the center remains irreducible

Tail corrections may improve overall RMSE while the center expert is selected
out. This would mean Local04D is already close to the noise floor in ordinary
parcels.

### C. Neither survives LOSO

If split-excluded selection falls back to Local04D, then the remaining residual
structure is not stable enough to exploit with the current data. Checkpoint 06
should then close rather than adding complexity post hoc.

## Selection gate

Candidate selection on the other 15 development splits requires:

- lower mean RMSE than Local04D;
- lower pooled RMSE than Local04D;
- center RMSE improvement of at least 0.002 t/ha;
- tail RMSE no worse than
  `max(0.005 t/ha, 1% of incumbent tail RMSE)`.

The selected method is then scored on the excluded 16th split.

The aggregate LOSO gate passes only if the split-excluded selected rule has:

- lower mean RMSE;
- lower pooled RMSE;
- lower center RMSE;
- tail RMSE within the frozen tolerance.

## If 06E passes

Do not tune further on the same 16 development splits.

Freeze exactly one challenger and create the already-reserved fresh
target-matched confirmation bank. Score only:

- Local04D;
- frozen 06E challenger.

Only after fresh confirmation may the 59 final target predictions change.

## If 06E fails

Retain Local04D and close the current Checkpoint 06 modeling path.

A new modeling attempt would require a genuinely new, predeclared source of
information or hypothesis, not another post-hoc variation of the same
center/tail corrections.

## Run

```powershell
git pull --ff-only
git lfs pull
conda activate geocebada
python -m pip install -e ".[dev,geo]"

python tools\run_checkpoint_06e.py --preflight
python tools\run_checkpoint_06e.py
```

Expected terminal decision:

```text
06E_LOSO_GATE = PASS / FAIL
```

Outputs are written to `reports/checkpoint_06e/`.
