# Checkpoint 05 — Global–Local Mixture Plan

**Status:** COMPLETED 2026-09-23 — NO PROMOTION  
**Results:** `docs/CHECKPOINT_05_FINDINGS.md`

The implemented protocol below was executed. The best development blend did
not survive the leave-one-development-split-out gate, so Local04D was retained
and the fresh confirmation bank was intentionally not scored.

## Why this is Checkpoint 05

Checkpoint 03 answered which global representations and model families are useful.
Checkpoint 04 answered how to exploit the fixed target set, locality, graph structure, and
external evidence.

Checkpoint 05 deliberately combines those two lines of evidence. It is therefore a new
checkpoint rather than a substage of 03 or 04.

## Incumbent

The incumbent entering Checkpoint 05 is the frozen Checkpoint 04F rule:

LocalRidge_C4_all_deterministic_Geo_k24_a30_p1

Target-matched mean RMSE: 0.487845.

## New hypothesis

The hypothesis is not that a more complicated model must be better. It is:

> Global experts from Checkpoint 03 and local/transductive experts from Checkpoint 04 may make
> partially different errors. A fully cross-fitted support-conditioned ensemble may exploit
> that complementarity without exposing any pseudo-target or hidden FIRA y to meta-training.

## Leakage boundary

Every outer pseudo-competition is treated as an untouched test set.

Inside the visible labels, base-expert predictions used to train the stack are themselves
cross-fitted. Global expert hyperparameters and fold-local expression discovery are fitted
only inside the corresponding cross-fit training subset.

Local and graph experts use only labels visible to that fold, while X-only topology may use all
197 covariate rows under the fixed-target transductive contract.

## Mixture-of-experts

For M experts and support vector s_i:

\[
w_{im}
=
\frac{\exp(\eta_{im})}
{\sum_{\ell=1}^M \exp(\eta_{i\ell})},
\qquad
\eta_{im} = \beta_{m0}+\beta_m^\top s_i,
\]

and

\[
\hat y_i
=
\sum_{m=1}^M w_{im}\hat y_{im}.
\]

The final expert is used as the reference logit for identifiability. Gate coefficients receive
L2 regularization selected only from OOF meta-training rows.

## Why several stackers are still included

Mixture-of-experts is the main new hypothesis, but simpler stackers are necessary controls.
If a simpler linear stack beats the support-conditioned gate, the project should not prefer
the more complicated explanation.

## Final promotion

Checkpoint 05 uses two distinct target-matched banks.

### Development bank

The original 16 frozen Checkpoint 04B target-matched splits are reused for model comparison,
meta-model tuning and the leave-one-development-split-out selector. They are deliberately not
treated as fresh evidence because Local04D itself was refined on this bank during 04D.1.

### Fresh confirmation bank

If and only if a non-incumbent challenger passes the development + LOSO gate, the runner
deterministically generates 16 new X-only target-matched masks using the same Checkpoint 04B
matching machinery but an unused random seed. The candidate is fixed before any yield from
this bank is scored.

Only two methods are evaluated for the promotion decision on this bank:

- fixed Local04D incumbent;
- the single challenger preselected from development.

The challenger is promoted only if it improves both mean and pooled RMSE on the fresh bank.
Exact duplicate masks from the original development bank are rejected.

This additional holdout layer is important because the original 16 target-matched masks have
already influenced 04D model refinement. It provides new pseudo-competition evidence without
using any hidden FIRA yield.

## Final export

After the promotion decision, the selected procedure is fitted using all 138 visible labels
and predicts the fixed 59 target parcels. The runner writes the final CSV, detailed
diagnostics, a JSON manifest and two local joblib bundles.

The lightweight `checkpoint05_app_bundle.joblib` contains only the fixed-target prediction
contract, candidate values, diagnostics and metadata; it intentionally contains no fitted
CatBoost/scikit-learn objects and is the preferred artifact for the later Streamlit app.

The full `checkpoint05_model.joblib` additionally preserves fitted global experts and
meta-models for exact modeling provenance. Both bundles are scoped to the fixed 59 targets,
exclude the cross-fitted meta-training label table, are ignored by Git, and are round-trip
verified before the runner reports PASS.
