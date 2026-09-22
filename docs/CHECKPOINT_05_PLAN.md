# Checkpoint 05 — Global–Local Mixture Plan

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

The full target-matched table identifies the strongest development candidate, but it is not
sufficient for promotion. The method-selection procedure is itself evaluated leave-one-
pseudo-split-out. A Checkpoint 05 model replaces Local04D only if that split-excluded selector
improves both mean and pooled RMSE relative to the frozen incumbent.

This rule is predeclared before the workstation run.
