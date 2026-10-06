# Checkpoint 06 plan — Tail-Aware Regime Refinement

**Status:** CLOSED / NO PROMOTION  
**Opened:** 2026-10-03  
**Track:** competition-only  
**Incumbent:** Local04D = LocalRidge_C4_all_deterministic_Geo_k24_a30_p1  
**Canonical current predictions:** reports/checkpoint_05/final_predictions.csv  
**Hidden FIRA target y may be read or scored:** NO

Checkpoint 06 is a narrow, hypothesis-driven reopening of modeling. It does not
reopen broad hyperparameter sweeps, arbitrary feature engineering or a new
model zoo.

The motivating observation came from a team member's inspection of residuals:
most parcels appeared to be predicted well, while a small number of very
low-yield and high-yield parcels seemed to account for unusually large errors.
The team further suspected that both CatBoost and Local04D may regress these
extreme parcels toward the center.

That observation is plausible but **has not yet been established rigorously**.
The first responsibility of Checkpoint 06 is to verify whether it is true,
quantify where it is true, identify which models and parcels exhibit it, and
determine whether the effect is predictable from observable X.

Only after that verification may 06 attempt a tail-aware correction.

## 1. Scientific question

Checkpoint 06 asks:

> Is the remaining squared error concentrated in predictable yield regimes,
> especially the low and high tails, and can those regimes be recognized from
> observable parcel covariates well enough to improve the fixed-target
> prediction without harming the well-predicted center?

The chain that must be demonstrated is:

    observable X
        -> recognizable agronomic/spatial/phenological regime
        -> systematic base-model shrinkage or residual bias
        -> tail-aware calibration or partially pooled correction
        -> lower honest target-matched RMSE

If a link fails, the corresponding more complicated model should not be built
or promoted.

## 2. Why tails may matter for RMSE

RMSE squares residuals:

    RMSE = sqrt(mean((y - y_hat)^2)).

For each parcel define residual e = y - y_hat, absolute error |e| and squared
error e^2. A small number of large residuals can therefore dominate total SSE
even when most parcels are predicted accurately.

Checkpoint 06 must report the cumulative SSE contribution of:

- top 5 residual parcels;
- top 10;
- top 10%;
- top 15%;
- top 20%.

The teammate claim is supported only if a small subset contributes a clearly
disproportionate share of SSE in honest out-of-fold / pseudo-competition
predictions, not merely in a fitted-on-all-data plot.

## 3. Distinguish six different concepts

Do not conflate:

1. extreme y — unusually low or high observed yield;
2. large residual — badly predicted parcel;
3. X-outlier — unusual covariate vector;
4. low-support parcel — few similar labeled neighbors;
5. natural X-regime — cluster or latent region of covariate space;
6. tail-risk parcel — X implies elevated probability of belonging to a yield tail.

A parcel can satisfy one, several or none of these.

The relevant question is not merely whether extreme y exists. It is whether
observable data contain enough information to anticipate where current models
fail.

## 4. Evidence entering Checkpoint 06

Current incumbent target-matched evidence:

    Local04D mean split RMSE   0.487845
    Local04D pooled RMSE       0.495741
    Local04D pooled MAE        0.342501

Checkpoint 04C.1 already found:

    CatBoost standalone target-matched   approximately 0.515–0.521
    Local + 10% CatBoost mean RMSE       0.487842
    Local04D mean RMSE                   0.487845
    Local + 10% CatBoost pooled RMSE     0.495915
    Local04D pooled RMSE                 0.495741
    Local/CatBoost residual correlation  0.9423
    LOSO controlled blend                0.488884

The high residual correlation is a warning that the two models often make
similar errors. It does not prove that their largest tail errors are identical
or that those errors are unpredictable from X.

Checkpoint 05 later showed that generic stacking and a support-conditioned
mixture-of-experts did not survive promotion. Checkpoint 06 must therefore not
repeat a generic stacker zoo. Its new information must be specifically
tail/regime related.

## 5. Validation boundary

All Checkpoint 06 claims must use honest predictions.

### 5.1 Primary validation

Reuse the target-matched pseudo-competition framework:

1. hide pseudo-target y;
2. keep pseudo-target X visible;
3. allow X-only transforms permitted by the transductive contract;
4. every y-aware step sees only visible training y;
5. freeze predictions;
6. reveal pseudo-target y only for scoring.

The historical 16 target-matched splits may be used for development and LOSO
selection.

### 5.2 Fresh confirmation

Checkpoint 06 must create a new deterministic target-matched confirmation bank
with an unused seed. The seed must be written to config before any scores are
inspected.

Only Local04D and one frozen challenger may be scored on fresh confirmation.

### 5.3 Stress tests

State-stratified and municipality-grouped folds remain stress tests. They are
not interchangeable with the target-matched objective.

### 5.4 Final all-138 refit

Training on all 138 labels is allowed only after selection. In-sample fit on all
138 may never be used as evidence of improvement.

The existing final predictors already use all 138 visible labels when producing
the 59 target predictions. Therefore "train on all data" is not itself a new
solution to tail error.

# Phase 06A — Verify the tail-error hypothesis

**Mandatory. No specialist model may be implemented before this phase is
reviewed.**

## 6. Comparable honest residual table

Construct target-matched honest predictions for at least:

- Local04D;
- CatBoost_C1;
- Graph04D as a locality/topology sensitivity model;
- a simple Local/CatBoost average only as a diagnostic.

A Checkpoint 03D global such as XGBoost may be included only after it is
evaluated under the same target-matched protocol. Do not mix 03D state/grouped
scores with target-matched residuals.

For every labeled parcel/model appearance record:

    ID_POLIGONO
    split_id
    observed_y
    prediction
    residual = observed_y - prediction
    absolute_error
    squared_error
    state
    municipality
    support metrics
    model disagreements

Because the 16 pseudo-splits reuse parcels, also aggregate by parcel:

- number of honest appearances;
- mean and median honest prediction;
- mean/median residual;
- mean squared error;
- worst absolute error;
- prediction variability across pseudo-splits.

Repeated appearances must not make one parcel dominate parcel-level diagnostics.

## 7. Quantify SSE concentration

For each base model report:

- total SSE;
- share from top 5 and top 10 error parcels;
- share from top 10%, 15% and 20%;
- concentration for overprediction vs underprediction;
- low-tail SSE;
- center SSE;
- high-tail SSE.

Exact largest-error IDs may be reported for diagnosis, but IDs must never
become prediction rules.

## 8. Tail definitions

Use two concepts.

### 8.1 Descriptive global tails

For retrospective description only, report q10/q90, q15/q85 and q20/q80 of
the full 138-y distribution.

### 8.2 Fold-valid modeling tails

Any tail classifier, calibrator or specialist must derive its thresholds from
visible training y only.

Primary modeling definition:

    low threshold  = q15(training-visible y)
    high threshold = q85(training-visible y)

q10/q90 is an extreme sensitivity analysis; q20/q80 is a wider sensitivity
analysis.

After a pseudo-target prediction is frozen, its revealed y may be categorized
relative to those training-derived thresholds for evaluation.

## 9. Test regression toward the mean

Use residual e = y - prediction.

Regression toward the mean predicts:

- low-y parcels: e < 0 because they are overpredicted;
- high-y parcels: e > 0 because they are underpredicted.

For Local04D and CatBoost report low/center/high:

- count;
- mean and median residual;
- RMSE;
- MAE;
- bootstrap confidence intervals by parcel;
- fraction of errors with the expected shrinkage sign.

Also fit the descriptive calibration relation:

    y = a + b * prediction + error.

A slope b > 1 is consistent with compressed predictions, but does not justify a
correction unless nested calibration improves held-out RMSE.

## 10. Determine whether the experts fail on the same parcels

Report:

- residual Pearson and Spearman correlation;
- squared-error correlation;
- top-5 and top-10 error-set overlap/Jaccard;
- residual-sign agreement;
- error conditional on Local-vs-CatBoost disagreement;
- parcels where one expert is accurate and the other is badly wrong.

A useful correction is more plausible if errors show either a shared systematic
bias that can be calibrated, or recognizable regimes in which one expert
consistently dominates.

# Phase 06B — Can the failure regime be predicted from X?

Checkpoint 06 is only actionable if the problem can be recognized before
seeing y.

## 11. Primary feature space

Start with G1_agronomic (350 deterministic features), because all three robust
03D finalists used G1 rather than the larger concatenated representations.

Relevant families include:

- phenology;
- phenology anomaly;
- water productivity;
- thermal;
- water timing;
- cross-domain interactions;
- soil profile/interactions;
- sensor agreement;
- nonlinear basis.

Add separate diagnostic variables:

- latitude/longitude or geographic embedding;
- nearest geographic and agronomic distance;
- geo/agro support;
- same-state and same-municipality fraction;
- Local/CatBoost disagreement;
- Graph disagreement;
- prediction instability across pseudo-splits.

Do not start with an unrestricted p >> n search over all 1,403 base features.

## 12. Family-level importance first

For each fold-safe tail/error-risk model test held-out contribution of groups:

    phenology
    phenology_anomaly
    water_productivity
    thermal
    water_timing
    soil
    sensor_agreement
    geography/support
    model_disagreement

Prefer family-level evidence because it is more stable and interpretable than
one winner among hundreds of correlated variables.

## 13. Stable individual-variable analysis

Only after family-level evidence exists, identify variables with multiple
diagnostics:

- fold-local Spearman association with y;
- association with signed residual;
- association with absolute/squared residual;
- standardized low-vs-center and high-vs-center differences;
- held-out permutation importance;
- selection frequency across folds/splits.

A feature is "important for tail identification" only if its signal is stable
across held-out fits. A large full-sample correlation is not enough.

SHAP may be used as a secondary explanation, not as the sole selection rule.

## 14. Tail-risk prediction

Test whether X predicts:

- low tail vs rest;
- high tail vs rest;
- either tail vs center;
- three-class low / center / high.

Primary q15/q85 is used for model training because q10/q90 may leave too few
examples.

Simple candidates only:

- class-weighted logistic / elastic-net logistic;
- shallow ExtraTrees;
- shallow CatBoost if class counts permit.

Report ROC-AUC, PR-AUC versus prevalence baseline, Brier score, confusion counts
and fold variance. OOF performance is mandatory.

## 15. Predict error risk directly

Construct honest residual targets and test whether X/support/disagreement can
predict:

- absolute residual;
- squared residual;
- residual sign.

A residual-risk target for a labeled row must come from a base prediction that
did not fit that row.

This phase answers whether failure itself has learnable structure, even if
yield-tail classification is weak.

# Phase 06C — Calibration before clustering

Calibration is lower variance than splitting 138 labels into smaller problems.

## 16. Identity-regularized linear calibration

Fit:

    corrected = a + b * base_prediction

with regularization toward a = 0 and b = 1.

A stable b > 1 would expand compressed predictions. Promotion still requires
held-out improvement.

## 17. Tail-stretch calibration

Around a fold-local center m:

    corrected = m + s * (prediction - m)

Predeclared candidate stretches:

    0.95, 1.00, 1.05, 1.10, 1.15, 1.20

Do not optimize a continuous stretch on the full development table.

## 18. Piecewise calibration

If residual plots show nonlinear shrinkage, allow one strongly regularized
piecewise-linear calibrator with at most two training-derived prediction
quantile knots.

Isotonic regression is diagnostic-only unless support proves sufficient.

## 19. Two-expert calibration

Use a low-dimensional correction:

    corrected =
        a
        + b_local * Local04D
        + b_cat * CatBoost
        + b_disagree * abs(Local04D - CatBoost)

Use Ridge/Huber regularization and nested fitting.

If this cannot beat Local04D, do not immediately escalate to a more flexible
gate using the same inputs.

# Phase 06D — Natural X-only regimes

Only enter this phase if 06A/06B show reproducible X-linked structure.

## 20. Clustering must be X-only

Do not cluster on true y and then route the 59 hidden parcels by an unavailable
quantity.

Clusters must be built only from observable X. After membership is frozen,
visible training y may be used to characterize yield/residual differences.

## 21. Primary clustering representation

Use robust-scaled G1_agronomic, with X-only PCA before clustering.

Predeclare a small PCA variance set such as 0.80 and 0.90.

Under the transductive competition contract, X-only geometry may use all 197 X.
No y may enter PCA or clustering.

## 22. Controlled clustering family

Primary candidates only:

    KMeans: K = 2, 3, 4
    GMM:    K = 2, 3, 4

Optional agglomerative clustering may be added only if diagnostics justify it.

Do not choose K because it maximizes separation of y means on all 138 labels.

Assess X structure with:

- silhouette;
- cluster balance;
- seed stability / adjusted Rand index;
- bootstrap stability;
- target-to-cluster support.

Then describe per cluster, using visible labels:

- y distribution;
- Local04D and CatBoost residuals;
- SSE contribution;
- geography/support;
- dominant agronomic features.

A "high-yield cluster" is credible only if membership was constructed without y
and its yield/residual behavior repeats out of sample.

# Phase 06E — Regime-aware corrections

Prefer partial pooling. Do not train independent complex regressors on tiny
tails.

## 23. Cluster-specific calibration

Fit global calibration plus shrunken cluster intercept/slope deviations. Cluster
effects must be regularized toward zero so all 138 observations still share
information.

## 24. Cluster residual correction

Use honest Local04D residuals as the target for a low-capacity Ridge/Huber
correction based on cluster, prediction, support and disagreement.

Final prediction is Local04D plus the predicted residual correction.

## 25. Soft X-regime mixture

If a GMM provides posterior probabilities, prefer probability-weighted regime
experts over a hard single-cluster routing decision.

## 26. Tail-risk gated specialist

Only if 06B shows real held-out tail predictability, fit a gate:

    g(X) = P(tail | X)

and combine a base predictor with a low-capacity tail correction using g(X).

The specialist may be a calibrated Local04D/CatBoost or residual correction.
Do not train a high-capacity tail regressor on two or five observations.

## 27. Asymmetric low/high correction

If low and high tails have opposite bias mechanisms, allow separate soft
probabilities P(low|X) and P(high|X) and separate shrunken corrections.

This is preferable to a single generic "outlier" class.

# Phase 06F — Explain which variables matter

Checkpoint 06 must produce an interpretable answer, not only a lower RMSE.

For every credible mechanism report:

    feature family
    feature
    direction in low tail
    direction in high tail
    signed-residual association
    absolute-residual association
    held-out importance
    fold selection frequency
    agronomic interpretation

The report should answer:

- Are high-yield misses associated with stronger/later greenness?
- Are low-yield misses associated with negative 2025-vs-history anomalies?
- Do water-productivity or thermal variables distinguish tails?
- Is geography doing most of the work?
- Are tail errors mostly low-support parcels?
- Does Local/CatBoost disagreement identify a regime?
- Are natural clusters merely state/municipality proxies?

## 28. Geography confounding check

Compare:

1. agronomic-only regime model;
2. geography/support-only model;
3. agronomic + geography model.

If agronomic variables lose held-out contribution after geography enters,
report that plainly. Do not describe a municipality proxy as a physiological
barley mechanism.

# Phase 06G — Evaluation and promotion

Overall competition RMSE remains the primary objective.

For every candidate report:

## Overall
- target-matched mean RMSE;
- pooled RMSE;
- pooled MAE;
- worst split RMSE.

## Tail-aware
- low-tail RMSE/MAE/bias;
- center RMSE/MAE/bias;
- high-tail RMSE/MAE/bias;
- combined-tail SSE;
- share of total SSE from tails;
- calibration slope/intercept.

## Robustness
- LOSO-selected mean and pooled RMSE;
- state-stratified stress;
- municipality-grouped stress;
- per-split win count.

## 29. Promotion requirements

A challenger may replace Local04D only if all hold:

1. 06A verifies a reproducible mechanism worth correcting.
2. Development mean and pooled target-matched RMSE both improve.
3. Combined low+high tail error improves in the intended direction.
4. Center RMSE is non-inferior. Before confirmation, freeze a tolerance;
   default proposal: max(0.005 t/ha, 1% of incumbent center RMSE).
5. LOSO mean and pooled RMSE both improve.
6. One preselected challenger beats Local04D on both overall metrics on a new
   fresh confirmation bank.
7. Tail behavior does not reverse on confirmation.
8. Hidden FIRA y is never used.

If any gate fails, Local04D remains canonical.

# Phase 06H — Explicit stop conditions

Close Checkpoint 06 as a negative result if:

- extreme y does not disproportionately contribute to honest SSE;
- low/high residual bias is not stable;
- the apparent "five outliers" are a single-fold artifact;
- tail membership is not predictably encoded in X;
- large-error risk is not predictable from X;
- X-only clusters are unstable or unrelated to repeatable residual structure.

Stop after simple calibration if it fails; do not automatically escalate.

Stop clustering if it only reproduces geography and adds no held-out residual
information.

A clean negative result is preferable to a post-hoc fix for five training IDs.

# Phase 06I — Required implementation order for the next agent

## Step 1 — audit provenance

Read:

    docs/CHECKPOINT_06_PLAN.md
    docs/CHECKPOINT_04C1_FINDINGS.md
    docs/CHECKPOINT_04D1_FINDINGS.md
    docs/CHECKPOINT_05_FINDINGS.md
    reports/checkpoint_04d1/
    reports/checkpoint_05/
    reports/checkpoint_03d/

Determine which honest residuals already exist and which must be regenerated.
Do not trust a file merely because it says OOF.

## Step 2 — implement diagnostics first

Expected future paths:

    src/geocebada/evaluation/checkpoint06.py
    tools/run_checkpoint_06.py
    configs/checkpoint06.yaml
    tests/test_checkpoint06.py

The first implementation should be capable of stopping after 06A/06B.

## Step 3 — preflight

Future command:

    python tools/run_checkpoint_06.py --preflight

Verify:

- 197/138/59 contract;
- competition input only;
- hidden 59 y missing;
- target-matched split provenance;
- honest base-prediction provenance;
- tail thresholds use training y only;
- IDs align one-to-one;
- fresh confirmation seed is unused.

## Step 4 — run 06A and stop for review

Required decision field:

    TAIL_HYPOTHESIS = SUPPORTED / NOT_SUPPORTED / AMBIGUOUS

NOT_SUPPORTED closes 06.

AMBIGUOUS means improve diagnostics, not jump to a flexible mixture model.

## Step 5 — 06B/06C

Only after support:

- family/variable analysis;
- tail/error-risk prediction;
- simple calibration.

Calibration is tested before clustering.

## Step 6 — 06D/06E only if justified

Keep the candidate universe small and predeclared.

## Step 7 — LOSO

Do not select a winner from the full development minimum alone.

## Step 8 — freeze one challenger

Freeze method, config and feature schema before confirmation.

## Step 9 — fresh confirmation

Score only Local04D and the frozen challenger.

## Step 10 — final all-label fit

Only if every gate passes:

- refit on all 138 labels;
- use permitted X-only transductive geometry;
- predict exactly 59 targets;
- preserve Checkpoint 05 outputs unchanged;
- write a new Checkpoint 06 final file.

# 30. Planned output namespace

Future generated outputs should include:

    reports/checkpoint_06/
      parcel_oof_residuals.csv
      tail_definition_summary.csv
      tail_sse_concentration.csv
      tail_bias_summary.csv
      expert_tail_overlap.csv
      feature_family_tail_importance.csv
      feature_tail_associations.csv
      tail_classifier_oof.csv
      error_risk_oof.csv
      calibration_summary.csv
      x_regime_assignments.csv
      x_regime_stability.csv
      x_regime_y_residual_summary.csv
      candidate_predictions.csv
      candidate_summary.csv
      loso_selection.csv
      loso_summary.csv
      confirmation_split_membership.csv
      confirmation_predictions.csv
      confirmation_summary.csv
      final_predictions.csv
      checkpoint_06_report.json
      checkpoint_06_report.md
      figures/

Recommended figures:

- observed vs honest prediction by tail;
- residual vs observed y;
- residual vs predicted y;
- cumulative SSE concentration;
- Local04D vs CatBoost residual scatter;
- feature-family importance;
- stable variable effects;
- PCA regime map;
- cluster-wise y/residual distributions;
- before/after calibration;
- per-split incumbent-vs-challenger delta.

# 31. Required tests/invariants

Future tests must verify:

1. tail thresholds come from training y only;
2. pseudo-target y is unavailable until scoring;
3. residual supervised targets come from honest predictions;
4. X-only clustering never reads y;
5. K/algorithm is not selected from full-sample y;
6. model-driving feature selection is fold-local;
7. parcel IDs remain one-to-one;
8. confirmation masks do not duplicate development masks;
9. final output has exactly 59 unique target IDs;
10. Checkpoint 05 artifacts are never overwritten.

# 32. Forbidden shortcuts

Do not:

- manually correct the five largest-error labeled parcels;
- hard-code tail parcel IDs;
- cluster on y and route hidden targets as though y were available;
- choose thresholds from the hidden 59;
- choose features on all 138 y and report non-nested CV;
- use in-sample residuals as honest error targets;
- compare 03D state/grouped scores directly to target-matched Local04D;
- search hundreds of cluster/model combinations and report only the best;
- replace Local04D for a tiny full-development gain;
- claim a tail mechanism without identifying observable variables that predict it.

# 33. Legitimate possible conclusions

A. **No tail problem.** The visual impression was misleading. Close 06.

B. **Tail problem but not predictable from X.** Treat as irreducible with current
information; retain Local04D.

C. **Predictable global shrinkage.** A simple calibration is sufficient; prefer
it over clustering.

D. **Predictable natural X-regimes.** Use partially pooled cluster-specific
correction.

E. **Predictable tail risk without clean clusters.** Use a soft probability
gate.

F. **One expert dominates in a recognizable regime.** Use a small explicit
Local/CatBoost regime-aware gate.

Prefer the simplest mechanism supported by held-out evidence.

# 34. State at plan creation

At 2026-10-03:

- the teammate's observation is not verified;
- no tail model has been fit;
- no cluster has been selected;
- no feature has been declared tail-important;
- no 59-target prediction has changed;
- Local04D remains incumbent;
- reports/checkpoint_05/final_predictions.csv remains canonical.

The 06A/06B evidence generator is implemented in:

    configs/checkpoint06.yaml
    src/geocebada/evaluation/checkpoint06.py
    tools/run_checkpoint_06.py
    tests/test_checkpoint06.py

Run preflight first and then the diagnostic runner. Do not implement 06C/06D
until the generated report is reviewed.


## 06C implementation status

06A/06B was executed on 2026-10-05 and returned
`TAIL_HYPOTHESIS = SUPPORTED`.

Observed evidence includes:

- Local04D top 10% error parcels explain 62.18% of equal-weight parcel SSE;
- Local04D low-tail mean residual = -0.358 t/ha;
- Local04D high-tail mean residual = +0.473 t/ha;
- CatBoost corroborates the same shrinkage direction;
- tail membership is predictably encoded in X, with the strongest combined-tail
  family classifier using nonlinear-basis features (mean split ROC-AUC 0.710);
- low/high classifiers are stronger separately, especially thermal,
  cross-domain, nonlinear-basis and water-productivity families.

Checkpoint 06C is now implemented in:

    configs/checkpoint06c.yaml
    src/geocebada/evaluation/checkpoint06c.py
    tools/run_checkpoint_06c.py
    tests/test_checkpoint06c.py

06C must be run before any clustering. It tests low-capacity calibration using
honest cross-fitted pseudo-train predictions and LOSO selection.


## 06D implementation status

06C failed the LOSO gate. Its best same-development candidate,
`LocalPiecewiseRidge_a100p0`, improved mean RMSE to 0.485375 and pooled RMSE
to 0.492747, but no candidate was eligible under leave-one-split-out selection;
Local04D was selected on all 16 held-out splits.

06D is therefore implemented to discover X-only agronomic regimes. Selection
uses no yield/residual information: G1 -> robust scaling -> PCA 0.80/0.90 ->
KMeans/GMM, K=2/3/4, with balance and seed/subsample stability gates. Only after
one clustering is frozen are Local04D/CatBoost residual effects, expert
advantage and geography confounding summarized.


## 06E implementation status

06D was executed and returned `REGIME_HYPOTHESIS = NOT_SUPPORTED`.

The selected X-only clustering itself was valid:

```text
KMeans
K = 4
G1 agronomic -> PCA 80%
silhouette = 0.1966
seed ARI mean = 0.8809
subsample ARI mean = 0.8204
```

However, the clusters did not explain Local04D failure:

```text
Local04D residual eta^2          0.0040
residual permutation p          0.9345
Local04D MSE eta^2              0.0119
MSE permutation p               0.7621
cluster residual mean range     0.0877 t/ha
```

Therefore Checkpoint 06 explicitly rejects cluster-specific specialists.

The remaining supported hypothesis is supervised low/high tail probability,
because 06B showed strong held-out classification:

```text
low tail  / water_productivity  mean split ROC-AUC ~ 0.889
high tail / thermal             mean split ROC-AUC ~ 0.805
```

Checkpoint 06E is implemented to combine:

- Local04D as the universal anchor;
- a center-only residual expert for non-outlier parcels;
- soft low/high tail gates;
- conservative low/high residual offsets.

The dedicated specification is `docs/CHECKPOINT_06E_PLAN.md`.

06E must pass split-excluded selection before any fresh confirmation or change
to the 59 target predictions.


## Final Checkpoint 06 closure

Checkpoint 06E completed and failed its LOSO gate.

The strongest center-only G1-PCA residual experts reduced center RMSE by about
0.01 t/ha on the aggregate development table, but this improvement was not
stable under leave-one-split-out selection. The split-excluded selector chose
Local04D on 13/16 holdouts, and aggregate LOSO mean, pooled, center and tail
RMSE were all slightly worse than the incumbent.

No fresh confirmation is run because the predeclared LOSO gate did not pass.

Checkpoint 06 is closed with no model promotion. Canonical findings:
`docs/CHECKPOINT_06_FINDINGS.md`.
