# Geographic Weighted Ridge v2 — matched exploratory experiment

**Status:** implementation available. Results must be produced on the local
workstation. No new validation scores or final-model promotion are asserted.

## Question

Can we improve the **existing geographic PCA20 weighted Ridge estimator**
by stabilizing its 24-observation local regressions, without introducing new
data or a larger feature representation?

This is a targeted model-objective experiment, **not** a new search of PCA rank,
spatial neighbors, or model families.

## Original estimator and proposed changes

For each query parcel, choose its **24 closest labeled parcels** by the same
normalized Haversine distance matrix used in the original model. With the
original 20-component C4 PCA representation (197 X-only rows) and geographic
distance d, compute raw weights:

```text
w_i = 1 / (d_i + 1e-6)
```

Original weighted local Ridge minimizes:

```text
sum_i w_i * (y_i - intercept - X_i @ beta)^2 + alpha * ||beta||^2
```

The original implementation is called **unchanged** for exact numerical
parity. The original model uses k=24, alpha=30, p=1.

The new variants are:

| Arm | Model modification | Tuned |
|---|---|---|
| `GeoPCA20_Ridge_Frozen` | Exactly original: raw weights, alpha=30 | No |
| `GeoPCA20_Ridge_InnerCV` | Exactly original objective, but alpha chosen using inner CV | Yes |
| `GeoPCA20_Ridge_NormalizedWeights` | Weights scaled so their sum equals 24 | Yes |
| `GeoPCA20_Ridge_EffectiveNAlpha` | Normalized weights, alpha increased by k/n_eff | Yes |
| `GeoPCA20_Ridge_EffectiveNAlphaShrink` | Adaptive alpha and output shrinkage toward geographic weighted neighbor mean | Yes |
| `GeoPCA20_Ridge_RobustIRLS` | Normalized weights, three Huber-like iteratively reweighted Ridge updates | Yes |

The effective sample size of the distance weights is:

```text
n_eff = (sum(w))^2 / sum(w^2), between 1 and 24
alpha_local = alpha * 24 / n_eff
```

The shrinkage arm combines the adaptive-Ridge estimate with the weighted
neighbor mean, assigning the Ridge estimate the fixed fraction
`n_eff/24`. This is a **prespecified heuristic** to reduce extrapolation
when only a few geographic neighbors effectively contribute.

The robust arm computes a Huber-like residual influence factor and refits
weighted Ridge three times. This is **an IRLS approximation**, not a claim
that sklearn HuberRegressor or a fully converged Huber objective was used.

### Crucial comparability detail

Normalizing sample weights changes Ridge's effective penalty even if alpha is
unchanged. Thus, comparing original alpha=30 directly to normalized alpha=30
would be misleading. **Every tunable arm, including the original raw-weight
Ridge, has the same seven-alpha grid:**

```text
0.3, 1, 3, 10, 30, 100, 300
```

The fitted PCA20 representation, k=24, geographic distances, inverse-distance
exponent p=1, data sources, outer masks and inner folds are held identical.
None of the proposed arms includes additional climate features, or new
satellite sources, or switches the model to an unrelated family.

## Fair validation and limits

1. Freeze the **16 historical Checkpoint 08 confirmation masks**: 97
   pseudo-training parcels, 41 pseudo-targets on each, 656 repeated outer
   predictions, 132 unique pseudo-target parcels.
2. On **each split's 97 labels only**, use identical **2 repetitions of
   shuffled 5-fold CV** for each of the five tunable arms. Every candidate
   alpha obtains 194 inner held-out predictions. The same inner masks are
   shared by all arms.
3. Select **one alpha per arm**, using mean squared prediction error pooled
   over its 194 inner predictions (reported as RMSE), with smallest alpha
   breaking ties. The original fixed control is not retuned.
4. Refit each arm using the 97 outer-training labels and predict exactly
   the same 41 outer pseudo-targets. The outer 41 y values are accessed only
   when recording validation errors, never in selection.
5. Report pooled RMSE, R², split-average/worst RMSE and the **paired delta
   MSE** against the frozen original, with a 95% bootstrap resampling unique
   parcels together with all their repeated validation predictions.
6. Export effective geographic neighbor count and weight concentration for
   X-only diagnostics; inspect errors on the same unique parcels. Do not
   select the method based on individual held-out parcel errors.

This makes the tuning budgets and train/test membership symmetric. The
original fixed predictor is also reported so we can compare the incumbent
with each tuned procedure.

**Limits:** Checkpoint 08 splits have already been examined in many
experiments; this is **exploratory**, not an independent new test. The
bootstrap groups repeated records by parcel ID but does not eliminate
spatial dependence. The all-197 PCA is transductive, as in the original;
we are *not* asserting strict inductive validation on unseen future areas.
The inner 5-fold fits use approximately 77 observed parcels versus 97 in
the outer fits. This reduces, but cannot eliminate, the train-size mismatch
compared with earlier inner 3-fold work. The observed RMSE of any new winner
must not be presented as an unbiased prospective error.

No prediction for the 59 actual unlabeled parcels is made or modified by this
experiment. The original final-predictions file is unchanged.

## PowerShell

```powershell
cd C:\Users\175199\GeoC
conda activate geocebada
git switch main
git pull --ff-only origin main
git lfs pull

python tools/run_geographic_ridge_v2.py --check
python tools/run_geographic_ridge_v2.py --smoke
python tools/run_geographic_ridge_v2.py --run
python tools/run_geographic_ridge_v2.py --summary
```

If interrupted before `--run` completes, rerun the **identical** command
with unchanged inputs and code; completed split files are verified and
reused. A change to inputs, masks, implementation or tuning settings
requires a fresh `--output` directory.

Results are saved under **gitignored**
`reports/geographic_ridge_v2/`:

- `run_spec.json`: SHA-256 source and code fingerprints plus frozen protocol.
- `confirm_XX__inner_cv.csv`: 5 methods × 7 alphas, 194 predictions per
  candidate, 10 inner folds.
- `confirm_XX__selection.json`: five alpha decisions made using only the
  97 outer-training y values.
- `confirm_XX__paired.csv`: six methods × 41 paired outer predictions.
- `confirm_XX__neighbors.csv`: effective-neighbor diagnostics (X only).
- `comparison_summary.csv`: pooled RMSE, R², mean/worst split RMSE.
- `paired_cluster_bootstrap.csv`: paired delta-MSE and parcel-cluster CIs.
- `per_parcel_error_diagnostics.csv`: individual parcel errors; **private**.

Do not force-add files from this output directory. Keep all observed parcel
y and per-parcel errors out of the public repository.

**Decision:** this experiment can identify a promising candidate, but no
automatic promotion. Validate any apparent improvement against independent
new parcel outcomes, or at minimum use a separately documented spatial
robustness analysis with explicit reuse limitations.
