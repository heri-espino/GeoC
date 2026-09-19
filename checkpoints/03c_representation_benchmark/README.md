# Checkpoint 03C.1 — Representation Benchmark

**Status:** implementation ready; workstation run pending  
**Opened:** 2026-09-18

Checkpoint 03C starts model evaluation after the closed 03A/03B feature-discovery phases.

The first stage intentionally asks a narrow question:

> Before tuning sophisticated models, which feature representation carries useful held-out
> signal under the exact same fixed model and validation settings?

## Representations

Both `clean` and `competition` tracks are evaluated:

```text
B0 = Feature Table v1
B1 = Agronomic Features v1 only
B2 = Empirical Features v1 only
B3 = base + agronomic
B4 = base + empirical
B5 = base + agronomic + empirical
B7 = B5 + fold-local discovered expressions
```

There is no B6 yet. Reduced/selected representations will be designed only after the first
representation results are observed.

`competition` contains clean + competition-mode features. `clean` contains only clean-mode
features.

CHIRPS is explicitly excluded from 03C.1 because its current Feature Table v1 representation
has unresolved QC.

## Fixed models

03C.1 deliberately reuses the same basic model forms as Checkpoint 02:

```text
Ridge10
  median imputation + missing indicators
  standardization
  Ridge(alpha=10)

ExtraTrees
  median imputation + missing indicators
  300 trees
  min_samples_leaf=3
  max_features=sqrt
```

The purpose is not to find the final estimator. Keeping models fixed makes representation
deltas easier to interpret.

CatBoost, ElasticNet, kernels, PCA/PLS and broader tuning are deferred to 03C.2 after the
representation benchmark narrows the candidate feature spaces.

## Validation

The exact frozen Checkpoint 02 folds are reused:

```text
fold_state_stratified
fold_municipality_grouped
```

No new split is generated.

Because grouped fold sizes are highly unequal, the output reports both:

- mean/std of the five fold metrics;
- pooled out-of-fold metrics over all 138 parcels.

The pooled OOF RMSE is particularly useful for understanding parcel-weighted error, while fold
means retain the equal-fold view used in Checkpoint 02.

## B7 leakage rule

B7 uses `FoldLocalExpressionAugmenter`.

For every outer fold:

```text
outer training rows
    |
    +-- fit primitive ranking with y_train
    +-- fit expression search with y_train
    +-- append selected expressions
    +-- fit imputer/scaler/model
    |
    v
outer validation rows
    |
    +-- apply training-fitted formulas
    +-- predict
```

Validation targets never participate in expression discovery.

The clean track automatically restricts the 03B primitive pool to clean variables. The
competition track may use all eligible configured primitives.

## Outputs

Running:

```powershell
python tools\run_checkpoint_03c.py
```

writes:

```text
reports/checkpoint_03c/
├── representation_manifest.json
├── representation_fold_metrics.csv
├── representation_summary.csv
├── representation_oof_predictions.csv
├── discovery_by_outer_fold.csv
├── checkpoint_03c_report.json
└── checkpoint_03c_report.md
```

No predictions for the final 59 hidden-target parcels are produced.

## Interpretation

For the same track/protocol/model:

```text
delta_oof_rmse_vs_B0 < 0
```

means the representation improved pooled OOF RMSE relative to Feature Table v1.

For B7:

```text
delta_oof_rmse_vs_B5 < 0
```

tests whether fold-local expression discovery adds signal beyond the complete deterministic
feature stack.

03C.1 does **not** select the final challenge model.

## Next stage

After the workstation run is reviewed, 03C.2 should retain only a small number of
representations and compare/tune model families such as:

- ElasticNet;
- CatBoost;
- ExtraTrees refinements;
- gradient boosting;
- polynomial and RBF kernels;
- fold-fitted PCA + Ridge;
- PLS.

Those learned transformations and tuning searches must remain inside training folds.
