# Checkpoint 02 — Features, diagnostics and frozen validation

**Checkpoint opened:** 2026-09-18  
**Project:** GeoCebada — Reto AgroCebada FIRA 2026  
**Status:** Feature Table v1 validated; diagnostic/fold/ablation pipeline implemented; full local
Checkpoint 02 run must still be executed and its generated report reviewed.

This checkpoint records the transition from data engineering to controlled modeling. It is
deliberately historical: each decision is tied to what was known at that moment, so future
changes can be distinguished from the original modeling contract.

---

## Historical record

### 2026-09-18 — Data foundation revalidated

The full workstation audit reported:

- Data Contract v2: **PASS**;
- 46 PASS, 0 WARN, 0 FAIL, 1 intentional SKIP;
- repository commit used for the feature build: `e650c12`.

The 12-parcel integration build also passed with WaPOR enabled.

### 2026-09-18 — Feature Table v1 built on the full workstation

Observed full-build result:

| Item | Result |
|---|---:|
| Parcels | 197 |
| Training | 138 |
| Prediction | 59 |
| All columns | 1,408 |
| CLEAN features | 694 |
| COMPETITION-only features | 709 |
| Total manifest features | 1,403 |
| WaPOR | included |

Independent validation confirmed:

- `parcel_features_clean.csv`: 197 × 699;
- `parcel_features_competition.csv`: 197 × 1,408;
- `parcel_features_all.csv`: 197 × 1,408;
- IDs are unique and complete;
- all 138 training targets exist;
- all 59 prediction targets remain hidden.

Thus the supervised problem is small-n/high-p:

[
n_{train}=138,qquad p_{clean}=694,qquad p_{competition}=1403.
]

This is why the project does **not** jump directly to a large tuned CatBoost run.

### 2026-09-18 — Checkpoint 02 contract created

The repository now defines a reproducible diagnostic/modeling gate through:

```text
configs/checkpoint02.yaml
src/geocebada/evaluation/parcel_modeling.py
tools/build_checkpoint_02.py
tests/test_checkpoint02.py
```

The generated run artifacts live under:

```text
reports/checkpoint_02/
```

---

## 1. Frozen feature-family inventory

Every manifest feature is classified into a stable family:

- BASIC;
- PRO;
- official climate;
- CHIRPS;
- SoilGrids;
- official topography;
- CEM 15 m;
- SIAP;
- WaPOR;
- geometry;
- administrative fields.

The pipeline writes both a row-level feature catalog and an aggregated inventory. Feature
family membership comes from the existing provenance manifest rather than notebook-specific
column lists.

Why: FIRA asks which indices, climate variables and combinations contribute most. Family-level
ablation must therefore be reproducible before individual feature importance is interpreted.

---

## 2. Train-versus-prediction diagnostics

The 59 prediction parcels expose covariates X but not yields y. Checkpoint 02 therefore measures,
for every numeric feature:

- train/prediction missingness;
- mean and standard deviation;
- min/max;
- 5%, 25%, 50%, 75%, 95% quantiles;
- fraction of prediction observations outside the training range;
- standardized mean difference (SMD);
- two-sample Kolmogorov-Smirnov statistic;
- FDR-adjusted p-value.

Default descriptive flags are:

[
|SMD|ge0.5,quad q_{FDR}<0.05,quad
|Delta missingness|ge0.10,quad
P(X_{pred}
otin range(X_{train}))ge0.10.
]

These flags are **diagnostic only**. They are not allowed to become an unlabeled-test feature
selector for the clean/scientific track.

---

## 3. Frozen cross-validation folds

Checkpoint 02 freezes two five-fold protocols with random seed 42.

### Primary: state-stratified folds

`fold_state_stratified` uses deterministic shuffled StratifiedKFold on `meta_estado`.

Purpose: each validation fold contains a comparable state composition while retaining enough
training data for the n=138 problem.

### Spatial robustness: municipality-grouped folds

`fold_municipality_grouped` keeps every state/municipality group entirely inside one fold,
using StratifiedGroupKFold when feasible and GroupKFold as a deterministic fallback.

Purpose: detect whether apparent performance depends on having geographically/administratively
similar parcels on both sides of a fold.

The generated `reports/checkpoint_02/cv_folds.csv` becomes an artifact. Once created and
reviewed, future model comparisons must reuse it. Regeneration requires the explicit
`--force-folds` flag.

---

## 4. Frozen ablations

Checkpoint 02 defines cumulative, interpretable ablations:

| Ablation | Meaning |
|---|---|
| A0 | geometry + administrative numeric fields |
| A1 | A0 + SoilGrids + official topography + CEM |
| A2 | A1 + historical SIAP |
| A3 | A2 + historical official climate |
| A4 | A3 + historical BASIC |
| A5 | all CLEAN numeric features |
| A6 | A5 + 2025 BASIC/PRO/climate/CHIRPS/WaPOR, excluding SIAP 2025 proxies |
| A7 | full COMPETITION numeric feature set, including SIAP 2025 proxies |

A6 versus A7 deliberately isolates the contribution of the contemporaneous SIAP 2025
municipal outcome proxy.

---

## 5. Initial model benchmark

This checkpoint is not final model selection. It runs only:

- DummyMean;
- Ridge with fixed alpha=10;
- ExtraTrees with 300 trees, minimum leaf size 3 and sqrt feature subsampling.

Median imputation and missing-value indicators are fitted **inside each fold**. Ridge scaling is
also fitted inside each fold.

Intentionally absent at this checkpoint:

- PCA;
- supervised feature selection;
- Optuna;
- CatBoost tuning;
- LightGBM/XGBoost tuning;
- prediction-set-driven model selection.

Why: first measure whether additional feature families actually add stable validation signal.
Only then is it sensible to spend model-selection capacity.

---

## 6. Run command

After updating the workstation to the branch/commit containing this checkpoint:

```powershell
python tools\build_checkpoint_02.py
```

For diagnostics/folds only:

```powershell
python tools\build_checkpoint_02.py --skip-models
```

The full run writes:

- `feature_catalog.csv`;
- `feature_inventory.csv`;
- `train_prediction_diagnostics.csv`;
- `cv_folds.csv`;
- `ablation_features.json`;
- `baseline_fold_scores.csv`;
- `baseline_summary.csv`;
- `checkpoint_02_report.json`;
- `checkpoint_02_report.md`.

The Markdown report is the dated run record. Results should be committed only after human review.

---

## 7. Completion criterion

Checkpoint 02 becomes closed when:

1. the full local pipeline passes on the 197-row table;
2. feature-family counts are reviewed;
3. train/prediction shift diagnostics are reviewed;
4. `cv_folds.csv` is frozen and committed;
5. baseline ablation results are reviewed under **both** fold protocols;
6. generated run report is committed;
7. only then is Checkpoint 03 — Modeling opened.

The next modeling stage may compare CatBoost, regularized linear models, boosting and optional
dimension-reduction pipelines, but all comparisons must reuse the frozen folds and preserve
clean-versus-competition provenance.
