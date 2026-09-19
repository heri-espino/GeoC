# Checkpoint 03 — Feature Discovery and Modeling

**Status:** In progress — 03A/03B closed; 03C.1 implementation ready  
**Current phase:** 03C.1 — Representation benchmark  
**Opened:** 2026-09-18

Checkpoint 03 begins after the frozen Feature Table v1 and Checkpoint 02 validation contract.

The phase is intentionally split so the project can distinguish:

```text
03A  agronomic nonlinear features
03B  empirical/data-driven feature discovery
03C  model comparison, kernels and tuning
```

The frozen validation assignments remain:

```text
reports/checkpoint_02/cv_folds.csv
```

with both `fold_state_stratified` and `fold_municipality_grouped`.

---

## 03A — Agronomic nonlinear features — CLOSED

03A created a separate, versioned, target-free feature layer with explicit agricultural
interpretation.

Canonical artifacts:

```text
data/processed/agronomic_features_v1/
  parcel_agronomic_features.csv
  feature_manifest.json
  build_report.json
  README.md
```

Canonical build:

```text
rows                    197
derived features        350
clean                    123
competition              227
missing fraction          0.0
infinite values           0
target used             false
CHIRPS used             false
```

Retained families:

```text
phenology              162
phenology_anomaly       41
water_productivity      40
thermal                 36
cross_domain            19
soil_profile            18
water_timing            10
sensor_agreement         9
soil_interaction         8
nonlinear_basis          7
```

See `docs/AGRONOMIC_FEATURES_V1.md` and
`checkpoints/03a_agronomic_features/README.md`.

---

## 03B — Empirical Feature Discovery — CLOSED

03B asks a different question:

> What useful nonlinear structure can be derived from the observed covariates themselves,
> without assuming an agronomic mechanism and without leaking the target?

The phase has two parts.

### 03B.1 Deterministic X-only empirical layer

The materialized layer is:

```text
data/processed/empirical_features_v1/
  parcel_empirical_features.csv
  feature_manifest.json
  build_report.json
  README.md
```

It contains only `ID_POLIGONO` plus deterministic row-wise predictors.

Implemented families:

- temporal-shape geometry;
- short-baseline historical condition;
- symmetric 2025-versus-history change;
- Sentinel-2/Planet temporal-shape similarity;
- normalized aggregate distribution geometry.

The short-baseline NDVI score is named `emp_vci_like_short__...`. It is inspired by VCI
normalization but is explicitly **not standard VCI**, because GeoCebada has only the short
2022-2024 historical reference rather than a long climatology.

No variable is called CWSI: the project does not have the canopy-temperature wet/dry reference
system required by the standard CWSI formulation.

Full formulas and caveats are in `docs/EMPIRICAL_FEATURES_V1.md`.

### 03B.2 Fold-local target-aware expression discovery

The target-aware part is **not** written to a global dataset.

`FoldLocalExpressionMiner` is fitted only on the training part of a fold. It:

1. median-imputes configured primitives using training rows only;
2. ranks primitives by absolute Spearman association with training `y`;
3. retains a small primitive set;
4. generates simple pairwise formulas;
5. ranks formulas using training `y`;
6. applies the selected formulas to validation X without seeing validation y.

The first-stage operation grammar is deliberately small:

```text
product
sum
difference, both directions
safe ratio, both directions
symmetric relative change
```

This is a controlled symbolic-like search, not unrestricted genetic programming.

A candidate expression should not be described as a new barley index merely because it has a
large training association. A genuinely interesting candidate must recur across fold fits and
show held-out predictive contribution, including under municipality-grouped robustness checks.

### Learned X-only representations

PCA, PLS, clustering and similar transforms do not use `y`, but they learn parameters from
multiple parcels. They therefore must also be fitted inside folds rather than materialized
globally from all 197 rows.

Kernels likewise remain model-layer representations.

Canonical 03B build:

```text
rows                    197
derived features        335
clean                     91
competition              244
missing fraction          0.0
infinite values           0
target used             false
global supervised expr  false
configured primitives     24
```

Family counts:

```text
temporal_shape              162
aggregate_geometry           72
symmetric_change             42
short_baseline_condition     41
sensor_shape_similarity      18
```

One `emp_histpos__landsat_vi6t__above_history_fraction` candidate was constant and dropped.
The workstation run passed. The deterministic artifacts and fold-local expression audit are committed. See `reports/checkpoint_03b/README.md` for the four conceptual recurrent motifs and their caveats.

---

## 03C — Model comparison — CURRENT

### 03C.1 — Representation benchmark — IMPLEMENTATION READY

03C.1 first compares representations with fixed Ridge10 and ExtraTrees baselines:

```text
B0 = Feature Table v1
B1 = Agronomic Features v1 only
B2 = Empirical Features v1 only
B3 = base + agronomic
B4 = base + empirical
B5 = base + agronomic + empirical
B7 = fold-local discovered-expression augmentation
```

03C.1 deliberately does not tune broad model families. After its workstation results are reviewed, 03C.2 may compare:

- Ridge / ElasticNet;
- ExtraTrees;
- CatBoost;
- HistGradientBoosting;
- LightGBM/XGBoost if justified;
- polynomial kernel;
- RBF kernel;
- PCA/PLS variants fitted inside folds.

Run 03C.1 with `python tools/run_checkpoint_03c.py`. It reports both mean-fold metrics and pooled OOF metrics because municipality-grouped folds are unequal. CHIRPS is excluded pending QC.

Because only 138 targets are observed, later searches should remain disciplined.

All preprocessing learned from data must remain inside the training fold:

- imputation;
- scaling;
- PCA/PLS;
- clustering;
- supervised feature selection;
- target-aware expression discovery;
- encoding;
- model tuning.

Do not regenerate the frozen folds because a result is inconvenient.

---

## Completion criteria for 03B

03B closes when:

- the deterministic empirical builder runs from a clean clone;
- output covers exactly 197 unique parcel IDs;
- no target or split column appears in the materialized layer;
- no retained empirical feature contains infinities;
- every retained feature has a manifest record;
- clean features use no target-year competition input;
- the short VCI-like variable is documented as nonstandard;
- fold-local expression discovery is covered by tests;
- configured discovery primitives exist after joining canonical layers;
- generated empirical artifacts are committed;
- Ruff, tests, function-index consistency and Sphinx all pass;
- agent/data/history/AI-use handoffs are synchronized.

Once these criteria are met, 03B is frozen and the project moves to 03C.
