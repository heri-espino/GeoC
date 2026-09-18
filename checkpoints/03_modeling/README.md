# Checkpoint 03 — Modeling

**Status:** In progress — 03A closed; 03B next  
**Current phase:** 03B — Model comparison  
**Opened:** 2026-09-18

Checkpoint 03 begins after the frozen Feature Table v1 and Checkpoint 02 validation contract.

The first phase intentionally does **not** tune CatBoost/LightGBM/Optuna yet. Before comparing
model families, the project creates an explicit agronomic nonlinear representation that can be
tested against the raw/source-derived Feature Table v1.

---

## 03A — Agronomic nonlinear features

### Goal

Create a separate, versioned, target-free dataset of nonlinear and interaction features with
clear agricultural interpretation.

The representation must answer:

1. Can biologically motivated temporal shape variables improve over monthly raw values?
2. Do current-season anomalies relative to historical parcel behavior add signal?
3. Do water, temperature, soil and crop-state interactions add predictive information?
4. Can a smaller set of interpretable nonlinear basis terms help regularized models?
5. Do these gains survive both frozen validation protocols?

### Architecture

Feature Table v1 remains frozen.

```text
data/processed/features_v1/
        |
        | deterministic X-only transforms
        v
data/processed/agronomic_features_v1/
```

Modeling can then evaluate:

```text
B0 = base only
B1 = agronomic only
B2 = base + agronomic
B3 = selected/reduced base + agronomic
```

The join key is always `ID_POLIGONO` and must validate one-to-one.

### Families

Agronomic Features v1 includes:

- phenology / curve shape;
- 2025-versus-historical remote-sensing anomalies;
- Sentinel-2/Planet agreement;
- thermal-time and heat-excess proxies;
- rainfall timing;
- WaPOR water-productivity/crop-water interactions;
- soil vertical-profile contrasts;
- soil nonlinear interactions;
- cross-domain productivity × crop-state × environment terms;
- a small set of target-free log basis transforms.

Full formulas, meanings, evidence classes, caveats and references are documented in
`docs/AGRONOMIC_FEATURES_V1.md`.

### Evidence policy

Each feature is labeled:

- `literature_backed`;
- `mechanistic_proxy`;
- `experimental`.

No experimental feature should be described as an established agronomic law.

### Target safety

The 03A builder does not inspect `RENDIMIENTO_T_HA`.

The output contains only:

```text
ID_POLIGONO
agro_*
```

Target-guided interaction mining is deferred. If later used, it must run inside training folds.

### CHIRPS

CHIRPS is deliberately excluded from 03A v1. Checkpoint 02 showed only one surviving CHIRPS
feature, so nonlinear CHIRPS transformations are deferred until that source receives QC.

### Validation contract

03A does not create new folds.

All later 03B comparisons must reuse:

```text
reports/checkpoint_02/cv_folds.csv
```

with both:

- `fold_state_stratified`;
- `fold_municipality_grouped`.

A feature family is not considered useful merely because it improves the easier protocol.

---

## 03B — Model comparison

03B starts only after 03A artifacts are built and validated.

Candidate families include:

- Ridge / ElasticNet;
- ExtraTrees;
- CatBoost;
- HistGradientBoosting;
- LightGBM/XGBoost if justified;
- polynomial kernel;
- RBF kernel;
- PCA/PLS comparisons where preprocessing is fit inside folds.

Kernel matrices are **not** stored in Agronomic Features v1 because kernels are model
representations whose scaling/hyperparameters belong inside CV.

---

## Completion criteria for 03A

03A can be marked closed when:

- the builder runs from a clean clone using versioned Feature Table v1;
- output has exactly 197 unique parcel IDs;
- no target/split column appears in the output;
- no infinite values exist;
- manifest documents every retained feature;
- clean features use no target-year competition inputs;
- generated artifacts are versioned;
- tests, Ruff, function index and Sphinx pass;
- project/data/agent handoffs are synchronized;
- AI use and bibliography are recorded.

### Canonical 03A build

The canonical build completed successfully from the versioned Feature Table v1.

```text
rows:                  197
derived features:      350
clean:                 123
competition:           227
missing fraction mean: 0.0
missing fraction max:  0.0
infinite values:       0
target used:           false
CHIRPS used:           false
```

Retained family counts:

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

Three candidate features were dropped because they were constant on all 197 parcels:

```text
agro_thermal__historical__heat_excess_25c_proxy
agro_thermal__2025__heat_excess_25c_proxy
agro_interaction__heat_x_positive_water_gap_2025
```

This is an empirical property of the current monthly climate representation, not evidence that
heat stress is agronomically irrelevant. The 25 C monthly-mean hinge simply carries no
cross-parcel variation here.

03A completion criteria are satisfied. The next task is 03B: compare base-only, agronomic-only,
base+agronomic and selected/reduced representations under both frozen CV protocols.
