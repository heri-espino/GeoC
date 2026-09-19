# Checkpoint 03B — Empirical Feature Discovery

**Status:** **Closed**  
**Opened:** 2026-09-18

Checkpoint 03B follows the closed Agronomic Features v1 layer.

Its goal is to discover nonlinear structure from the observed covariates while keeping two
very different operations separate:

1. deterministic X-only transformations that may be materialized globally;
2. target-aware formula discovery that must be fitted inside training folds.

## Deterministic processed layer

The versioned output is:

```text
data/processed/empirical_features_v1/
├── parcel_empirical_features.csv
├── feature_manifest.json
├── build_report.json
└── README.md
```

The table contains only `ID_POLIGONO` plus empirical predictors.

Implemented X-only families:

- temporal-shape geometry;
- short-baseline historical condition;
- symmetric 2025-versus-history change;
- Sentinel-2/Planet shape similarity;
- aggregate distribution geometry.

The formulas and caveats are documented in `docs/EMPIRICAL_FEATURES_V1.md`.

## VCI-like feature

A short-baseline NDVI condition score is included, inspired by VCI normalization.

It is deliberately named `emp_vci_like_short__...` because GeoCebada has only a short
2022-2024 historical reference and therefore does not satisfy the usual long-climatology
interpretation of standard VCI.

No feature is labeled CWSI because the available temperature products are not canopy
temperature with wet/dry reference baselines.

## Target-aware discovery

`FoldLocalExpressionMiner` provides the supervised part of 03B.

It may use `RENDIMIENTO_T_HA` only inside its `fit` call on an outer training fold.

The first-stage search is intentionally small:

```text
rank curated primitives using training y
        ↓
keep a small primitive subset
        ↓
generate pairwise formulas
        ↓
rank formulas using training y
        ↓
apply selected formulas to validation X
```

Operations:

```text
product
sum
difference in both directions
safe ratio in both directions
symmetric relative change
```

The selected formulas are not written as a global processed table.

## Learned X-only transforms

PCA, PLS, KMeans/environmental clustering and kernels are not part of the materialized 03B
dataset because they learn cross-row parameters.

They must be fitted inside CV and belong to the next modeling phase.

## Canonical build result

The versioned build from Feature Table v1 passed with:

```text
rows                         197
derived features             335
clean                         91
competition                  244
mean/max missing fraction    0.0
infinite values              0
target used                  false
supervised formulas global  false
short VCI-like standard VCI false
CHIRPS used                  false
```

Retained family counts:

```text
temporal_shape              162
aggregate_geometry           72
symmetric_change             42
short_baseline_condition     41
sensor_shape_similarity      18
```

One candidate was constant and therefore removed:

```text
emp_histpos__landsat_vi6t__above_history_fraction
```

This only states that the specific 2025 Landsat VI6T historical-range exceedance fraction has
no cross-parcel variation in the current representation.

The fold-local discovery pool contains 24 configured primitives drawn from source-derived,
agronomic and empirical variables.

## Acceptance criteria

03B closed after the workstation run confirmed that:

1. deterministic empirical builder runs from a clean clone;
2. output covers exactly the same 197 parcel IDs;
3. no target or split column appears in the output;
4. no retained feature contains infinities;
5. every retained feature has a manifest record;
6. clean features use no 2025 source;
7. the short VCI-like feature is explicitly documented as nonstandard/short-baseline;
8. the fold-local expression miner is tested to fit only from training X/y;
9. configured supervised primitives exist after joining the canonical layers;
10. generated empirical artifacts are committed;
11. Ruff, tests, function-index consistency and Sphinx pass;
12. agent handoffs/history/AI-use docs are synchronized.

The workstation run passed. The deterministic empirical layer contains 335 features and the fold-local audit selected 88 unique expressions, with 7 raw expression strings recurring in at least 3/5 folds under both protocols. Those seven collapse to four conceptual primitive-pair motifs because several are monotone ratio reparameterizations. See `reports/checkpoint_03b/README.md` for interpretation. Checkpoint 03C is next.
