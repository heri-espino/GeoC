# Checkpoint 03B — Empirical Feature Discovery

**Status:** implementation in progress  
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

## Acceptance criteria

03B closes when:

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

After 03B closes, **Checkpoint 03C — Modeling** should compare the source, agronomic and
empirical representations and then test fold-fitted PCA/PLS, kernels and controlled model
families using the frozen Checkpoint 02 folds.
