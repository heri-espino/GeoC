# Empirical Features v1

This directory contains the deterministic **X-only empirical feature layer** for Checkpoint 03B.

The layer is separate from both:

- `data/processed/features_v1/` — source-derived parcel covariates;
- `data/processed/agronomic_features_v1/` — explicit agronomic hypotheses.

It contains only:

- `ID_POLIGONO`;
- deterministic empirical predictors built from existing X values.

It never contains:

- `RENDIMIENTO_T_HA`;
- `CONJUNTO`;
- target-selected interactions;
- globally fitted PCA/clustering;
- formulas chosen using all 138 labels.

## Artifacts

```text
data/processed/empirical_features_v1/
├── parcel_empirical_features.csv
├── feature_manifest.json
├── build_report.json
└── README.md
```

The canonical validated build contains **197 parcels × 336 columns** = `ID_POLIGONO` plus **335 derived features**: **91 clean** and **244 competition**. Retained missingness is 0.0 and no infinite values are present.

## Families

The deterministic layer includes 335 retained features:

- temporal curve geometry;
- short-baseline historical condition;
- symmetric 2025-versus-history change;
- Sentinel-2/Planet temporal-shape similarity;
- normalized aggregate distribution geometry.

The short-baseline NDVI condition feature is explicitly named
`emp_vci_like_short__...`. It is inspired by VCI-style historical normalization, but it is
**not standard VCI** because the historical reference is only 2022-2024 and is not a long
climatology.

One candidate was dropped as constant: `emp_histpos__landsat_vi6t__above_history_fraction`. This means no parcel exceeded that short historical upper range under this specific summary; it is not a biological conclusion.

## Target-aware discovery

The supervised portion of 03B is not materialized here.

`FoldLocalExpressionMiner` uses `y` only inside the training fold and selects simple
candidate expressions from a curated primitive set. Validation rows are transformed using the
already selected formulas and training-fold medians.

This distinction prevents a formula discovered from all 138 labels from being treated as if it
had been pre-specified.

## Join contract

```python
from geocebada.features import join_empirical_features

joined = join_empirical_features(base, empirical)
```

The join validates exact one-to-one parcel coverage.

## Rebuild

```powershell
python tools\build_empirical_features_v1.py
```

The build uses only versioned processed inputs. No large external raw downloads are required.

See:

- `docs/EMPIRICAL_FEATURES_V1.md`;
- `checkpoints/03b_empirical_feature_discovery/README.md`;
- `configs/empirical_features_v1.yaml`.
