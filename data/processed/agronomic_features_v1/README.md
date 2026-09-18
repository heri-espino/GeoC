# Agronomic Features v1

This directory contains the **additive agronomic/nonlinear feature layer** used after
Feature Table v1.

The layer is deliberately separate from `data/processed/features_v1/`.

It contains only:

- `ID_POLIGONO`;
- deterministic X-only agronomic/nonlinear predictors.

Canonical validated build (2026-09-18): **197 parcels × 352 columns** = `ID_POLIGONO`
plus **351 derived features**. The manifest contains **123 clean** and **228 competition**
features. No retained derived feature is missing on the canonical 197 rows.

It does **not** contain:

- `RENDIMIENTO_T_HA`;
- `CONJUNTO`;
- hidden prediction targets;
- target-selected interactions.

## Artifacts

```text
data/processed/agronomic_features_v1/
├── parcel_agronomic_features.csv
├── feature_manifest.json
├── build_report.json
└── README.md
```

## Join contract

Join to any one-row-per-parcel Feature Table v1 representation using:

```python
from geocebada.features import join_agronomic_features

joined = join_agronomic_features(base, agronomic)
```

The merge is validated as one-to-one on `ID_POLIGONO`.

## Provenance

Every derived feature is documented in `feature_manifest.json` with:

- feature family;
- clean/competition mode;
- source columns;
- formula;
- plain-language meaning;
- agronomic rationale;
- evidence class;
- bibliography keys;
- caveats.

The human-readable methodology is in
`docs/AGRONOMIC_FEATURES_V1.md`.

The bibliography is in
`docs/references/agronomic_features_v1.bib`.

## Rebuild

From a clean clone with Feature Table v1 already versioned:

```powershell
python tools\build_agronomic_features_v1.py
```

No large raw/external data downloads are required.

## Modeling

Do not automatically assume that every agronomic feature improves prediction.

Checkpoint 03B should compare at least:

```text
base only
agronomic only
base + agronomic
selected/reduced base + agronomic
```

under the frozen folds in `reports/checkpoint_02/cv_folds.csv`.

Target-guided selection must happen inside training folds.
