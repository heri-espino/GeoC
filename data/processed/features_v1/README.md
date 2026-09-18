# Feature Table v1

This directory is the **canonical model-ready input** for GeoCebada modeling.

Unlike the large/reproducible raw external downloads under `data/raw/external/`, these
small parcel-level artifacts are intentionally versioned so collaborators, notebooks and
AI agents can reproduce Checkpoint 02+ modeling from a clean clone of the repository.

## Expected artifacts

```text
data/processed/features_v1/
├── parcel_features_clean.csv
├── parcel_features_competition.csv
├── parcel_features_all.csv
├── feature_manifest.json
├── build_report.json
└── README.md
```

Observed validated contract as of 2026-09-18:

| Artifact | Contract |
|---|---|
| `parcel_features_clean.csv` | 197 rows × 699 columns; 694 clean features + metadata |
| `parcel_features_competition.csv` | 197 rows × 1,408 columns; 1,403 total features + metadata |
| `parcel_features_all.csv` | 197 rows × 1,408 columns |
| `feature_manifest.json` | 694 clean + 709 competition-only = 1,403 features |
| `build_report.json` | deterministic build metadata, missingness and split summary |

All three tables must satisfy:

- exactly 197 unique `ID_POLIGONO`;
- 138 `ENTRENAMIENTO`;
- 59 `PREDICCION`;
- all training targets populated;
- all prediction targets missing;
- no hidden-target reconstruction;
- no infinite numeric values.

## Source of truth and regeneration

These files are **derived artifacts**, not source evidence. If they conflict with
`data/source/`, the source evidence wins.

Regenerate from the full workstation data with:

```powershell
python tools\build_parcel_feature_table.py
```

Then validate with the documented Feature Table v1 checks before replacing the versioned
artifacts.

The generation code, provenance rules and clean/competition definitions live in:

- `configs/features_v1.yaml`
- `src/geocebada/features/parcel.py`
- `tools/build_parcel_feature_table.py`
- `docs/FEATURE_TABLE_V1.md`

## Modeling use

Checkpoint 02 and later modeling should consume these versioned files directly.

- `clean`: defensible historical/static track.
- `competition`: clean plus rule-permitted full-season/contemporaneous 2025 covariates.
- Never use the 59 hidden targets.
- Fit learned preprocessing, PCA, feature selection and other trainable transforms inside
  frozen CV folds.

The frozen Checkpoint 02 folds and diagnostics are under `reports/checkpoint_02/`.
