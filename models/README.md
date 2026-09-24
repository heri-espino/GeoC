# Models

Broad modeling is closed. The competition-facing method remains
`Local04D = LocalRidge_C4_all_deterministic_Geo_k24_a30_p1`, retained by the
completed Checkpoint 05 promotion gate.

The project now has two deliberately separate model contracts:

1. **fixed-59 competition contract** — exact frozen predictions for the 59 FIRA
   target parcels;
2. **generic global deployment contract** — the verified Checkpoint 03D
   XGBoost ONNX estimator.

These must not be described as the same model.

## Fixed-59 Checkpoint 05 bundles

The completed Checkpoint 05 runner writes two local bundles:

```text
models/final/checkpoint05_app_bundle.joblib
models/final/checkpoint05_model.joblib
```

Both are intentionally ignored by Git. The **app bundle** is the preferred
fixed-target artifact for Python/Streamlit display: it contains the frozen 59
predictions, candidate predictions, diagnostics, manifest, feature schema and
target IDs, without fitted CatBoost/scikit-learn objects.

The **model bundle** additionally preserves fitted global experts and
meta-models for reproducibility.

The canonical versioned competition table is:

```text
reports/checkpoint_05/final_predictions.csv
```

Checkpoint 05 completed without promotion, so this table retains Local04D.

Public helpers:

```python
from geocebada.models import (
    fixed_target_diagnostics,
    fixed_target_predictions,
    load_checkpoint05_bundle,
    verify_fixed_target_bundle,
)
```

The fixed-target export scope is:

```text
fixed_59_competition_targets
```

It is not an arbitrary-new-parcel inference service.

## Checkpoint 03D global ONNX

Checkpoint 03D completed on 2026-09-24. Scientific finalist rank 1 was HistGB
on G1 agronomic, but that family was not ONNX-convertible with the pinned
stable converter stack. XGBoost G1, scientific rank 2, became the
highest-ranked deployable finalist.

Local artifacts:

```text
models/final/checkpoint03d_global.joblib
models/final/checkpoint03d_global.onnx
models/final/checkpoint03d_global.onnx.json
```

They remain gitignored. The versioned audit is:

```text
reports/checkpoint_03d/checkpoint_03d_report.json
```

The final ONNX round trip passed over 138 rows with maximum absolute difference
`3.814697265625e-06`. The JSON manifest stores the exact raw feature order and
median-imputation statistics.

This ONNX model is a **global deployment estimator**. It does not reproduce the
Local04D fixed-target competition rule.

## Provenance requirements

Final production artifacts must record:

- commit SHA and config;
- exact feature inputs;
- validation and promotion-gate evidence;
- random seeds;
- target-support diagnostics where applicable;
- exact 59-parcel linkage for fixed-target outputs;
- preprocessing schema and numerical round-trip checks for deployment models.

Do not store source data, credentials or private raw datasets here.
