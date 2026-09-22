# Models

Checkpoint 05 is the active final modeling phase.

The fixed-target transductive objective remains defined in
`docs/TRANSDUCTIVE_OBJECTIVE.md`. Checkpoint 04F supplies the incumbent
Local04D predictions; Checkpoint 05 is allowed to replace them only if the
predeclared development, LOSO and fresh-confirmation gates pass.

## Final Checkpoint 05 bundle

The long workstation runner writes two local bundles:

    models/final/checkpoint05_app_bundle.joblib
    models/final/checkpoint05_model.joblib

Both are intentionally ignored by Git. The **app bundle** is the preferred
fixed-target artifact for the later Python/Streamlit interface: it contains no
fitted CatBoost/scikit-learn estimator objects, only the frozen 59 predictions,
candidate predictions, diagnostics, manifest, feature schema and target IDs.
That keeps fixed-target display independent of the training stack's optional
model dependencies.

The **model bundle** is the full reproducibility artifact. It additionally
contains fitted global experts and fitted meta-models and therefore requires
the corresponding modeling dependencies when deserialized.

The bundle contains:

- the selected final method and manifest;
- the exact 59 final predictions;
- the actual-target candidate prediction table;
- app-facing diagnostics;
- fitted global experts from the final 138-label fit;
- fitted meta-models;
- target/training IDs and feature schema metadata.

The current export scope is deliberately:

    fixed_59_competition_targets

It guarantees exact reproduction of the competition result. It is not yet a
generic arbitrary-new-parcel inference service.

The runner performs a joblib round-trip verification before declaring the
checkpoint complete. The exported final values must reproduce the selected
candidate to numerical tolerance.

## Loading from Python

Use the public helpers:

    from geocebada.models import (
        fixed_target_diagnostics,
        fixed_target_predictions,
        load_checkpoint05_bundle,
        verify_fixed_target_bundle,
    )

    bundle = load_checkpoint05_bundle(
        "models/final/checkpoint05_app_bundle.joblib"
    )
    predictions = fixed_target_predictions(bundle)
    diagnostics = fixed_target_diagnostics(bundle)
    verification = verify_fixed_target_bundle(bundle)

The later app should use these helpers rather than unpickling the bundle and
depending on internal keys directly.

## Provenance requirements

Final production artifacts must record:

- commit SHA;
- config;
- exact feature inputs;
- pseudo-competition validation evidence;
- random seeds;
- promotion/confirmation gate results;
- target-support diagnostics;
- exact 59-parcel prediction linkage.

Do not store source data, credentials or private raw datasets here.
