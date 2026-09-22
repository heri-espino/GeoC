# Models

Checkpoint 05 is the active final modeling phase.

The fixed-target transductive objective remains defined in
`docs/TRANSDUCTIVE_OBJECTIVE.md`. Checkpoint 04F supplies the incumbent
Local04D predictions; Checkpoint 05 is allowed to replace them only if the
predeclared development, LOSO and fresh-confirmation gates pass.

## Final Checkpoint 05 bundle

The long workstation runner writes:

    models/final/checkpoint05_model.joblib

This file is intentionally ignored by Git. It is a serialized reproducibility
bundle for the later Python/Streamlit application, not a source-data artifact.

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
        "models/final/checkpoint05_model.joblib"
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
