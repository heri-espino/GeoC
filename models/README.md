# Models

Checkpoint 04 is the active modeling phase.

Model artifacts in this directory must serve the fixed-target transductive reconstruction
objective described in `docs/TRANSDUCTIVE_OBJECTIVE.md`.

The final model need not be a single global estimator. Valid artifact types include:

- global PLS/Ridge/CatBoost/tree models;
- target-specific local models;
- graph/semi-supervised models over the 197 known parcels;
- covariate-shift/domain-adaptation models;
- calibrated mixtures or ensembles whose weights are justified by transductive pseudo-test RMSE.

Historical Checkpoint 03 candidate definitions remain frozen in their configs/reports.

Final production artifacts should record:

- commit SHA;
- config;
- exact feature/trajectory inputs;
- pseudo-competition validation evidence;
- random seeds;
- external-data provenance;
- target-support diagnostics;
- exact 59-parcel prediction table linkage.

Do not store source data here.
