# Checkpoint 05 — Final Global–Local Mixture

**Status:** CLOSED / COMPLETED — NO PROMOTION

Checkpoint 05 is the only modeling stage reopened after Checkpoint 04F. It exists because
Checkpoint 03 learned strong global experts and Checkpoint 04 learned strong local/transductive
experts, but the project had not yet tested a leakage-safe stack that combines both families
inside the exact pseudo-competition protocol.

The incumbent remains:

    LocalRidge_C4_all_deterministic_Geo_k24_a30_p1

with target-matched mean RMSE 0.487845.

Checkpoint 05 evaluates whether global and local errors are sufficiently complementary to
justify replacing that incumbent.

## Completed result

The run completed on 2026-09-23. The best development challenger was
`LocalE13_w0p10` (90% Local04D + 10% E13), with mean RMSE 0.487243 and
pooled RMSE 0.495261 versus Local04D at 0.487845 and 0.495741.

The predeclared LOSO method-selection gate scored 0.487934 mean RMSE and
0.495933 pooled RMSE, slightly worse than Local04D on both metrics. The gate
therefore failed, the fresh confirmation bank was intentionally not scored,
and Local04D remains the final method.

Canonical interpretation: `docs/CHECKPOINT_05_FINDINGS.md`.  
Canonical post-05 prediction file: `reports/checkpoint_05/final_predictions.csv`.


## Competition-only data contract

Checkpoint 05 is pinned to
`data/processed/features_v1/parcel_features_competition.csv`. The historical
`parcel_features_all.csv` artifact is not an active input for this stage. A base feature may
still be available in both the clean and competition availability modes; that does not change
the active track. The operative rule is that model construction runs from the competition
table and the competition representation policy only.

The runner validates this contract before either preflight or the full scientific run, and a
regression test prevents the config from silently drifting back to `all`.

## Base experts

Global experts are refit and tuned strictly inside the visible labels of each pseudo-split:

- PLS on C0 base;
- Ridge on C3 base + agronomic + fold-local discovery;
- CatBoost on C1 agronomic, averaged across three seeds after inner tuning;
- ExtraTrees on C1 agronomic.

Transductive experts are frozen from Checkpoint 04D.1:

- Local04D: C4 / geographic / k=24 / alpha=30 / inverse-distance power=1;
- Graph04D: GeoAgro25 / k=6 / lambda=8.

The historical global ensembles E13 and E123 are reconstructed inside the same outer
pseudo-splits rather than borrowing the old Checkpoint 03 OOF predictions.

## Cross-fitting contract

For each outer pseudo-competition:

1. hide only y for the pseudo-target rows;
2. generate repeated OOF predictions for every base expert on the visible labels;
3. tune global experts inside each cross-fit training fold;
4. train the meta-model only on those OOF base predictions;
5. refit each base expert on all visible labels;
6. predict the outer pseudo-targets;
7. apply the already-trained meta-model.

No outer pseudo-target y enters tuning, cross-fitting, gating, or stacking.

## Meta-models

Checkpoint 05 compares:

- fixed E13/E123 global ensembles;
- fixed small Local + E13/E123 blends;
- convex non-negative stacking;
- Ridge stacking;
- Elastic Net stacking;
- Huber stacking;
- shallow ExtraTrees stacking;
- shallow histogram gradient boosting stacking;
- a support-conditioned softmax mixture-of-experts.

The mixture-of-experts learns parcel-specific convex weights:

    w_im = softmax(beta_m^T s_i)

where s_i contains leakage-safe support descriptors such as nearest geographic/agronomic
distance, relative support, visible same-state/same-municipality fractions, and Local–Graph
disagreement.

## Promotion rule

A lower full-development RMSE is not sufficient.

Promotion uses two gates.

First, on the original 16 target-matched development splits, the challenger must improve mean
and pooled RMSE and the leave-one-development-split-out selector must also improve both metrics
relative to Local04D.

Second, only after that gate passes, the runner generates a **fresh 16-split X-only
target-matched confirmation bank** with a predeclared unused seed. The challenger is frozen
before these labels are scored. It replaces Local04D only if it improves both mean and pooled
RMSE on the fresh bank as well.

Exact duplicate masks from the original development bank are rejected. Otherwise Checkpoint
05 retains Local04D.

## Run

GPU is the default and automatic CPU fallback is disabled.

Before committing the workstation to the long run, use the fail-fast preflight:

    git pull
    conda activate geocebada
    python tools\run_checkpoint_05.py --preflight

This checks required inputs, the 197/138/59 contract, configured representations and split bank, constructs the complete fresh X-only confirmation bank and verifies its 16 unique 41-row masks against the development bank, reproduces Local04D/Graph04D exactly against frozen 04F, and checks CatBoost installation/GPU visibility without fitting any global expert, stacker or mixture-of-experts. Then start the final run:

    python tools\run_checkpoint_05.py

If interrupted after one or more completed outer splits:

    python tools\run_checkpoint_05.py --resume

### Nested PLS feasibility guard

A workstation run on 2026-09-22 exposed an edge case after 29/34 outer splits:
municipality-grouped meta cross-fitting produced a 34-label training block whose
inner CV contained an 18-row training fold, while the inherited PLS grid still
contained `n_components=20`. Scikit-learn correctly rejected that candidate.

Checkpoint 05 now computes, independently inside each nested fit,

    K_max = min(n_features, minimum inner-training-fold size)

and removes only component candidates above `K_max`. This is an X/sample-size
feasibility constraint, not target-based tuning. Larger folds retain the original
20-component candidate when it is valid.

The split-level partial artifacts are written only after a complete outer split.
Therefore a crash during the next split can be resumed without recomputing the
already completed splits.

Intentional CPU execution requires:

    python tools\run_checkpoint_05.py --catboost-task-type CPU --confirm-cpu y

## Final artifacts

The runner writes:

    reports/checkpoint_05/
      base_oof_predictions.csv
      pseudo_predictions.csv
      split_metrics.csv
      protocol_summary.csv
      residual_correlation.csv
      meta_selection_details.csv
      loso_selection.csv
      loso_predictions.csv
      loso_summary.csv
      confirmation_split_membership.csv
      confirmation_predictions.csv
      confirmation_summary.csv
      actual_candidates.csv
      final_predictions.csv
      final_prediction_diagnostics.csv
      model_manifest.json
      checkpoint_05_report.json
      checkpoint_05_report.md

It also writes two local serialized bundles:

    models/final/checkpoint05_app_bundle.joblib
    models/final/checkpoint05_model.joblib

The lightweight app bundle contains the fixed 59 prediction/diagnostic contract
without fitted estimator objects and is the preferred input for the later
Streamlit interface. The full model bundle additionally preserves fitted global
experts and meta-models for reproducibility. Both are ignored by Git.

Before the runner reports PASS, it serializes and reloads both bundles and
verifies that the stored 59-row final table reproduces the selected actual
candidate to numerical tolerance.

Until Checkpoint 05 finishes and the promotion gate is evaluated, the canonical competition
prediction file remains Checkpoint 04F.


## Postscript — Checkpoint 03D global closure

After Checkpoint 05 closed without promotion, Checkpoint 03D was reopened only
to test whether the small 03C global grids had limited the experts supplied to
05. It completed on 2026-09-24.

Robust 03D finalists:

```text
HistGBLarge G1     state 0.549924   grouped 0.715339
XGBoostLarge G1    state 0.525847   grouped 0.733080
LightGBMLarge G1   state 0.540812   grouped 0.748478
```

These are frozen Checkpoint 03 state/grouped protocols, not the target-matched
Checkpoint 05 protocol. They do not retroactively invalidate Checkpoint 05 or
replace Local04D. Any use of the new globals in the competition solution
requires the separate narrow 05B protocol.
