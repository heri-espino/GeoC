# Budgeted Reference26 Ridge search: 25 → 8 → 3

## Stop the expensive 250 × 138 job safely

Press **Ctrl+C once** in the PowerShell window, and allow child workers time
to terminate. A second Ctrl+C is appropriate only if the first leaves the
process unresponsive. Previously completed per-(heldout, hyperparameter)
predictions are already stored atomically under
`reports/reference26_ridge250_loo/fold_XXX/kYY_aZ.csv`. Do **not** delete,
rename or move this directory. Terminating a current worker may lose only
its in-progress fit; completed CSV files persist.

The progress line `[GRID 64/3450] fold=003: 10 new ...` means **64 completed
task batches of 10 predictions each**, not 640 different leave-one-out
test parcels. The original runner schedules by parcel first, so the
~640 completed predictions may span only a handful of distinct parcels.
The old `--summary` deliberately requires all 250 × 138 predictions and
cannot give a valid intermediate score.

## New independent script and protected reuse

The original 250-grid script and its saved results **remain unchanged**.
The new `tools/run_reference26_ridge_budget.py`:

1. Audits the old 250-grid directory to count completed, distinct parcels
   and reusable predictions among 25 planned configurations;
2. Validates original input/model/runner hashes in its run_spec.json and
   reuses only **numerically validated** original prediction CSVs, copying
   them to a separate Git-ignored location; nothing is rewritten in the
   original 250-grid directory;
3. Runs each remaining fixed (geographic neighbors, Ridge alpha) pair
   using the exact old native Reference26 fit and 137/1 parcel isolation,
   saving output after every fit;
4. Computes paired RMSE and MAE on **identical, predeclared sets of parcel
   positions** at each fidelity stage. It never compares a model scored on
   12 parcels directly to another model scored on 46 parcels.

The fixed 25-candidate coarse grid:

| Parameter | Values |
|---|---|
| Geographic neighbors `k` | 12, 20, 24, 32, 48 |
| Ridge `alpha` | 3.2, 10, 30, 100, 300 |

Includes original (24,30), which must reproduce saved Reference26
predictions on the first fold to 1e-7 t/ha.

## Commands (from GeoC root in activated geocebada environment)

~~~powershell
# Only AFTER stopping the old command:
git pull --ff-only origin experiment/nuria-champions-loocv

# Audit old completed work. No fitting.
python tools/run_reference26_ridge_budget.py --audit

# Stage A: all 25 configurations on the same first 12 held-out parcels
python tools/run_reference26_ridge_budget.py --screen --jobs 12

# Stage B: top 7 nonbaseline configs + original (24,30), 46 parcels each
python tools/run_reference26_ridge_budget.py --advance --jobs 12

# Stage C: top 2 nonbaseline configs + original (24,30), all 138 parcels
python tools/run_reference26_ridge_budget.py --final --jobs 12

# Print all available stage scoreboards, without training
python tools/run_reference26_ridge_budget.py --summary
~~~

`--jobs` accepts any positive integer; real concurrency is bounded
by the remaining tasks and available RAM. Default `--batch-size 5` reuses
climate model fits in a process-local batch. Each stage can be interrupted
and rerun; previously completed predictions remain available. Do not
change the source, input bundle, runner, or grid while a stage has begun:
run-spec fingerprint checks will stop incompatible resumption.

## Cost and expected deliverables

In the absence of reuse, this runs roughly:

| Stage | Total model–parcel pairs for that stage | New fits after previous stage |
|---|---:|---:|
| 25 candidates × 12 parcels | 300 | 300 |
| 8 candidates × 46 parcels | 368 | 272 |
| 3 candidates × 138 parcels | 414 | 276 |
| Total distinct configurations tested | 25 | **848 model fits** |

The last figure includes the 12-fold screening work and reuse across
stages: 300 + 272 + 276 = 848. This is dramatically smaller than 34,500.
On a machine with 12 equally productive workers and the previous
observed ~120 seconds per single candidate fit, the idealized estimate
would be roughly **2.4 hours**, before operating-system and resource
overheads, and minus successfully reused fits. It is not a guarantee.

Private outputs are stored in
`reports/reference26_ridge_budget/`, with aggregate metric files:

- `screen_scores.csv` — RMSE/MAE on first 12 parcels for 25 models;
- `advance_scores.csv` — on first 46 for eight models;
- `final_scores.csv` — on all 138 for the three finalists;
- `*_selection.json`, `run_spec.json`, and private fold-level predictions.

For stage ranking, all candidates share exactly the same held-out IDs
and original yields. Each stage additionally displays the saved original
Reference26 RMSE on **that same subset**, never the full-138 benchmark
when discussing a 12- or 46-parcel score.

## Statistical caveats and further tuning

The first 12 rows may be unrepresentative; therefore screening is noisy,
and successive elimination can discard a good configuration. More
importantly, stage rankings reuse observed held-out labels for adaptive
model selection. The full 138-fold finalist results are useful for
competition-focused model choice, but **not independent, unbiased
performance confirmation**. Fully nested hyperparameter selection or
genuinely new labeled parcels would be required for that claim.

For only two hyperparameters and a 25-configuration budget, this
coarse grid plus successive halving is a simpler, controllable alternative
to Bayesian optimization. A later Optuna TPE or Gaussian-process surrogate
can use early scored configurations (e.g. on the common 12 or 46 holdouts)
as warm-starting evidence, but must use **the same evaluation fidelity
within each optimizer trial**, with additional caution about adaptive
selection bias.

The old full 250-grid files are kept intact to preserve provenance and
allow future experiments with a larger computation budget.
