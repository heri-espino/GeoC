# Reference26 geological gate grid: first 46 held-out parcels

This is a **separate exploratory experiment**, not a replacement for the original
two-model 138-fold LOOCV. The goal is to test whether changing **Reference26's
geological gate** improves prediction, while keeping **Compact14** as a control.
The original full comparison `tools/run_selected_champions_loo.py` is unchanged.

## Exactly the same 46 parcels

The teammate's frozen `data_n/processed/current_champions/v1/train138.parquet`
contains 138 labeled rows. Every candidate predicts the **first 46 rows of
that Parquet table in its original order**, using the other 137 rows as training
observations in each outer fold. This is one-third of the LOOCV folds, NOT a
random third and NOT a new train/test partition of size 92/46. The first 46
are not necessarily the lowest 46 numeric polygon IDs.

The internal four-fold CV has seed `20261018` and is run exclusively within
each fold's 137 labeled training parcels. Its gate selection never sees that
outer held-out yield. No data or pretrained models from GeoC's C4/Local04D
pipeline are read. Reference26's global climate model, local expert and
R3 correction remain the teammate's native implementations.

## Candidate families

| Candidate | Search inside each 137-label outer fold | Model reported |
|---|---|---|
| Original | Original K-means, k=4 or 8 | Reference26 original K-means |
| Expanded K-means | k = 2, 4, 6, 8, 12 | Reference26 with expanded k |
| DBSCAN | min_samples = 3, 5; radius = 0.75, 1.25, 2, 3 × train-only median k-distance (8 combos) | Reference26 with density clusters |
| Continuous geological support | Strength = 0.25, 0.5, 1, 2, 4 (5 combos) | Reference26 with distance-decaying global/local weight |
| Global gate | One regularized global mixing weight | Reference26 without geological clusters |
| Fixed control | Original five-feature compact expert | Compact14, unchanged |

Each candidate **only replaces the geological mixing gate** in the original
Reference26 recipe. K-means and DBSCAN operate on the same 10 geologic
composition features. DBSCAN labels points not near a *training core sample*
as outliers and backs off to the global mixing weight. The DBSCAN radius is
measured in standardized geological-feature units, **not kilometers**.

We retain the teammate's shrinkage formula for per-cluster mixing weights.
All candidate hyperparameters are chosen inside training folds by mean
squared error before testing the one excluded parcel. The output includes
the chosen clustering setting, number of DBSCAN clusters, noise fraction
and other fold-specific gate diagnostics where applicable.

No CatBoost or GPU training is involved. These models use numerical Ridge,
Elastic Net and other small scikit-learn estimators. Parallel speed-up is
obtained by fitting independent **outer folds** simultaneously on CPU.

## PowerShell (from GeoC root)

~~~powershell
conda activate geocebada
git pull --ff-only origin experiment/nuria-champions-loocv

# Confirm teammate's bundle and original model source are available
python tools/run_peer_gate_grid_loo46.py --check

# One parcel evaluated under every gate plus the Compact14 control
python tools/run_peer_gate_grid_loo46.py --smoke --jobs 1

# First 46 identical held-out parcels, with user-selected parallelism
python tools/run_peer_gate_grid_loo46.py --run --jobs 12

# Print results later, without retraining
python tools/run_peer_gate_grid_loo46.py --summary
~~~

`--jobs` accepts **any positive integer**, without an artificial cap of 8.
The actual worker count is naturally bounded by the number of remaining
outer folds (at most 46). Start at a reasonable value for available memory;
e.g. 8 or 12 before considering 24 or 46. Each process loads the source
bundle and creates its own nested training cache. A large `--jobs` can
exhaust RAM, induce swapping or slow the machine.

Optional fast isolations:

~~~powershell
# Original + DBSCAN only; separate new output, different method grid.
python tools/run_peer_gate_grid_loo46.py --smoke --jobs 1 --variants original,dbscan --output reports/peer_gate_grid_loo46_dbscan_only
~~~

**Do not reuse the same output directory for a different variants list or
modified script.** All outputs are private and ignored under the default
`reports/peer_gate_grid_loo46/`; if overriding `--output`, make sure the
new location is ignored too before running.

## Outputs and interpretation

- `smoke/`: independent single-fold smoke output, never included in the
  46-fold scores.
- `fold_001.csv` … `fold_046.csv`: prediction per method for one held-out
  parcel; written atomically and reused if verified after interruption.
- `gate_001.json` etc.: inner-train-only hyperparameter selection and
  DBSCAN cluster/noise diagnostics.
- `loo46_predictions.csv`: 46 paired held-out yield predictions.
- `loo46_scores.csv`: pooled RMSE, MAE, R², mean error and paired difference
  in mean squared error relative to the original Reference26.
- `run_spec.json`: strict input/model code, grid, and fold-protocol hashes;
  changes cannot mix with old results.

**Statistical caveat:** the first 46 rows may be geographically or otherwise
unrepresentative, so these are *46/138 selected LOOCV fold predictions*,
not a proper 138-fold LOOCV estimate. Choosing a model after comparing
these 46 outer scores makes its winning RMSE optimistic; only the
hyperparameters chosen inside the 137-label training folds are protected
against outer-fold target leakage. Keep the original Reference26 and
Compact14 benchmark unchanged for comparison. No results have been
computed on the user's private workstation data in this editing session.
