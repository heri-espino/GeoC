# LOOCV for teammate's two champions

**Only Reference26 and Compact14.** No GeoC model, Local04D, PCA,
DBSCAN, CatBoost, B1 weight selection or new feature engineering.

## Inputs

- Her original source code: `../todoseboso/src/current_champions.py`
  (from `heri-espino/todoseboso`).
- Her original private exported inputs:
  `GeoC/data_n/processed/current_champions/v1/` (the manifest, 138-row
  Parquet table, groups and precomputed geographical distances).
- Output: `reports/selected_champions_loo/`, ignored by Git.

The script verifies the bundle's original file hashes using her own
`load_bundle(..., verify_saved_models=False)`; no saved model is reused.
This experiment does **not** load any GeoC baseline data, PCA, or original
target tables; it simply calls her `ColdExperts.full_fit` for each fold,
then her `predict_saved` for the two selected models.

## Commands (PowerShell, from GeoC)

~~~powershell
git pull --ff-only origin experiment/nuria-champions-loocv

# Check the 138-row bundle and original model imports.
python tools/run_selected_champions_loo.py --check

# One held-out parcel to verify cold training.
python tools/run_selected_champions_loo.py --smoke --jobs 1

# 138 held-out parcels, two workers; raise to 4 if RAM permits.
python tools/run_selected_champions_loo.py --run --jobs 2

# After 138 fold CSV files are present.
python tools/run_selected_champions_loo.py --summary
~~~

## Exactly what changes from her original refit

1. For each outer parcel, hold out its row and **mask its yield to NaN**
   before any fitting. All other 137 yields remain available.
2. Four deterministic shuffled inner folds are defined on those 137
   training positions (seed `20261018`).
3. Fit **her original complete Reference26 and Compact14 recipes** through
   `ColdExperts.full_fit`. This includes climate modeling, local Ridge,
   geological gate and R3 strength selection as originally implemented.
   B1's extra nested calibration fits in `refit_fold` are not executed.
4. Predict the one masked parcel through her original `predict_saved`.
   Save exactly two predictions alongside its observed yield, *after*
   fitting. Only this held-out observation is used for scoring.

Each fold has a distinct private cache and atomic CSV output; completed
folds are verified and resumed. Use `--jobs 2` or `--jobs 4` to process
**independent outer folds** concurrently on CPU; per-fold numerical
libraries are capped at one thread.

Files: `fold_001.csv`...`fold_138.csv`, `loo_summary.csv`,
`loo_predictions.csv`. The one-fold smoke lives under
`reports/selected_champions_loo/smoke/` and does not constitute
a LOOCV performance estimate.

The full 138-fold RMSE/MAE/R2 cannot be claimed until all folds finish.
Selection based on previously explored yields remains exploratory
rather than independently confirmed.

## If the script reports a failure

The dependency check is now **only the teammate's original bundle and
code**. GeoC's PCA parity failure 5.556 was caused by the earlier,
unnecessary Local04D comparison and cannot arise from this script.

If a fold fails during her nested training, keep the traceback: do not
change her hyperparameters or the 26 input features to silence it.
