# Public aggregate peer-model results

This directory contains **only model-level metrics**, suitable for GitHub.
Files are created on the workstation with:

~~~powershell
python tools/publish_peer_results.py --export
~~~

Run the command now to preserve the previously completed 138-fold
Reference26/Compact14 LOOCV and 46-fold geological gate comparison. Rerun
it after the full 138-fold DBSCAN experiment and four-model paired comparison.

Expected outputs (generated only if the corresponding private summary exists):

- `reference_compact_loo138.csv` — exact recovered Reference26/Compact14 RMSE, MAE, R² and bias
- `historic_local04d_climate_loo138.csv` — original Local04D and climate augmentations from the earlier separate LOOCV
- `geological_gates_loo46.csv` — fixed first-46 comparisons
- `dbscan_loo138.csv` — complete geological DBSCAN CV
- `four_model_comparison_loo138.csv` — four models evaluated on exactly matched 138 heldouts
- `legacy_five_model_loo138.csv` — optional five-model LOOCV, when present
- `sources.json` — paths and hashes of the *aggregate summary* sources

Only the aggregate schema's model labels, held-out count, training labels
per fit, RMSE, MAE, R², bias, and aggregate MSE differences are exported.
All `reports/...` model caches, held-out parcel IDs, observed yields,
per-parcel predictions and raw `data_n/` inputs stay ignored by Git.

To publish:

~~~powershell
git add results/peer_models/
git commit -m "Archive aggregate CV results"
git push origin experiment/nuria-champions-loocv
~~~

The results are initially stored locally; GitHub does not receive them
until the user commits and pushes. Previously supplied rounded results
are also preserved at `docs/PEER_LOOCV_RECORDED_RESULTS.md`.
