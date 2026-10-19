# Selected-model LOOCV: Reference26, Compact14, Local04D

**Goal:** Compare only the teammate's two selected champions with GeoC Local04D
under exactly the same 138-parcel leave-one-parcel-out folds.

**Status:** Runner and synthetic QA prepared. Real data are local only, and
no CPU/GPU speedup or RMSE has been measured in this GitHub session.

## Why this runner is faster than the earlier scripts

Use `tools/run_selected_champions_loo.py` instead of either
`run_peer_geology_dbscan.py` or `run_peer_champions_loo.py` for the user's
current goal. The original peer `refit_fold` does four extra fully nested
outer-training fits to select the secondary B1 ensemble. We do not need B1.

This new runner calls **the original**
`src.current_champions.ColdExperts.full_fit`, retaining the train-only
CT3 global basis, seven-/five-feature local experts, geological gate for
Reference26, and R3 residual correction, with its original nested tuning.
It predicts Reference26 and Compact14 only and separately refits the frozen
Local04D reference on the same 137 labels. It skips the additional B1
model-selection fits rather than approximating the selected models.

Reference26/Compact14 use scikit-learn's Ridge, Elastic Net, KMeans and
related numeric operations; **neither selected candidate trains CatBoost
or XGBoost**. No CUDA/GPU code is used here. The practical acceleration is
parallelizing independent outer parcels on CPU, without oversubscribing BLAS.

## Run on Windows

Data bundle is under ignored root
`data_n/processed/current_champions/v1`; model source in the sibling
`../todoseboso` checkout. The peer import chain may require its recorded
research dependencies (including `exactextract` and `seaborn`).
Only execute trusted team `joblib` data, never arbitrary pickle sources.

~~~powershell
cd C:\Users\175199\GeoC
conda activate geocebada
git fetch origin
git switch experiment/nuria-champions-loocv
git pull --ff-only origin experiment/nuria-champions-loocv

# Fast input alignment test; no refits
python tools/run_selected_champions_loo.py --check

# One held-out parcel, all THREE models; checks that nested refitting works
python tools/run_selected_champions_loo.py --smoke --jobs 1

# Full 138-fold LOOCV. Start with 2 workers; use 4 if RAM permits.
python tools/run_selected_champions_loo.py --run --jobs 2

# Once all 138 folds are complete
python tools/run_selected_champions_loo.py --summary
~~~

If `--run` is interrupted, run it again and it will skip completed
fold CSVs. For isolated subranges use `--from-fold 1 --through-fold 30`
etc. Outer processes write distinct per-fold cache directories and atomic
CSV files. `--jobs 4` increases memory pressure because separate workers
load the peer feature bundle and the GeoC C4 representation; benchmark
`--jobs 2` first. Numeric-library thread counts are set to 1 per process.
On Windows, spawned Python workers can have nontrivial startup cost.
Use CPU/task-manager measurements to decide between 2 and 4 workers.
This cannot accelerate an individual outer fold as though it were 4 folds.

Output path (ignored by Git):
`reports/selected_champions_loo/`. A separate `smoke/` subfolder keeps
the one-fold smoke out of the 138-fold results. Each completed fold emits
`fold_001.csv` and its own `peer_cache/fold_001` work directory. A full
run writes `loo_predictions.csv` and `loo_summary.csv`. No official
59-parcel predictions are touched.

The runner refuses a changed model/data fingerprint in the same output
folder. Use `--output reports/selected_champions_loo_v2` for changed
code/bundle/protocol and add that output to `.gitignore` locally before
running. Do not merge or promote based only on smoke or the repeatedly
studied 138 labels.

## Validation and interpretation

- Training: 137 original known yields; exactly one excluded outer y.
- Peer cold refit includes nested cross-fitted training-only selection.
- GeoC C4 PCA20 Local04D uses target-free 197-parcel X, the usual
  24-neighbor local geographical Ridge, alpha 30, inverse-distance power 1.
- Pooled RMSE, MAE, R2 and bias are reported for all 138 unique heldouts.
- No B1, CT3-only or DBSCAN arm is presented as an extra contender.
- A three-way paired performance difference is **exploratory**, not
  independent confirmation: her champion recipes were previously
  selected using the same 138 observed labels.
- Actual package versions and a successful local smoke remain necessary.

## 2026-10-18 PCA parity repair

The first workstation `--check` correctly stopped with
`20-PC numerical parity failure (5.556)`. This was traced to the
peer bridge requesting 160 PCA axes even though Local04D originally
fits **exactly 20**. Separate PCA fits can have different SVD signs,
and the prefix is not the frozen representation used in Checkpoint 04A.
The peer bridge now calls `load_pca_data(..., max_components=20)`.
The strict historical 20-PC parity and explained-variance checks
remain intact; there is **no tolerance relaxation, model modification
or replacement with unverified PCA**. Synthetic contract test:
`tests/test_peer_local04d_pca20.py`.

After pulling the updated branch, run
`python tools/run_selected_champions_loo.py --check` again before
the smoke. Pandas DataFrame fragmentation warnings were from repeatedly
constructing 160 PCA columns and should be reduced by the 20-PC fit.
If strict parity still fails, share the new error before training.

