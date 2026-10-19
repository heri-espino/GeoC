# Reference26: separate fixed K-means hyperparameter trials

This experiment changes **only the number of geological K-means clusters**
in the teammate's original **Reference26** model. It does **not** build
Compact14 + K-means, does not alter Reference26's seven-feature local Ridge,
C-T3 climate branch, or native R3 selection, and does not train a new B1.

The original Reference26 chooses k = 4 or 8 *inside each training fold*.
The new experiment evaluates each requested fixed k separately, so the table
will contain `Reference26_KMeans_k4`,
`Reference26_KMeans_k8`, `Reference26_KMeans_k14`,
`Reference26_KMeans_k26`, etc. The original internally tuned Reference26
is read **from already saved full-138 LOOCV predictions**, never retrained.

## Commands (PowerShell, from GeoC root)

~~~powershell
git pull --ff-only origin experiment/nuria-champions-loocv

# Verify original source, 138-row private bundle and saved Reference26 predictions
python tools/run_reference26_kmeans_grid.py --check

# Original 137/1 fit, first parcel, every fixed k; no CV summary claim
python tools/run_reference26_kmeans_grid.py --smoke --jobs 1

# Same first 46 held-out parcels, one row PER fixed k
python tools/run_reference26_kmeans_grid.py --run --jobs 12

# Read finished results without any training
python tools/run_reference26_kmeans_grid.py --summary
~~~

Default fixed k values: **4, 8, 12, 14, 16, 20, 26**. You can pass an explicit
list via `--ks`, for example `--ks 4,8,14,26`. Different k-lists require
a *different* `--output` directory; strict fingerprinting prevents mixing
results from different runs. All private outputs remain ignored by Git.
`--jobs` accepts **any positive integer**, and the number of concurrent
workers is bounded only by remaining folds (and practical RAM limitations).
Within each outer process, numerical libraries use a single thread.

If the 46-fold experiment identifies candidates worth examining, do a
**separate 138-fold experiment** on predeclared candidates:

~~~powershell
python tools/run_reference26_kmeans_grid.py --run --folds 138 --ks 14,26 --jobs 12 --output reports/reference26_kmeans_fixed_138
python tools/run_reference26_kmeans_grid.py --summary --folds 138 --ks 14,26 --output reports/reference26_kmeans_fixed_138
~~~

This runs the two specified configurations on all 138 parcels; it does not
rerun the original Reference26, which was already evaluated.

## Protocol

Each fixed-k configuration uses the **same first 46 rows of the teammate's
original 138-row Parquet bundle** (or all 138 with `--folds 138`). For each
held-out parcel, the yield is masked; 137 training labels are passed to the
original `src.current_champions.ColdExperts.full_fit` and its native R3
tuning. Only `ColdBases.shares` is replaced: the same training-only
geological composition representation and original
`src.conditional_local_phases.ExpertGate("clusters", k)` are used with one
fixed k rather than inner-selecting 4 or 8.

The `Reference26_original` comparator is loaded from
`reports/selected_champions_loo/loo_predictions.csv`. The experiment
validates the complete 138-ID set and **all observed yields**, then aligns
predictions by ID before computing scores. No held-out label affects the
fixed-k fit.

For each k the output table includes RMSE, MAE, R², bias, and paired
difference in mean squared error relative to original Reference26, with
an individual, self-explanatory model name. Private fold predictions,
selected fitted cluster counts, and caches stay in
`reports/reference26_kmeans_fixed_46/`.

**Interpretation:** Because the first 46 held-out labels have already been
used in prior searches, choosing the lowest-RMSE k from this table is
exploratory and the selected performance is optimistic. A follow-up full-138
LOOCV describes performance on more labeled parcels but is *not an
independent test* after selection. A more rigorous improvement claim would
require untouched labeled observations or a fully nested selection design.

Large k (e.g. 26) implies few points per geological cluster with only
137 labeled parcels. The original smoothing/shrinkage inside
`ExpertGate` remains unchanged, and `fitted_clusters` diagnostics
show when fewer distinct clusters can be formed than requested.
