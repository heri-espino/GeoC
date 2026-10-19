# Reference26 — 250 geographic Ridge configurations across full LOOCV

This experiment evaluates exactly **250 fixed configurations** of the original teammate's
`Reference26` architecture against the already-saved **138-parcel LOOCV** baseline.

- **10 geographic neighbor counts:** `8, 12, 16, 20, 24, 28, 32, 40, 48, 64`.
- **25 positive Ridge alpha values:** `0.1, 0.15, 0.22, 0.32, 0.46, 0.68, 1,
  1.5, 2.2, 3.2, 4.6, 6.8, 10, 15, 22, 30, 46, 68, 100, 150, 220,
  300, 460, 680, 1000`.
- **10 × 25 = 250 configurations**, including the original `(k=24, alpha=30)`.
- **138 outer folds** each hold out one original teammate parcel, train with the other 137
  labeled observations, and run the teammate's original nested `ColdExperts.full_fit`.
- **No rerun of the original baseline for comparison:** original `Reference26`
  predictions come from `reports/selected_champions_loo/loo_predictions.csv`.
  The `(24,30)` candidate is nevertheless included as a *numerical parity
  control*: its 138 predictions must match the saved baseline to `1e-7` t/ha.

Only local Ridge neighbor count and alpha change. The C-T3 climate/Elastic Net
model, original K-means geological weight selection (k=4/8), five- and seven-
feature roles and native R3 strength selection remain unchanged. The original
teammate source `../todoseboso` is not modified. Local Ridge is implemented
via two separate original code paths: one used by internal fits
(`src.geoc_signal_research.local_prediction`) and one used for test-time
predictions (`src.current_champions.predict_local_ridge`). The runner overrides
**both** using identical neighbor/alpha values. It verifies the original
`(24,30)` path on the first known parcel before expensive execution, and checks
**all 138** original-parameter predictions at final scoring.

## Steps (PowerShell, GeoC directory)

~~~powershell
conda activate geocebada
git pull --ff-only origin experiment/nuria-champions-loocv

# Read teammate bundle and original 138-fold predictions; no model fits.
python tools/run_reference26_ridge250_loo.py --check

# Refit one original (24,30) fold to check native numerical parity;
# also exercise one other candidate on the same held-out parcel.
python tools/run_reference26_ridge250_loo.py --smoke

# All 250 configurations, all 138 heldout parcels; any positive --jobs.
python tools/run_reference26_ridge250_loo.py --run --jobs 12

# Read already computed results, no training.
python tools/run_reference26_ridge250_loo.py --summary
~~~

The worker count has **no artificial upper limit** (naturally bounded by
remaining outer-fold batches), but nested fits can consume considerable memory.
You can run `--jobs 4`, `--jobs 12`, `--jobs 24`, etc. Too many workers
may slow the machine by swapping, and all models run on CPU; CatBoost/GPU is
not applicable to this experiment.

The default batch size is 10 candidate configurations per task
(`--batch-size 10`). To reduce latency until visible progress,
use `--batch-size 5`; for better reuse of train-only climate fits,
try `--batch-size 20`. **Changing only --jobs or --batch-size does not
change model results or invalidate saved folds.** Train-only C-T3 model
objects are shared in memory *within a single fold/batch*, not across target
holdouts or from cached outputs with known labels. Every local Ridge model,
geological gate, and R3 correction is refitted for its own configuration.

## Practical compute and resumability

This is **34,500 full outer fits** (250 × 138), not 250 cheap Ridge
estimators. Each full fit runs nested model selection and can cost far more
than a standalone Ridge solve. Even with parallelism it may take many hours
or days; no runtime is asserted until the local smoke test is measured.

For each pair `(fold, k, alpha)`, the runner writes a private CSV under
`reports/reference26_ridge250_loo/fold_XXX/`. After a successful
prediction, its per-candidate nested model cache is discarded to avoid
unbounded disk growth. If interrupted, rerun the **same `--run` command**:
all valid completed predictions will be skipped. Do not delete or publish
the run provenance (`run_spec.json`) while fold CSVs remain.

Final outputs:

- `reports/reference26_ridge250_loo/loo138_summary_250.csv` — 251
  aggregate rows: original Reference26 + each of the 250 fixed configurations,
  sorted by RMSE.
- `reports/reference26_ridge250_loo/loo138_predictions_private.csv` —
  aligned parcel-level outcomes and forecasts; **never commit** this file.
- `run_spec.json` — source hashes, fixed 250-pair grid and CV protocol.
- `smoke/` — separate sanity checks and numerical parity observations.

All these local outputs are ignored by Git. Aggregate summary scores can
subsequently be exported under `results/peer_models/` without parcel IDs
or label-bearing fold outputs.

## Methodological interpretation

Each individual configuration receives a valid **138-fold LOOCV** estimate,
on exactly the same heldouts as the frozen original. However, looking at
all **250 held-out RMSEs and then choosing the minimum** reuses the outer
labels for model selection. The winning minimum is optimistically biased
as an estimate of future generalization. In this competition-focused
setting, that is useful *exploratory selection*, not independent confirmation.
If an unbiased performance estimate of the selected hyperparameter search is
needed, use outer LOOCV with **training-only inner hyperparameter selection**
or untouched new labels. Do not describe this exhaustive comparison as
nested 250-candidate hyperparameter selection: original *model* internals
are nested; the **cross-candidate winner is selected post hoc**.

This is deliberately a separate experiment from the geological K-means
cluster-number search and DBSCAN, which did not improve on the original
138-fold Reference26 RMSE of approximately **0.468186 t/ha**.
