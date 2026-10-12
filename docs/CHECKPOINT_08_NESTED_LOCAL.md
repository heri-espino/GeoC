# Checkpoint 08 — final symmetric nested local comparison

**Status:** implementation prepared; no workstation training or confirmed results yet.
Scope: compare exactly Local04D and Local07, both tuned on their own
training-only inner folds. Keep frozen Local04D as the third reference.
Never modify the canonical 59-target predictions automatically.

## Why this experiment

Checkpoint 07 pooled development RMSE: Local04D = 0.495741 t/ha,
Local07 = 0.523081 t/ha. Those estimates come from the same 16 target-matched
splits, but selection was asymmetric: historical Local04D came from a
broad previous development search; Local07 was tuned within each outer fold.
We therefore use a newly frozen, X-only confirmation bank and independently
tune both families within each new outer training set.

## Preregistered design

- 197 parcels = 138 with reported yields + 59 real prediction parcels
  without labels. The 59 missing yields must never be read or scored.
- 16 fresh confirmation splits, each 97 training / 41 pseudo-target labels,
  generated from X-only municipality/state strata. Candidate sampling
  minimizes overlap with the original development splits. All memberships
  are created before the runner loads any yield data.
- Inner tuning: threefold shuffled KFold on the 97 *training-only* yields.
  Default 96 Optuna TPE trials per family per outer split. The 96-trial
  budget searches the spaces; it does not enumerate them exhaustively.
- Local04D_nested: original 04A C4 transductive 197-parcel PCA embedding,
  original PCA score standardization, haversine distance normalization,
  and original inverse-distance weighted per-query Ridge implementation.
  k = 8,12,16,20,24,30,36,48;
  alpha = 0.3,1,3,10,30,100,300,1000;
  power = 0,0.5,1,1.5,2 (320 combinations).
  Original (24,30,1) always included.
- Local07_nested: latest 07B full X; fold-local selection of
  balanced raw256/raw512 or PCA16/PCA32/PCA64;
  k = 8,12,16,20,24,30,36,48;
  alpha = 0.3,1,3,10,30,100,300,1000;
  geographic share = 0,0.25,0.5,0.75,1 (1,600 combinations).
  All imputation/scaling/PCA are fold-local.
- Local04D_frozen: the original k=24/alpha=30/power=1 reference, refit
  on each 97-label outer training subset; no selection on held-out y.
- Evaluations: pooled and mean-split outer RMSE; worst split; paired
  delta-MSE versus the frozen incumbent; parcel-cluster bootstrap 95% CI
  resampling unique parcel IDs with their repeated predictions together.

This remains a *repeated pseudo-target validation*, not a test on new years
or statistically independent agricultural observations.

## Run on Windows (PowerShell)

Run from the GeoC repository root after Checkpoint 07 has finished.
The pre-existing 21,445-column 07B features and old 04A embeddings must be
present on the workstation. No downloads, Prithvi inference or GPU are needed.

~~~powershell
cd C:\Users\175199\GeoC
git fetch origin
git switch experiment/08-nested-local-comparison
conda activate geocebada

# Freeze X-only confirmation splits. Re-running checks equivalence.
python tools/run_checkpoint_08_local.py --prepare
python tools/run_checkpoint_08_local.py --check

# Isolated smoke, one split / two trials per family:
python tools/run_checkpoint_08_local.py --smoke

# Full search. Rerun the same command after any workstation reboot.
python tools/run_checkpoint_08_local.py --run --trials 96

# Full comparison; refuses incomplete splits:
python tools/run_checkpoint_08_local.py --summary --trials 96
~~~

This is a foreground worker; closing PowerShell or shutting down the VM can
interrupt it. SQLite Optuna trials and split prediction outputs are persisted;
rerunning the exact command resumes. Do not run two workers concurrently.
Local Ridge is CPU-based; a CUDA GPU is not required.

The run fingerprints every source file, confirmation memberships,
implementation code and search settings. To change the budget, seed, data
or code after starting, choose a different --output directory instead
of mixing incompatible trials.

## Artifacts (all local, gitignored)

- reports/checkpoint_08/confirmation_membership.csv
- reports/checkpoint_08/run_spec.json
- reports/checkpoint_08/confirm_NN__Local04D_nested.sqlite
- reports/checkpoint_08/confirm_NN__Local07_nested.sqlite
- reports/checkpoint_08/confirm_NN__paired.csv
- reports/checkpoint_08/comparison_summary.csv
- reports/checkpoint_08/paired_cluster_bootstrap.csv

These per-parcel outputs include observed yields and must not be accidentally
published to the public repository. A separate sanitized aggregate summary
may be prepared for approved sharing.

## Decision rule

Negative delta-MSE relative to frozen Local04D and a parcel-bootstrap
95% interval entirely below zero support an improvement under this
pseudo-target evaluation. Also inspect pooled and mean split RMSE, worst
split, and the magnitude of any benefit. Intervals crossing zero are
inconclusive. Do not post-hoc tune another blend or threshold on the
confirmation bank. No automatic promotion or rewriting of the frozen
59-row forecasts at reports/checkpoint_05/final_predictions.csv.
