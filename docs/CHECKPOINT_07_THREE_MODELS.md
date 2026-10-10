# Checkpoint 07 — Three selected models, one unattended run

**Scope:** Only (1) optimized CatBoost on full HLS/SMAP-derived features,
(2) Local07, preserving Local04D's parcel-neighborhood Ridge principle but
learning the representation/neighborhood using new features, and
(3) genuinely pretrained Prithvi-EO-2.0-TL frozen image embeddings with
cross-validated Ridge head.

No XGBoost, LightGBM, KRR model zoo, retraining of raw data, or loading
hidden parcel y. The old Local04D baseline and 59 unknown parcel estimates
remain unchanged.

## Existing preparation

The workstation already generated 197 HLS parcel chips (4 selected time
frames), 29,592 HLS parcel observations, 84,316 SMAP rows and the full 07B
197-row table (~21,445 X-only variables). Do not repeat these downloads or
their processing.

The first independent fully nested 07C tool already exists at
`tools/run_checkpoint_07c.py`. We reuse its **CatBoost-only** path.
The other families in that older, generic script are intentionally excluded
from the new three-model orchestrator.

## What each of the three branches does

**CatBoost07:** 48 Optuna trials (by default) inside each of the 16 frozen
97-train/41-pseudo-target splits, with inner threefold CV, optimizing over
raw 256/512 balanced features and fold-local PCA16/32/64. GPU required.
The saved Local04D predictions are compared on the **same** 41 holdouts.

**Local07:** locally weighted Ridge regression fitted for each query
parcel, like Local04D, but neighborhood similarity combines geographic
distance and distances in the new X representation. Inner CV selects
representation, nearest-neighbor count, alpha and geographic-versus-feature
distance share. Previous Local04D results remain frozen for comparison.

**Prithvi07:** 197 local, full 4 × 6 × 224 × 224 HLS NPZ chips are fed through
the pretrained IBM/NASA **Prithvi-EO-2.0-300M-TL** encoder (600M-TL
optional). The code requires the existing pretrained .pt checkpoint and
config.json. It fails rather than substituting randomly initialized weights.
All 197 frozen embeddings are cached per parcel, then a compact Ridge head
is tuned using **only training labels** in each of the 16 splits. It records
global and parcel-overlapping token-pool features. GPU is required for
Prithvi inference. Freezing the encoder prevents training hundreds of
millions of parameters on only 138 known yield observations.

The full three-model comparison also considers conservative **fixed**
10%, 25% and 50% blends against frozen Local04D. These are *not*
supervised residual calibrators, and no such model is claimed.

Official Prithvi architecture and pretraining information:
https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL
TerraTorch official usage:
https://torchgeo.org/terratorch/1.2.10/guide/quick_start/

## One-time dependencies

With the pre-existing conda environment activated, install only missing
Python requirements. CUDA-enabled PyTorch should be verified BEFORE any
change, rather than overwriting a functioning GPU installation:

```powershell
git pull --ff-only
conda activate geocebada
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -m pip install -e ".[dev,models,checkpoint07,prithvi]"
```

The `prithvi` extra installs TerraTorch and PyTorch if absent. On Windows,
if this changes a working CUDA PyTorch build or preflight sees no CUDA,
restore the official CUDA-enabled PyTorch build for the workstation before
starting. These packages do **not** download the ~180 GiB raw data.

## Operator commands (Windows PowerShell)

Run from repository root. **Only checks**:

```powershell
.\tools\start_checkpoint_07.ps1 -Check
```

Optional real smoke (CatBoost, Local07 small tests and one **real** pretrained
Prithvi GPU image), no replacement of the full experiment:

```powershell
.\tools\start_checkpoint_07.ps1 -Smoke
```

**One command to start all three models unattended:**

```powershell
.\tools\start_checkpoint_07.ps1 -Start
```

The script runs preflight first, then starts exactly one independent worker,
and returns control to PowerShell. The worker executes **CatBoost → Local07 →
Prithvi** sequentially, with one stage log and separate checkpoints for
each. No user interaction or two-hour restarts are needed. One branch's
failure is recorded, but the orchestrator attempts remaining branches and
summarizes all failure states. Run the **same** `-Start` after correcting a
problem, or after a VM reboot, to resume. Do not launch two processes.

Optional stronger CatBoost/Local07 budgets, or 600M pretrained version:

```powershell
.\tools\start_checkpoint_07.ps1 -Start -CatBoostTrials 80 -LocalTrials 80
.\tools\start_checkpoint_07.ps1 -Start -PrithviModel 600 -OutputDirectory reports/checkpoint_07_three/exp_600m
```

Choose the budget **before** the first complete run. Fingerprints will
prevent mixing a later, incompatible budget or model version into the
same existing output directory. Use `-OutputDirectory` to start a
different experiment. Default is 48 trials per outer fold for CatBoost
and 48 for Local07. This is computationally intensive and may require
many hours, depending on GPU/model.

To inspect status only when convenient:

```powershell
.\tools\start_checkpoint_07.ps1 -Status
```

## Run safety and output paths

The independent worker logs, JSON state, stage Optuna studies and
predictions remain local under `reports/checkpoint_07_three/`
(gitignored):

- `status.json`: `running`, `completed` or `failed` for each stage.
- `worker.pid`: last known process ID (not proof of survival after reboot).
- `worker_*.out.log`, `worker_*.err.log`: orchestration output.
- `catboost/stage.log`: CatBoost search and Local04D comparisons.
- `local07/stage.log`: updated local-model fitting and search.
- `prithvi/stage.log`: frozen encoder/inference and Ridge head diagnostics.
- `prithvi/300M/shards/`: per-parcel frozen embeddings for safe resume.
- `three_model_summary.csv`: summary of the completed branches on
  identical target-matched splits, including Local04D reference.

A persisted `run_spec.json` hashes source features, target/split/baseline
data, chip manifest and modeling code. It prevents accidentally mixing
incompatible runs. Per-family Optuna studies also persist.

Closing the initial terminal after `-Start` does not intentionally kill
the detached worker. **Windows logoff, VM shutdown, OS restart or system
policy can kill it**. Re-running `-Start` resumes; no claims are made
that it will magically survive power loss or auto-start after reboot.
Keep the VM powered on and disable sleep for the run if permitted.

The training code does not perform Git add, Git push, or upload outputs.
The local processing data remain intact. Do not publish model predictions
or parcel-level data without checking permissions.

## Scientific validation boundaries

There are only 138 parcel yield labels. A 300-million-parameter
pretrained encoder does not create thousands of new labeled examples.
The final comparison is **development evidence**, not promotion. All
hyperparameter choices are made inside nested folds of 97 training
labels. The held-out 41 labels are used only to score predictions. The 16
repeated splits share many parcels, so they do not give 16 independent
estimates of performance.

The next task, after collecting the results, is to assess which of
CatBoost07/Local07/Prithvi07 actually reduces paired RMSE and/or has
complementary residuals, then perform one fresh untouched confirmation
before choosing a replacement for Local04D.

Important limitation: the TerraTorch Prithvi weight loader and full
four-frame output contract have to pass a **real local `-Smoke`**. CI
cannot validate pretrained GPU execution without the multi-gigabyte
weights and actual HLS chips. Do not skip that smoke on an unverified
TerraTorch/CUDA environment.
