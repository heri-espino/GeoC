# Paired LOOCV: Local04D versus Nuria's current champions

**Status (2026-10-18):** comparison runner and synthetic contract tests implemented
on a separate branch. **No actual peer 138-fold LOOCV run or new RMSE has been
computed.** The peer's private 26-feature bundle is not in either GitHub repo.
Do not merge this experiment with checkpoint histories, publish parcel-level
outcomes, or change the 59 canonical GeoC forecasts.

## Why this is worth testing

Source repositories:

- GeoC: https://github.com/heri-espino/GeoC
- Peer source (fork): https://github.com/heri-espino/todoseboso
- Peer recipe: src/current_champions.py; scripts/521_current_champions.py.
- Peer evidence: results/current_champions/v1 and results/comparison_97_41/v1.
- GeoC anchor: src/geocebada/evaluation/checkpoint10_pca.py,
  src/geocebada/evaluation/checkpoint04d1.py, frozen C4 PCA20 k24 alpha30 p1.

The peer repo reports the following **mean of 16 split metrics** on **its
own** 97-training/41-test benchmark (RMSE, t/ha): Local04D/L40 0.4893,
Reference26 0.4625, Compact14 0.4661. This is an internally comparable
benchmark, but **not** GeoC's Checkpoint 08 0.481936 pooled RMSE, which uses
different masks/aggregation. The peer's municipality/spatial-block scores also
favor her current champions over the L40 comparator. Their 65 original
folds/16 fresh masks repeatedly reuse the same 138 known yields.

The GeoC standalone frozen LOOCV control on 138/137+1 reported RMSE
0.485133 t/ha in the user's past run (private
reports/area_weighted_climate_loo/loo_summary.csv). This number is a separate
benchmark and **cannot be compared** against peer 97/41 numbers.

## Actual architecture (no surrogate reimplementation)

Peer [README](https://github.com/heri-espino/todoseboso/blob/main/README.md)
and [recipe](https://github.com/heri-espino/todoseboso/blob/main/src/current_champions.py)
specify four return values:

- CT3: compact four-covariate tuned Elastic Net reference.
- Reference26: CT3 with local-seven expert, geographic/geologic gate and R3
  residual correction.
- Compact14: CT3 with local-five expert and R3 correction, without Planet
  or the geological gate.
- B1: inner-cross-fitted, weight-constrained combination of CT3/Reference26;
  sensitivity only, not a selected champion.

The script imports the peer's **original** refit_fold and predict_saved
functions from a clean local checkout. It does not copy the hundreds of
research scripts into GeoC, replace peer's models with a four-feature proxy,
load a model previously fit on all 138 labels, or use its saved 81 OOF folds
as though they were 138 LOOCV folds.

**Why a private bundle is required:** the fork deliberately excludes
data/processed/current_champions/v1. This bundle includes 26 engineered
predictors for the 138 training IDs, group definitions and exactly aligned
geographic distances. GeoC's C4 features alone do not reproduce all 26
peer features (notably geology and R3). The peer README explicitly states
that a Git clone alone cannot reproduce training. Request an authorized
bundle from the teammate (her scripts/521_current_champions.py export,
executed where her historical inputs are available). The LOOCV runner does
not need pre-fitted peer joblib models; its bundle loader skips those hashes.
Do **not** publish bundle files or individual labeled forecasts.

## Paired protocol

For each of the **same 138 original labeled parcel IDs**:

1. GeoC reconstructs the frozen Local04D C4 representation using target-free
   X from all 197 parcels and exact normalized geographic distances.
2. Peer input IDs, reported yields, groups and distance dimensions are
   checked against GeoC. No positional ID assumptions across repos.
3. One parcel is held out. The peer's held-out **y is replaced by NaN
   before calling any peer training method**; the other 137 labels are
   available. GeoC fits Local04D on those same 137 labels.
4. Peer refit_fold performs the original multi-level training and
   weight/penalty selection with **four deterministic inner folds on the
   137 training labels**, never loading its historical fit or reusing
   the original 65/16 OOF predictions.
5. Frozen outer predictions are saved together for GeoC Local04D, CT3,
   Reference26, Compact14 and B1; only then is the withheld observed
   yield used for scoring.

This estimates performance of a **new 4-fold inner-selection version**
of the peer's recipe under 137-label LOOCV. It does not claim to exactly
replicate her original 65-fold inner protocol or to certify that the best
configuration under the 97/41 task is best under LOOCV.

All folds share the same metric convention: pooled RMSE/MAE/R2 over 138
distinct held-out parcels. Differences are paired by ID, including
per-parcel win fractions and percentile parcel-bootstrap intervals for
delta RMSE (peer minus GeoC); the latter **do not** correct for spatial
correlation. Spatial holdout protocols remain a separate, essential
robustness check.

## Running on the workstation (PowerShell)

From your existing GeoC checkout (read-only against main; use branch/PR):

~~~powershell
cd C:\Users\175199\GeoC
conda activate geocebada
git fetch origin
git switch experiment/nuria-champions-loocv
git pull --ff-only origin experiment/nuria-champions-loocv

# Peer checkout lives outside GeoC, NOT copied into it:
if (!(Test-Path ..\todoseboso)) {
    git clone https://github.com/heri-espino/todoseboso.git ..\todoseboso
}

# The following MUST already exist locally, provided securely by the teammate:
# ..\todoseboso\data\processed\current_champions\v1\manifest.json
# ..\todoseboso\data\processed\current_champions\v1\train138.parquet
# ..\todoseboso\data\processed\current_champions\v1\groups.csv
# ..\todoseboso\data\processed\current_champions\v1\distance.npy
# ..\todoseboso\data\processed\current_champions\v1\expected_oof.parquet
# ..\todoseboso\data\processed\current_champions\v1\splits.json

# Ensure peer runtime dependencies are installed in this environment.
# Peer recipe originally used scikit-learn 1.7.2, pyarrow, threadpoolctl, etc.

python tools/run_peer_champions_loo.py --check
python tools/run_peer_champions_loo.py --smoke

# Heavy, resumable; writes a completed result after each parcel:
python tools/run_peer_champions_loo.py --run

# Full 138-fold score; refuses incomplete coverage:
python tools/run_peer_champions_loo.py --summary
~~~

To split work across multiple sessions without running concurrently, use
--run --from-fold 1 --through-fold 20 and then 21..138. Rerunning the same
command resumes only incomplete folds. A changed peer commit, changed bundle
manifest, changed GeoC data/code or changed validation source requires a
**new --output** directory; stored SHA256 provenance prevents mixed runs.

**Runtime:** each outer fold may invoke deeply nested training and several
local experts. 138 cold refits can be expensive. Smoke first. No benchmark
duration or performance is claimed in GitHub CI.

## Output and safety

Ignored local directory: reports/peer_champions_loo/. Each fold is an atomic
fold_001.csv..fold_138.csv (one ID with observed and five predictions).
The runner writes loo_predictions.csv, loo_summary.csv,
loo_paired_differences.csv and run_spec.json only for complete coverage;
smoke writes its own non-promoting subdirectory.

- C4 PCA and geographic predictions must match the GeoC original definition.
- All 138 peer labels must equal the official GeoC labels before fitting.
- Never read or use the 59 hidden yields.
- Do not fit 26-feature supervised preprocessing using all 138 targets.
- Do not compare the 97/41 table directly with LOOCV.
- No winner is promoted on these previously analyzed 138 outcomes.
- Any actual replacement of 59 final forecasts requires separately
  constructed 197-row peer inputs and appropriate confirmation.

**Next execution blocker:** obtain the trusted, private peer feature bundle.
No data or model files have been merged into GeoC by this PR.
