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

**Your local input location:** place the downloaded private files in
`GeoC/data_n/`. The runner automatically looks for a frozen manifest and
accompanying files in the following layouts (all relative to the GeoC root):

- `data_n/manifest.json` (bundle files directly in `data_n/`)
- `data_n/v1/manifest.json`
- `data_n/current_champions/v1/manifest.json`
- `data_n/processed/current_champions/v1/manifest.json`
- `data_n/data/processed/current_champions/v1/manifest.json`

It prefers `data_n/` over a bundle in an external peer checkout. If you
extracted the downloaded folder elsewhere, pass `--bundle <directory>`;
this accepts either the direct bundle or one of the parent layouts above.
The folder `data_n/` is fully ignored by Git, as are local per-parcel
`reports/peer_champions_loo/` outputs. **No private data should be staged.**
If only raw inputs were downloaded, a valid `manifest.json`,
`train138.parquet`, `groups.csv`, `distance.npy`, `splits.json`, and
`expected_oof.parquet` must first be exported using the peer project's
own supported process. Merely renaming raw inputs to `data_n/` does not
construct the 26-feature champion bundle.

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
From your existing GeoC checkout (use the experiment branch, not `main`):

~~~powershell
cd C:\Users\175199\GeoC
conda activate geocebada
git fetch origin
git switch experiment/nuria-champions-loocv
git pull --ff-only origin experiment/nuria-champions-loocv

# Private data already lives under GeoC/data_n/ and is ignored.
git check-ignore data_n/
git status --short

# Only the modeling source code is needed from the fork.
# Clone beside GeoC if it is not already there (do NOT move the private files):
if (!(Test-Path ..\todoseboso\src\current_champions.py)) {
    git clone https://github.com/heri-espino/todoseboso.git ..\todoseboso
}

# Parquet bundle loading requires pyarrow (or another parquet engine).
python -m pip install pyarrow threadpoolctl

# Validate peer bundle hashes, IDs, yields, 26 features, and GeoC C4 parity.
python tools/run_peer_champions_loo.py --check

# Fit one held-out parcel to catch runtime errors before the long run.
python tools/run_peer_champions_loo.py --smoke

# Full run: 138 independent held-out parcels. Resumes completed folds.
python tools/run_peer_champions_loo.py --run
python tools/run_peer_champions_loo.py --summary
~~~

The default `--bundle` resolves the first valid champion export inside
`GeoC/data_n/`, so no path flag is needed for the standard layouts.
If the actual bundle is under a different extracted folder, use
`--bundle <path>` with `--check`, `--smoke`, and `--run`.
A successful `--check` is required before training; it is not proof that
all 138 model refits will succeed. The peer's model-selection code is
computationally expensive and has not been executed in this change.

If `git ls-files data_n/` returns anything, these files were tracked
*before* the ignore rule; remove them from the index without deleting local
copies: `git rm -r --cached -- data_n/`. Inspect the staged changes
before committing; never push private datasets.

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

**Next execution step:** run `--check` against your local `data_n/`
copy and correct any missing bundle files. The workstation's private
folder cannot be inspected from this GitHub branch. No refits or private
data files have been committed or merged by this PR.
