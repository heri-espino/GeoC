# Geology-gate sensitivity: KMeans versus DBSCAN (pre-LOOCV)

**Status:** code and synthetic QA committed on
`experiment/nuria-champions-loocv`, **not executed on the local private
dataset**. No superiority or new yield RMSE is claimed.

## What the 14/26 comparison actually means

The original Reference26 uses ten *geological composition* features for
KMeans 4/8 clusters, not parcel-coordinate clustering. The separate
24-nearest-neighbor local Ridge **does** use geographical coordinates.
The baseline prediction is

\[
\widehat y_{26}(x)=C(x)+g_{\rm geo}(x)[E(x)-C(x)]+s\,m(x)\delta_{R3}(x).
\]

\(C\) is global climatic C-T3; \(E\) is the local seven-feature expert.
The geology gate \(g\) is a shrinkage-regularized convex weight. Compact14
has a different **global** mixture of C-T3 and its five-feature local expert:
**it does not use the geological KMeans gate**. The original B1 is a
cross-fitted mixture of C-T3 and Reference26. Read
[peer Reference26's block-level method](https://github.com/heri-espino/todoseboso/blob/main/documentation/models/Reference26/blocks.md)
and `src/base_distribution_correction.py`.

This experiment replaces **only the gate of Reference26**, and
re-fits every corresponding nested expert and R3 correction using the
original source code from the peer clone. The parent Reference26
and DBSCAN26 remain directly comparable. CT3, Compact14 and the
frozen GeoC Local04D remain controls. B1 is re-derived using each
Reference26 version; B1_DBSCAN is *not* the original B1.

## DBSCAN setup

- Input: exactly the same ten geological-composition features, with the
  **training-only** source `RoleSpace` transform, never parcel yield.
  No geographical coordinate or altitude is added to this clustering.
- Candidate `min_samples`: 3 and 5.
- Candidate relative radii: 0.75, 1.25, 2 and 3 times the training median
  k-neighbor distance (k = min_samples - 1 other samples).
- Epsilon therefore has units of transformed geological-feature distance,
  **not** meters, degrees or kilometers.
- DBSCAN may create no clusters. In this case, it falls back to the global
  mixing weight; it does not force a geological membership.
- DBSCAN does not have a natural `predict` method. A held-out parcel is
  assigned to the cluster of the *nearest training core point* if and only
  if its distance is within the fitted epsilon. Otherwise it uses the
  global mixing weight. Border-point assignments are deterministic.
- For each density cluster, the same shrinkage penalty as the original
  KMeans gate, `20 × mean[(E_OOF - C_OOF)^2]`, regularizes its weight.
- All **eight** (2 × 4) DBSCAN candidates are scored via inner
  cross-fitting within the 137 labeled outer-training parcels. The
  held-out yield is masked before **any** expert or gate fitting; its
  value is used only to compare predictions afterwards.
- This changes a single choice within Reference26; it does **not**
  change the 24-nearest-neighbor geography or PCA/C4 baseline.

DBSCAN candidate selection is based on inner MSE, with deterministic
ties resolved by smaller min_samples and radius. DBSCAN noise fraction,
cluster count, epsilon and all candidate scores for each outer fit
are saved privately to `gate_001.json`, etc.

## Commands on the Windows workstation

The private input location is `GeoC/data_n/`. The current champion bundle
under this tree must contain `manifest.json`, `train138.parquet`,
`groups.csv`, `distance.npy`, `splits.json` and `expected_oof.parquet`.
The fork's source code is still required in the sibling `todoseboso`
checkout. The code rejects an incomplete bundle.

~~~powershell
cd C:\Users\175199\GeoC
conda activate geocebada
git fetch origin
git switch experiment/nuria-champions-loocv
git pull --ff-only origin experiment/nuria-champions-loocv

if (!(Test-Path ..\todoseboso\src\current_champions.py)) {
    git clone https://github.com/heri-espino/todoseboso.git ..\todoseboso
}
python -m pip install pyarrow threadpoolctl

# Only inspect geological clusters: KMeans 4/8 and DBSCAN 8 density settings.
# No supervised fitting or use of parcel yield in the cluster diagnostics.
python tools/run_peer_geology_dbscan.py --check

# One target parcel excluded, ALL controls and both Reference26 gates refitted.
# This may be expensive because each pipeline is nested and cold-refitted.
python tools/run_peer_geology_dbscan.py --smoke

# Optional preliminary comparison on three distinct held-out parcels.
# NOT sufficient to claim general superiority or to select a champion.
python tools/run_peer_geology_dbscan.py --run --folds 3
~~~

Output under ignored `reports/peer_geology_dbscan/`:

- `geometry/geological_cluster_diagnostics.csv`: numbers/sizes of
  KMeans and DBSCAN clusters, DBSCAN noise fractions and actual radii.
- `smoke/fold_001.csv` and `smoke/preliminary_errors.csv`:
  all controls and both gates compared on the same one held-out parcel.
- `fold_001.csv`.., `gate_001.json`, `preliminary_predictions.csv`,
  `preliminary_errors.csv` for a 3-fold exploratory run. Cached folds
  are reuse-protected by a provenance manifest and checked identities.
- No underlying private data, individual yields or fold errors are
  published to GitHub.

**Selection warning:** three folds are only a functional diagnostic. Their
RMSE has extreme sampling uncertainty and must not be compared to the
138-fold Local04D result or to peer 97/41 validation. A larger *predeclared*
paired evaluation is needed to establish whether this new gate improves
predictive error. Never select a DBSCAN radius using outer-heldout errors.
The existing 138-fold benchmark remains available after this ablation;
do not promote a model from the smoke.

**Secondary idea (not implemented):** run coordinate-based DBSCAN using
haversine-distance radii in kilometers and use clusters only as an
*explicitly separate spatial-neighbor constraint*. This is a different
scientific experiment with different noise handling and should not
be mixed with the geological gate comparison.
