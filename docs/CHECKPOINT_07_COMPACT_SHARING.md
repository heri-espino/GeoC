# Checkpoint 07 — Share small, reproducible derived snapshots

**Status:** optional local export tool implemented; snapshots not yet
generated on the Windows workstation or pushed by the workstation owner.

The raw archive is about 180 GiB and should remain on the workstation.
We can nevertheless share the small, reproducible X-only data products that
allow collaborators or another machine to continue the modeling experiment.

## What can be published

| Artifact | Source | Repository destination | Git storage |
|---|---|---|---|
| HLS parcel-date panel | `data/processed/checkpoint_07/hls_parcel_observations.csv` | `data/processed/checkpoint_07_compact/hls_parcel_observations.csv.gz` | LFS |
| SMAP parcel-daily panel | `data/processed/checkpoint_07/smap_parcel_daily.csv` | `data/processed/checkpoint_07_compact/smap_parcel_daily.csv.gz` | LFS |
| Full expanded X table, optional | `reports/checkpoint_07/highdim/expanded_parcel_features.csv` | `data/processed/checkpoint_07_compact/expanded_parcel_features.csv.gz` | LFS |
| Manifest with SHA-256 checksums, no absolute paths | inferred from the above | `reports/checkpoint_07/shared/checkpoint07_portable_manifest.json` | Git |
| Processing QA summary | HLS / SMAP processing JSON | `reports/checkpoint_07/shared/checkpoint07_quality_summary.json` | Git |
| Small development smoke benchmark (if present) | `reports/checkpoint_07/highdim/smoke/benchmark_summary.csv` | `reports/checkpoint_07/shared/checkpoint07_smoke_benchmark.csv` | Git |

These are distinct from the large ignored local sources. Git LFS already
tracks all files under `data/**` according to `.gitattributes`, so no
`git lfs track` invocation or extra attribute patterns are needed.

**Excluded:** full global SMAP HDF5s, HLS/Landsat/Sentinel GeoTIFFs, Prithvi
network weights and spatial chips, source polygons and all supervised yield
labels. Any source with `RENDIMIENTO_T_HA`, `CONJUNTO` or a `siap_2025`
field is rejected. Beware: parcel-level X metadata still contain stable IDs,
source scene IDs and coarse location-derived data. Confirm your data-sharing
permissions before pushing to a publicly visible GitHub repository.

## Run once to share HLS and SMAP compact panels

From the cloned repo in Windows PowerShell:

```powershell
git pull --ff-only
conda activate geocebada
git lfs install

# Confirm source tables and inspect sizes; no files created.
python tools\\publish_checkpoint_07_light.py --check

# Compress and produce SHA-256 manifest and sanitized QA/smoke reports.
python tools\\publish_checkpoint_07_light.py --build

# Re-read the compressed tables and verify both SHA-256 digests.
python tools\\publish_checkpoint_07_light.py --verify

# Review EXACTLY what Git would see; do not add raw sources.
git status --short
git check-attr filter -- data/processed/checkpoint_07_compact/hls_parcel_observations.csv.gz
```

Default max is **25 MiB per compressed file and 60 MiB total**. The script
fails before publishing the new set if it exceeds the caps. Compression is
streaming (no full 84k-row daily panel held in memory). Original inputs are
neither modified nor deleted. Gzip timestamps and embedded filenames are
normalized to make copies reproducible; the manifest documents hashes.

The command exports both HLS and SMAP, regardless of whether the full
high-dimensional model run has completed. The **smoke** benchmark is marked
as one split; it is not evidence that the MLP outperforms Local04D.

## Explicitly stage and push the allowlisted outputs

After verifying your sharing permissions and inspecting the manifest:

```powershell
git add -- data/processed/checkpoint_07_compact/hls_parcel_observations.csv.gz
git add -- data/processed/checkpoint_07_compact/smap_parcel_daily.csv.gz
git add -- reports/checkpoint_07/shared/checkpoint07_portable_manifest.json
git add -- reports/checkpoint_07/shared/checkpoint07_quality_summary.json
git add -- reports/checkpoint_07/shared/checkpoint07_smoke_benchmark.csv

git diff --cached --stat
git lfs ls-files
git commit -m "data: share compact HLS and SMAP parcel panels with QC"
git push origin main
```

The last report CSV exists when the 07B smoke test has already completed.
If there is no smoke benchmark yet, omit that corresponding `git add`
command. **Never use `git add -A` for this sharing operation.** Do not
stage or force-add `data/raw/checkpoint_07/`,
`data/processed/checkpoint_07/`, `models/checkpoint_07/` or
`reports/checkpoint_07/highdim/` directly.

To check that Git LFS is active for the archived X data, `git check-attr
filter` should report `lfs`; `git lfs ls-files` should list these
archives after staging. Once pushed, another workstation can use normal
`git lfs pull` to retrieve the small snapshots.

## Later: include the complete high-dimensional X table

The log from October 10 showed **9,195 features in a smoke run** using only
60 interaction products and one held-out development split. Do **not**
publish that smoke table as a final feature table.

Once the full feature builder finishes:

```powershell
python tools\\run_checkpoint_07_highdim.py --stage build --require-hls --require-smap

python tools\\publish_checkpoint_07_light.py --check --include-highdim
python tools\\publish_checkpoint_07_light.py --build --include-highdim
python tools\\publish_checkpoint_07_light.py --verify

git add -- data/processed/checkpoint_07_compact/
git add -- reports/checkpoint_07/shared/
git diff --cached --stat
git lfs ls-files
git commit -m "data: add full high-dimensional X-only checkpoint07 table"
git push origin main
```

Do not rerun the full feature builder if its complete output already
exists. Use `--include-highdim` only after a successful non-smoke build.
If the compressed feature table exceeds the default per-file cap, keep it
local or deliberately specify larger `--max-file-mib` and
`--max-total-mib` with clear repository-size review.

## Observed Checkpoint 07 results (October 10, 2026)

- 197 HLS scenes -> **29,592** parcel-date observations and 197 parcels
  with four quality-selected Prithvi chip frames.
- 214 SMAP daily HDF5 files -> **84,316** parcel-overpass rows,
  **29,006** recommended-quality observations (34.4%).
- Reported **15** unique SMAP grid cells for AM and **15** for PM;
  many parcels share a cell. The AM file used coordinate datasets,
  while the PM used the EPSG:6933 fixed global grid. Compare spatial
  alignment before claiming fine-scale moisture information.
- 07B smoke: **9,195** expanded features, one 41-parcel held-out split,
  four candidates; lowest RMSE **0.615971 t/ha** (PCA 0.80 → small MLP).
  This is not a final RMSE comparison against the incumbent.
- The MLP smoke raised a scikit-learn maximum-iteration warning at 120
  iterations; that limit is intentionally smaller than in the full run.

Always keep raw source copies, acquisition manifests and processing code,
even after uploading a compact snapshot.
