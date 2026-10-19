# Three-regime synthetic Gaussian-bridge experiment

**2026-10-19 — implemented, awaiting a real VM run.** Competition-only, exploratory; no new Reference26 refits, official 59-target change, or publication of individual yields.

## Hypothesis and architecture

All 13 checked true 138-fold LOOCV model configurations have the same two largest residuals (27–28% of their total squared error). Two original difficult parcels might represent a rare agronomic regime rather than incorrect labels. Test two small synthetic Ridge specialists, leaving the current Reference26 mean-imputation model untouched for ordinary parcels.

The two anchor observations are chosen post hoc based on previous LOOCV errors; hence any comparison on the same 138 labels is exploratory, not independent confirmation.

For anchors with real standardized feature vectors z_a, z_b and reported yields y_a, y_b, generate feature paths at evenly spaced t in [0,1]:

```math
z(t)=(1-t)z_a+t z_b + 2 sigma L B(t),    B(t)=W(t)-t W(1).
```

Here W is d-dimensional Brownian motion and L L^T is an X-only covariance estimate (70% empirical covariance plus 30% identity). Its perturbation covariance is

```math
Cov(epsilon(s),epsilon(t))=4 sigma^2 (min(s,t)-s t) (L L^T).
```

Noise is exactly zero at the original endpoints, maximized at t=1/2, and correlated along each bridge path. The artificial response is **not a measured yield**:

```math
y_tilde(t)=(1-t)y_a+t y_b.
```

Train specialist 1 on 0<=t<=1/2 and specialist 2 on 1/2<=t<=1, with separate regularized Ridge models (16 paths, 17 t steps by default).

A feature-only gate routes to a specialist only if the new parcel is nearer to the synthetic cloud than to ordinary real training samples (adjustable distance ratio), is within a radius of the synthetic cloud measured relative to training nearest-neighbor spacing, and lies near the anchor segment. The correct specialist depends on projected t. Otherwise, original Reference26 predictions are unchanged exactly. A declared blending weight allows a partial correction rather than full replacement.

This is an interpolation assumption, not a claim that synthetic rows are independent crop yields, or that a path in model-feature space is physically plausible. No hidden y is accessed.

## Leakage and special case of only two labels

For every original heldout parcel, reuse its genuine saved Reference26 outer prediction. Generate augmented training examples only using anchor labels that remain in the 137 allowed outer-training labels. **When either anchor is held out, disable both specialists and use the untouched baseline.** This is necessary: otherwise the heldout yield would leak into the synthetic training set. Consequently, this diagnostic cannot claim unbiased recovery of either of the two anchors; it evaluates whether similar *other* parcels and the 59 fixed targets can benefit.

The selected anchor IDs and grid were inspired by these same 138 historical folds; model selection and performance are post-hoc even with proper label masking. Previous Checkpoint 06 and 09 specialists/robustification failed to improve their respective validation protocols; this is a separate geometric synthetic-data experiment, not a proven improvement.

## How to run

From GeoC on the VM, with the original private champion bundle and sibling Todoseboso deployment features:

```powershell
cd C:\Users\175199\GeoC
conda activate geocebada
git fetch origin
git switch experiment/synthetic-bridge-three-regime

python -m pytest tests/test_synthetic_bridge_three_regime.py -q
python tools/run_synthetic_bridge_three_regime.py --check
python tools/run_synthetic_bridge_three_regime.py --smoke
python tools/run_synthetic_bridge_three_regime.py --run
python tools/run_synthetic_bridge_three_regime.py --summary
```

Grid: sigma=0/0.05/0.15/0.30, gate ratio=0.6/1.0/1.4, blend=0.5/1.0, radius=1.5, Ridge alpha=5, fixed seed 20261019. The full 138-fold reference model LOOCV is read, not rerun. Only the lightweight synthetic experts are refitted per heldout fold.

Private local outputs under **reports/synthetic_bridge_three_regime/** (Git ignored):
- comparison_aggregate.csv: 138 RMSE; separately 136 ordinary parcel RMSE, MAE, gate changes and worsening count; model parameters
- parcel_predictions_private.csv: heldout parcel IDs, actual yields and individual new forecasts, DO NOT commit
- run_spec.json: experiment hashes and method warnings.

Optional comparison on 59 target parcels, only after the original 138-label Reference26 final fit has written a validated private 59-row prediction file:

```powershell
python tools/run_synthetic_bridge_three_regime.py --predict59 --sigma 0.15 --gate-ratio 1.0 --blend 1.0
```

This creates private **predictions59_private.csv**, refuses to replace an existing output, and does not publish anything or change the frozen final model. Inspect whether any of the 59 actually fall into the synthetic region. Zero routed predictions is a meaningful possible result.

## Decision

Compare against original Reference26 (138 RMSE ~0.468138 t/ha). Prioritize RMSE on 136 other parcels, fraction unaffected, severity of deterioration, and number of 59 targets entering the synthetic gate. Even a post-hoc improvement on these 138 folds is not independent evidence that the new model improves hidden-target accuracy. No promotion before review of the experiment and a separate validation strategy.


## Private anchor selection and diagnostic visualization

The real endpoint identifiers are **not embedded in the final Python
source**. The runner reads the private, gitignored file
`reports/loocv_hard_parcels/diagnostic_manifest_private.json` from
the completed shared hard-case analysis (the key
`two_highest_consensus_private`). If that file is absent, run
`python tools/analyze_loocv_hard_parcels.py --run` using the
previous analysis branch or supply `--anchor-file` pointing to a
compatible private manifest. Identifiers and yields are never included
in the public aggregate comparison tables.

Visualize the real 197 X parcels plus both synthetic clouds in a
two-dimensional PCA projection, alongside the theoretical noise
variance curve:

```powershell
python tools/plot_synthetic_bridge.py --sigma 0.15
```

Private outputs (both gitignored):
- `reports/synthetic_bridge_three_regime/bridge_cloud_pca_private.png`
- `reports/synthetic_bridge_three_regime/bridge_expected_variance.png`

A PCA view cannot validate that mixtures of soil/climate/phenology
features correspond to physically possible fields; it helps detect
obviously misplaced synthetic clouds.
