# Checkpoint 04 — Transductive Competition Modeling

**Status:** OPEN — 04A COMPLETE, 04B COMPLETE, 04D.1 COMPLETE, 04E.1 IMPLEMENTED / RUN PENDING  
**Opened:** 2026-09-19

Checkpoint 04 begins after the closure of Checkpoint 03 as the **First Modeling Delivery**. Read `docs/TRANSDUCTIVE_OBJECTIVE.md` before implementing this checkpoint.

## Objective

Reconstruct the 59 fixed hidden FIRA parcel yields using all rule-permitted observable information from the complete 197-parcel universe.

This checkpoint intentionally changes the modeling doctrine from conventional inductive generalization to fixed-target transductive inference.

## Frozen baseline from Checkpoint 03

```text
E123 equal top 3:      state RMSE 0.5121, municipality RMSE 0.7481
E13 PLS + CatBoost:    state RMSE 0.5082, municipality RMSE 0.7488
```

These are baseline development scores. Historical 03 configs/reports must not be modified.

## 04A — Target-set topology and similarity — COMPLETE

Build a deterministic diagnostic layer for the relationship between the 59 targets and the 138 labeled parcels.

Required outputs should include, where feasible:

- geographic nearest-neighbor distances;
- same-municipality/state coverage;
- feature-space nearest-neighbor distances;
- agronomic-representation similarity;
- temporal-curve Pearson/Spearman correlation;
- cross-correlation / best lag;
- DTW or another justified curve distance;
- train-vs-target adversarial AUC;
- local density/support and extrapolation flags;
- Moran's I or related spatial autocorrelation diagnostics for observed y;
- spatial autocorrelation of baseline OOF residuals;
- empirical curves linking X similarity to absolute yield difference.

The 59 parcels should receive a target-specific predictability/support profile rather than be treated as exchangeable.

### 04A implementation

The first runnable 04A implementation is now committed:

```text
configs/checkpoint04a.yaml
src/geocebada/evaluation/checkpoint04a.py
src/geocebada/visualization/checkpoint04a.py
tools/run_checkpoint_04a.py
tests/test_checkpoint04a.py
```

Run it from the repository root:

```powershell
python tools\run_checkpoint_04a.py
```

It writes the complete result set under `reports/checkpoint_04a/`: target and leave-one-out
neighbor tables, temporal-pair diagnostics, all 138-choose-2 labeled-pair comparisons,
adversarial shift scores, Moran statistics, the 59-row support profile, a Markdown/JSON report,
summary plots and one Planet-NDVI best-neighbor panel for each target.

The temporal first pass uses the canonical April-October 2025 monthly trajectories already
derived from BASIC/PRO. It preserves sensor identity and adds Pearson/Spearman similarity,
best-lag correlation and normalized DTW. Raw capture-level alignment can be added after these
results show whether additional temporal resolution is worth the complexity.

The workstation run is complete and committed. Canonical interpretation: `docs/CHECKPOINT_04A_FINDINGS.md`.

Key evidence: adversarial AUC 0.4892; geographic distance vs absolute yield difference Spearman rho 0.4882; observed-y Moran's I 0.6797 (p=0.001); municipality-grouped E123 residual Moran's I 0.5483 (p=0.001); 5 high-support, 13 extrapolation and 41 intermediate/mixed targets.


## 04B — Transductive pseudo-competition validation — COMPLETE

Create repeated pseudo-target splits from the 138 labels. For each split, pseudo-target X is visible throughout X-only transductive learning while pseudo-target y is hidden until scoring.


### 04B implementation

The runnable validation layer is now committed:

```text
configs/checkpoint04b.yaml
src/geocebada/evaluation/checkpoint04b.py
src/geocebada/visualization/checkpoint04b.py
tools/run_checkpoint_04b.py
tests/test_checkpoint04b.py
```

Run from the repository root:

```powershell
python tools\run_checkpoint_04b.py
```

The primary protocol creates 16 repeated X-only target-matched pseudo-competitions with 41
pseudo-targets each, approximating the real 59/197 target fraction. A secondary state-matched
random protocol and both frozen Checkpoint 02 state/municipality folds remain as stress tests.

The method suite is intentionally lightweight enough to establish the validation doctrine
before 04C GPU tuning. It includes global/state/municipality baselines, geographic kNN,
agronomic-space kNN, mixed geographic+agronomic(+temporal) kNN, transductive-PCA Ridge,
C0 PLS, query-specific local Ridge, graph-Laplacian regression and fixed global/local blends.

All split selection is X-only. A separate dynamic X-only support score is recomputed inside
every pseudo split, and support-tier winners are used to propose method routing for the actual
59 targets without inspecting hidden y.

At least two split families should be retained:

1. target-matched pseudo-competition splits designed to resemble the actual 59 X distribution;
2. legacy spatial/state stress tests for robustness.

Every 04C–04F model must produce reproducible pseudo-target predictions and RMSE.

## 04C — Higher-compute global models

Revisit strong global families with a larger compute budget. Candidate work includes:

- CatBoost with larger iteration budget, early stopping, depth/regularization/column-sampling search and multiple seeds;
- PLS component search;
- Ridge/ElasticNet variants;
- ExtraTrees/boosting where transductive validation supports them;
- carefully justified feature subsets and representations.

GPU should be used where supported.

## 04D.1 — Local/graph refinement — COMPLETE

04D.1 operationalizes the strongest 04B evidence instead of reopening a generic model search.
It reuses the exact committed 04B pseudo-target memberships and exhaustively refines:

- direct graph-Laplacian regression over five geography/agronomy topologies;
- graph correction of cross-fitted PLS4 residuals;
- local Ridge over C1/C4 embeddings, three distance topologies, neighborhood sizes,
  regularization strengths and inverse-distance powers;
- fixed GeoKNN10 and PLS4 anchors.

Primary search uses the 16 target-matched pseudo-competition splits. Hyperparameter selection
is evaluated leave-one-pseudo-split-out, and support-tier routing is also validated on a split
excluded from route fitting. Only finalists are then evaluated on state-random and frozen
state/municipality stress splits.

Run:

```powershell
python tools\run_checkpoint_04d1.py
```

Outputs go to `reports/checkpoint_04d1/`. Candidate predictions for the real 59 are retained
for disagreement analysis and later 04F blending, but are explicitly not the final submission.

## 04D — Local, graph and domain-adaptation models

Evaluate target-specific kNN/local regression, mixed spatial + phenological distance metrics, graph-based regression on all 197 parcels, local/global mixtures of experts, covariate-shift / importance weighting, target-aware training weights derived from X only and uncertainty/support-dependent blending.

Model choice and blending must be justified by pseudo-competition RMSE, not subjective visual inspection of the 59 final predictions.

## 04D.1 completed evidence

Canonical interpretation: `docs/CHECKPOINT_04D1_FINDINGS.md`.

```text
best target-matched Local Ridge                  RMSE 0.4878
LOSO model selection                            RMSE 0.4884
LOSO support-tier routing                       RMSE 0.4858
GraphDirect GeoAgro25 k6 lambda8 target-matched RMSE 0.4949
GraphDirect GeoAgro25 k6 lambda8 grouped stress RMSE 0.5279
```

The local optimum is stable across nearby k/alpha settings. The direct graph is
less competitive on the primary target-matched mean but substantially more
robust under municipality-grouped stress.

## 04E — External-evidence enrichment

Highest priority is a fresh SIAP audit focused on the closest valid contemporaneous proxy: `Cebada grano` + relevant cycle + `Temporal` + correct municipality.

Also evaluate CHIRPS after QC, WaPOR, SoilGrids, terrain and richer public daily weather/stress data where a concrete non-redundant hypothesis exists.

Do not create thousands of external features without an ablation plan.

### 04E.1 implementation — SIAP localization

The first external-evidence stage is implemented and deliberately narrow. It
audits raw SIAP records without collapsing cycle or modality, constructs an
explicit specificity hierarchy, and validates external-prior methods on the
frozen 04B splits.

Primary scope:

```text
Cebada grano + Primavera-Verano + Temporal + exact CVEGEO + 2025
```

Broader 2025 grain-barley scopes are retained only as named fallbacks. The
runner compares direct SIAP, affine calibration, SIAP-anchored local residuals,
SIAP-anchored graph residuals and fixed blends against the frozen 04D.1 local
and graph baselines.

Run:

```powershell
python tools\run_checkpoint_04e1.py
```

Outputs go to `reports/checkpoint_04e1/`.

## 04F — Final transductive ensemble

Only after 04B is stable:

- assemble pseudo-test predictions from surviving global/local/graph/domain-adaptation models;
- evaluate blending/stacking with the same transductive validation contract;
- inspect target-specific support and uncertainty;
- fit the final rule using all 138 labels and all 197 X;
- generate exactly one canonical table of 59 reconstructed yields keyed by `ID_POLIGONO`.

## Completion criteria

Checkpoint 04 closes only when target-set topology is characterized, transductive pseudo-competition validation is reproducible, every final component has pseudo-test evidence, external evidence and SIAP semantics are documented, final fit uses all 138 observed labels and may use all 197 X as allowed, exactly 59 final yields are generated with IDs preserved, final model/table provenance is frozen, and project handoffs/history/AI-use documentation are synchronized.
