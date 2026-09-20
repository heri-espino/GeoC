# Checkpoint 04 — Transductive Competition Modeling

**Status:** OPEN  
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

## 04A — Target-set topology and similarity — IMPLEMENTED, RUN PENDING

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

## 04B — Transductive pseudo-competition validation

Create repeated pseudo-target splits from the 138 labels. For each split, pseudo-target X is visible throughout X-only transductive learning while pseudo-target y is hidden until scoring.

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

## 04D — Local, graph and domain-adaptation models

Evaluate target-specific kNN/local regression, mixed spatial + phenological distance metrics, graph-based regression on all 197 parcels, local/global mixtures of experts, covariate-shift / importance weighting, target-aware training weights derived from X only and uncertainty/support-dependent blending.

Model choice and blending must be justified by pseudo-competition RMSE, not subjective visual inspection of the 59 final predictions.

## 04E — External-evidence enrichment

Highest priority is a fresh SIAP audit focused on the closest valid contemporaneous proxy: `Cebada grano` + relevant cycle + `Temporal` + correct municipality.

Also evaluate CHIRPS after QC, WaPOR, SoilGrids, terrain and richer public daily weather/stress data where a concrete non-redundant hypothesis exists.

Do not create thousands of external features without an ablation plan.

## 04F — Final transductive ensemble

Only after 04B is stable:

- assemble pseudo-test predictions from surviving global/local/graph/domain-adaptation models;
- evaluate blending/stacking with the same transductive validation contract;
- inspect target-specific support and uncertainty;
- fit the final rule using all 138 labels and all 197 X;
- generate exactly one canonical table of 59 reconstructed yields keyed by `ID_POLIGONO`.

## Completion criteria

Checkpoint 04 closes only when target-set topology is characterized, transductive pseudo-competition validation is reproducible, every final component has pseudo-test evidence, external evidence and SIAP semantics are documented, final fit uses all 138 observed labels and may use all 197 X as allowed, exactly 59 final yields are generated with IDs preserved, final model/table provenance is frozen, and project handoffs/history/AI-use documentation are synchronized.
