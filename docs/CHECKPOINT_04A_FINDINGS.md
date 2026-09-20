# Checkpoint 04A Findings — Fixed-target topology and support

**Status:** COMPLETE  
**Run committed:** 2026-09-20  
**Result commit:** `77f5b90a3288dc7365902ed918baa6b132d3ec52`

The canonical generated report is
`reports/checkpoint_04a/checkpoint_04a_report.md`. This note records the interpretation that
should guide Checkpoint 04B onward.

## Main result

The 59 fixed target parcels are not globally separable from the 138 labeled parcels, but the
labeled yields have strong spatial structure and a nontrivial subset of the 59 targets has weak
local support. The most useful similarity signal in 04A is geographic proximity, followed by
the compact agronomic representation. Monthly temporal-shape similarity by itself is much
weaker.

This shifts the next phase toward **spatially aware, local and target-specific transductive
validation**, rather than toward a generic domain-adaptation correction or a temporal-neighbor
rule used in isolation.

## Global train-target shift

The adversarial classifier distinguishing labeled from target parcels produced:

```text
cross-validated AUC = 0.4892
```

This is effectively no global discrimination in the tested 20-PC deterministic representation.
The 59 targets therefore do not look like a separate population in aggregate.

Individual feature shifts still exist. The largest standardized mean differences are roughly
0.5 in magnitude and are dominated by historical Sentinel-2 seasonal variables and derived
shape features. These differences are useful for parcel-level support diagnostics but do not
constitute evidence of a large global covariate shift.

Implication: global importance-weighting/domain adaptation should remain a candidate in 04D,
but it should not be assumed to be the main source of improvement.

## Similarity versus observed yield difference

Using all `138 choose 2 = 9,453` labeled-labeled parcel pairs:

| Metric | Spearman rho with absolute yield difference | Interpretation |
|---|---:|---|
| geographic distance | 0.4882 | strongest observed similarity-transfer signal |
| C1 agronomic PCA distance | 0.2200 | useful secondary structure |
| C4 all-deterministic PCA distance | 0.1833 | useful but diluted by high-dimensional mixed features |
| C0 base PCA distance | 0.1673 | weaker than agronomic-only |
| temporal DTW | 0.0763 | weak |
| temporal Pearson similarity | -0.0503 | weak |
| temporal best-lag correlation | -0.0482 | weak |

For distance metrics, a positive rho is the useful direction: farther parcels tend to differ
more in yield. For correlation/similarity metrics, a negative rho is the useful direction:
higher similarity tends to correspond to smaller yield differences.

### Important erratum

The first generated 04A report labels the expected direction for
`temporal_pearson_mean` as `positive`. That label is a display bug only; the statistic is
correct. Pearson similarity, like best-lag correlation, should have expected direction
`negative` with absolute yield difference. The code has been corrected after the run.

## Spatial autocorrelation

Observed yield has strong positive spatial autocorrelation under the 5-nearest-neighbor spatial
graph:

```text
Moran's I = 0.6797
permutation p = 0.001
expected I under spatial randomness = -0.0073
```

This is one of the most important 04A findings. Geographic location is not merely a nuisance
variable: nearby labeled parcels contain substantial information about one another's observed
yield.

The frozen Checkpoint 03 ensemble residuals show a revealing protocol dependence:

| Residual signal | Moran's I | permutation p |
|---|---:|---:|
| E13 state-stratified OOF | -0.0470 | 0.314 |
| E123 state-stratified OOF | -0.0375 | 0.429 |
| E13 municipality-grouped OOF | 0.5589 | 0.001 |
| E123 municipality-grouped OOF | 0.5483 | 0.001 |

When nearby/municipality-related parcels are allowed to support each other through the
state-stratified protocol, residual spatial autocorrelation largely disappears. When
municipalities are held out as groups, substantial spatial structure remains in the errors.

Interpretation: the first delivery models were already exploiting some local structure, but
their ability to reconstruct a parcel deteriorates when its local administrative/spatial
context is absent from training. This is direct motivation for 04B pseudo-target designs that
match the real target support geometry and for 04D local/graph models.

## Target support taxonomy

The 59 target parcels were classified diagnostically as:

| Support class | Targets |
|---|---:|
| A — high support | 5 |
| B — feature-supported | 2 |
| C — geographically supported | 5 |
| D — extrapolation | 13 |
| E — mixed support | 34 |

The support score is not a predicted yield. It ranks how well a target is represented by
labeled parcels across geography, deterministic feature space, monthly temporal behavior,
adversarial overlap and consistency of labeled-neighbor yields.

### Especially well-supported targets

The highest support scores include:

```text
AGC_088  0.8623
AGC_190  0.7435
AGC_126  0.7246
AGC_128  0.7217
AGC_086  0.6899
```

Three targets are particularly notable because the same labeled parcel is their nearest
neighbor geographically, in deterministic feature space and in monthly temporal behavior:

```text
AGC_088 -> AGC_165
AGC_190 -> AGC_006
AGC_126 -> AGC_111
```

These are natural first candidates for testing query-specific local estimators in 04B/04D.

### Weak-support targets

The lowest support scores include:

```text
AGC_173  0.0870
AGC_029  0.1246
AGC_120  0.1667
AGC_183  0.1884
AGC_003  0.2232
```

`AGC_173` and `AGC_029` are especially conspicuous: both have very large deterministic
feature-space distances to their nearest labeled parcel and adversarial target probabilities
near one. These parcels should not receive the same local-neighbor trust as the high-support
targets.

Support is also heterogeneous by geography. In the current diagnostic score, Singuilucan has
4 of 7 targets classified as extrapolation, Cuautepec de Hinojosa has 2 of 4, while Almoloya
has no extrapolation targets among its four targets. This must be treated as a support
diagnostic rather than a statement about expected yield.

## Temporal result

The monthly 2025 temporal comparison used 11 sensor-specific trajectories from Sentinel-2,
Landsat and Planet, with Pearson/Spearman correlation, best monthly lag and normalized DTW.

Temporal curves can identify visually compelling parcel analogues, and several high-support
targets have temporal nearest neighbors that agree with geographic/feature neighbors.
Nevertheless, across all labeled-labeled pairs the relationship between monthly temporal
similarity and yield difference is weak compared with geography and agronomic feature space.

Therefore:

- do not use temporal correlation alone as the primary predictor;
- retain temporal similarity as one component of a mixed metric;
- defer capture-level DTW/cross-correlation expansion until 04B shows that temporal information
  adds pseudo-test RMSE improvement after geography and agronomic structure are controlled.

## Implications for SIAP and external evidence

04A increases the value of geographically localized external evidence rather than decreasing
it. Strong spatial yield autocorrelation and municipality-sensitive residual structure make
the existing SIAP 2025 municipal information particularly relevant.

Checkpoint 04E should still audit the most specific defensible SIAP proxy available:

```text
Cebada grano
+ correct agricultural cycle
+ Temporal modality
+ exact municipality/CVEGEO
+ 2025 contemporaneous outcome when available
```

Local precipitation, water-stress, soil and topographic evidence should be evaluated in the
same target-specific framework.

## Required 04B design response

04B must now test whether the observed topology actually reduces RMSE. The pseudo-competition
framework should include:

1. repeated 30%-sized pseudo-target masks with pseudo-target X visible;
2. target-matched masks that resemble the real 59 in support score, municipality/state and
   multivariate geometry;
3. spatial/local baselines such as geographic kNN, agronomic-space kNN, weighted local
   regression and global-plus-local mixtures;
4. graph/Laplacian methods over all visible X;
5. the frozen E13/E123 models as reference baselines;
6. evaluation stratified by support tier so a method may be used only where it helps;
7. explicit testing of SIAP/local external priors rather than assuming they help.

A central hypothesis for 04B is now:

```text
high-support target -> local/spatial information may outperform or improve the global model
low-support target  -> rely more heavily on robust global/external-prior components
```

This hypothesis must be judged by pseudo-target RMSE, never by the appearance of the 59 final
predictions.
