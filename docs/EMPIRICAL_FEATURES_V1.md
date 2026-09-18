# Empirical Features v1

Empirical Features v1 is the **data-driven, target-free** feature layer for Checkpoint 03B.

It follows Agronomic Features v1 but answers a different question:

> What nonlinear structure can be extracted from the observed covariates themselves without
> first assuming a crop mechanism or looking at yield?

The deterministic layer is stored separately:

```text
data/processed/features_v1/              # source-derived covariates
data/processed/agronomic_features_v1/    # explicit agronomic hypotheses
data/processed/empirical_features_v1/    # empirical X-only geometry
```

All three tables remain one row per parcel and join by `ID_POLIGONO`.

Target-aware formula discovery is **not** written to the processed dataset. It is implemented as
a fold-local transformer and may only be fitted on the training portion of a validation fold.

---

## 1. Why a separate empirical layer?

Agronomic Features v1 encodes relationships that can be explained before model fitting:
phenology, water productivity, thermal time, soil interactions and similar concepts.

Empirical Features v1 instead searches for useful mathematical summaries of the observed
covariate geometry:

- temporal roughness and curvature;
- persistence and turning behavior;
- short-baseline relative condition;
- stable relative change for signed indices;
- cross-sensor shape similarity;
- normalized distribution geometry.

This separation allows later model comparisons to ask whether performance comes from:

```text
source variables
agronomic knowledge
empirical X-only structure
fold-local target-aware discovery
or combinations of them
```

without silently mixing the provenance of the predictors.

---

## 2. Temporal-shape geometry

For each configured April-October series, the layer computes descriptors that are distinct from
the phenology summaries already stored in Agronomic Features v1.

### Total variation

```{math}
TV(x)=\sum_{t=2}^{T}|x_t-x_{t-1}|.
```

This measures total month-to-month movement regardless of direction.

### Roughness

Let the local slope be

```{math}
s_t=\frac{x_t-x_{t-1}}{m_t-m_{t-1}}.
```

Then

```{math}
R(x)=\frac{1}{T-2}\sum_t |s_t-s_{t-1}|.
```

A smooth one-directional seasonal trajectory has lower roughness than one with repeated
accelerations and reversals.

### Curvature energy

```{math}
C(x)=\frac{1}{T-2}\sum_t (s_t-s_{t-1})^2.
```

This emphasizes larger changes in local slope.

### Lag-1 autocorrelation

```{math}
\rho_1=\operatorname{corr}(x_1,\ldots,x_{T-1},
                             x_2,\ldots,x_T).
```

It summarizes month-to-month persistence.

### Linear-trend R²

A straight line is fitted against calendar month. The resulting R² quantifies how much of the
seasonal signal is approximately monotone/linear versus curved.

### Turning fraction

The sign of each local slope is recorded. The feature is the fraction of adjacent slopes that
change sign. It is a compact measure of how often the trajectory reverses direction.

### Peak sharpness

```{math}
P(x)=x_{(1)}-x_{(2)},
```

where (x_{(1)}) and (x_{(2)}) are the largest and second-largest monthly values.

### Center of mass in time

Each row is range-normalized to nonnegative weights

```{math}
w_t=\frac{x_t-\min x}{\max x-\min x},
```

then

```{math}
M(x)=\frac{\sum_t m_t w_t}{\sum_t w_t}.
```

This is an empirical timing descriptor, not a biological growth stage.

### Shape entropy

After normalizing the same weights to probabilities (p_t),

```{math}
H(x)=
-\frac{\sum_t p_t\log p_t}{\log T}.
```

Values near one indicate a diffuse seasonal signal; lower values indicate concentration in
fewer months.

These features are mathematical summaries rather than published barley indices.

---

## 3. Short-baseline historical condition

The project has only a short remote-sensing history. For each parcel, the 2025 monthly value can
still be positioned relative to the observed historical parcel range:

```{math}
Q_{m}=
\frac{x_{2025,m}-x_{hist,min}}
     {x_{hist,max}-x_{hist,min}}.
```

For NDVI the same quantity is scaled by 100 and named
`emp_vci_like_short__...`.

This construction is inspired by the logic of the Vegetation Condition Index (VCI), where
current vegetation condition is normalized relative to historical extrema
[Bokusheva2016; Serban2025].

It is **not standard VCI** for this project because:

- the historical reference is only 2022-2024;
- extrema are parcel-level source summaries rather than a long climatology;
- the project intentionally leaves values outside the historical range unclipped.

The feature names therefore include `vci_like_short`, not `vci`.

The same range-position idea is used for other indices under the generic
`emp_histpos__...` namespace without claiming they are established vegetation-condition
indices.

---

## 4. Stable symmetric change

A simple ratio such as (x_{2025}/x_{hist}) can become unstable when an index is signed and its
historical value is close to zero.

Checkpoint 03B therefore adds the target-free contrast

```{math}
D(a,b)=\frac{2(a-b)}{|a|+|b|+\epsilon}.
```

For each 2025-versus-history pair the layer summarizes:

- seasonal mean;
- root-mean-square change;
- maximum absolute change;
- late-season mean;
- fraction of positive monthly changes;
- fraction of sign reversals through the season.

This is an empirical mathematical contrast, not a standard agronomic index.

---

## 5. Cross-sensor temporal similarity

Sentinel-2 and Planet observe the same crop cycle through different instruments and processing
chains. Instead of treating only absolute level differences as features, 03B compares their
seasonal shapes using:

- Pearson correlation;
- Spearman rank correlation;
- cosine similarity;
- RMSE normalized by the combined observed range;
- mean absolute symmetric difference;
- absolute difference in peak month.

The purpose is not to assert radiometric equivalence. The variables expose whether two sensor
streams tell a similar temporal story for the parcel.

---

## 6. Aggregate distribution geometry

Feature Table v1 already contains source summaries such as mean, standard deviation, minimum
and maximum. 03B converts selected groups into scale-normalized descriptors.

Examples include:

```{math}
CV^* = \frac{sd}{|mean|+\epsilon},
```

```{math}
R_{sd} = \frac{sd}{max-min+\epsilon},
```

and the position of the mean within its observed range:

```{math}
Q_{mean} = \frac{mean-min}{max-min+\epsilon}.
```

A signed asymmetry basis is also stored:

```{math}
A = \frac{max+min-2mean}{max-min+\epsilon}.
```

These may be useful to regularized linear models that otherwise see the four source summaries
as unrelated columns.

---

## 7. What is safe to materialize globally?

The committed empirical dataset contains only deterministic row-wise functions of X.

It does **not** use:

- `RENDIMIENTO_T_HA`;
- target correlation;
- target-guided pair selection;
- PCA fitted on all 197 rows;
- clustering fitted on all 197 rows;
- supervised symbolic regression.

This distinction matters even for X-only learned transforms. PCA, KMeans and similar methods
learn parameters from the distribution of multiple parcels. For rigorous validation, they
should be fitted on the training partition and then applied to the validation partition.

Therefore they belong in the later model pipeline rather than the global processed table.

---

## 8. Fold-local target-aware expression discovery

The second part of 03B is implemented by:

```python
from geocebada.features import FoldLocalExpressionMiner
```

The transformer is deliberately small and auditable.

During `fit(X_train, y_train)` it:

1. reads only a curated primitive set;
2. median-imputes using training rows only;
3. ranks primitives by absolute Spearman association with the training target;
4. retains a small number of primitives;
5. generates simple two-variable formulas;
6. ranks those formulas using the training target;
7. stores only the selected formulas and training medians.

During `transform(X_valid)` it applies the already selected expressions and never receives
`y_valid`.

Allowed first-stage operations are:

```{math}
x_i x_j,
\qquad
x_i+x_j,
\qquad
x_i-x_j,
```

```{math}
\frac{x_i}{|x_j|+\epsilon},
```

and

```{math}
\frac{2(x_i-x_j)}
{|x_i|+|x_j|+\epsilon}.
```

Both orientations are considered for difference and safe ratio.

This is **not yet unrestricted symbolic regression**. It is a controlled depth-one expression
search designed for (n=138).

A recent remote-sensing study explicitly used symbolic regression to discover vegetation
indices for crop biomass estimation [Lou2025]. That supports testing formula discovery as a
methodological direction, but it does not validate any GeoCebada expression in advance.

---

## 9. What would count as a candidate new barley-yield index?

A high training correlation is not sufficient.

A formula would become scientifically interesting only if its structure is stable across
training folds and its predictive contribution survives held-out evaluation, especially the
municipality-grouped protocol.

For example, if independently fitted folds repeatedly recover an expression combining crop
condition and water productivity, that recurrence is evidence worth investigating.

Until then, outputs of `FoldLocalExpressionMiner` should be called:

```text
candidate expressions
fold-local discovered features
empirical formulas
```

not a new agronomic index.

A final named index would require a separate stability analysis and ideally external validation.

---

## 10. Why CWSI is not reconstructed from the available variables

Crop Water Stress Index has direct relevance to malt barley, and a 2024 study related CWSI to
malt-barley yield and quality [King2024].

However, the empirical CWSI formulation in that work uses canopy temperature and wet/dry
reference temperatures. GeoCebada currently has air-temperature products, not the required
canopy-temperature reference system.

The project therefore does **not** create a variable named CWSI from air temperature.

---

## 11. Leakage contract

The materialized table may use all 197 X rows because every retained transformation is
deterministic and row-wise.

Target-aware discovery follows a stricter rule:

```text
outer train
   |
   +-- fit primitive ranking
   +-- fit expression search
   +-- fit imputation/scaling/model
   |
   v
outer validation
   |
   +-- transform using train-fitted state only
   +-- score
```

No formula discovered using all 138 labeled parcels may later be treated as if it had been
pre-specified for validation on those same parcels.

---

## 12. Build

From the repository root:

```powershell
conda activate geocebada
python tools\build_empirical_features_v1.py
```

The builder consumes the versioned Feature Table v1 and validates that all configured fold-local
discovery primitives exist after joining Feature Table v1, Agronomic Features v1 and Empirical
Features v1.

No external raw downloads are required.

---

## References

Citation keys are versioned in
`docs/references/empirical_features_v1.bib`.

- Bokusheva2016 — vegetation health indices and drought-related yield losses.
- Serban2025 — VCI/TCI/VHI in agricultural drought and crop-yield analysis.
- King2024 — malt barley yield and quality response to CWSI.
- Lou2025 — symbolic regression for discovery of vegetation indices.
