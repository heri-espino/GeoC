# Agronomic Features v1

Agronomic Features v1 is an **additive, target-free feature layer** derived from the canonical
Feature Table v1.

It exists to encode nonlinear and cross-domain hypotheses that have agricultural meaning
without modifying or replacing the frozen source-derived representation in
`data/processed/features_v1/`.

The layer is intentionally stored separately:

```text
data/processed/features_v1/
    parcel_features_all.csv
    ...

data/processed/agronomic_features_v1/
    parcel_agronomic_features.csv
    feature_manifest.json
    build_report.json
    README.md
```

To use both representations, join one-to-one on `ID_POLIGONO`.

```python
from geocebada.features import join_agronomic_features

model_table = join_agronomic_features(base_table, agronomic_table)
```

No feature in this layer is selected using `RENDIMIENTO_T_HA`. Any later target-guided
interaction search or feature selection must occur inside training folds.

---

## Design principles

### Feature Table v1 remains immutable

Feature Table v1 records source-derived parcel covariates. Agronomic Features v1 records
**hypotheses built from those covariates**.

Keeping them separate makes it possible to compare:

```text
B0 = Feature Table v1 only
B1 = Agronomic Features v1 only
B2 = Feature Table v1 + Agronomic Features v1
B3 = selected/reduced base + selected agronomic families
```

under the same frozen validation folds.

### No combinatorial polynomial explosion

With roughly 1,400 numeric source features, all pairwise second-order products would create
nearly one million interactions. This layer instead creates a compact set of transformations
whose interpretation can be stated before looking at the target.

### Clean/competition provenance is inherited

A derived feature is `clean` only when it uses historical/static inputs.

Any feature using 2025 BASIC, PRO, target-year official climate, WaPOR or another
competition-mode parent is `competition`.

### CHIRPS is excluded from v1

The Checkpoint 02 audit showed that only one CHIRPS feature survived Feature Table v1.
Checkpoint 03A therefore does **not** build agronomic variables from CHIRPS. This avoids making
the nonlinear layer depend on a source whose extraction still needs QC.

---

## Evidence classes

Every feature in `feature_manifest.json` has an `evidence` field.

| Evidence | Meaning |
|---|---|
| `literature_backed` | The feature family or relationship is directly motivated by published crop/barley literature. |
| `mechanistic_proxy` | The formula has a clear agronomic/physical interpretation, but the exact algebraic form is a project proxy rather than a standard published index. |
| `experimental` | A target-free nonlinear basis or interaction worth testing predictively; it must not be presented as an established agronomic law. |

Every manifest record also stores:

- source input columns;
- formula;
- plain-language meaning;
- rationale;
- clean/competition mode;
- references;
- caveats.

The bibliography used by this layer is versioned at
`docs/references/agronomic_features_v1.bib`.

---

## 1. Phenology and seasonal curve shape

The satellite tables already contain monthly values. Agronomic Features v1 adds descriptors of
the **shape and timing** of the April–October signal.

Configured series include:

- historical Sentinel-2 NDVI, EVI, LAI, FAPAR, NDWI and MSI;
- historical Landsat VI6T;
- 2025 Sentinel-2 NDVI, EVI, LAI, FAPAR, NDWI and MSI;
- 2025 Landsat VI6T;
- 2025 Planet NDVI, EVI, LAI and MSAVI.

For each series:

### Seasonal mean

```math
\bar x = \frac{1}{n}\sum_t x_t
```

Represents average seasonal crop state.

### Seasonal AUC

```math
AUC \approx \sum_t \frac{x_t+x_{t+1}}{2}(t_{t+1}-t_t)
```

Represents integrated seasonal signal.

### Amplitude

```math
A = \max_t x_t - \min_t x_t
```

Represents seasonal dynamic range.

### Peak month

```math
t_{peak}=\arg\max_t x_t
```

Represents timing of maximum observed crop signal.

### Early and late means

```math
\bar x_{early}=\operatorname{mean}(Apr,May,Jun)
```

```math
\bar x_{late}=\operatorname{mean}(Aug,Sep,Oct)
```

These allow early and late crop states to contribute differently.

### June-to-August change

```math
\Delta_{Jun\rightarrow Aug}=x_{Aug}-x_{Jun}
```

A compact mid-season growth/decline descriptor.

### Green-up and senescence slopes

Linear slopes are fitted across calendar month values:

```math
x_t = \alpha + \beta t
```

for April–July and August–October separately.

These are not claims that crop development is literally linear. They are compact basis terms
that summarize directional change.

### Why

Remote sensing studies in barley show that relationships between vegetation indices and yield
depend on phenological stage rather than being constant over the season. The layer therefore
preserves curve timing and shape rather than relying only on a global average
[Chanev2025; Sharifi2020; Mirosavljevic2018].

---

## 2. 2025-versus-historical crop anomalies

For matching historical and 2025 Sentinel-2 series, and Landsat VI6T, the layer creates:

```text
season_mean_delta
season_mean_ratio
auc_delta
august_delta
late_season_delta
peak_month_shift
```

Examples:

```math
\Delta \bar x = \operatorname{mean}_m(x_{2025,m}-x_{hist,m})
```

```math
R_x = \frac{\bar x_{2025}}{\bar x_{hist}+\epsilon}
```

```math
\Delta AUC = AUC_{2025}-AUC_{hist}
```

Interpretation:

> persistent parcel/site signal is separated from how unusual the 2025 crop appears relative
> to its own historical remote-sensing profile.

The NDWI seasonal ratio is deliberately omitted. NDWI is a signed normalized index and its
historical seasonal mean can approach or cross zero, making a ratio numerically unstable and
difficult to interpret. For NDWI the layer keeps difference/AUC/timing anomaly terms instead.

These variables are `competition` because they use full-season 2025 covariates.

---

## 3. Cross-sensor agreement

For NDVI, EVI and LAI, Sentinel-2 and Planet 2025 summaries are compared with:

```text
season_mean_difference
season_mean_ratio
monthly_rmse
```

For example:

```math
RMSE_{sensor}=
\sqrt{\frac{1}{n}\sum_m
(x^{S2}_m-x^{Planet}_m)^2}
```

These are **experimental** features. They do not assume that Sentinel-2 and Planet products are
radiometrically interchangeable. Instead they expose whether independent sensor streams tell a
similar or divergent story for a parcel.

---

## 4. Thermal development

For historical official climate and 2025 climate, the layer derives monthly:

```math
T_{mean,m}=\frac{T_{max,m}+T_{min,m}}{2}
```

and:

```math
DTR_m=T_{max,m}-T_{min,m}.
```

It also creates seasonal averages.

### GDD proxies

The project uses two **declared scenarios**, not target-fitted thresholds:

```math
GDD_{proxy}(T_b)=
\sum_m d_m
\max(T_{mean,m}-T_b,0)
```

for:

```text
Tb = 0 C
Tb = 5 C
```

where `d_m` is the number of days in the month.

These are explicitly named `gdd_proxy` because our official climate inputs are monthly.
They are not equivalent to true daily GDD.

Cereal phenology is commonly represented through accumulated thermal time, and barley
phenology is strongly linked to meteorological conditions
[Hajkova2019; McMaster2005; McMaster1997].

### Heat-excess basis

A fixed, non-target-fitted 25 C scenario is included:

```math
H_{25}=
\sum_m d_m\max(T_{mean,m}-25,0).
```

This is labeled a **mechanistic proxy**, not a physiological critical-temperature estimate.
Monthly means can miss short heat waves.

---

## 5. Rainfall timing

For both historical and 2025 official precipitation:

```text
early_sum      = Apr + May + Jun
mid_sum        = Jun + Jul + Aug
late_sum       = Aug + Sep + Oct
season_cv      = sd(monthly precip) / abs(mean(monthly precip))
late_fraction  = late_sum / season_sum
```

The rationale is stage dependence: barley sensitivity to water deficit changes through crop
development, so timing can matter beyond total rainfall
[Bello2022; Mogensen1980; Zhang2015].

---

## 6. Water productivity and crop-water interactions

WaPOR AETI and NPP are combined with official rainfall and satellite crop state.

### NPP per AETI

```math
WP_{NPP}=\frac{NPP}{AETI+\epsilon}
```

This is a remote-sensing **water-productivity proxy**, not grain water-use efficiency.

### AETI relative to precipitation

```math
R_{ET/P}=\frac{AETI}{P+\epsilon}
```

This is not called a drought index. AETI can exceed concurrent precipitation because of stored
soil water, irrigation and other water sources.

### Atmospheric water-balance proxy

```math
WB=P-AETI.
```

Again, this is a simple proxy and not a complete soil-water balance.

### Crop-state interactions

Monthly terms include:

```math
NPP\times NDVI
```

and:

```math
LAI\times AETI.
```

Seasonal analogues are also created.

WaPOR is designed for remote-sensing water-productivity analysis, while barley literature
supports stage-dependent interactions between water availability, evapotranspiration and yield
[Bello2022; Mogensen1980; Zhang2015; Dhonthi2024].

---

## 7. Soil profile and nonlinear soil terms

For every SoilGrids property available in Feature Table v1:

```text
bdod
cec
cfvo
clay
nitrogen
phh2o
sand
silt
soc
```

the layer creates a vertical contrast:

```math
G_p = p_{0-30}-p_{30-60}
```

and an upper/deeper ratio:

```math
R_p = \frac{p_{0-30}}{p_{30-60}+\epsilon}.
```

Additional compact interactions include:

```text
SOC × CEC
SOC × nitrogen
clay × SOC
clay × CEC
bulk density × clay
clay / sand
clay + silt
pH²
```

The pH-squared term does **not** assume a fixed optimum. It simply provides a quadratic basis
that a regularized linear model can combine with the original pH variable.

Barley yield response to soil chemistry and exchangeable cations can be nonlinear; the exact
interaction terms here are therefore documented as either mechanistic proxies or experimental
basis terms rather than causal laws [Holland2021].

---

## 8. Cross-domain agronomic interactions

A deliberately small set combines different biological/environmental domains.

### Historical productivity × current crop condition

```math
Y_{SIAP,hist}\times \Delta NDVI_{2025}
```

```math
Y_{SIAP,hist}\times \Delta EVI_{2025}
```

```math
Y_{SIAP,hist}\times NPP_{2025}
```

```math
Y_{SIAP,hist}\times WP_{NPP,2025}
```

Interpretation: regional historical productivity provides a baseline while the 2025 parcel
signal represents current crop condition.

### Terrain × climate

```text
slope × precipitation
elevation × seasonal temperature
```

### Soil × climate

```text
clay × precipitation
SOC × precipitation
```

### Soil fertility × crop state

```text
soil nitrogen × NDVI
```

### Crop-state interactions

```text
NDVI × NDWI
EVI × LAI
```

### Combined heat/water candidate

```math
H_{25}\times \max(AETI-P,0)
```

This final term is explicitly experimental. `AETI-P` is not equivalent to plant water stress.

All of these are predictive hypotheses that must earn their place through the frozen CV
protocols.

---

## 9. Generic nonlinear basis terms

A small number of non-negative skewed variables receive:

```math
z=\log(1+x)
```

including parcel area, historical SIAP area/production, precipitation, AETI and NPP.

These terms are useful mainly for linear/regularized models because they can represent
diminishing effects without generating a combinatorial polynomial expansion.

They are classified as `experimental`.

---

## Why kernels are not stored here

Polynomial, RBF and other kernels are model transformations:

```math
K(x,z)
```

Their representation depends on scaling and model hyperparameters such as degree and
`gamma`.

They therefore belong in Checkpoint 03B model pipelines, not in the processed feature dataset.

The distinction is intentional:

```text
Agronomic Features v1
    explicit, interpretable X transformations
    stored and versioned

Kernel methods
    learned/model representation
    fitted/evaluated inside CV
```

---

## Leakage and validation rules

Agronomic Features v1 itself is target-free.

Allowed at build time:

- deterministic transformations of X;
- cross-domain products defined before target inspection;
- 197-row X-only construction;
- clean/competition provenance assignment.

Not allowed at build time:

- ranking interactions by correlation with yield;
- selecting transformations because they score well on all 138 labeled parcels;
- choosing thresholds from full-target performance;
- hidden-target reconstruction.

Any target-guided feature selection belongs inside training folds in Checkpoint 03B.

All model comparisons must reuse:

```text
reports/checkpoint_02/cv_folds.csv
```

for the frozen state-stratified and municipality-grouped protocols.

---

## Build

From the project root:

```powershell
conda activate geocebada
python tools\build_agronomic_features_v1.py
```

The builder consumes only versioned Feature Table v1, so no large external raw downloads are
required.

---

## References

Citation keys correspond to
`docs/references/agronomic_features_v1.bib`.

- Chanev2025 — barley phenological stage, Sentinel-2 and yield prediction.
- Sharifi2020 — barley yield prediction using satellite and meteorological information.
- Mirosavljevic2018 — NDVI variation and grain yield in winter barley.
- Hajkova2019 — meteorological drivers of spring-barley phenology.
- McMaster2005 — development and phenology of temperate cereals including barley.
- McMaster1997 — growing-degree-day formulation and interpretation.
- Bello2022 — malt barley response to water stress at different growth stages.
- Mogensen1980 — barley drought sensitivity by growth stage and relative evapotranspiration.
- Zhang2015 — evapotranspiration/water dynamics and wheat/barley grain yield.
- Holland2021 — nonlinear barley yield response to soil exchangeable cations.
- Dhonthi2024 — WaPOR-based water-productivity analysis.
