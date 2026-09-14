# Visual Explorer

`app/pages/1_Visual_Explorer.py` is the high-density exploratory page of GeoCebada Lab. It is intended to help the team discover data-quality issues, spatial patterns, multivariate structure, possible feature ideas, and distribution shift without moving reusable scientific logic into notebooks.

## Data sources

The page can load the official yield/split table, the bundled BASIC CSV, the bundled PRO CSV, or an uploaded CSV. When `ID_POLIGONO` is available, BASIC/PRO can inherit missing parcel metadata from the official split table through `attach_yield_split_metadata`. This never fills the hidden yield for `PREDICCION` parcels.

## Shared filters

Filters are applied once and the resulting subset is reused across every visualization. Current filters include train/prediction membership, parcel-ID substring search, low-cardinality categorical selections, numeric range filters, and an optional complete-case filter.

The filtered row count, retained fraction, unique parcel count, and missing-cell count remain visible above the visualization tabs.

## Views

### Map

The map loads the official parcel geometry archive, joins the filtered data to parcel geometry, and permits an arbitrary variable to color the parcels. Additional fields can be placed in hover tooltips.

Repeated rows are reduced to parcel level for display using a generic mean/first aggregation. This is only an exploratory map summary. Scientifically meaningful temporal aggregation must still be implemented as an explicit feature recipe or pipeline function.

The map does not guess CRS. If the geometry source lacks CRS metadata, mapping stops until the CRS is resolved from a source-backed reference.

### Pairplot

The interactive scatter matrix supports up to eight selected numeric variables and optional color/group encoding. It is intended for rapid identification of non-linearity, clusters, interactions, outliers, and train/prediction separation.

### Correlations

The correlation heatmap supports Spearman, Pearson, and Kendall correlations. Correlation structure is exploratory and does not establish causality or independence.

### Bivariate

The bivariate explorer allows arbitrary numeric X/Y selection, optional grouping, and an optional OLS trendline.

### Outliers

The outlier view reports univariate IQR flags and provides a boxplot for inspection. IQR flags are never automatic deletion rules; agronomic extremes may be valid and predictive.

### Groups

Low-cardinality variables can be used to compare mean, median, standard deviation, extrema, and counts across selected numeric variables.

### Missingness

The page reports missingness after filtering and exposes a descriptive-statistics table. This is useful for identifying whether missingness patterns are concentrated in a particular split, parcel subset, region, or other group.

## Methodological guardrails

The Visual Explorer is intentionally permissive for exploratory work, but results must not bypass the project's leakage rules. In particular:

- prediction-set covariates can be inspected, but hidden targets must never be fabricated or inferred from external leakage;
- supervised model selection remains based on labeled-data validation;
- the agricultural cycle represented by `RENDIMIENTO_T_HA` is still unresolved;
- temporal features must not be promoted until their relationship to the target harvest date is source-backed;
- repeated rows from the same parcel must not be treated as independent train/validation observations;
- spatial structure can make random cross-validation optimistic;
- map reprojection requires source CRS metadata.

## Reusable functions

The page delegates reusable operations to the package rather than reimplementing them in Streamlit. Relevant functions include `filter_frame`, `aggregate_to_parcels`, `load_parcels`, `pairplot_figure`, `correlation_heatmap`, `outlier_summary`, and `parcel_map_figure`. Search `docs/FUNCTION_INDEX.md` before adding overlapping helpers.
