# `geocebada` Python library

This directory is the reusable codebase for GeoCebada. Team notebooks and the web app should **import functionality from this package** instead of copying data-loading, feature-engineering, statistical inference, visualization or evaluation code into notebooks.

## Design rule

If code is useful in more than one notebook, experiment, model, or application path, it belongs here.

```text
src/geocebada/
├── __init__.py
├── paths.py              # repository/data path resolution
├── config.py             # YAML configuration
├── data/
│   ├── files.py          # discovery and multi-file ingestion
│   ├── targets.py        # official yield/split loader and validation
│   └── rasters.py        # GeoTIFF discovery/inventory/metadata
├── geo/
│   └── crs.py            # CRS safeguards and reprojection helpers
├── features/
│   └── interactive.py    # reproducible feature recipes and joins
├── statistics/
│   └── inference.py      # association tests, multiplicity and assumptions
├── visualization/
│   └── exploration.py    # reusable Plotly figures and data-quality views
├── evaluation/
│   └── regression.py     # early regression benchmark utilities
└── models/               # final training/inference interfaces
```

## Usage from notebooks

Install the repository once in editable mode from the project root:

```bash
pip install -e ".[dev,geo]"
```

Then notebooks can import the package normally:

```python
from geocebada.data import load_yield_split, partition_yield_split
from geocebada.paths import source_path

split = load_yield_split()
train, prediction = partition_yield_split(split)
```

For many CSV files:

```python
from geocebada.data import concatenate_csvs

frame = concatenate_csvs(
    "some/directory",
    recursive=True,
    source_column="source_file",
)
```

For climate rasters:

```python
from geocebada.data import build_raster_inventory
from geocebada.paths import source_path

precip_dir = source_path("climate", "precipitation")
inventory = build_raster_inventory(precip_dir)
```

For statistical exploration:

```python
from geocebada.statistics import correlation_screen, linear_regression_diagnostics

screen = correlation_screen(train, "RENDIMIENTO_T_HA", correction="fdr_bh")
diagnostics = linear_regression_diagnostics(train, "RENDIMIENTO_T_HA", ["AREA_HA"])
```

For reproducible feature recipes:

```python
from geocebada.features import FeatureRecipe, apply_feature_recipe

recipe = FeatureRecipe(
    name="ndvi_mean",
    value_column="NDVI",
    group_column="ID_POLIGONO",
    aggregation="mean",
    date_column="date",
    start="2025-05-01",
    end="2025-08-31",
)
feature = apply_feature_recipe(long_table, recipe)
```

Interactive feature recipes are exploratory specifications. If a feature becomes part of the canonical model, promote its logic into the appropriate stable feature module and cover it with tests.

## GeoCebada Lab contract

`app/main.py` is an interface over this library, not a second implementation of the analysis pipeline. The current laboratory provides:

- dataset/schema and missingness exploration;
- bivariate Pearson/Spearman association;
- multiple-hypothesis screening with FDR/Holm/Bonferroni correction;
- linear-regression residual, heteroscedasticity and VIF diagnostics;
- interactive grouped feature recipes, including temporal mean/max/slope/AUC;
- exploratory random-CV regression benchmarks;
- reusable Plotly visualizations.

The app automatically excludes hidden official prediction targets from supervised statistical/model analyses. Random CV in the laboratory remains exploratory until the project establishes spatial/grouped validation.

## Public API and function index

Before creating a helper, search:

- `docs/FUNCTION_INDEX.md` for the quick searchable catalog;
- the Sphinx API reference under `docs/api/`;
- the package source when implementation details are needed.

After adding, renaming or deleting a **public** function/class, regenerate the quick index:

```bash
python tools/generate_function_index.py
```

Public reusable functions should:

1. have a clear docstring;
2. include type hints;
3. avoid machine-specific paths;
4. validate important assumptions rather than silently guessing;
5. receive tests when behavior is stable;
6. live in the narrowest appropriate module;
7. be exported from the relevant package `__init__.py` when intended for common use.

Do not create a new helper merely to save one or two lines in one notebook. The package should centralize **stable, reused behavior**, not every exploratory operation.

## Naming and source-of-truth rules

- canonical parcel ID: `ID_POLIGONO`;
- canonical target: `RENDIMIENTO_T_HA`;
- `data/source/` is immutable;
- CRS and temporal assumptions must be explicit;
- learned preprocessing must ultimately live inside leakage-safe model/CV pipelines;
- statistical significance is not evidence of causality;
- multiple exploratory tests require multiplicity control and explicit interpretation;
- spatial independence cannot be assumed for georeferenced parcels.

See root `AGENTS.md`, `.ai_handoff`, and `data/.ai_handoff` for project-wide constraints.
