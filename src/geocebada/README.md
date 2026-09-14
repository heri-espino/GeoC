# `geocebada` Python library

This directory is the reusable codebase for GeoCebada. Team notebooks and the web app should **import functionality from this package** instead of copying data-loading, feature-engineering or evaluation code into notebooks.

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
├── features/             # stable feature engineering
├── models/               # training/inference interfaces
└── evaluation/           # CV, metrics and diagnostics
```

The last three packages will grow as the modeling pipeline is defined.

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
- learned preprocessing must ultimately live inside leakage-safe model/CV pipelines.

See root `AGENTS.md`, `.ai_handoff`, and `data/.ai_handoff` for project-wide constraints.
