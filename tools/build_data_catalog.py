#!/usr/bin/env python
"""Build a machine-readable data catalog and tiny real-data fixtures for GeoCebada.

The catalog captures columns, dtypes, CRS, raster dimensions and NetCDF structure so new
pipelines can be designed against a small contract instead of repeatedly loading all data.
Generated samples are deterministic development fixtures, not training/evaluation datasets.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = Path("data/data_catalog.json")
DEFAULT_SAMPLES = Path("data/samples")

SOURCE_GROUPS: dict[str, Path] = {
    "official_split": Path(
        "data/source/tabular/ID_area_rendimiento_70_30_Reto_AgroCebada.csv"
    ),
    "official_basic": Path("data/source/tabular/Conjunto_datos_BASICO_AgroCebada2026.csv"),
    "official_pro": Path("data/source/tabular/Conjunto_datos_PRO_AgroCebada.csv"),
    "official_parcels": Path(
        "data/source/geospatial/Parcelas_Reto_AGC_CONJUNTO_70_30.zip"
    ),
    "official_climate": Path("data/source/climate"),
    "official_topography": Path("data/source/topography"),
    "external_inegi_municipios": Path("data/raw/external/inegi_municipios"),
    "external_chirps_daily": Path("data/raw/external/chirps_v3_daily"),
    "external_wapor": Path("data/raw/external/wapor_v3"),
    "external_soilgrids": Path("data/raw/external/soilgrids_250m"),
    "external_siap": Path("data/raw/external/SIAP"),
    "external_inegi_cem": Path("data/raw/external/Inegi"),
}

TABULAR_SUFFIXES = {".csv", ".xlsx", ".xls"}
RASTER_SUFFIXES = {".tif", ".tiff"}
VECTOR_SUFFIXES = {".geojson", ".gpkg", ".shp"}
NETCDF_SUFFIXES = {".nc"}
RECOGNIZED_SUFFIXES = TABULAR_SUFFIXES | RASTER_SUFFIXES | VECTOR_SUFFIXES | NETCDF_SUFFIXES
DATE_NAME_PATTERN = re.compile(r"(^|_)(fecha|date|datetime|time|timestamp)(_|$)", re.I)


def portable(path: Path, root: Path) -> str:
    """Return a forward-slash path relative to the repository root."""
    return path.resolve().relative_to(root.resolve()).as_posix()


def json_value(value: Any) -> Any:
    """Convert common NumPy/Pandas scalars to JSON-compatible values."""
    if value is None or value is pd.NA:
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if np.isnan(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, float) and np.isnan(value):
        return None
    return value if isinstance(value, (str, int, float, bool)) else str(value)


def evenly_spaced_indices(length: int, count: int) -> list[int]:
    """Return deterministic unique indices spanning ``[0, length)``."""
    if length <= 0 or count <= 0:
        return []
    if length <= count:
        return list(range(length))
    return sorted(set(np.linspace(0, length - 1, num=count, dtype=int).tolist()))


def dataframe_profile(frame: pd.DataFrame, *, sample_limit: int) -> dict[str, Any]:
    """Describe columns and values from an in-memory tabular profile sample."""
    columns: list[dict[str, Any]] = []
    for name in frame.columns:
        series = frame[name]
        non_null = series.dropna()
        examples = [json_value(value) for value in non_null.drop_duplicates().head(3).tolist()]
        record: dict[str, Any] = {
            "name": str(name),
            "dtype": str(series.dtype),
            "non_null_in_profile": int(series.notna().sum()),
            "missing_in_profile": int(series.isna().sum()),
            "unique_in_profile": int(non_null.nunique(dropna=True)),
            "examples": examples,
        }
        if pd.api.types.is_numeric_dtype(series) and not non_null.empty:
            numeric = pd.to_numeric(non_null, errors="coerce").dropna()
            if not numeric.empty:
                record["numeric_range_in_profile"] = {
                    "min": json_value(numeric.min()),
                    "max": json_value(numeric.max()),
                }
        if DATE_NAME_PATTERN.search(str(name)) and not non_null.empty:
            parsed = pd.to_datetime(non_null, errors="coerce", dayfirst=True)
            parsed = parsed[parsed.notna()]
            if not parsed.empty:
                record["date_range_in_profile"] = {
                    "min": parsed.min().isoformat(),
                    "max": parsed.max().isoformat(),
                }
        columns.append(record)
    return {
        "profile_rows": int(len(frame)),
        "column_count": int(len(frame.columns)),
        "columns": columns,
        "profile_row_limit": sample_limit,
    }


def read_csv_profile(path: Path, sample_limit: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read enough of a CSV to infer its schema without loading the full table."""
    attempts = [
        {"encoding": "utf-8", "sep": ","},
        {"encoding": "utf-8-sig", "sep": ","},
        {"encoding": "latin-1", "sep": ","},
        {"encoding": "utf-8", "sep": None, "engine": "python"},
        {"encoding": "latin-1", "sep": None, "engine": "python"},
    ]
    error: Exception | None = None
    for kwargs in attempts:
        try:
            frame = pd.read_csv(path, nrows=sample_limit, low_memory=False, **kwargs)
            if len(frame.columns) == 1 and kwargs.get("sep") == ",":
                continue
            profile = dataframe_profile(frame, sample_limit=sample_limit)
            profile["encoding_used"] = kwargs["encoding"]
            profile["separator_mode"] = kwargs.get("sep") or "auto"
            return frame, profile
        except Exception as exc:  # pragma: no cover - depends on source encoding.
            error = exc
    raise RuntimeError(f"Could not parse CSV {path}: {error}")


def excel_profile(path: Path, sample_limit: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Profile an Excel workbook and return its first sheet as the fixture source."""
    workbook = pd.ExcelFile(path)
    sheet_profiles: dict[str, Any] = {}
    first_frame: pd.DataFrame | None = None
    for sheet_name in workbook.sheet_names:
        frame = pd.read_excel(path, sheet_name=sheet_name, nrows=sample_limit)
        if first_frame is None:
            first_frame = frame
        sheet_profiles[sheet_name] = dataframe_profile(frame, sample_limit=sample_limit)
    return first_frame if first_frame is not None else pd.DataFrame(), {
        "sheet_names": workbook.sheet_names,
        "sheets": sheet_profiles,
    }


def select_tabular_sample(frame: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    """Select a deterministic sample preserving parcel/date diversity when possible."""
    if frame.empty or len(frame) <= max_rows:
        return frame.copy()

    if "ID_POLIGONO" in frame.columns:
        ids = sorted(frame["ID_POLIGONO"].dropna().astype(str).unique())
        chosen_ids = [ids[index] for index in evenly_spaced_indices(len(ids), min(8, len(ids)))]
        rows_per_id = max(1, max_rows // max(1, len(chosen_ids)))
        pieces: list[pd.DataFrame] = []
        for parcel_id in chosen_ids:
            part = frame.loc[frame["ID_POLIGONO"].astype(str) == parcel_id].copy()
            if "fecha_captura" in part.columns:
                part = part.sort_values("fecha_captura")
            indices = evenly_spaced_indices(len(part), min(rows_per_id, len(part)))
            pieces.append(part.iloc[indices])
        result = pd.concat(pieces, ignore_index=True) if pieces else frame.head(max_rows)
        return result.head(max_rows)

    indices = evenly_spaced_indices(len(frame), max_rows)
    return frame.iloc[indices].copy().reset_index(drop=True)


def inspect_tabular(
    path: Path,
    *,
    root: Path,
    sample_limit: int,
    sample_rows: int,
    sample_dir: Path | None,
    sample_name: str,
) -> dict[str, Any]:
    """Inspect one CSV/Excel file and optionally emit a tiny CSV fixture."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        frame, profile = read_csv_profile(path, sample_limit)
    else:
        frame, profile = excel_profile(path, sample_limit)

    result: dict[str, Any] = {
        "path": portable(path, root),
        "format": suffix.lstrip("."),
        "bytes": path.stat().st_size,
        "profile": profile,
    }
    if sample_dir is not None:
        sample = select_tabular_sample(frame, sample_rows)
        output = sample_dir / "tabular" / f"{sample_name}.csv"
        output.parent.mkdir(parents=True, exist_ok=True)
        sample.to_csv(output, index=False)
        result["sample"] = portable(output, root)
        result["sample_rows"] = int(len(sample))
    return result


def raster_profile(path: Path) -> dict[str, Any]:
    """Return structural raster metadata without loading the full raster."""
    import rasterio

    with rasterio.open(path) as dataset:
        bounds = dataset.bounds
        return {
            "driver": dataset.driver,
            "crs": str(dataset.crs) if dataset.crs else None,
            "dimensions": [dataset.width, dataset.height],
            "band_count": dataset.count,
            "dtypes": list(dataset.dtypes),
            "nodata": json_value(dataset.nodata),
            "resolution": [dataset.res[0], dataset.res[1]],
            "bbox": [bounds.left, bounds.bottom, bounds.right, bounds.top],
        }


def write_raster_sample(path: Path, output: Path, window_size: int) -> None:
    """Write a small center window of a raster preserving its georeferencing."""
    import rasterio
    from rasterio.windows import Window

    output.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path) as source:
        width = min(window_size, source.width)
        height = min(window_size, source.height)
        col_off = max(0, (source.width - width) // 2)
        row_off = max(0, (source.height - height) // 2)
        window = Window(col_off, row_off, width, height)
        array = source.read(window=window)
        profile = source.profile.copy()
        profile.update(
            width=width,
            height=height,
            transform=source.window_transform(window),
        )
        with rasterio.open(output, "w", **profile) as target:
            target.write(array)


def inspect_raster(
    path: Path,
    *,
    root: Path,
    sample_dir: Path | None,
    sample_name: str,
    window_size: int,
) -> dict[str, Any]:
    """Inspect one raster and optionally write a tiny real-data GeoTIFF fixture."""
    result: dict[str, Any] = {
        "path": portable(path, root),
        "format": path.suffix.lower().lstrip("."),
        "bytes": path.stat().st_size,
        "profile": raster_profile(path),
    }
    if sample_dir is not None:
        output = sample_dir / "raster" / f"{sample_name}.tif"
        write_raster_sample(path, output, window_size)
        result["sample"] = portable(output, root)
    return result


def vector_profile(path: Path) -> tuple[Any, dict[str, Any]]:
    """Return a GeoDataFrame and structural metadata for one vector layer."""
    import geopandas as gpd

    source: Any = f"zip://{path.resolve()}" if path.suffix.lower() == ".zip" else path
    frame = gpd.read_file(source)
    bounds = frame.total_bounds.tolist() if not frame.empty else [None] * 4
    profile = {
        "feature_count": int(len(frame)),
        "crs": str(frame.crs) if frame.crs else None,
        "geometry_types": sorted(frame.geometry.geom_type.dropna().unique().tolist()),
        "bbox": [json_value(value) for value in bounds],
        "columns": [
            {"name": str(name), "dtype": str(frame[name].dtype)} for name in frame.columns
        ],
    }
    return frame, profile


def inspect_vector(
    path: Path,
    *,
    root: Path,
    sample_dir: Path | None,
    sample_name: str,
    sample_features: int,
) -> dict[str, Any]:
    """Inspect one vector layer and optionally write a tiny GeoJSON fixture."""
    frame, profile = vector_profile(path)
    result: dict[str, Any] = {
        "path": portable(path, root),
        "format": path.suffix.lower().lstrip("."),
        "bytes": path.stat().st_size,
        "profile": profile,
    }
    if sample_dir is not None and not frame.empty:
        indices = evenly_spaced_indices(len(frame), min(sample_features, len(frame)))
        output = sample_dir / "vector" / f"{sample_name}.geojson"
        output.parent.mkdir(parents=True, exist_ok=True)
        frame.iloc[indices].copy().to_file(output, driver="GeoJSON")
        result["sample"] = portable(output, root)
        result["sample_features"] = len(indices)
    return result


def inspect_netcdf(
    path: Path,
    *,
    root: Path,
    sample_dir: Path | None,
    sample_name: str,
    max_dim: int,
) -> dict[str, Any]:
    """Inspect NetCDF in an isolated process and optionally emit a tiny fixture.

    Isolation is intentional: GDAL/rasterio and h5py may load different HDF5
    runtimes on Windows. The dedicated worker imports xarray/h5py independently
    and also repairs surrogate-escaped textual metadata in the generated
    fixture without modifying the source NetCDF.
    """
    import subprocess
    import sys

    output: Path | None = None

    if sample_dir is not None:
        output = sample_dir / "netcdf" / f"{sample_name}.nc"
        output.parent.mkdir(parents=True, exist_ok=True)

    worker = Path(__file__).with_name("_netcdf_catalog_worker.py")

    command = [
        sys.executable,
        str(worker),
        str(path),
        "--max-dim",
        str(max_dim),
    ]

    if output is not None:
        command.extend(["--output", str(output)])

    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )

    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"NetCDF worker failed: {detail}")

    payload = json.loads(completed.stdout)

    result: dict[str, Any] = {
        "path": portable(path, root),
        "format": "netcdf",
        "bytes": path.stat().st_size,
        "profile": payload["profile"],
    }

    repairs = payload.get("metadata_repairs", [])

    if repairs:
        result["metadata_repairs"] = repairs
        result["metadata_repair_count"] = len(repairs)

    if output is not None:
        result["sample"] = portable(output, root)

    return result


def recognized_files(path: Path, *, include_zip: bool = False) -> list[Path]:
    """List recognized data files under a file/directory source path."""
    if path.is_file():
        return [path]
    if not path.is_dir():
        return []
    suffixes = set(RECOGNIZED_SUFFIXES)
    if include_zip:
        suffixes.add(".zip")
    return sorted(
        item for item in path.rglob("*") if item.is_file() and item.suffix.lower() in suffixes
    )


def representative_files(files: list[Path], maximum: int = 3) -> list[Path]:
    """Choose deterministic first/middle/last-like representatives."""
    if len(files) <= maximum:
        return files
    return [files[index] for index in evenly_spaced_indices(len(files), maximum)]


def sample_slug(group_name: str, path: Path, index: int) -> str:
    """Create a stable short name for a generated fixture."""
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem).strip("_")
    return f"{group_name}__{index + 1:02d}__{stem}"


def inspect_file(
    path: Path,
    *,
    group_name: str,
    index: int,
    root: Path,
    sample_dir: Path | None,
    tabular_profile_rows: int,
    sample_rows: int,
    raster_window: int,
    vector_features: int,
    netcdf_max_dim: int,
) -> dict[str, Any]:
    """Dispatch structural inspection by file type, returning errors as metadata."""
    suffix = path.suffix.lower()
    slug = sample_slug(group_name, path, index)
    try:
        if suffix in TABULAR_SUFFIXES:
            return inspect_tabular(
                path,
                root=root,
                sample_limit=tabular_profile_rows,
                sample_rows=sample_rows,
                sample_dir=sample_dir,
                sample_name=slug,
            )
        if suffix in RASTER_SUFFIXES:
            return inspect_raster(
                path,
                root=root,
                sample_dir=sample_dir,
                sample_name=slug,
                window_size=raster_window,
            )
        if suffix in VECTOR_SUFFIXES or (
            suffix == ".zip" and group_name == "official_parcels"
        ):
            return inspect_vector(
                path,
                root=root,
                sample_dir=sample_dir,
                sample_name=slug,
                sample_features=vector_features,
            )
        if suffix in NETCDF_SUFFIXES:
            return inspect_netcdf(
                path,
                root=root,
                sample_dir=sample_dir,
                sample_name=slug,
                max_dim=netcdf_max_dim,
            )
        return {
            "path": portable(path, root),
            "format": suffix.lstrip(".") or "unknown",
            "bytes": path.stat().st_size,
        }
    except Exception as exc:  # pragma: no cover - source/driver dependent.
        return {
            "path": portable(path, root),
            "format": suffix.lstrip(".") or "unknown",
            "bytes": path.stat().st_size,
            "inspection_error": f"{type(exc).__name__}: {exc}",
        }


def group_record(
    group_name: str,
    relative_path: Path,
    *,
    root: Path,
    sample_dir: Path | None,
    tabular_profile_rows: int,
    sample_rows: int,
    raster_window: int,
    vector_features: int,
    netcdf_max_dim: int,
) -> dict[str, Any]:
    """Describe one logical source and inspect a few representative files."""
    absolute = root / relative_path
    files = recognized_files(absolute, include_zip=group_name == "official_parcels")
    extensions = Counter(path.suffix.lower() or "[none]" for path in files)
    representatives: list[Path] = []
    for suffix in sorted(extensions):
        same_type = [path for path in files if (path.suffix.lower() or "[none]") == suffix]
        representatives.extend(representative_files(same_type, maximum=3))
    representatives = sorted(set(representatives))

    profiles = [
        inspect_file(
            path,
            group_name=group_name,
            index=index,
            root=root,
            sample_dir=sample_dir,
            tabular_profile_rows=tabular_profile_rows,
            sample_rows=sample_rows,
            raster_window=raster_window,
            vector_features=vector_features,
            netcdf_max_dim=netcdf_max_dim,
        )
        for index, path in enumerate(representatives)
    ]
    return {
        "source": group_name,
        "path": relative_path.as_posix(),
        "available": bool(files),
        "file_count": len(files),
        "total_bytes": sum(path.stat().st_size for path in files),
        "extensions": dict(sorted(extensions.items())),
        "file_names": [portable(path, root) for path in files],
        "representative_profiles": profiles,
    }


def build_catalog(
    root: Path,
    *,
    source_groups: Mapping[str, Path] = SOURCE_GROUPS,
    sample_dir: Path | None = None,
    tabular_profile_rows: int = 5000,
    sample_rows: int = 48,
    raster_window: int = 32,
    vector_features: int = 8,
    netcdf_max_dim: int = 4,
) -> dict[str, Any]:
    """Build the project-wide data structure catalog."""
    resolved_root = root.resolve()
    resolved_sample_dir = None
    if sample_dir is not None:
        resolved_sample_dir = sample_dir if sample_dir.is_absolute() else resolved_root / sample_dir
        # Reuse the fixture directory instead of deleting it wholesale.
        # OneDrive/Windows may temporarily lock generated subdirectories.
        # Fixture filenames are deterministic and are overwritten in place.
        resolved_sample_dir.mkdir(parents=True, exist_ok=True)

    sources = [
        group_record(
            name,
            relative,
            root=resolved_root,
            sample_dir=resolved_sample_dir,
            tabular_profile_rows=tabular_profile_rows,
            sample_rows=sample_rows,
            raster_window=raster_window,
            vector_features=vector_features,
            netcdf_max_dim=netcdf_max_dim,
        )
        for name, relative in source_groups.items()
    ]
    errors = sum(
        "inspection_error" in profile
        for source in sources
        for profile in source["representative_profiles"]
    )
    return {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "repository_root": ".",
        "purpose": "Structure catalog and tiny real-data fixtures; not model training data.",
        "settings": {
            "tabular_profile_rows": tabular_profile_rows,
            "sample_rows": sample_rows,
            "raster_window": raster_window,
            "vector_features": vector_features,
            "netcdf_max_dim": netcdf_max_dim,
        },
        "summary": {
            "source_count": len(sources),
            "available_source_count": sum(source["available"] for source in sources),
            "inspection_error_count": errors,
        },
        "sources": sources,
    }


def write_catalog(catalog: Mapping[str, Any], output: Path) -> None:
    """Write the catalog as stable readable UTF-8 JSON."""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_sample_readme(sample_dir: Path) -> None:
    """Document the intended use of generated mini fixtures."""
    text = """# Mini muestras reales de datos

Esta carpeta se genera con `python tools/build_data_catalog.py`. Contiene recortes
deterministas y pequenos de los datos reales para desarrollar, depurar y probar loaders y
feature engineering sin leer los archivos completos.

**No son un dataset de entrenamiento**, no preservan distribuciones estadisticas y no deben
usarse para estimar metricas. El contrato completo esta en `data/data_catalog.json`.

Subcarpetas posibles: `tabular/`, `vector/`, `raster/` y `netcdf/`.
"""
    output = sample_dir / "README.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--no-samples", action="store_true")
    parser.add_argument("--tabular-profile-rows", type=int, default=5000)
    parser.add_argument("--sample-rows", type=int, default=48)
    parser.add_argument("--raster-window", type=int, default=32)
    parser.add_argument("--vector-features", type=int, default=8)
    parser.add_argument("--netcdf-max-dim", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    """Build data catalog plus optional tiny fixtures."""
    args = parse_args()
    root = args.root.expanduser().resolve()
    output = args.output.expanduser()
    if not output.is_absolute():
        output = root / output
    sample_dir: Path | None = None if args.no_samples else args.samples.expanduser()
    if sample_dir is not None and not sample_dir.is_absolute():
        sample_dir = root / sample_dir

    catalog = build_catalog(
        root,
        sample_dir=sample_dir,
        tabular_profile_rows=args.tabular_profile_rows,
        sample_rows=args.sample_rows,
        raster_window=args.raster_window,
        vector_features=args.vector_features,
        netcdf_max_dim=args.netcdf_max_dim,
    )
    write_catalog(catalog, output)
    if sample_dir is not None:
        write_sample_readme(sample_dir)

    summary = catalog["summary"]
    print(
        f"Wrote {output} "
        f"({summary['available_source_count']}/{summary['source_count']} sources available; "
        f"{summary['inspection_error_count']} inspection errors)."
    )
    if sample_dir is not None:
        print(f"Mini fixtures: {sample_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
