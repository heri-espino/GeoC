#!/usr/bin/env python
"""Build a versionable inventory of locally available external data.

The data under ``data/raw/`` remain local and ignored by Git. This utility
records their reproducibility metadata—paths, sizes, SHA256 checksums, inferred
dates and lightweight geospatial metadata—in ``data/external_manifest.json``.

Examples
--------
Build the standard inventory::

    python tools/build_external_data_manifest.py

Inspect another checkout without changing the default output location::

    python tools/build_external_data_manifest.py --root C:\\path\\to\\GeoCebada
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = Path("data/external_manifest.json")

SOURCE_PATHS = {
    "inegi_municipios": Path("data/raw/external/inegi_municipios"),
    "chirps_diario": Path("data/raw/external/chirps_v3_daily"),
    "wapor": Path("data/raw/external/wapor_v3"),
    "soilgrids": Path("data/raw/external/soilgrids_250m"),
    "siap_local": Path("data/raw/external/SIAP"),
    "inegi_cem_local": Path("data/raw/external/Inegi"),
}

DATE_PATTERN = re.compile(r"(?<!\d)(20\d{2})-(\d{2})-(\d{2})(?!\d)")
YEAR_PATTERN = re.compile(r"(?<!\d)(20\d{2})(?!\d)")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA256 checksum of ``path`` without loading it all at once."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def infer_filename_date(path: Path) -> str | None:
    """Infer an ISO date from a filename, using January 1 for year-only names."""
    date_match = DATE_PATTERN.search(path.name)
    if date_match:
        return date_match.group(0)
    year_match = YEAR_PATTERN.search(path.name)
    return f"{year_match.group(1)}-01-01" if year_match else None


def raster_metadata(path: Path) -> dict[str, Any] | None:
    """Return inexpensive GeoTIFF metadata, or ``None`` if the file is not readable."""
    try:
        import rasterio
    except ImportError:
        return {"inspection_error": "rasterio is not installed"}

    try:
        with rasterio.open(path) as dataset:
            bounds = dataset.bounds
            return {
                "driver": dataset.driver,
                "crs": str(dataset.crs) if dataset.crs else None,
                "dimensions": [dataset.width, dataset.height],
                "band_count": dataset.count,
                "bbox": [bounds.left, bounds.bottom, bounds.right, bounds.top],
                "resolution": [dataset.res[0], dataset.res[1]],
            }
    except Exception as exc:  # pragma: no cover - depends on local GDAL drivers.
        return {"inspection_error": f"{type(exc).__name__}: {exc}"}


def netcdf_metadata(path: Path) -> dict[str, Any] | None:
    """Return NetCDF subdataset metadata when GDAL's NetCDF driver is available."""
    try:
        from osgeo import gdal
    except ImportError:
        return {"inspection_error": "GDAL Python bindings are not installed"}

    try:
        gdal.DontUseExceptions()
        dataset = gdal.OpenEx(str(path), gdal.OF_RASTER)
        if dataset is None:
            return {"inspection_error": "GDAL could not open the NetCDF file"}
        subdatasets = dataset.GetMetadata("SUBDATASETS")
        descriptions = [
            value
            for key, value in sorted(subdatasets.items())
            if key.endswith("_DESC")
        ]
        dimensions = sorted(
            {
                match.group(1)
                for description in descriptions
                if (match := re.search(r"\[([^]]+)\]", description))
            }
        )
        variables = [description.rsplit(" ", 1)[-1] for description in descriptions]
        return {
            "driver": dataset.GetDriver().ShortName,
            "root_dimensions": [dataset.RasterXSize, dataset.RasterYSize],
            "root_band_count": dataset.RasterCount,
            "subdataset_count": len(descriptions),
            "subdataset_dimensions": dimensions,
            "variables": variables,
        }
    except Exception as exc:  # pragma: no cover - depends on local GDAL drivers.
        return {"inspection_error": f"{type(exc).__name__}: {exc}"}


def file_record(path: Path, repo_root: Path) -> dict[str, Any]:
    """Build a portable metadata record for one local external-data file."""
    suffix = path.suffix.lower() or None
    record: dict[str, Any] = {
        "path": path.relative_to(repo_root).as_posix(),
        "bytes": path.stat().st_size,
        "extension": suffix,
        "sha256": sha256_file(path),
    }
    if inferred_date := infer_filename_date(path):
        record["date_inferred"] = inferred_date
    if suffix in {".tif", ".tiff"}:
        record["raster"] = raster_metadata(path)
    elif suffix == ".nc":
        record["netcdf"] = netcdf_metadata(path)
    return record


def source_record(source: str, relative_path: Path, repo_root: Path) -> dict[str, Any]:
    """Summarize one known external source, including an empty-source record."""
    directory = repo_root / relative_path
    portable_path = relative_path.as_posix()
    if not directory.is_dir():
        return {
            "source": source,
            "path": portable_path,
            "available": False,
            "file_count": 0,
            "total_bytes": 0,
            "files": [],
        }

    files = sorted(path for path in directory.rglob("*") if path.is_file())
    records = [file_record(path, repo_root) for path in files]
    dates = sorted(record["date_inferred"] for record in records if "date_inferred" in record)
    extensions = Counter(record["extension"] or "[none]" for record in records)
    result: dict[str, Any] = {
        "source": source,
        "path": portable_path,
        "available": True,
        "file_count": len(records),
        "total_bytes": sum(record["bytes"] for record in records),
        "extensions": dict(sorted(extensions.items())),
        "files": records,
    }
    if dates:
        result["date_range_inferred"] = {"min": dates[0], "max": dates[-1]}
    return result


def build_manifest(
    repo_root: Path,
    source_paths: Mapping[str, Path] = SOURCE_PATHS,
) -> dict[str, Any]:
    """Return a manifest for the known external-data sources under ``repo_root``."""
    resolved_root = repo_root.resolve()
    return {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "repository_root": ".",
        "sources": [
            source_record(source, relative_path, resolved_root)
            for source, relative_path in source_paths.items()
        ],
    }


def write_manifest(manifest: Mapping[str, Any], output: Path) -> None:
    """Write ``manifest`` as stable, readable UTF-8 JSON."""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    """Parse manifest-builder command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="GeoCebada repository root.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Manifest path, relative to --root unless absolute.",
    )
    return parser.parse_args()


def main() -> int:
    """Build and write the external-data manifest."""
    args = parse_args()
    repo_root = args.root.expanduser().resolve()
    output = args.output.expanduser()
    if not output.is_absolute():
        output = repo_root / output
    manifest = build_manifest(repo_root)
    write_manifest(manifest, output)
    available = sum(source["available"] for source in manifest["sources"])
    print(f"Wrote {output} ({available}/{len(manifest['sources'])} sources available).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
