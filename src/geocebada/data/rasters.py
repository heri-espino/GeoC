"""Raster discovery, filename parsing and lightweight metadata helpers."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from geocebada.data.files import discover_files

_RASTER_RE = re.compile(
    r"^(?P<variable>PREC|Tmin|Tmax)_(?P<year>\d{4})_(?P<month>\d{2})\.tif$",
    flags=re.IGNORECASE,
)


def discover_geotiffs(directory: str | Path, *, recursive: bool = True) -> list[Path]:
    """Return sorted GeoTIFF files below ``directory``."""

    return discover_files(directory, pattern="*.tif", recursive=recursive)


def parse_raster_name(path: str | Path) -> dict[str, str | int] | None:
    """Parse official climate raster names such as ``PREC_2022_01.tif``.

    Returns ``None`` for filenames that do not follow the confirmed CHIRPS /
    CHIRTS-ERA5 naming convention.
    """

    name = Path(path).name
    match = _RASTER_RE.match(name)
    if match is None:
        return None

    variable = match.group("variable")
    canonical = {"prec": "PREC", "tmin": "Tmin", "tmax": "Tmax"}[variable.lower()]
    return {
        "variable": canonical,
        "year": int(match.group("year")),
        "month": int(match.group("month")),
    }


def build_raster_inventory(
    directory: str | Path,
    *,
    recursive: bool = True,
    include_unparsed: bool = False,
) -> pd.DataFrame:
    """Build a tabular inventory of GeoTIFF files.

    The returned columns are ``path``, ``name``, ``variable``, ``year`` and
    ``month``. This function reads filenames only and does not load raster data.
    """

    rows: list[dict[str, Any]] = []
    for path in discover_geotiffs(directory, recursive=recursive):
        parsed = parse_raster_name(path)
        if parsed is None and not include_unparsed:
            continue
        rows.append(
            {
                "path": path,
                "name": path.name,
                "variable": None if parsed is None else parsed["variable"],
                "year": None if parsed is None else parsed["year"],
                "month": None if parsed is None else parsed["month"],
            }
        )

    return pd.DataFrame(rows, columns=["path", "name", "variable", "year", "month"])


def raster_metadata(path: str | Path) -> dict[str, Any]:
    """Return core metadata for one raster without reading its full pixel array.

    ``rasterio`` is an optional dependency installed with ``geocebada[geo]``.
    """

    try:
        import rasterio
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "raster_metadata requires rasterio. Install GeoCebada with `pip install -e '.[geo]'`."
        ) from exc

    raster_path = Path(path).expanduser().resolve()
    if not raster_path.is_file():
        raise FileNotFoundError(raster_path)

    with rasterio.open(raster_path) as dataset:
        return {
            "path": raster_path,
            "crs": None if dataset.crs is None else dataset.crs.to_string(),
            "width": dataset.width,
            "height": dataset.height,
            "count": dataset.count,
            "dtypes": tuple(dataset.dtypes),
            "nodata": dataset.nodata,
            "resolution": tuple(dataset.res),
            "bounds": tuple(dataset.bounds),
        }
