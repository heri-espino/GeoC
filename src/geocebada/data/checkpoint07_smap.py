"""Extract parcel-region SMAP daily soil moisture from SPL3SMP_E v006.

HDF5 groups may omit latitude/longitude coordinate rasters. The NASA NSIDC
EASE-Grid 2.0 Global (EPSG:6933) fixed geometry provides a safe geolocation
fallback ONLY for a validated 1624 x 3856 global-grid shape. PM science fields
have a _pm suffix in v006.

Only sample individual source cells from each daily file; do not read global
soil-moisture arrays into memory. Original HDF5 files remain unchanged.
"""

from __future__ import annotations

import csv
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np

SMAP_DAY = re.compile(r"_(\d{8})_")
GROUPS = (
    "Soil_Moisture_Retrieval_Data_AM",
    "Soil_Moisture_Retrieval_Data_PM",
)

# NSIDC EASE-Grid 2.0 global 9-km geometry, not its 2000x2000 polar variant.
# Source: https://nsidc.org/data/ease (NSIDC EASE Grids guide).
# Use derived spacing from the canonical 3856-column global extent; some
# SMAP user-guide tables report an inconsistent nominal 9,024.31 m.
EASE2_GLOBAL_SHAPE = (1624, 3856)
EASE2_GLOBAL_UL_X = -17367530.44
EASE2_GLOBAL_UL_Y = 7314540.83
EASE2_GLOBAL_RES = 2 * abs(EASE2_GLOBAL_UL_X) / EASE2_GLOBAL_SHAPE[1]


def parse_smap_day(name: str) -> date | None:
    match = SMAP_DAY.search(name)
    if match is None:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y%m%d").date()
    except ValueError:
        return None


def _dataset_scalar(ds: Any, row: int, col: int) -> float:
    """Decode one HDF5 grid cell, honoring fill flags before any scale/offset."""
    raw = float(ds[row, col])
    for key in ("_FillValue", "missing_value"):
        fill = ds.attrs.get(key)
        if fill is not None and np.any(np.isclose(raw, np.asarray(fill, dtype=float))):
            return float("nan")
    if not np.isfinite(raw):
        return float("nan")
    scale = float(np.asarray(ds.attrs.get("scale_factor", 1.0)).flat[0])
    offset = float(np.asarray(ds.attrs.get("add_offset", 0.0)).flat[0])
    return raw * scale + offset


def _read_reference_grid(group: Any) -> tuple[np.ndarray, np.ndarray]:
    """Read optional explicit latitude/longitude arrays if BOTH are present."""
    if "latitude" not in group or "longitude" not in group:
        raise ValueError("SMAP group missing latitude/longitude datasets")
    return np.asarray(group["latitude"][:]), np.asarray(group["longitude"][:])


def _science_field(group: Any, stem: str, suffix: str) -> Any | None:
    """v006 PM names are suffixed with '_pm', unlike corresponding AM fields."""
    candidate_names = (
        (f"{stem}_pm", stem) if suffix == "PM" else (stem, f"{stem}_am")
    )
    for name in candidate_names:
        if name in group:
            return group[name]
    return None


def _parcel_nearest_ease2_global(
    shape: tuple[int, int],
    parcels: list[tuple[str, Any]],
) -> dict[str, tuple[int, int, float]]:
    """Compute nearest source cell in EPSG:6933 without allocating a huge grid.

    The upper-left reference is the global NSIDC EASE2 grid, NOT geographic
    lat/lon degrees. Candidate 3x3 cells are compared by geodesic distance
    to the polygon centroid. Refuse any unknown/subset/polar array shape.
    """
    from pyproj import Geod, Transformer

    if tuple(shape) != EASE2_GLOBAL_SHAPE:
        raise ValueError(
            "Cannot geolocate SMAP without coordinate arrays: expected "
            f"full EASE2 global {EASE2_GLOBAL_SHAPE}, got {tuple(shape)}. "
            "A spatial subset or polar grid requires its own geotransform."
        )
    to_grid = Transformer.from_crs("EPSG:4326", "EPSG:6933", always_xy=True)
    to_geo = Transformer.from_crs("EPSG:6933", "EPSG:4326", always_xy=True)
    geod = Geod(ellps="WGS84")
    result: dict[str, tuple[int, int, float]] = {}
    height, width = shape
    for pid, polygon in parcels:
        longitude, latitude = float(polygon.centroid.x), float(polygon.centroid.y)
        x, y = to_grid.transform(longitude, latitude)
        if not (np.isfinite(x) and np.isfinite(y)):
            raise ValueError(f"Invalid projected parcel centroid for {pid}")
        approx_col = int(np.floor((x - EASE2_GLOBAL_UL_X) / EASE2_GLOBAL_RES))
        approx_row = int(np.floor((EASE2_GLOBAL_UL_Y - y) / EASE2_GLOBAL_RES))
        if not (0 <= approx_col < width and 0 <= approx_row < height):
            raise ValueError(f"Parcel centroid outside SMAP global grid: {pid}")
        nearest: tuple[float, int, int] | None = None
        for row in range(max(0, approx_row - 1), min(height, approx_row + 2)):
            for col in range(max(0, approx_col - 1), min(width, approx_col + 2)):
                cell_x = EASE2_GLOBAL_UL_X + (col + 0.5) * EASE2_GLOBAL_RES
                cell_y = EASE2_GLOBAL_UL_Y - (row + 0.5) * EASE2_GLOBAL_RES
                cell_lon, cell_lat = to_geo.transform(cell_x, cell_y)
                _, _, metres = geod.inv(longitude, latitude, cell_lon, cell_lat)
                candidate = (float(metres), row, col)
                if nearest is None or candidate < nearest:
                    nearest = candidate
        assert nearest is not None
        metres, row, col = nearest
        result[str(pid)] = (row, col, metres / 1000.0)
    return result


def _parcel_lookup_from_group(
    group: Any, parcels: list[tuple[str, Any]], suffix: str,
) -> tuple[dict[str, tuple[int, int, float]], str]:
    """Use geolocation arrays if supplied; else shape-verified fixed grid."""
    science = _science_field(group, "soil_moisture", suffix)
    if science is None:
        raise ValueError(f"SMAP {suffix} group has no soil_moisture field")
    if science.ndim != 2:
        raise ValueError("Expected 2D soil-moisture array.")
    if "latitude" in group and "longitude" in group:
        lat, lon = _read_reference_grid(group)
        if lat.shape != science.shape or lon.shape != science.shape:
            raise ValueError("SMAP coordinate arrays do not match moisture shape")
        return _parcel_nearest_cells(lat, lon, parcels), "coordinate_datasets"
    # No safe fallback for partial geolocation: mixed-format corruption.
    if ("latitude" in group) != ("longitude" in group):
        raise ValueError(f"SMAP {suffix} group has incomplete lat/lon coordinates")
    return _parcel_nearest_ease2_global(science.shape, parcels), "EPSG6933_fixed_global"


def _parcel_nearest_cells(
    latitude: np.ndarray,
    longitude: np.ndarray,
    parcels: list[tuple[str, Any]],
    *,
    search_padding_deg: float = 1.0,
) -> dict[str, tuple[int, int, float]]:
    """Find closest valid 9-km grid cell to each parcel centroid in local subset."""
    from scipy.spatial import cKDTree

    if latitude.shape != longitude.shape or latitude.ndim != 2:
        raise ValueError("SMAP latitude/longitude must be congruent 2D grids.")
    centroids = [
        (str(pid), float(geom.centroid.y), float(geom.centroid.x))
        for pid, geom in parcels
    ]
    min_lat = min(lat for _, lat, _ in centroids) - search_padding_deg
    max_lat = max(lat for _, lat, _ in centroids) + search_padding_deg
    min_lon = min(lon for _, _, lon in centroids) - search_padding_deg
    max_lon = max(lon for _, _, lon in centroids) + search_padding_deg
    region = (
        np.isfinite(latitude)
        & np.isfinite(longitude)
        & (latitude >= min_lat)
        & (latitude <= max_lat)
        & (longitude >= min_lon)
        & (longitude <= max_lon)
    )
    indices = np.column_stack(np.nonzero(region))
    if not len(indices):
        raise RuntimeError("No SMAP grid cells near GeoC parcels.")

    cos_lat = float(np.cos(np.deg2rad(np.mean([v[1] for v in centroids]))))
    coords = np.column_stack(
        [latitude[region].astype(float), longitude[region].astype(float) * cos_lat]
    )
    tree = cKDTree(coords)
    queries = np.asarray(
        [(lat, lon * cos_lat) for _, lat, lon in centroids], dtype=float
    )
    _, nearest_indices = tree.query(queries)
    result = {}
    for (pid, lat, lon), ix in zip(centroids, nearest_indices, strict=True):
        row, col = map(int, indices[int(ix)])
        # Approximate centre-to-centre great-circle distance for diagnostics.
        delta_lat = float(latitude[row, col]) - lat
        delta_lon = (float(longitude[row, col]) - lon) * cos_lat
        distance_km = float(np.hypot(delta_lat, delta_lon) * 111.2)
        result[pid] = (row, col, distance_km)
    return result


def _usable_moisture(
    moisture: float,
    quality_flag: float,
) -> bool:
    """Recommended quality flags 0 or 8; plausible soil moisture 0..0.6."""
    # Bits 0-2: recommended, attempted, successful (all zero for valid).
    return (
        np.isfinite(moisture)
        and 0.0 <= moisture <= 0.6
        and np.isfinite(quality_flag)
        and 0 <= quality_flag < 65534
        and float(quality_flag).is_integer()
        and (int(quality_flag) & 0b111) == 0
    )


def run_smap(
    root: Path,
    parcels: list[tuple[str, Any]],
    output: Path,
    *,
    max_files: int = 0,
) -> dict[str, Any]:
    """Stream 9-km AM and PM measurements to a parcel-date CSV."""
    import h5py

    files = sorted(
        (parse_smap_day(p.name), p)
        for p in root.rglob("*.h5")
        if parse_smap_day(p.name) is not None
    )
    if max_files:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(f"No dated SPL3SMP_E HDF5 files in {root}")
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "hdf5_files": len(files),
        "rows": 0,
        "recommended_quality_rows": 0,
        "unique_grid_cells": {},
        "geolocation_methods": {},
        "available_overpasses": [],
        "missing_science_groups_by_file": 0,
        "source_grid_resolution": "nominal 9 km; global EASE2 EPSG:6933",
    }
    destination = output / "smap_parcel_daily.csv"
    with h5py.File(files[0][1], "r") as first:
        index: dict[str, dict[str, tuple[int, int, float]]] = {}
        for group_name in GROUPS:
            if group_name not in first:
                continue
            suffix = group_name.rsplit("_", 1)[-1]
            lookups, method = _parcel_lookup_from_group(
                first[group_name], parcels, suffix
            )
            index[group_name] = lookups
            report["geolocation_methods"][group_name] = method
            report["available_overpasses"].append(suffix)
            report["unique_grid_cells"][group_name] = len({
                pair[:2] for pair in lookups.values()
            })
        if not index:
            raise ValueError(
                "No global AM/PM soil-moisture groups in first SMAP HDF5."
            )

    fields = [
        "ID_POLIGONO", "date", "overpass", "soil_moisture_m3_m3",
        "retrieval_qual_flag", "recommended_quality", "smap_grid_row",
        "smap_grid_col", "distance_to_cell_center_km", "source_file",
    ]
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for number, (day, path) in enumerate(files, 1):
            with h5py.File(path, "r") as h5:
                for group_name, lookups in index.items():
                    if group_name not in h5:
                        continue
                    group = h5[group_name]
                    suffix = group_name.rsplit("_", 1)[-1]
                    moisture_ds = _science_field(group, "soil_moisture", suffix)
                    quality_ds = _science_field(group, "retrieval_qual_flag", suffix)
                    if moisture_ds is None or quality_ds is None:
                        report["missing_science_groups_by_file"] += 1
                        continue
                    if moisture_ds.shape != quality_ds.shape:
                        raise ValueError(
                            f"Mismatched science/quality grids in {path}: "
                            f"{group_name}"
                        )
                    if any(
                        row >= moisture_ds.shape[0] or col >= moisture_ds.shape[1]
                        for row, col, _ in lookups.values()
                    ):
                        raise ValueError(
                            f"Incompatible SMAP grid shape in {path}: {group_name}"
                        )
                    measurements: dict[tuple[int, int], tuple[float, float]] = {}
                    for row, col in sorted({pair[:2] for pair in lookups.values()}):
                        val = _dataset_scalar(moisture_ds, row, col)
                        flag = _dataset_scalar(quality_ds, row, col)
                        measurements[(row, col)] = (val, flag)
                    for parcel_id, (row, col, distance) in lookups.items():
                        val, flag = measurements[(row, col)]
                        accepted = _usable_moisture(val, flag)
                        writer.writerow({
                            "ID_POLIGONO": parcel_id,
                            "date": day.isoformat(),
                            "overpass": group_name.rsplit("_", 1)[-1],
                            "soil_moisture_m3_m3": val if accepted else "",
                            "retrieval_qual_flag": flag,
                            "recommended_quality": int(accepted),
                            "smap_grid_row": row,
                            "smap_grid_col": col,
                            "distance_to_cell_center_km": distance,
                            "source_file": path.name,
                        })
                        report["rows"] += 1
                        if accepted:
                            report["recommended_quality_rows"] += 1
            if number % 20 == 0 or number == len(files):
                print(
                    f"[SMAP {number}/{len(files)}] "
                    f"parcel-overpass rows={report['rows']}",
                    flush=True,
                )
    if report["rows"] == 0:
        destination.unlink(missing_ok=True)
        raise ValueError(
            "SMAP produced no parcel rows: check HDF5 science field names."
        )
    report["csv"] = str(destination)
    return report
