"""Extract parcel-region SMAP daily soil moisture from huge SPL3SMP_E HDF5s.

Reads full lat/lon grid only ONCE for local lookup; then samples a few cells
per daily file. Raw SMAP archives are never modified or loaded as dense 3D
stacks. The resulting soil moisture is a regional 9-km signal, not field scale.
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


def parse_smap_day(name: str) -> date | None:
    match = SMAP_DAY.search(name)
    if match is None:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y%m%d").date()
    except ValueError:
        return None


def _dataset_scalar(ds: Any, row: int, col: int) -> float:
    value = float(ds[row, col])
    scale = float(ds.attrs.get("scale_factor", 1.0))
    offset = float(ds.attrs.get("add_offset", 0.0))
    return value * scale + offset


def _read_reference_grid(group: Any) -> tuple[np.ndarray, np.ndarray]:
    if "latitude" not in group or "longitude" not in group:
        raise ValueError("SMAP group missing latitude/longitude datasets")
    return np.asarray(group["latitude"][:]), np.asarray(group["longitude"][:])


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
    return (
        np.isfinite(moisture)
        and 0.0 <= moisture <= 0.6
        and quality_flag in (0.0, 8.0)
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
        "source_grid_resolution": "nominal 9 km",
    }
    destination = output / "smap_parcel_daily.csv"
    with h5py.File(files[0][1], "r") as first:
        index: dict[str, dict[str, tuple[int, int, float]]] = {}
        for group_name in GROUPS:
            if group_name not in first:
                continue
            lat, lon = _read_reference_grid(first[group_name])
            index[group_name] = _parcel_nearest_cells(lat, lon, parcels)
            report["unique_grid_cells"][group_name] = len({
                pair[:2] for pair in index[group_name].values()
            })

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
                    if (
                        "soil_moisture" not in group
                        or "retrieval_qual_flag" not in group
                    ):
                        continue
                    measurements: dict[tuple[int, int], tuple[float, float]] = {}
                    for row, col, _ in set(lookups.values()):
                        val = _dataset_scalar(group["soil_moisture"], row, col)
                        flag = _dataset_scalar(
                            group["retrieval_qual_flag"], row, col
                        )
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
    report["csv"] = str(destination)
    return report
