"""Exact polygon-cell overlap climate features and target-free time transforms.

Unlike all_touched raster means, cells are weighted by valid intersected area
in EPSG:6933. This does NOT infer subpixel meteorology.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from geocebada.features.parcel import ID_COLUMN, canonicalize_parcels

CLIMATE_RE = re.compile(r"^(PREC|Tmin|Tmax)_(20\d{2})_(\d{2})\.tif$", re.IGNORECASE)
YEARS = (2022, 2023, 2024, 2025)
MONTHS = (4, 5, 6, 7, 8, 9, 10)
VARIABLES = ("prec", "tmin", "tmax")


def climate_inventory(directory: Path) -> dict[tuple[str, int, int], Path]:
    """Index all mandatory official monthly climate rasters."""
    if not directory.is_dir():
        raise FileNotFoundError(f"Climate directory not found: {directory}")
    files: dict[tuple[str, int, int], Path] = {}
    for path in directory.rglob("*.tif"):
        match = CLIMATE_RE.fullmatch(path.name)
        if match is None:
            continue
        variable, year, month = match.group(1).lower(), int(match.group(2)), int(match.group(3))
        if year in YEARS and month in MONTHS:
            key = (variable, year, month)
            if key in files:
                raise ValueError(f"Ambiguous duplicate monthly raster {key}: {path}")
            files[key] = path
    missing = {(v, y, m) for v in VARIABLES for y in YEARS for m in MONTHS} - files.keys()
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} official climate rasters, e.g. {sorted(missing)[:8]}"
        )
    return dict(sorted(files.items()))


def area_weighted_climate_raster(parcels, path: Path) -> pd.DataFrame:
    """Exact area-weighted and like-for-like all-touched pixel averages.

    Zero-area boundary touches do not contribute to either estimator. The
    current pipeline uses all_touched=True, but not intersection weights.
    """
    try:
        import rasterio
        from pyproj import Transformer
        from rasterio.mask import mask
        from shapely.geometry import Polygon, mapping
        from shapely.ops import transform
    except ImportError as exc:
        raise ImportError("Install the geo extra to extract climate rasters.") from exc

    geo = canonicalize_parcels(parcels)[[ID_COLUMN, "geometry"]].copy()
    if geo.crs is None:
        raise ValueError("Parcel CRS must be declared and verified.")
    if geo[ID_COLUMN].astype(str).duplicated().any():
        raise ValueError("Duplicate parcel ID in vector geometry.")
    rows = []
    with rasterio.open(path) as raster:
        if raster.crs is None:
            raise ValueError(f"Raster CRS unspecified: {path}")
        grid_geo = geo.to_crs(raster.crs)
        project = Transformer.from_crs(raster.crs, "EPSG:6933", always_xy=True).transform
        for record in grid_geo.itertuples(index=False):
            pid = str(getattr(record, ID_COLUMN))
            polygon = record.geometry
            if polygon is None or polygon.is_empty or not polygon.is_valid:
                raise ValueError(f"Missing or invalid polygon: {pid}")
            parcel_area = float(transform(project, polygon).area)
            if not np.isfinite(parcel_area) or parcel_area <= 0:
                raise ValueError(f"Invalid equal-area parcel geometry: {pid}")
            try:
                band, affine = mask(
                    raster, [mapping(polygon)], crop=True, filled=False,
                    all_touched=True, indexes=1,
                )
            except ValueError:
                rows.append({
                    ID_COLUMN: pid, "weighted_mean": np.nan, "unweighted_mean": np.nan,
                    "coverage": 0., "n_cells": 0, "effective_cells": 0.,
                })
                continue
            values = np.asarray(band.data, dtype=float)
            valid = (~np.ma.getmaskarray(band)) & np.isfinite(values)
            areas, samples = [], []
            for row, col in np.argwhere(valid):
                cell = Polygon([
                    affine * (col, row), affine * (col + 1, row),
                    affine * (col + 1, row + 1), affine * (col, row + 1),
                ])
                intersection = polygon.intersection(cell)
                if intersection.is_empty:
                    continue
                area = float(transform(project, intersection).area)
                if np.isfinite(area) and area > 0:
                    areas.append(area)
                    samples.append(float(values[row, col]))
            if not areas:
                rows.append({
                    ID_COLUMN: pid, "weighted_mean": np.nan, "unweighted_mean": np.nan,
                    "coverage": 0., "n_cells": 0, "effective_cells": 0.,
                })
                continue
            weights = np.asarray(areas, dtype=float)
            samples_array = np.asarray(samples, dtype=float)
            weights = weights / weights.sum()
            rows.append({
                ID_COLUMN: pid,
                "weighted_mean": float(np.dot(weights, samples_array)),
                "unweighted_mean": float(samples_array.mean()),
                "coverage": float(min(1., sum(areas) / parcel_area)),
                "n_cells": int(len(samples)),
                "effective_cells": float(1. / np.square(weights).sum()),
            })
    return pd.DataFrame(rows).sort_values(ID_COLUMN).reset_index(drop=True)


def temporal_features(frame: pd.DataFrame, *, prefix: str) -> pd.DataFrame:
    """Monthly precipitation timing, heat proxies and historical anomalies."""
    result = frame[[ID_COLUMN]].copy()
    rain = np.column_stack([
        frame[f"{prefix}__prec__2025_m{m:02d}"] for m in MONTHS
    ]).astype(float)
    tmin = np.column_stack([
        frame[f"{prefix}__tmin__2025_m{m:02d}"] for m in MONTHS
    ]).astype(float)
    tmax = np.column_stack([
        frame[f"{prefix}__tmax__2025_m{m:02d}"] for m in MONTHS
    ]).astype(float)
    tmean = (tmin + tmax) / 2.
    days = np.array([30, 31, 30, 31, 31, 30, 31], dtype=float)
    expressions = {
        "rainfall_early_sum": rain[:, :3].sum(axis=1),
        "rainfall_mid_sum": rain[:, 2:5].sum(axis=1),
        "rainfall_late_sum": rain[:, 4:].sum(axis=1),
        "rainfall_season_sum": rain.sum(axis=1),
        "rainfall_monthly_sd": rain.std(axis=1),
        "rainfall_wettest_month": rain.max(axis=1),
        "rainfall_driest_month": rain.min(axis=1),
        "rainfall_late_fraction": rain[:, 4:].sum(axis=1) / (rain.sum(axis=1) + 1.),
        "rainfall_max_2month": np.stack([
            rain[:, i] + rain[:, i + 1] for i in range(6)
        ], axis=1).max(axis=1),
        "temperature_mean_season": tmean.mean(axis=1),
        "temperature_diurnal_range": (tmax - tmin).mean(axis=1),
        "temperature_max_monthly_mean": tmean.max(axis=1),
        "gdd_monthly_proxy_base0": (np.maximum(tmean, 0.) * days).sum(axis=1),
        "gdd_monthly_proxy_base5": (np.maximum(tmean - 5., 0.) * days).sum(axis=1),
        "gdd_monthly_proxy_base10": (np.maximum(tmean - 10., 0.) * days).sum(axis=1),
    }
    for name, series in expressions.items():
        result[f"{prefix}__{name}"] = series
    for variable in VARIABLES:
        for month in MONTHS:
            current = frame[f"{prefix}__{variable}__2025_m{month:02d}"]
            historical = pd.concat([
                frame[f"{prefix}__{variable}__{year}_m{month:02d}"]
                for year in YEARS[:-1]
            ], axis=1).mean(axis=1)
            result[f"{prefix}__{variable}_anomaly_m{month:02d}"] = current - historical
    return result


def build_climate_representations(
    parcels, sources: dict[tuple[str, int, int], Path],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Weighted, unweighted, and temporal blocks, with extraction quality."""
    geo = canonicalize_parcels(parcels)
    ids = geo[ID_COLUMN].astype(str).sort_values().tolist()
    weighted = pd.DataFrame({ID_COLUMN: ids})
    unweighted = pd.DataFrame({ID_COLUMN: ids})
    diagnostics = []
    for (variable, year, month), path in sorted(sources.items()):
        pixels = area_weighted_climate_raster(geo, path)
        name = f"{variable}__{year}_m{month:02d}"
        wx = f"area_climate__{name}"
        ux = f"touched_climate__{name}"
        indexed = pixels.set_index(ID_COLUMN)
        weighted[wx] = weighted[ID_COLUMN].map(indexed["weighted_mean"])
        unweighted[ux] = unweighted[ID_COLUMN].map(indexed["unweighted_mean"])
        if year == 2025:
            for row in pixels.itertuples(index=False):
                diagnostics.append({
                    ID_COLUMN: str(getattr(row, ID_COLUMN)),
                    "variable": variable, "month": month,
                    "coverage": row.coverage, "n_cells": row.n_cells,
                    "effective_cells": row.effective_cells,
                    "absolute_extraction_difference": abs(
                        row.weighted_mean - row.unweighted_mean
                    ),
                })
    temporal = temporal_features(weighted, prefix="area_climate")
    return weighted, unweighted, temporal, pd.DataFrame(diagnostics)
