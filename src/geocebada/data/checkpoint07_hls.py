"""Checkpoint 07A: stream HLS windows into parcel panels and Prithvi chips.

Never load full granules. All outputs are X-only and preserve per-source QA.
The parquet/CSV panel is a longitudinal table, not a supervised training table.
"""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

HLS_RE = re.compile(
    r"^(HLS\.(L30|S30)\.T[^.]+\.(\d{7})T\d{6}\.v2\.0)\.(.+)\.tif$",
    re.IGNORECASE,
)
BANDS = {
    "L30": ("B02", "B03", "B04", "B05", "B06", "B07"),
    "S30": ("B02", "B03", "B04", "B8A", "B11", "B12"),
}
WINDOWS = (
    ("2025-04-15", "2025-05-31"),
    ("2025-06-01", "2025-07-15"),
    ("2025-07-16", "2025-08-31"),
    ("2025-09-01", "2025-10-15"),
)
# HLS Fmask v2: cloud=1, cloud-adjacent=2, shadow=3, snow=4, water=5.
# Water is excluded from the rainfed-field vegetation mask.
QA_REJECT_BITS = (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4) | (1 << 5)
CHIP_SIZE = 224
CHIP_PIXEL_M = 30


def parse_scene_name(name: str) -> tuple[str, str, date, str] | None:
    """Return scene ID, L30/S30, acquisition day and band."""
    match = HLS_RE.match(name)
    if not match:
        return None
    scene, sensor, year_doy, band = match.groups()
    day = datetime.strptime(year_doy, "%Y%j").date()
    return scene, sensor.upper(), day, band


def scan_hls(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Group source files without reading pixels; reject incomplete scenes."""
    scenes: dict[str, dict[str, Any]] = {}
    for path in root.rglob("*.tif"):
        parsed = parse_scene_name(path.name)
        if parsed is None:
            continue
        scene_id, sensor, day, band = parsed
        entry = scenes.setdefault(
            scene_id,
            {"scene_id": scene_id, "sensor": sensor, "date": day, "files": {}},
        )
        if band in entry["files"]:
            raise ValueError(f"Duplicate HLS band for {scene_id}: {band}")
        entry["files"][band] = path

    ready = []
    incomplete = []
    for scene in sorted(scenes.values(), key=lambda s: (s["date"], s["scene_id"])):
        required = (*BANDS[scene["sensor"]], "Fmask")
        missing = sorted(set(required) - set(scene["files"]))
        if missing:
            incomplete.append(
                {"scene_id": scene["scene_id"], "missing_bands": missing}
            )
        else:
            ready.append(scene)
    return ready, incomplete


def clear_land_mask(fmask: np.ndarray, nodata: int = 255) -> np.ndarray:
    """Exclude HLS v2 cloud, adjacency, shadow, snow, water and fill pixels."""
    qa = np.asarray(fmask).astype(np.uint8)
    return (qa != nodata) & ((qa & QA_REJECT_BITS) == 0)


def band_valid(raw: np.ma.MaskedArray, quality: np.ndarray) -> np.ndarray:
    """Preserve negative possible HLS reflectance and remove nodata/saturation."""
    values = np.asarray(raw.data)
    return (
        quality
        & ~np.ma.getmaskarray(raw)
        & (values != 12000)
        & (values > -1000)
        & (values < 10000)
    )


def _stats(prefix: str, values: np.ndarray) -> dict[str, float]:
    if not len(values):
        return {
            f"{prefix}_{item}": float("nan")
            for item in ("mean", "std", "p10", "p50", "p90")
        }
    return {
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_std": float(values.std()),
        f"{prefix}_p10": float(np.quantile(values, 0.1)),
        f"{prefix}_p50": float(np.quantile(values, 0.5)),
        f"{prefix}_p90": float(np.quantile(values, 0.9)),
    }


def summarize_scene_for_parcel(
    scene: dict[str, Any],
    parcel_id: str,
    geometry_wgs84: Any,
) -> dict[str, Any] | None:
    """Read polygon windows from one scene, never whole full-resolution rasters."""
    import rasterio
    from pyproj import Transformer
    from rasterio.errors import WindowError
    from rasterio.features import geometry_mask, geometry_window
    from shapely.geometry import mapping
    from shapely.ops import transform

    with rasterio.open(scene["files"]["Fmask"]) as qa_ds:
        if qa_ds.crs is None:
            raise ValueError(f"Missing raster CRS: {scene['scene_id']}")
        transformer = Transformer.from_crs(
            "EPSG:4326", qa_ds.crs, always_xy=True
        )
        polygon = transform(transformer.transform, geometry_wgs84)
        if polygon.is_empty:
            return None
        try:
            win = geometry_window(qa_ds, [mapping(polygon)], pad_x=0, pad_y=0)
        except (WindowError, ValueError):
            return None
        if win.width <= 0 or win.height <= 0:
            return None
        interior = geometry_mask(
            [mapping(polygon)],
            out_shape=(int(win.height), int(win.width)),
            transform=qa_ds.window_transform(win),
            invert=True,
            all_touched=True,
        )
        n_polygon = int(interior.sum())
        if not n_polygon:
            return None
        qa = qa_ds.read(1, window=win, masked=False)
        good_qa = clear_land_mask(qa, nodata=qa_ds.nodata or 255)
        quality = interior & good_qa
        row: dict[str, Any] = {
            "ID_POLIGONO": str(parcel_id),
            "scene_id": scene["scene_id"],
            "date": scene["date"].isoformat(),
            "sensor": scene["sensor"],
            "n_pixels_polygon": n_polygon,
            "n_pixels_qa_clear": int(quality.sum()),
            "qa_clear_fraction": float(quality.sum() / n_polygon),
        }
        reflectances = []
        combined = quality.copy()
        for semantic, band in zip(
            ("blue", "green", "red", "nir", "swir1", "swir2"),
            BANDS[scene["sensor"]],
            strict=True,
        ):
            with rasterio.open(scene["files"][band]) as ds:
                if ds.crs != qa_ds.crs or ds.transform != qa_ds.transform:
                    raise ValueError(
                        f"Unaligned HLS band {band} in {scene['scene_id']}"
                    )
                raw = ds.read(1, window=win, masked=True)
                valid = band_valid(raw, quality)
                # HLS packed DN=reflectance * 10000. Prithvi uses packed DN.
                # The Local07 panel uses physical reflectance.
                refl = raw.data.astype(np.float32) * 0.0001
                row.update(_stats(semantic, refl[valid]))
                reflectances.append(refl)
                combined &= valid
        row["n_pixels_all_bands_valid"] = int(combined.sum())
        row["valid_fraction"] = float(combined.sum() / n_polygon)
        red, nir, swir1 = (
            reflectances[2],
            reflectances[3],
            reflectances[4],
        )
        ndvi_denom = nir + red
        ndwi_denom = nir + swir1
        ndvi_mask = combined & (np.abs(ndvi_denom) > 1e-6)
        ndwi_mask = combined & (np.abs(ndwi_denom) > 1e-6)
        ndvi = (nir[ndvi_mask] - red[ndvi_mask]) / ndvi_denom[ndvi_mask]
        ndwi = (nir[ndwi_mask] - swir1[ndwi_mask]) / ndwi_denom[ndwi_mask]
        row.update(_stats("ndvi", ndvi))
        row.update(_stats("ndwi_nir_swir1", ndwi))
        return row


def choose_prithvi_scenes(
    rows: list[dict[str, Any]],
    *,
    min_valid_fraction: float = 0.05,
) -> list[dict[str, Any] | None]:
    """Choose one X-only quality-ranked scene per fixed 2025 window."""
    by_window: list[dict[str, Any] | None] = []
    for start, end in WINDOWS:
        candidates = [
            row
            for row in rows
            if start <= str(row["date"]) <= end
            and float(row["valid_fraction"]) >= min_valid_fraction
        ]
        if not candidates:
            by_window.append(None)
            continue
        midpoint = date.fromisoformat(start) + (
            date.fromisoformat(end) - date.fromisoformat(start)
        ) / 2
        by_window.append(
            min(
                candidates,
                key=lambda row: (
                    -float(row["valid_fraction"]),
                    -int(row["n_pixels_all_bands_valid"]),
                    abs((date.fromisoformat(row["date"]) - midpoint).days),
                    str(row["scene_id"]),
                ),
            )
        )
    return by_window


def load_parcels(source: Path, *, max_parcels: int = 0) -> list[tuple[str, Any]]:
    """Load immutable polygons and normalize official truncated ID field."""
    import geopandas as gpd

    gdf = gpd.read_file(source)
    if gdf.crs is None:
        raise ValueError("Parcel source CRS is missing.")
    gdf = gdf.to_crs(4326)
    if len(gdf) != 197:
        raise ValueError(f"Expected 197 official parcels, found {len(gdf)}")
    id_col = "ID_POLIGONO" if "ID_POLIGONO" in gdf else "ID_POLIGON"
    if id_col not in gdf:
        raise ValueError("Official parcel ID field is missing.")
    if gdf[id_col].isna().any() or gdf[id_col].astype(str).duplicated().any():
        raise ValueError("Missing or duplicate official parcel IDs.")
    if gdf.geometry.isna().any() or gdf.geometry.is_empty.any():
        raise ValueError("Missing/empty official parcel polygon.")
    parcels = sorted(
        [(str(i), geom) for i, geom in zip(gdf[id_col], gdf.geometry, strict=True)]
    )
    return parcels[:max_parcels] if max_parcels else parcels


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", value)


def _write_panel_merged(shards: Path, target: Path) -> dict[str, int]:
    """Merge on-disk scene shards into one compact CSV without retaining all rows."""
    paths = sorted(shards.glob("*.json"))
    target.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    ids: set[str] = set()
    fieldnames: list[str] | None = None
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer: csv.DictWriter | None = None
        for path in paths:
            rows = json.loads(path.read_text(encoding="utf-8"))
            for row in rows:
                if writer is None:
                    fieldnames = list(row)
                    writer = csv.DictWriter(handle, fieldnames=fieldnames)
                    writer.writeheader()
                assert writer is not None
                writer.writerow(row)
                row_count += 1
                ids.add(str(row["ID_POLIGONO"]))
    return {
        "scene_shards": len(paths),
        "panel_rows": row_count,
        "parcels_with_observations": len(ids),
    }


def run_panel(
    scenes: list[dict[str, Any]],
    parcels: list[tuple[str, Any]],
    output: Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Checkpoint 07A2 parcel-date raster feature extraction, resumable per scene."""
    shards = output / "hls_scene_shards"
    shards.mkdir(parents=True, exist_ok=True)
    written = 0
    reused = 0
    for index, scene in enumerate(scenes, start=1):
        dst = shards / (scene["scene_id"] + ".json")
        if dst.is_file() and not force:
            reused += 1
            continue
        rows = []
        for parcel_id, polygon in parcels:
            row = summarize_scene_for_parcel(scene, parcel_id, polygon)
            if row is not None:
                rows.append(row)
        tmp = dst.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(rows, allow_nan=True), encoding="utf-8")
        tmp.replace(dst)
        written += 1
        if index % 10 == 0 or index == len(scenes):
            print(
                f"[HLS panel {index}/{len(scenes)}] new={written} "
                f"reused={reused} rows_in_scene={len(rows)}",
                flush=True,
            )
    report = _write_panel_merged(
        shards, output / "hls_parcel_observations.csv"
    )
    return {"processed_scenes": written, "reused_scenes": reused, **report}


def _load_panel_by_parcel(path: Path) -> dict[str, list[dict[str, Any]]]:
    by_parcel: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            by_parcel[str(row["ID_POLIGONO"])].append(row)
    return by_parcel


def _prithvi_chip(
    selected: list[dict[str, Any]],
    scenes_by_id: dict[str, dict[str, Any]],
    polygon_wgs84: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Four aligned 224px chips in raw HLS DN, plus valid/parcel masks."""
    import rasterio
    from affine import Affine
    from pyproj import Transformer
    from rasterio.enums import Resampling
    from rasterio.features import geometry_mask
    from rasterio.vrt import WarpedVRT
    from shapely.geometry import mapping
    from shapely.ops import transform

    reference = scenes_by_id[str(selected[0]["scene_id"])]
    with rasterio.open(reference["files"]["Fmask"]) as first:
        crs = first.crs
        if crs is None:
            raise ValueError("Reference HLS scene missing CRS.")
    to_projected = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    projected = transform(to_projected.transform, polygon_wgs84)
    cx, cy = projected.centroid.x, projected.centroid.y
    # Shared georeferenced grid across all 4 frames (no temporal pixel drift).
    affine = (
        Affine.translation(
            cx - (CHIP_SIZE / 2) * CHIP_PIXEL_M,
            cy + (CHIP_SIZE / 2) * CHIP_PIXEL_M,
        )
        * Affine.scale(CHIP_PIXEL_M, -CHIP_PIXEL_M)
    )
    parcel_mask = geometry_mask(
        [mapping(projected)],
        transform=affine,
        out_shape=(CHIP_SIZE, CHIP_SIZE),
        invert=True,
        all_touched=True,
    )
    frames = np.zeros((4, 6, CHIP_SIZE, CHIP_SIZE), dtype=np.int16)
    valid = np.zeros((4, CHIP_SIZE, CHIP_SIZE), dtype=np.uint8)
    for t, item in enumerate(selected):
        scene = scenes_by_id[str(item["scene_id"])]
        with rasterio.open(scene["files"]["Fmask"]) as src:
            with WarpedVRT(
                src, crs=crs, transform=affine, width=CHIP_SIZE,
                height=CHIP_SIZE, resampling=Resampling.nearest,
            ) as vrt:
                qa = vrt.read(1, masked=True)
        pixel_good = clear_land_mask(qa.filled(255)) & ~np.ma.getmaskarray(qa)
        for band_i, band in enumerate(BANDS[scene["sensor"]]):
            with rasterio.open(scene["files"][band]) as src:
                with WarpedVRT(
                    src, crs=crs, transform=affine, width=CHIP_SIZE,
                    height=CHIP_SIZE, resampling=Resampling.bilinear,
                ) as vrt:
                    raw = vrt.read(1, masked=True)
            pixel_good &= band_valid(raw, pixel_good)
            frames[t, band_i] = raw.filled(0).astype(np.int16)
        # Invalid pixels are stored as 0 but are NEVER considered measured zero.
        frames[t, :, ~pixel_good] = 0
        valid[t] = pixel_good.astype(np.uint8)
    return frames, valid, parcel_mask.astype(np.uint8)


def run_chips(
    scenes: list[dict[str, Any]],
    parcels: list[tuple[str, Any]],
    output: Path,
    *,
    min_valid_fraction: float = 0.05,
    force: bool = False,
) -> dict[str, Any]:
    """Save one compressed NPZ per parcel; missing 4-frame sequences are reported."""
    panel_path = output / "hls_parcel_observations.csv"
    if not panel_path.is_file():
        raise FileNotFoundError(
            "Run --stage panel before --stage chips; HLS panel is missing."
        )
    rows_by_parcel = _load_panel_by_parcel(panel_path)
    scenes_by_id = {scene["scene_id"]: scene for scene in scenes}
    target = output / "prithvi_chips"
    target.mkdir(parents=True, exist_ok=True)
    metadata: list[dict[str, Any]] = []
    complete = 0
    incomplete = 0
    for index, (parcel_id, polygon) in enumerate(parcels, 1):
        choices = choose_prithvi_scenes(
            rows_by_parcel.get(parcel_id, []),
            min_valid_fraction=min_valid_fraction,
        )
        row: dict[str, Any] = {
            "ID_POLIGONO": parcel_id,
            "available_windows": int(sum(c is not None for c in choices)),
            "complete": all(c is not None for c in choices),
            "dates": [c["date"] if c else None for c in choices],
            "scene_ids": [c["scene_id"] if c else None for c in choices],
            "selected_valid_fractions": [
                float(c["valid_fraction"]) if c else None for c in choices
            ],
        }
        if not row["complete"]:
            incomplete += 1
            metadata.append(row)
            continue
        selected = [c for c in choices if c is not None]
        if any(c["scene_id"] not in scenes_by_id for c in selected):
            row["complete"] = False
            row["error"] = "selected_scene_missing_from_current_inventory"
            incomplete += 1
            metadata.append(row)
            continue
        filename = target / f"{_safe_name(parcel_id)}.npz"
        if filename.is_file() and not force:
            row["chip_path"] = filename.name
            row["reused"] = True
        else:
            frames, valid, parcel_mask = _prithvi_chip(
                selected, scenes_by_id, polygon
            )
            tmp = filename.with_suffix(".npz.tmp")
            # Passing an already-open file avoids numpy appending .npz to .tmp.
            with tmp.open("wb") as stream:
                np.savez_compressed(
                    stream,
                    hls_dn=frames,
                    valid_mask=valid,
                    parcel_mask=parcel_mask,
                    acquisition_dates=np.asarray(row["dates"], dtype="U10"),
                )
            tmp.replace(filename)
            row["chip_path"] = filename.name
            row["reused"] = False
            row["chip_valid_pixel_fractions"] = (
                valid.mean(axis=(1, 2)).astype(float).tolist()
            )
        metadata.append(row)
        complete += 1
        if index % 10 == 0 or index == len(parcels):
            print(
                f"[Prithvi chips {index}/{len(parcels)}] "
                f"complete={complete} missing_windows={incomplete}",
                flush=True,
            )
    output_path = output / "prithvi_chip_manifest.json"
    output_path.write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return {
        "parcels_with_4_quality_frames": complete,
        "parcels_missing_windows": incomplete,
        "chip_manifest": str(output_path),
    }
