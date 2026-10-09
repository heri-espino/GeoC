#!/usr/bin/env python3
"""Inventory Checkpoint 07 raw acquisitions without reading raster pixels.

This utility is safe to run while Sentinel-1 is downloading. It walks directory
entries and optionally inspects a few raster headers; it does not load 180 GB
into memory, delete raw inputs, train models, or access hidden FIRA yields.

Usage:
    python tools/audit_checkpoint_07_storage.py
    python tools/audit_checkpoint_07_storage.py --sample-raster-headers 8
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = ROOT / "data" / "raw" / "checkpoint_07"
DEFAULT_MODELS = ROOT / "models" / "checkpoint_07"
DEFAULT_REPORT = ROOT / "reports" / "checkpoint_07" / "storage_inventory.json"

# HLS L30 and S30 have sensor-specific band IDs for NIR and SWIR.
HLS_REQUIRED = {
    "L30": {"B02", "B03", "B04", "B05", "B06", "B07", "Fmask"},
    "S30": {"B02", "B03", "B04", "B8A", "B11", "B12", "Fmask"},
}
HLS_PATTERN = re.compile(
    r"^(HLS\.(L30|S30)\.T[^.]+\.\d{7}T\d{6}\.v2\.0)\.(.+)\.tif$",
    re.IGNORECASE,
)


def _gb(value: int) -> float:
    return round(value / (1024**3), 3)


def _iter_files(root: Path):
    if root.is_dir():
        yield from (path for path in root.rglob("*") if path.is_file())


def _hls_band_inventory(files: list[Path]) -> dict[str, Any]:
    granules: dict[str, set[str]] = defaultdict(set)
    sensor_by_granule: dict[str, str] = {}
    unmatched = 0
    for path in files:
        match = HLS_PATTERN.match(path.name)
        if match is None:
            unmatched += 1
            continue
        granule, sensor, band = match.groups()
        granules[granule].add(band)
        sensor_by_granule[granule] = sensor.upper()

    missing_scenes: list[dict[str, Any]] = []
    counts = Counter(sensor_by_granule.values())
    for granule, bands in sorted(granules.items()):
        missing = HLS_REQUIRED[sensor_by_granule[granule]] - bands
        if missing:
            missing_scenes.append({"granule": granule, "missing": sorted(missing)})
    return {
        "parsed_granules": len(granules),
        "granules_by_sensor": dict(sorted(counts.items())),
        "granules_missing_prithvi_bands": len(missing_scenes),
        "missing_examples": missing_scenes[:12],
        "unmatched_filename_count": unmatched,
        "note": (
            "Presence of all seven required files is not proof of valid "
            "pixels, radiometry, or cloud-free parcel coverage."
        ),
    }


def _file_inventory(root: Path) -> tuple[dict[str, Any], list[Path]]:
    source_dirs: dict[str, dict[str, Any]] = {}
    hls_files: list[Path] = []
    top_large: list[dict[str, Any]] = []
    grand_bytes = 0
    grand_files = 0
    if not root.is_dir():
        return {
            "exists": False,
            "total_files": 0,
            "total_gib": 0,
            "sources": {},
            "largest_files": [],
        }, []

    for directory in sorted(root.iterdir()):
        if not directory.is_dir():
            continue
        count = 0
        total_bytes = 0
        for path in _iter_files(directory):
            info = path.stat()
            count += 1
            total_bytes += info.st_size
            if directory.name == "hls_v2" and path.suffix.lower() == ".tif":
                hls_files.append(path)
            relative = path.relative_to(root).as_posix()
            item = {"path": relative, "gib": _gb(info.st_size), "bytes": info.st_size}
            top_large.append(item)
            if len(top_large) > 24:
                top_large.sort(key=lambda entry: entry["bytes"], reverse=True)
                top_large.pop()

        source_dirs[directory.name] = {"files": count, "gib": _gb(total_bytes)}
        grand_files += count
        grand_bytes += total_bytes

    top_large.sort(key=lambda entry: entry["bytes"], reverse=True)
    for entry in top_large:
        entry.pop("bytes")
    return {
        "exists": True,
        "total_files": grand_files,
        "total_gib": _gb(grand_bytes),
        "sources": source_dirs,
        "largest_files": top_large[:12],
    }, hls_files


def _raster_headers(paths: list[Path], sample_size: int) -> list[dict[str, Any]]:
    if sample_size == 0:
        return []
    try:
        import rasterio
    except ImportError as exc:
        raise RuntimeError("Rasterio is required to inspect optional headers.") from exc
    entries: list[dict[str, Any]] = []
    for path in paths[:sample_size]:
        try:
            with rasterio.open(path) as ds:
                entries.append({
                    "name": path.name,
                    "shape": [ds.height, ds.width],
                    "dtype": ds.dtypes[0],
                    "crs": str(ds.crs),
                    "nodata": ds.nodata,
                    "tiled": ds.is_tiled,
                    "block_shape": list(ds.block_shapes[0]),
                })
        except Exception as exc:
            entries.append({"name": path.name, "error": type(exc).__name__})
    return entries


def build_inventory(
    raw: Path,
    models: Path,
    *,
    sample_raster_headers: int = 0,
) -> dict[str, Any]:
    raw_stats, hls_files = _file_inventory(raw)
    model_stats, _ = _file_inventory(models)
    if raw.exists():
        disk = shutil.disk_usage(raw)
    else:
        disk = shutil.disk_usage(raw.parent if raw.parent.exists() else ROOT)

    status_path = raw / "acquisition_manifests" / "source_status.json"
    try:
        source_status = (
            json.loads(status_path.read_text(encoding="utf-8"))
            if status_path.is_file() else {}
        )
    except (ValueError, OSError):
        source_status = {"error": "could_not_read_source_status"}

    selected = sorted(hls_files, key=lambda p: p.name)
    if selected and sample_raster_headers:
        # Inspect multiple dates/sensors rather than many bands from one granule.
        sample = [
            selected[index]
            for index in range(
                0,
                len(selected),
                max(1, len(selected) // sample_raster_headers),
            )
        ][:sample_raster_headers]
    else:
        sample = []

    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "raw_root": str(raw),
        "models_root": str(models),
        "raw": raw_stats,
        "model_weights": model_stats,
        "disk_free_gib": _gb(disk.free),
        "disk_total_gib": _gb(disk.total),
        "source_status": source_status,
        "hls": _hls_band_inventory(hls_files),
        "sample_raster_headers": _raster_headers(sample, sample_raster_headers),
        "rules": [
            "Read raster windows, never load complete scenes into memory.",
            "Do not delete raw files until derived data QA and checksums pass.",
            "Status 'complete' means the downloader exited, not that coverage is valid.",
            "Status 'running' alone cannot prove the process is still alive.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--models", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--sample-raster-headers", type=int, default=0)
    args = parser.parse_args()
    if args.sample_raster_headers < 0 or args.sample_raster_headers > 50:
        parser.error("--sample-raster-headers must be between 0 and 50")
    payload = build_inventory(
        args.raw,
        args.models,
        sample_raster_headers=args.sample_raster_headers,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    print("Checkpoint 07 storage inventory: DONE")
    for source, details in payload["raw"]["sources"].items():
        print(f"  {source}: {details['gib']} GiB, {details['files']} files")
    print(f"  total raw: {payload['raw']['total_gib']} GiB")
    print(f"  model weights: {payload['model_weights']['total_gib']} GiB")
    print(f"  disk free: {payload['disk_free_gib']} GiB")
    print(f"  HLS parsed granules: {payload['hls']['parsed_granules']}")
    print(f"  HLS scenes with missing 6-band+Fmask files: "
          f"{payload['hls']['granules_missing_prithvi_bands']}")
    print(f"  report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
