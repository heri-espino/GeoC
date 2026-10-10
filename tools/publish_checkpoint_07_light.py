#!/usr/bin/env python3
"""Prepare reproducible lightweight Checkpoint 07 snapshots for optional Git LFS.

The script NEVER stages, commits, pushes, or reads hidden parcel yields.
It streams approved X-only CSVs to deterministic gzip files with size caps.
Metadata reports contain no absolute workstation paths.

PowerShell:
    python tools/publish_checkpoint_07_light.py --check
    python tools/publish_checkpoint_07_light.py --build
    python tools/publish_checkpoint_07_light.py --verify
    python tools/publish_checkpoint_07_light.py --build --include-highdim

The user separately reviews/stages ONLY the generated allowlisted paths.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RAW_PROCESSED = Path("data/processed/checkpoint_07")
X_FULL = Path("reports/checkpoint_07/highdim/expanded_parcel_features.csv")
PORTABLE_DATA = Path("data/processed/checkpoint_07_compact")
PORTABLE_REPORTS = Path("reports/checkpoint_07/shared")

BASE_INPUTS = {
    "hls_parcel_observations.csv.gz":
        RAW_PROCESSED / "hls_parcel_observations.csv",
    "smap_parcel_daily.csv.gz":
        RAW_PROCESSED / "smap_parcel_daily.csv",
}
HIGH_DIM_NAME = "expanded_parcel_features.csv.gz"
MANIFEST_NAME = "checkpoint07_portable_manifest.json"
QC_NAME = "checkpoint07_quality_summary.json"
SMOKE_NAME = "checkpoint07_smoke_benchmark.csv"

FORBIDDEN_EXACT = {
    "RENDIMIENTO_T_HA", "CONJUNTO", "RENDIMIENTO", "PRODUCCION_T",
    "TARGET", "YIELD",
}


def sources(include_highdim: bool) -> dict[str, Path]:
    items = dict(BASE_INPUTS)
    if include_highdim:
        items[HIGH_DIM_NAME] = X_FULL
    return items


def _check_header(src: Path) -> tuple[int, list[str]]:
    if not src.is_file():
        raise FileNotFoundError(f"Missing input {src}. Process locally first.")
    with src.open("rb") as raw:
        if raw.read(48).startswith(b"version https://git-lfs"):
            raise ValueError(f"Git LFS pointer instead of a real source: {src}")
    with src.open("r", encoding="utf-8-sig", newline="") as stream:
        header = next(csv.reader(stream), [])
    if "ID_POLIGONO" not in header:
        raise ValueError(f"Missing ID_POLIGONO in {src}")
    banned = [
        col for col in header
        if col.upper() in FORBIDDEN_EXACT
        or "siap_2025" in col.lower()
        or col.lower().startswith(("target_", "yield_"))
    ]
    if banned:
        raise ValueError(f"Not safe to share {src}: banned columns {banned[:10]}")
    if len(header) != len(set(header)):
        raise ValueError(f"Duplicate source columns in {src}")
    return len(header), header


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _compress(
    src: Path,
    destination: Path,
    *,
    max_bytes: int,
) -> dict[str, Any]:
    source_digest = hashlib.sha256()
    source_size = 0
    with src.open("rb") as stream, destination.open("wb") as out:
        # mtime=0, filename='' makes output deterministic across computers.
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=out, compresslevel=9, mtime=0
        ) as compressor:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                source_size += len(chunk)
                source_digest.update(chunk)
                compressor.write(chunk)
    compressed_size = destination.stat().st_size
    if compressed_size > max_bytes:
        raise ValueError(
            f"Compressed {src.name}: {compressed_size / (1024**2):.2f} MiB "
            f"exceeds cap {max_bytes / (1024**2):.1f} MiB. "
            "Keep it local or explicitly raise --max-file-mib."
        )
    return {
        "name": destination.name,
        "source": src.relative_to(ROOT).as_posix(),
        "relative_path": (PORTABLE_DATA / destination.name).as_posix(),
        "source_bytes": source_size,
        "compressed_bytes": compressed_size,
        "source_sha256": source_digest.hexdigest(),
        "compressed_sha256": _sha256_file(destination),
    }


def _summary() -> dict[str, Any]:
    hls_path = ROOT / RAW_PROCESSED / "processing_report.json"
    smap_path = ROOT / RAW_PROCESSED / "smap_processing_report.json"
    quality: dict[str, Any] = {
        "note": "X-only processing summary; no parcel-level yield labels",
        "hls": {},
        "smap": {},
        "highdim_smoke": {},
    }
    if hls_path.is_file():
        hls = json.loads(hls_path.read_text(encoding="utf-8"))
        panel = hls.get("panel") or {}
        chips = hls.get("chips") or {}
        quality["hls"] = {
            "source_scenes": hls.get("source_scenes"),
            "source_parcels": hls.get("source_parcels"),
            "panel_rows": panel.get("panel_rows"),
            "parcels_with_observations": panel.get("parcels_with_observations"),
            "parcels_with_4_quality_frames":
                chips.get("parcels_with_4_quality_frames"),
            "parcels_missing_windows": chips.get("parcels_missing_windows"),
        }
    if smap_path.is_file():
        smap = json.loads(smap_path.read_text(encoding="utf-8"))
        quality["smap"] = {
            key: smap.get(key)
            for key in (
                "hdf5_files", "rows", "recommended_quality_rows",
                "unique_grid_cells", "geolocation_methods",
                "available_overpasses", "missing_science_groups_by_file",
                "source_grid_resolution",
            )
        }
        n = smap.get("rows")
        good = smap.get("recommended_quality_rows")
        if isinstance(n, int) and isinstance(good, int) and n > 0:
            quality["smap"]["recommended_quality_fraction"] = good / n
    smoke = ROOT / "reports/checkpoint_07/highdim/smoke/run_report.json"
    if smoke.is_file():
        payload = json.loads(smoke.read_text(encoding="utf-8"))
        eval_data = payload.get("evaluation") or {}
        quality["highdim_smoke"] = {
            "smoke": True,
            "n_splits": eval_data.get("n_splits"),
            "n_models": eval_data.get("n_models"),
            "best_development_model": eval_data.get("best_development_model"),
            "best_development_pooled_rmse":
                eval_data.get("best_development_pooled_rmse"),
            "warning": "One development split is not final RMSE evidence.",
        }
    return quality


def _export_smoke_csv(destination: Path) -> bool:
    source = ROOT / "reports/checkpoint_07/highdim/smoke/benchmark_summary.csv"
    if not source.is_file():
        return False
    with source.open("r", encoding="utf-8-sig", newline="") as reader:
        rows = list(csv.DictReader(reader))
        columns = (
            "model", "n_splits", "n_rows_repeated", "rmse_pooled",
            "rmse_split_mean", "rmse_split_worst",
        )
        if not rows or any(c not in rows[0] for c in columns):
            raise ValueError("Unexpected 07B smoke benchmark CSV schema.")
        if any(row.get("n_splits") != "1" for row in rows):
            raise ValueError("Expected exactly one development split in smoke.")
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row[name] for name in columns})
    return True


def build(
    *,
    include_highdim: bool,
    max_file_mib: float,
    max_total_mib: float,
) -> dict[str, Any]:
    max_file_bytes = int(max_file_mib * 1024**2)
    max_total_bytes = int(max_total_mib * 1024**2)
    if max_file_bytes < 1 or max_total_bytes < 1:
        raise ValueError("Positive size caps required.")

    selection = sources(include_highdim)
    for name, relative in selection.items():
        count, _ = _check_header(ROOT / relative)
        print(f"[07 SHARE] {name}: {count:,} X-only columns", flush=True)

    output = ROOT / PORTABLE_DATA
    reports = ROOT / PORTABLE_REPORTS
    output.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix=".checkpoint07_share_", dir=output))
    try:
        entries = []
        total = 0
        for name, relative in selection.items():
            compressed = scratch / name
            meta = _compress(
                ROOT / relative, compressed, max_bytes=max_file_bytes
            )
            total += meta["compressed_bytes"]
            if total > max_total_bytes:
                raise ValueError(
                    f"Portable set exceeds {max_total_mib:.1f} MiB. "
                    "Exclude --include-highdim or raise cap explicitly."
                )
            entries.append(meta)

        summary = _summary()
        created_at = datetime.now(UTC).isoformat()
        manifest = {
            "schema_version": 1,
            "created_at_utc": created_at,
            "format": "deterministic gzip CSV, UTF-8",
            "scope": "197-parcel X-only HLS/SMAP; optional FULL 07B X matrix",
            "git_storage": "data/** Git LFS, reports/** normal Git",
            "n_compressed_files": len(entries),
            "total_compressed_bytes": total,
            "files": entries,
            "excludes": [
                "raw HDF5/GeoTIFF/Prithvi weights or chips",
                "parcel polygons", "rendimiento labels", "SIAP 2025 outcome",
                "full per-parcel model predictions",
            ],
        }

        # All size/safety checks have passed. Promote files into allowlisted
        # unignored paths. No Git invocation is performed.
        for meta in entries:
            os.replace(scratch / meta["name"], output / meta["name"])
        (reports / MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (reports / QC_NAME).write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        has_smoke = _export_smoke_csv(reports / SMOKE_NAME)
        print(
            f"[07 SHARE] Built {len(entries)} compressed X-only sources, "
            f"{total / 1024**2:.2f} MiB total; "
            f"smoke_metrics={has_smoke}; no Git staging/push.",
            flush=True,
        )
        return manifest
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def verify() -> int:
    path = ROOT / PORTABLE_REPORTS / MANIFEST_NAME
    if not path.is_file():
        raise FileNotFoundError(f"Build the compact snapshots first: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        actual = ROOT / entry["relative_path"]
        if not actual.is_file():
            raise FileNotFoundError(actual)
        if _sha256_file(actual) != entry["compressed_sha256"]:
            raise ValueError(f"Compressed checksum mismatch: {actual}")
        uncompressed_hash = hashlib.sha256()
        with gzip.open(actual, "rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                uncompressed_hash.update(chunk)
        if uncompressed_hash.hexdigest() != entry["source_sha256"]:
            raise ValueError(f"Uncompressed checksum mismatch: {actual}")
        print(f"[07 SHARE] Verified {actual.relative_to(ROOT)}")
    return len(manifest["files"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    exclusive = parser.add_mutually_exclusive_group(required=True)
    exclusive.add_argument("--check", action="store_true")
    exclusive.add_argument("--build", action="store_true")
    exclusive.add_argument("--verify", action="store_true")
    parser.add_argument(
        "--include-highdim", action="store_true",
        help="Also pack FULL 07B expanded 197-parcel X table; never smoke X",
    )
    parser.add_argument("--max-file-mib", type=float, default=25.0)
    parser.add_argument("--max-total-mib", type=float, default=60.0)
    args = parser.parse_args()

    if args.verify:
        count = verify()
        print(f"Checkpoint 07 portable snapshots: PASS ({count} sources)")
        return 0
    selection = sources(args.include_highdim)
    for name, src in selection.items():
        count, _ = _check_header(ROOT / src)
        print(
            f"[07 SHARE] {name}: {count:,} X columns; "
            f"raw size={(ROOT / src).stat().st_size / 1024**2:.2f} MiB"
        )
    print(
        f"[07 SHARE] Optional 07B full X included: {args.include_highdim}. "
        "No local 2025 SIAP outcome or parcel yields allowed."
    )
    if args.check:
        return 0
    build(
        include_highdim=args.include_highdim,
        max_file_mib=args.max_file_mib,
        max_total_mib=args.max_total_mib,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
