#!/usr/bin/env python3
"""Checkpoint 07A3: compress raw 9-km SMAP HDF5 archive to parcel-daily CSV.

Example:
  python tools/run_checkpoint_07_smap.py --preflight
  python tools/run_checkpoint_07_smap.py --max-files 2
  python tools/run_checkpoint_07_smap.py
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from geocebada.data.checkpoint07_hls import load_parcels
from geocebada.data.checkpoint07_smap import parse_smap_day, run_smap

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw", type=Path,
        default=ROOT / "data" / "raw" / "checkpoint_07"
        / "smap_spl3smp_e_v006",
    )
    parser.add_argument(
        "--parcels", type=Path,
        default=ROOT / "data" / "source" / "geospatial"
        / "Parcelas_Reto_AGC_CONJUNTO_70_30.zip",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "data" / "processed" / "checkpoint_07",
    )
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.max_files < 0:
        parser.error("--max-files cannot be negative")

    raw = args.raw.expanduser().resolve()
    if not raw.is_dir():
        raise FileNotFoundError(f"SMAP directory not found: {raw}")
    files = [p for p in raw.rglob("*.h5") if parse_smap_day(p.name)]
    parcels = load_parcels(args.parcels.expanduser().resolve())
    print(f"SMAP dated HDF5 files: {len(files)}")
    print(f"Parcel polygons: {len(parcels)}")
    if not files:
        raise FileNotFoundError("No SMAP HDF5 files found.")
    if args.preflight:
        print("Checkpoint 07A3 SMAP preflight: PASS")
        return 0
    output = args.output.expanduser().resolve()
    if args.max_files:
        output = output / "smoke_smap"
        print(f"SMOKE MODE — isolated output: {output}")

    report = run_smap(raw, parcels, output, max_files=args.max_files)
    report["generated_at_utc"] = datetime.now(UTC).isoformat()
    report["smoke"] = bool(args.max_files)
    (output / "smap_processing_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print("Checkpoint 07A3 SMAP processing: DONE")
    print(f"Rows: {report['rows']}")
    print(f"Recommended quality: {report['recommended_quality_rows']}")
    print(f"Output: {report['csv']}")
    print("Raw HDF5 files were not altered or deleted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
