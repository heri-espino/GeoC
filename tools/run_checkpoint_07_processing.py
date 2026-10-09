#!/usr/bin/env python3
"""Checkpoint 07A2: HLS raster windows -> longitudinal panel + Prithvi chips.

No downloads, no hidden FIRA y, no 180-GB in-memory reads.
Examples:
  python tools/run_checkpoint_07_processing.py --preflight
  python tools/run_checkpoint_07_processing.py --stage panel --max-scenes 3 --max-parcels 3
  python tools/run_checkpoint_07_processing.py --stage all
  python tools/run_checkpoint_07_processing.py --stage chips
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from geocebada.data.checkpoint07_hls import (
    load_parcels,
    run_chips,
    run_panel,
    scan_hls,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw", type=Path,
        default=ROOT / "data" / "raw" / "checkpoint_07" / "hls_v2",
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
    parser.add_argument(
        "--stage", choices=("panel", "chips", "all"), default="all",
    )
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--max-scenes", type=int, default=0)
    parser.add_argument("--max-parcels", type=int, default=0)
    parser.add_argument("--min-valid-fraction", type=float, default=0.05)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.max_scenes < 0 or args.max_parcels < 0:
        parser.error("--max-scenes and --max-parcels cannot be negative")
    if not 0 < args.min_valid_fraction <= 1:
        parser.error("--min-valid-fraction must be in (0, 1]")

    raw = args.raw.expanduser().resolve()
    if not raw.is_dir():
        raise FileNotFoundError(f"HLS directory not found: {raw}")
    scenes, incomplete = scan_hls(raw)
    parcels = load_parcels(args.parcels.expanduser().resolve())
    if not scenes:
        raise RuntimeError("No complete HLS v2 scenes found.")
    print(f"HLS scenes with required bands and Fmask: {len(scenes)}")
    print(f"HLS scenes missing required bands: {len(incomplete)}")
    print(f"Official parcel polygons: {len(parcels)}")
    print("HLS bands: L30 B02/B03/B04/B05/B06/B07; "
          "S30 B02/B03/B04/B8A/B11/B12")

    # Development smoke tests must never contaminate the full 197-parcel cache.
    smoke = bool(args.max_scenes or args.max_parcels)
    output = args.output.expanduser().resolve()
    if smoke:
        output = output / "smoke"
        scenes = scenes[:args.max_scenes] if args.max_scenes else scenes
        parcels = parcels[:args.max_parcels] if args.max_parcels else parcels
        print(
            f"SMOKE MODE — isolated output: {output} "
            f"({len(scenes)} scenes, {len(parcels)} parcels)"
        )
    elif len(parcels) != 197:
        raise ValueError("Full processing requires exactly 197 parcels.")

    if args.preflight:
        print("Checkpoint 07A2 HLS preflight: PASS")
        return 0

    output.mkdir(parents=True, exist_ok=True)
    result: dict[str, object] = {
        "stage": args.stage,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "smoke": smoke,
        "source_scenes": len(scenes),
        "source_parcels": len(parcels),
        "incomplete_scenes": incomplete,
    }
    if args.stage in {"panel", "all"}:
        result["panel"] = run_panel(
            scenes, parcels, output, force=args.force
        )
    if args.stage in {"chips", "all"}:
        result["chips"] = run_chips(
            scenes,
            parcels,
            output,
            min_valid_fraction=args.min_valid_fraction,
            force=args.force,
        )
    (output / "processing_report.json").write_text(
        json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print("Checkpoint 07A2 HLS processing: DONE")
    print(f"Output: {output}")
    print(f"Report: {output / 'processing_report.json'}")
    if "panel" in result:
        print(f"Panel: {result['panel']}")
    if "chips" in result:
        print(f"Chips: {result['chips']}")
    print("Raw HLS files were not altered or deleted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
