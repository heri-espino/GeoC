#!/usr/bin/env python3
"""Run Local07 on HLS/SMAP features with the frozen Local04D benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from geocebada.data.targets import load_yield_split
from geocebada.evaluation.checkpoint07_local import run_local07
from geocebada.evaluation.checkpoint07c import load_development

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--trials", type=int, default=48)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument(
        "--features", type=Path,
        default=ROOT / "reports/checkpoint_07/highdim/expanded_parcel_features.csv",
    )
    parser.add_argument(
        "--targets", type=Path,
        default=ROOT / "data/source/tabular/ID_area_rendimiento_70_30_Reto_AgroCebada.csv",
    )
    parser.add_argument(
        "--membership", type=Path,
        default=ROOT / "reports/checkpoint_04b/pseudo_competition_splits.csv",
    )
    parser.add_argument(
        "--baseline", type=Path,
        default=ROOT / "reports/checkpoint_06/parcel_oof_residuals.csv",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "reports/checkpoint_07_three/local07")
    args = parser.parse_args()

    if args.trials < 1 or args.inner_folds < 2:
        raise ValueError("Local07 requires trials >=1 and inner-folds >=2.")
    for path in (args.features, args.targets, args.membership, args.baseline):
        if not path.is_file():
            raise FileNotFoundError(f"Missing Local07 input: {path}")
    cols = set(pd.read_csv(args.features, nrows=0).columns)
    if not {"ID_POLIGONO", "base_centroid_lat", "base_centroid_lon"}.issubset(cols):
        raise ValueError("Local07 lacks required parcel centroid coordinates.")
    if args.preflight:
        print("[LOCAL07] preflight PASS; geography + 07B data available.")
        return 0

    x = pd.read_csv(args.features, low_memory=False)
    target = load_yield_split(path=args.targets, validate=True)
    member = pd.read_csv(args.membership)
    incumbent = pd.read_csv(args.baseline)
    data = load_development(x, target, member, incumbent)
    output = args.output / "smoke" if args.smoke else args.output
    output.mkdir(parents=True, exist_ok=True)
    ids = sorted(data.splits)
    if args.smoke:
        ids = ids[:1]
    for index, split_id in enumerate(ids, 1):
        report = run_local07(
            data, split_id=split_id, output=output,
            trials=min(2, args.trials) if args.smoke else args.trials,
            inner_folds=2 if args.smoke else args.inner_folds, seed=args.seed,
        )
        print(
            f"[LOCAL07 {index}/{len(ids)}] {split_id}: "
            f"new={report['outer_rmse_local07']:.5f} "
            f"Local04D={report['outer_rmse_local04d']:.5f}",
            flush=True,
        )
    reports = sorted(output.glob("target_matched_*__local07.json"))
    if not args.smoke and len(reports) != 16:
        raise RuntimeError(f"Only {len(reports)}/16 Local07 splits finished.")
    (output / "run_report.json").write_text(
        json.dumps({
            "n_completed_splits": len(reports), "expected_splits": len(ids),
            "used_hidden_y": False, "incumbent_modified": False,
        }, indent=2)+"\n", encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
