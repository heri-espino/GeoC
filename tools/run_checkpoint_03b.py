#!/usr/bin/env python
"""Run the complete Checkpoint 03B pipeline locally.

This script intentionally does two things, in order:

1. rebuild and validate the deterministic X-only empirical feature layer;
2. run the fold-local target-aware expression recurrence audit using the
   frozen Checkpoint 02 folds.

It does not train or select a final predictive model. That belongs to 03C.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from geocebada.paths import find_project_root

ROOT = Path(__file__).resolve().parents[1]


def _run(root: Path, script: str) -> None:
    subprocess.run(
        [sys.executable, str(root / "tools" / script)],
        cwd=root,
        check=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--skip-feature-build",
        action="store_true",
        help="Reuse the committed empirical feature table and run only expression discovery.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    if not (root / "pyproject.toml").is_file():
        root = find_project_root(root)

    if not args.skip_feature_build:
        print("[03B 1/2] Building deterministic empirical features...")
        _run(root, "build_empirical_features_v1.py")
    else:
        print("[03B 1/2] Skipping empirical feature rebuild.")

    print("[03B 2/2] Running fold-local expression discovery...")
    _run(root, "run_expression_discovery_03b.py")

    print("")
    print("Checkpoint 03B local pipeline finished.")
    print("Inspect:")
    print("  data/processed/empirical_features_v1/build_report.json")
    print("  reports/checkpoint_03b/expression_discovery_report.md")
    print("  reports/checkpoint_03b/expression_stability.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
