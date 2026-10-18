"""Validate or refresh the lightweight GeoCebada website's published CSV snapshots.

This script copies already published result tables. It never trains models, accesses
hidden targets, or downloads raw geospatial data. Run from any working directory.
"""
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "reports/checkpoint_07/shared/three_model_summary.csv": "website/data/checkpoint07_summary.csv",
    "reports/checkpoint_07/shared/checkpoint08_comparison_summary.csv": (
        "website/data/checkpoint08_summary.csv"
    ),
    "reports/checkpoint_07/shared/checkpoint08_paired_cluster_bootstrap.csv": (
        "website/data/checkpoint08_bootstrap.csv"
    ),
    "reports/checkpoint_05/final_predictions.csv": "website/data/predictions.csv",
    # Contexto, soporte y variables (anexos y explorador).
    "reports/checkpoint_02/baseline_summary.csv": (
        "website/data/checkpoint02_baseline_summary.csv"
    ),
    "reports/checkpoint_03b/expression_stability.csv": (
        "website/data/checkpoint03b_expression_stability.csv"
    ),
    "reports/checkpoint_04b/protocol_summary.csv": (
        "website/data/checkpoint04b_protocol_summary.csv"
    ),
    "reports/checkpoint_05/protocol_summary.csv": (
        "website/data/checkpoint05_protocol_summary.csv"
    ),
    "reports/checkpoint_04f/final_prediction_diagnostics.csv": (
        "website/data/final_prediction_diagnostics.csv"
    ),
    "reports/checkpoint_06/feature_family_tail_importance.csv": (
        "website/data/checkpoint06_feature_family_tail_importance.csv"
    ),
    "reports/checkpoint_06/feature_family_error_risk.csv": (
        "website/data/checkpoint06_feature_family_error_risk.csv"
    ),
}


def sync(*, write: bool = False) -> list[str]:
    """Return mismatched paths; copy source bytes on an explicitly requested write."""
    changes: list[str] = []
    for source_name, target_name in SOURCES.items():
        source, target = ROOT / source_name, ROOT / target_name
        if not source.is_file():
            raise FileNotFoundError(f"Canonical website data source missing: {source}")
        source_bytes = source.read_bytes()
        if not source_bytes.strip():
            raise ValueError(f"Empty canonical website source: {source}")
        if not target.exists() or target.read_bytes() != source_bytes:
            changes.append(target_name)
            if write:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source_bytes)
    return changes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true", help="Fail if a copy is stale")
    group.add_argument("--write", action="store_true", help="Copy canonical public CSV bytes")
    args = parser.parse_args()
    changed = sync(write=args.write)
    if changed:
        print(("Synchronized" if args.write else "Out of sync") + ": " + ", ".join(changed))
    else:
        print("Website data snapshots match canonical artifacts.")
    return 1 if changed and args.check else 0


if __name__ == "__main__":
    raise SystemExit(main())
