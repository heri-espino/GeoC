#!/usr/bin/env python
"""Build the deterministic empirical feature layer for Checkpoint 03B."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from geocebada.features import (
    build_empirical_feature_layer,
    join_agronomic_features,
    join_empirical_features,
    validate_empirical_feature_layer,
)
from geocebada.paths import find_project_root

ROOT = Path(__file__).resolve().parents[1]


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping in {path}.")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def build_empirical_tables(
    *,
    root: Path,
    config: dict[str, Any],
    output_directory: Path | None = None,
) -> dict[str, Any]:
    """Build, validate and write Empirical Features v1 artifacts."""

    inputs = config["inputs"]
    outputs = config["outputs"]
    identity = config["identity"]

    feature_directory = root / str(inputs["feature_directory"])
    source_table_path = feature_directory / str(inputs["feature_table"])
    source_manifest_path = feature_directory / str(inputs["feature_manifest"])

    agronomic_directory = root / str(inputs["agronomic_directory"])
    agronomic_table_path = agronomic_directory / str(inputs["agronomic_table"])
    agronomic_manifest_path = agronomic_directory / str(inputs["agronomic_manifest"])

    if output_directory is None:
        output_directory = root / str(outputs["directory"])
    output_directory.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(source_table_path)
    agronomic = pd.read_csv(agronomic_table_path)
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    agronomic_manifest = json.loads(
        agronomic_manifest_path.read_text(encoding="utf-8")
    )

    expected_rows = int(config["validation"]["expected_rows"])
    id_column = str(identity["id_column"])
    if len(frame) != expected_rows:
        raise ValueError(f"Expected {expected_rows} input rows, found {len(frame)}.")
    if frame[id_column].duplicated().any():
        raise ValueError("Feature Table v1 contains duplicate parcel IDs.")

    table, records, diagnostics = build_empirical_feature_layer(frame, config)
    validate_empirical_feature_layer(
        table,
        expected_ids=frame[id_column],
        id_column=id_column,
    )

    if str(identity["target_column"]) in table.columns:
        raise ValueError("Empirical layer must not contain the target column.")
    if str(identity["split_column"]) in table.columns:
        raise ValueError("Empirical layer should contain only ID plus derived predictors.")

    joined = join_agronomic_features(frame, agronomic)
    joined = join_empirical_features(joined, table)
    primitive_columns = list(config["supervised_discovery"]["primitive_columns"])
    missing_primitives = [
        column for column in primitive_columns if column not in joined.columns
    ]
    if missing_primitives:
        raise KeyError(
            "Configured supervised discovery primitives are missing: "
            f"{missing_primitives}"
        )

    table_path = output_directory / str(outputs["feature_csv"])
    manifest_path = output_directory / str(outputs["manifest_json"])
    report_path = output_directory / str(outputs["build_report_json"])

    table.to_csv(table_path, index=False)
    manifest = {
        "schema_version": int(config["schema_version"]),
        "generated_at": datetime.now(UTC).isoformat(),
        "id_column": id_column,
        "source_feature_table": str(source_table_path.relative_to(root)),
        "source_feature_manifest": str(source_manifest_path.relative_to(root)),
        "source_manifest_schema_version": source_manifest.get("schema_version"),
        "agronomic_feature_table": str(agronomic_table_path.relative_to(root)),
        "agronomic_feature_manifest": str(
            agronomic_manifest_path.relative_to(root)
        ),
        "agronomic_manifest_schema_version": agronomic_manifest.get(
            "schema_version"
        ),
        "feature_count": len(records),
        "features": sorted(records, key=lambda item: item["column"]),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report = {
        "schema_version": int(config["schema_version"]),
        "generated_at": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(root),
        "source_feature_table_sha256": _sha256(source_table_path),
        "source_feature_manifest_sha256": _sha256(source_manifest_path),
        "source_agronomic_table_sha256": _sha256(agronomic_table_path),
        "source_agronomic_manifest_sha256": _sha256(agronomic_manifest_path),
        "rows": diagnostics["rows"],
        "features": diagnostics["features"],
        "family_counts": diagnostics["family_counts"],
        "mode_counts": diagnostics["mode_counts"],
        "dropped_all_null": diagnostics["dropped_all_null"],
        "dropped_constant": diagnostics["dropped_constant"],
        "missing_fraction_mean": diagnostics["missing_fraction_mean"],
        "missing_fraction_max": diagnostics["missing_fraction_max"],
        "infinite_values": diagnostics["infinite_values"],
        "target_used_for_materialized_features": False,
        "supervised_discovery_materialized_globally": False,
        "configured_fold_local_primitive_count": len(primitive_columns),
        "short_vci_like_is_standard_vci": False,
        "chirps_used_for_feature_construction": False,
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/empirical_features_v1.yaml"),
    )
    parser.add_argument("--output-directory", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    if not (root / "pyproject.toml").is_file():
        root = find_project_root(root)

    config_path = args.config if args.config.is_absolute() else root / args.config
    config = _load_yaml(config_path)

    output_directory = args.output_directory
    if output_directory is not None and not output_directory.is_absolute():
        output_directory = root / output_directory

    report = build_empirical_tables(
        root=root,
        config=config,
        output_directory=output_directory,
    )
    print("Empirical Feature Layer v1: PASS")
    print(f"Rows: {report['rows']}")
    print(f"Features: {report['features']}")
    print(f"Modes: {report['mode_counts']}")
    print(f"Families: {report['family_counts']}")
    print(
        "Missing fraction mean/max: "
        f"{report['missing_fraction_mean']:.6f}/"
        f"{report['missing_fraction_max']:.6f}"
    )
    print(f"Dropped all-null: {report['dropped_all_null']}")
    print(f"Dropped constant: {report['dropped_constant']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
