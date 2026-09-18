#!/usr/bin/env python
"""Build GeoCebada parcel-level feature tables.

Default mode uses the complete local official + external data tree and writes exactly
197 rows. --fixture runs the same feature logic against the committed 12-parcel
integration fixtures for end-to-end development checks.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from geocebada.data.external import (
    build_admin_mapping,
    load_inegi_municipalities,
    load_siap_barley_history,
)
from geocebada.data.official import load_basic_data, load_pro_data
from geocebada.data.targets import load_yield_split
from geocebada.features.parcel import (
    ID_COLUMN,
    SPLIT_COLUMN,
    TARGET_COLUMN,
    build_base_features,
    build_cem15_features,
    build_daily_chirps_features,
    build_official_climate_features,
    build_satellite_features,
    build_siap_features,
    build_soilgrids_features,
    build_static_raster_features,
    build_wapor_features,
    canonicalize_parcels,
    merge_feature_blocks,
    validate_parcel_feature_table,
)
from geocebada.geo.parcels import load_parcels
from geocebada.paths import find_project_root

ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping in {path}.")
    return payload


def _fixture_inputs(root: Path) -> dict[str, Any]:
    import geopandas as gpd

    directory = root / "data" / "samples" / "integration"
    return {
        "split": pd.read_csv(directory / "split.csv"),
        "parcels": gpd.read_file(directory / "parcels.geojson"),
        "admin": pd.read_csv(directory / "admin.csv", dtype={"cvegeo": "string"}),
        "basic": pd.read_csv(directory / "basic.csv"),
        "pro": pd.read_csv(directory / "pro.csv"),
        "siap": pd.read_csv(directory / "siap.csv", dtype={"cvegeo": "string"}),
        "climate_paths": sorted((directory / "climate").glob("*.tif")),
        "chirps_paths": sorted((directory / "chirps_daily").glob("*.tif")),
        "topography_paths": sorted((directory / "topography").glob("*.tif")),
        "soil_paths": sorted((directory / "soilgrids").glob("*.tif")),
        "cem_paths": sorted((directory / "cem").glob("*.tif")),
        "wapor_paths": sorted((directory / "wapor").glob("*.nc")),
    }


def _full_inputs(
    root: Path,
    contract: dict[str, Any],
) -> dict[str, Any]:
    split = load_yield_split(root=root)
    parcels = canonicalize_parcels(load_parcels(root=root))
    municipalities = load_inegi_municipalities(contract, root=root)
    admin = build_admin_mapping(parcels, municipalities)

    return {
        "split": split,
        "parcels": parcels,
        "admin": admin,
        "basic": load_basic_data(root=root),
        "pro": load_pro_data(root=root),
        "siap": load_siap_barley_history(contract, root=root),
        "climate_paths": sorted(
            (root / contract["official"]["climate"]["root"]).rglob("*.tif")
        ),
        "chirps_paths": sorted(
            (root / contract["external"]["chirps_daily"]["root"]).glob("*.tif")
        ),
        "topography_paths": sorted(
            (root / contract["official"]["topography"]["root"]).rglob("*.tif")
        ),
        "soil_paths": sorted(
            (root / contract["external"]["soilgrids"]["root"]).glob("*.tif")
        ),
        "cem_paths": sorted(
            (root / contract["external"]["inegi_cem_15m"]["root"]).rglob("*.tif")
        ),
        "wapor_paths": [
            root / contract["external"]["wapor"]["root"] / name
            for name in contract["external"]["wapor"]["expected_files"]
        ],
    }


def _topography_specs(
    paths: list[Path],
    stats: list[str],
) -> list[dict[str, Any]]:
    specs = []
    for path in paths:
        stem = path.stem.casefold()
        if "elevacion" in stem:
            name = "topo_official__elevation"
        elif "pendiente" in stem:
            name = "topo_official__slope"
        else:
            name = f"topo_official__{re.sub(r'[^a-z0-9]+', '_', stem).strip('_')}"
        specs.append({"name": name, "path": path, "stats": stats})
    return specs


def _soil_specs(
    paths: list[Path],
    contract: dict[str, Any],
    stats: list[str],
) -> list[dict[str, Any]]:
    spec = contract["external"]["soilgrids"]
    pattern = re.compile(
        r"^(?P<prop>[a-z0-9]+)_(?P<depth>0-5cm|5-15cm|15-30cm|30-60cm)_Q0\.5\.tif$"
    )
    result = []
    for path in paths:
        match = pattern.match(path.name)
        if match is None:
            continue
        prop = match.group("prop")
        depth = match.group("depth")
        result.append(
            {
                "property": prop,
                "depth": depth,
                "path": path,
                "stats": stats,
                "divisor": float(spec["properties"][prop]["divisor"]),
                "crs_override": spec["source_crs_when_header_missing"],
            }
        )
    return result


def _dated_chirps(paths: list[Path]) -> list[tuple[pd.Timestamp, Path]]:
    pattern = re.compile(r"chirps_v3_(\d{4}-\d{2}-\d{2})\.tif$")
    result = []
    for path in paths:
        match = pattern.match(path.name)
        if match:
            result.append((pd.Timestamp(match.group(1)), path))
    return result


def _remove_all_null_features(
    frame: pd.DataFrame,
    provenance: list[dict[str, str]],
) -> tuple[pd.DataFrame, list[dict[str, str]], list[str]]:
    protected = {
        ID_COLUMN,
        SPLIT_COLUMN,
        TARGET_COLUMN,
        "meta_estado",
        "meta_municipio",
    }
    candidates = [column for column in frame.columns if column not in protected]
    dropped = [column for column in candidates if frame[column].isna().all()]
    cleaned = frame.drop(columns=dropped)
    keep = set(cleaned.columns)
    records = [record for record in provenance if record["column"] in keep]
    return cleaned, records, dropped


def _feature_columns_by_mode(
    provenance: list[dict[str, str]],
) -> tuple[list[str], list[str]]:
    by_column = {record["column"]: record for record in provenance}
    clean = sorted(
        column for column, record in by_column.items() if record["mode"] == "clean"
    )
    competition = sorted(by_column)
    return clean, competition


def _write_outputs(
    master: pd.DataFrame,
    provenance: list[dict[str, str]],
    *,
    output_directory: Path,
    output_spec: dict[str, str],
    build_report: dict[str, Any],
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    metadata = [
        column
        for column in [
            ID_COLUMN,
            SPLIT_COLUMN,
            TARGET_COLUMN,
            "meta_estado",
            "meta_municipio",
        ]
        if column in master.columns
    ]
    clean_features, competition_features = _feature_columns_by_mode(provenance)

    master.to_csv(output_directory / output_spec["master_csv"], index=False)
    master[[*metadata, *clean_features]].to_csv(
        output_directory / output_spec["clean_csv"],
        index=False,
    )
    master[[*metadata, *competition_features]].to_csv(
        output_directory / output_spec["competition_csv"],
        index=False,
    )

    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "metadata_columns": metadata,
        "target_column": TARGET_COLUMN,
        "split_column": SPLIT_COLUMN,
        "features": sorted(provenance, key=lambda record: record["column"]),
    }
    (output_directory / output_spec["manifest_json"]).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_directory / output_spec["build_report_json"]).write_text(
        json.dumps(build_report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_tables(
    *,
    root: Path,
    feature_config: dict[str, Any],
    contract: dict[str, Any],
    fixture: bool,
    include_wapor: bool,
    output_directory: Path,
) -> dict[str, Any]:
    """Build all parcel feature blocks and write clean/competition tables."""

    inputs = _fixture_inputs(root) if fixture else _full_inputs(root, contract)
    split = inputs["split"]
    parcels = canonicalize_parcels(inputs["parcels"])
    admin = inputs["admin"]

    blocks = []
    provenance: list[dict[str, str]] = []

    base, records = build_base_features(
        split,
        parcels,
        admin=admin,
        projected_area_crs=feature_config["base"]["projected_area_crs"],
    )
    provenance.extend(records)

    windows = feature_config["windows"]
    sat_cfg = feature_config["satellite"]
    for frame, prefix in [
        (inputs["basic"], "sat_basic"),
        (inputs["pro"], "sat_pro"),
    ]:
        block, records = build_satellite_features(
            frame,
            source_prefix=prefix,
            target_start=str(windows["target_start"]),
            target_end=str(windows["target_end"]),
            historical_start=str(windows["historical_start"]),
            historical_end=str(windows["historical_end"]),
            historical_months=sat_cfg["historical_months"],
            target_months=sat_cfg["target_months"],
            aggregations=sat_cfg["summary_aggregations"],
        )
        blocks.append(block)
        provenance.extend(records)

    climate_cfg = feature_config["official_climate"]
    block, records = build_official_climate_features(
        parcels,
        inputs["climate_paths"],
        historical_years=climate_cfg["historical_years"],
        target_year=int(climate_cfg["target_year"]),
        months=climate_cfg["months"],
    )
    blocks.append(block)
    provenance.extend(records)

    chirps_cfg = feature_config["chirps_daily"]
    block, records = build_daily_chirps_features(
        parcels,
        _dated_chirps(inputs["chirps_paths"]),
        rain_day_threshold_mm=float(chirps_cfg["rain_day_threshold_mm"]),
        heavy_rain_threshold_mm=float(chirps_cfg["heavy_rain_threshold_mm"]),
        rolling_window_days=int(chirps_cfg["rolling_window_days"]),
    )
    blocks.append(block)
    provenance.extend(records)

    topo_cfg = feature_config["topography"]
    block, records = build_static_raster_features(
        parcels,
        _topography_specs(inputs["topography_paths"], topo_cfg["zonal_stats"]),
        source="official_topography",
    )
    blocks.append(block)
    provenance.extend(records)

    soil_cfg = feature_config["soilgrids"]
    block, records = build_soilgrids_features(
        parcels,
        _soil_specs(inputs["soil_paths"], contract, soil_cfg["zonal_stats"]),
        depth_weights_cm=soil_cfg["depth_weights_cm"],
    )
    blocks.append(block)
    provenance.extend(records)

    cem_cfg = feature_config["cem15"]
    block, records = build_cem15_features(
        parcels,
        inputs["cem_paths"],
        stats=cem_cfg["zonal_stats"],
    )
    blocks.append(block)
    provenance.extend(records)

    siap_cfg = feature_config["siap"]
    block, records = build_siap_features(
        admin,
        inputs["siap"],
        crop=siap_cfg["crop"],
        historical_end_year=int(siap_cfg["historical_end_year"]),
        recent_years=siap_cfg["recent_years"],
        competition_year=int(siap_cfg["competition_year"]),
    )
    blocks.append(block)
    provenance.extend(records)

    if include_wapor:
        block, records = build_wapor_features(parcels, inputs["wapor_paths"])
        blocks.append(block)
        provenance.extend(records)

    master = merge_feature_blocks(base, *blocks)
    master, provenance, dropped_all_null = _remove_all_null_features(master, provenance)

    validation = feature_config["validation"]
    if fixture:
        expected_rows = len(split)
        expected_train = int(split[SPLIT_COLUMN].eq("ENTRENAMIENTO").sum())
        expected_prediction = int(split[SPLIT_COLUMN].eq("PREDICCION").sum())
    else:
        expected_rows = int(validation["expected_full_rows"])
        expected_train = int(validation["expected_train_rows"])
        expected_prediction = int(validation["expected_prediction_rows"])

    validate_parcel_feature_table(
        master,
        expected_rows=expected_rows,
        expected_train=expected_train,
        expected_prediction=expected_prediction,
    )

    clean_features, competition_features = _feature_columns_by_mode(provenance)
    numeric_features = master[competition_features].select_dtypes(include=[np.number])
    missing_fraction = numeric_features.isna().mean().sort_values(ascending=False)

    report = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "fixture_mode": fixture,
        "include_wapor": include_wapor,
        "rows": len(master),
        "columns_all": len(master.columns),
        "features_clean": len(clean_features),
        "features_competition_total": len(competition_features),
        "dropped_all_null_features": dropped_all_null,
        "top_missing_numeric_features": {
            column: round(float(value), 6)
            for column, value in missing_fraction.head(30).items()
        },
        "split_counts": {
            str(key): int(value)
            for key, value in master[SPLIT_COLUMN].value_counts().items()
        },
    }

    _write_outputs(
        master,
        provenance,
        output_directory=output_directory,
        output_spec=feature_config["outputs"],
        build_report=report,
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--feature-config",
        type=Path,
        default=Path("configs/features_v1.yaml"),
    )
    parser.add_argument(
        "--data-contract",
        type=Path,
        default=Path("configs/data_contract_v2.yaml"),
    )
    parser.add_argument(
        "--fixture",
        action="store_true",
        help="Build a 12-row integration-fixture preview instead of the full 197 rows.",
    )
    parser.add_argument(
        "--skip-wapor",
        action="store_true",
        help="Skip WaPOR extraction; useful for minimal CI environments.",
    )
    parser.add_argument("--output-directory", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    if not (root / "pyproject.toml").is_file():
        root = find_project_root(root)

    feature_path = (
        args.feature_config
        if args.feature_config.is_absolute()
        else root / args.feature_config
    )
    contract_path = (
        args.data_contract
        if args.data_contract.is_absolute()
        else root / args.data_contract
    )
    feature_config = load_yaml(feature_path)
    contract = load_yaml(contract_path)

    if args.output_directory is not None:
        output_directory = args.output_directory
        if not output_directory.is_absolute():
            output_directory = root / output_directory
    elif args.fixture:
        output_directory = root / "data" / "processed" / "features_v1_fixture"
    else:
        output_directory = root / feature_config["outputs"]["directory"]

    report = build_tables(
        root=root,
        feature_config=feature_config,
        contract=contract,
        fixture=args.fixture,
        include_wapor=not args.skip_wapor,
        output_directory=output_directory,
    )

    print(f"Parcel feature table build: PASS ({report['rows']} rows)")
    print(f"All columns: {report['columns_all']}")
    print(f"Clean features: {report['features_clean']}")
    print(f"Competition features total: {report['features_competition_total']}")
    print(f"Output: {output_directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
