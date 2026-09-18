#!/usr/bin/env python
"""Audit GeoCebada against the machine-readable Data Contract v2.

The audit is read-only. It validates the official 197-parcel contract and, when
local external data are present, checks source completeness, temporal coverage,
CRS/schema assumptions and join keys before feature engineering begins.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = Path("configs/data_contract_v2.yaml")
DEFAULT_JSON = Path("reports/data_audit_v2.json")
DEFAULT_MARKDOWN = Path("reports/data_audit_v2.md")
STATUSES = ("pass", "warn", "fail", "skip")


def portable(path: Path, root: Path) -> str:
    """Return a path relative to the repository root when possible."""
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def load_contract(path: Path) -> dict[str, Any]:
    """Load and minimally validate the Data Contract v2 YAML."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Contract must contain a mapping: {path}")
    if payload.get("schema_version") != 2:
        raise ValueError("Data Contract schema_version must be 2.")
    return payload


def check(
    name: str,
    status: str,
    summary: str,
    *,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create one JSON-serializable audit check."""
    if status not in STATUSES:
        raise ValueError(f"Unknown audit status: {status}")
    return {
        "name": name,
        "status": status,
        "summary": summary,
        "details": dict(details or {}),
    }


def bool_check(
    name: str,
    condition: bool,
    pass_summary: str,
    fail_summary: str,
    *,
    details: Mapping[str, Any] | None = None,
    failure_status: str = "fail",
) -> dict[str, Any]:
    """Return a pass/fail-like check from a boolean condition."""
    return check(
        name,
        "pass" if condition else failure_status,
        pass_summary if condition else fail_summary,
        details=details,
    )


def drop_empty_unnamed(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop source-CSV trailing empty Unnamed columns only."""
    columns = [
        column
        for column in frame.columns
        if str(column).startswith("Unnamed:") and frame[column].isna().all()
    ]
    return frame.drop(columns=columns)


def read_csv_flexible(
    path: Path,
    *,
    usecols: Iterable[str] | None = None,
    nrows: int | None = None,
) -> pd.DataFrame:
    """Read CSVs from official/SIAP sources with conservative encoding fallback."""
    attempts = ("utf-8", "utf-8-sig", "latin-1")
    last_error: Exception | None = None
    for encoding in attempts:
        try:
            return pd.read_csv(
                path,
                encoding=encoding,
                low_memory=False,
                usecols=usecols,
                nrows=nrows,
            )
        except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as exc:
            last_error = exc
    raise RuntimeError(f"Could not parse {path}: {last_error}")


def parse_strict_dates(series: pd.Series, fmt: str) -> pd.Series:
    """Parse dates with the source-declared format; invalid values become NaT."""
    return pd.to_datetime(series, format=fmt, errors="coerce")


def date_range(start: date, end: date) -> list[date]:
    """Return all dates in a closed interval."""
    days = (end - start).days
    return [start + timedelta(days=offset) for offset in range(days + 1)]


def normalize_text(value: Any) -> str:
    """Normalize labels for robust accent/case/whitespace comparisons."""
    if pd.isna(value):
        return ""
    normalized = unicodedata.normalize("NFKD", str(value))
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    return " ".join(normalized.casefold().split())


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Hash a file without loading it entirely into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_split(
    root: Path,
    contract: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], pd.DataFrame | None]:
    """Audit the official target/split table."""
    spec = contract["official"]["split"]
    identity = contract["identity"]
    path = root / spec["path"]
    if not path.is_file():
        return [check("official_split.exists", "fail", f"Missing {portable(path, root)}")], None

    frame = drop_empty_unnamed(pd.read_csv(path, low_memory=False))
    id_col = identity["id_column"]
    target_col = identity["target_column"]
    split_col = identity["split_column"]
    train_value = identity["train_value"]
    pred_value = identity["prediction_value"]

    required = set(spec["required_columns"])
    missing_columns = sorted(required.difference(frame.columns))
    checks = [
        bool_check(
            "official_split.columns",
            not missing_columns,
            "Required target/split columns are present.",
            f"Missing required columns: {missing_columns}",
            details={"missing_columns": missing_columns},
        ),
        bool_check(
            "official_split.rows",
            len(frame) == int(spec["expected_rows"]),
            f"Split contains {len(frame)} parcel rows.",
            f"Expected {spec['expected_rows']} rows, found {len(frame)}.",
            details={"rows": len(frame)},
        ),
    ]
    if missing_columns:
        return checks, frame

    train_mask = frame[split_col].eq(train_value)
    pred_mask = frame[split_col].eq(pred_value)
    ids = frame[id_col]
    checks.extend(
        [
            bool_check(
                "official_split.ids",
                ids.notna().all() and ids.is_unique,
                f"{id_col} is non-null and unique.",
                f"{id_col} contains nulls or duplicates.",
                details={
                    "unique_ids": int(ids.nunique(dropna=True)),
                    "missing_ids": int(ids.isna().sum()),
                    "duplicate_rows": int(ids.duplicated(keep=False).sum()),
                },
            ),
            bool_check(
                "official_split.counts",
                int(train_mask.sum()) == int(identity["expected_train_parcels"])
                and int(pred_mask.sum()) == int(identity["expected_prediction_parcels"]),
                "Official split is 138 training + 59 prediction parcels.",
                "Training/prediction counts differ from the contract.",
                details={
                    "train": int(train_mask.sum()),
                    "prediction": int(pred_mask.sum()),
                },
            ),
            bool_check(
                "official_split.target_visibility",
                frame.loc[train_mask, target_col].notna().all()
                and frame.loc[pred_mask, target_col].isna().all(),
                "Targets are observed only for training parcels.",
                "Target missingness does not match the official split.",
                details={
                    "train_missing_targets": int(frame.loc[train_mask, target_col].isna().sum()),
                    "prediction_exposed_targets": int(
                        frame.loc[pred_mask, target_col].notna().sum()
                    ),
                },
            ),
        ]
    )
    return checks, frame


def audit_satellite(
    root: Path,
    contract: Mapping[str, Any],
    kind: str,
    split_ids: set[str] | None,
    split_lookup: Mapping[str, str] | None,
) -> list[dict[str, Any]]:
    """Audit one complete BASIC/PRO longitudinal table."""
    spec = contract["official"][kind]
    id_col = contract["identity"]["id_column"]
    path = root / spec["path"]
    prefix = f"official_{kind}"
    if not path.is_file():
        return [check(f"{prefix}.exists", "fail", f"Missing {portable(path, root)}")]

    frame = pd.read_csv(path, low_memory=False)
    required = set(spec["required_columns"])
    missing_columns = sorted(required.difference(frame.columns))
    checks = [
        bool_check(
            f"{prefix}.shape",
            len(frame) == int(spec["expected_rows"])
            and len(frame.columns) == int(spec["expected_columns"]),
            f"{kind.upper()} shape matches {len(frame):,} × {len(frame.columns)}.",
            f"{kind.upper()} shape differs from the Data Contract v2.",
            details={"rows": len(frame), "columns": len(frame.columns)},
        ),
        bool_check(
            f"{prefix}.columns",
            not missing_columns,
            "Required longitudinal metadata columns are present.",
            f"Missing required columns: {missing_columns}",
            details={"missing_columns": missing_columns},
        ),
    ]
    if missing_columns:
        return checks

    ids = set(frame[id_col].dropna().astype(str).unique())
    id_ok = len(ids) == int(spec["expected_parcels"])
    if split_ids is not None:
        id_ok = id_ok and ids == split_ids
    checks.append(
        bool_check(
            f"{prefix}.parcel_coverage",
            id_ok,
            f"{kind.upper()} covers all {len(ids)} official parcels.",
            f"{kind.upper()} parcel coverage differs from the official split.",
            details={
                "unique_ids": len(ids),
                "missing_from_table": sorted((split_ids or set()) - ids),
                "extra_ids": sorted(ids - (split_ids or ids)),
            },
        )
    )

    parsed = parse_strict_dates(frame[spec["date_column"]], spec["date_format"])
    invalid_dates = int(parsed.isna().sum())
    date_min = parsed.min()
    date_max = parsed.max()
    expected_min = pd.Timestamp(spec["expected_date_min"])
    expected_max = pd.Timestamp(spec["expected_date_max"])
    checks.append(
        bool_check(
            f"{prefix}.dates",
            invalid_dates == 0 and date_min == expected_min and date_max == expected_max,
            (
                f"Dates parse strictly as {spec['date_format']} and span "
                f"{date_min.date()} to {date_max.date()}."
            ),
            "Date parsing/range differs from the contract.",
            details={
                "invalid_dates": invalid_dates,
                "min": None if pd.isna(date_min) else date_min.date().isoformat(),
                "max": None if pd.isna(date_max) else date_max.date().isoformat(),
            },
        )
    )

    observed_sensors = {str(value) for value in frame["sensor"].dropna().unique()}
    expected_sensors = set(map(str, spec["expected_sensors"]))
    sensor_counts = frame["sensor"].value_counts(dropna=False)
    checks.append(
        bool_check(
            f"{prefix}.sensors",
            observed_sensors == expected_sensors,
            f"Observed sensors match contract: {sorted(observed_sensors)}.",
            "Observed sensor set differs from the contract.",
            details={
                "observed": sorted(observed_sensors),
                "expected": sorted(expected_sensors),
                "row_counts": {
                    str(key): int(value) for key, value in sensor_counts.items()
                },
            },
        )
    )

    duplicate_keys = int(
        frame.duplicated([id_col, spec["date_column"], "sensor"], keep=False).sum()
    )
    checks.append(
        check(
            f"{prefix}.observation_keys",
            "pass" if duplicate_keys == 0 else "warn",
            (
                "Parcel/date/sensor keys are unique."
                if duplicate_keys == 0
                else f"{duplicate_keys:,} rows share a parcel/date/sensor key."
            ),
            details={"rows_in_duplicate_keys": duplicate_keys},
        )
    )

    feature_columns = [
        column
        for column in frame.columns
        if column not in {id_col, spec["date_column"], "sensor", "porcentaje_nubosidad"}
    ]
    missing_by_sensor: dict[str, Any] = {}
    for sensor, part in frame.groupby("sensor", dropna=False):
        rates = part[feature_columns].isna().mean().sort_values(ascending=False)
        missing_by_sensor[str(sensor)] = {
            "rows": len(part),
            "top_missing_rates": {
                str(column): round(float(value), 6)
                for column, value in rates.head(12).items()
            },
        }

    coverage_by_split: dict[str, Any] = {}
    if split_lookup is not None:
        working = frame[[id_col]].copy()
        working["_split"] = working[id_col].astype(str).map(split_lookup)
        working["_date"] = parsed
        for split_value, part in working.groupby("_split", dropna=False):
            counts = part.groupby(id_col).size()
            unique_dates = part.groupby(id_col)["_date"].nunique()
            coverage_by_split[str(split_value)] = {
                "rows": int(len(part)),
                "parcels": int(part[id_col].nunique()),
                "median_rows_per_parcel": float(counts.median()),
                "median_unique_dates_per_parcel": float(unique_dates.median()),
                "min_rows_per_parcel": int(counts.min()),
                "max_rows_per_parcel": int(counts.max()),
            }

    checks.append(
        check(
            f"{prefix}.coverage_profile",
            "pass",
            "Recorded sensor-specific missingness and train/prediction coverage diagnostics.",
            details={
                "missing_by_sensor": missing_by_sensor,
                "coverage_by_split": coverage_by_split,
            },
        )
    )
    return checks


def canonicalize_parcel_columns(frame: Any, contract: Mapping[str, Any]) -> Any:
    """Return parcels with canonical ID/area names without mutating source data."""
    spec = contract["official"]["parcels"]
    renamed = frame.copy()
    id_col = contract["identity"]["id_column"]
    if id_col not in renamed.columns:
        alias = next((name for name in spec["id_aliases"] if name in renamed.columns), None)
        if alias is not None:
            renamed = renamed.rename(columns={alias: id_col})
    if "area_ha" not in renamed.columns:
        alias = next((name for name in spec["area_aliases"] if name in renamed.columns), None)
        if alias is not None:
            renamed = renamed.rename(columns={alias: "area_ha"})
    return renamed


def audit_parcels(
    root: Path,
    contract: Mapping[str, Any],
    split_ids: set[str] | None,
) -> tuple[list[dict[str, Any]], Any | None]:
    """Audit official parcel geometry and canonical field normalization."""
    try:
        import geopandas as gpd
    except ImportError:
        return [check("official_parcels.dependencies", "fail", "GeoPandas is required.")], None

    spec = contract["official"]["parcels"]
    path = root / spec["path"]
    if not path.is_file():
        return [check("official_parcels.exists", "fail", f"Missing {portable(path, root)}")], None

    source = f"zip://{path.resolve()}" if path.suffix.lower() == ".zip" else path
    frame = canonicalize_parcel_columns(gpd.read_file(source), contract)
    id_col = contract["identity"]["id_column"]
    observed_crs = None if frame.crs is None else frame.crs.to_string()
    ids = set(frame[id_col].dropna().astype(str)) if id_col in frame else set()
    valid_mask = frame.geometry.notna() & ~frame.geometry.is_empty & frame.geometry.is_valid
    geometry_types = sorted(frame.geometry.geom_type.dropna().unique().tolist())

    checks = [
        bool_check(
            "official_parcels.rows",
            len(frame) == int(spec["expected_features"]),
            f"Parcel archive contains {len(frame)} features.",
            f"Expected {spec['expected_features']} parcel features, found {len(frame)}.",
            details={"features": len(frame)},
        ),
        bool_check(
            "official_parcels.crs",
            observed_crs == str(spec["expected_crs"]),
            f"Parcel CRS is {observed_crs}.",
            f"Parcel CRS {observed_crs!r} differs from {spec['expected_crs']}.",
            details={"crs": observed_crs},
        ),
        bool_check(
            "official_parcels.geometry",
            bool(valid_mask.all()),
            "All parcel geometries are present, non-empty and valid.",
            "Some parcel geometries are missing, empty or invalid.",
            details={
                "valid": int(valid_mask.sum()),
                "invalid_or_empty": int((~valid_mask).sum()),
                "geometry_types": geometry_types,
            },
        ),
        bool_check(
            "official_parcels.ids",
            id_col in frame
            and frame[id_col].notna().all()
            and frame[id_col].is_unique
            and (split_ids is None or ids == split_ids),
            "Parcel IDs are unique and match the official split.",
            "Parcel ID coverage differs from the official split.",
            details={
                "unique_ids": len(ids),
                "missing_from_geometry": sorted((split_ids or set()) - ids),
                "extra_geometry_ids": sorted(ids - (split_ids or ids)),
            },
        ),
        bool_check(
            "official_parcels.area_alias",
            "area_ha" in frame.columns,
            "Parcel area field normalizes to area_ha.",
            "No documented parcel area alias was found.",
        ),
    ]
    return checks, frame


def expected_months(start: str, end: str) -> set[tuple[int, int]]:
    """Return all year/month pairs in an inclusive YYYY-MM interval."""
    periods = pd.period_range(pd.Period(start, freq="M"), pd.Period(end, freq="M"), freq="M")
    return {(period.year, period.month) for period in periods}


def audit_official_climate(
    root: Path,
    contract: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Audit official monthly climate filename continuity and CRS."""
    spec = contract["official"]["climate"]
    directory = root / spec["root"]
    if not directory.is_dir():
        return [check("official_climate.exists", "fail", f"Missing {portable(directory, root)}")]

    pattern = re.compile(r"^(PREC|Tmin|Tmax)_(\d{4})_(\d{2})\.tif$", re.I)
    files = sorted(directory.rglob("*.tif"))
    observed: dict[str, set[tuple[int, int]]] = defaultdict(set)
    unparsed: list[str] = []
    for path in files:
        match = pattern.match(path.name)
        if match is None:
            unparsed.append(portable(path, root))
            continue
        variable = {"prec": "PREC", "tmin": "Tmin", "tmax": "Tmax"}[match.group(1).lower()]
        observed[variable].add((int(match.group(2)), int(match.group(3))))

    months = expected_months(str(spec["start"]), str(spec["end"]))
    count_ok = (
        len(observed["PREC"]) == int(spec["expected_precip_months"])
        and len(observed["Tmin"]) == int(spec["expected_tmin_months"])
        and len(observed["Tmax"]) == int(spec["expected_tmax_months"])
    )
    continuity_ok = all(observed[name] == months for name in ("PREC", "Tmin", "Tmax"))
    checks = [
        bool_check(
            "official_climate.months",
            count_ok and continuity_ok and not unparsed,
            "Official climate has complete monthly PREC/Tmin/Tmax coverage for 2022–2025.",
            "Official climate monthly coverage is incomplete or has unexpected TIFFs.",
            details={
                "PREC": len(observed["PREC"]),
                "Tmin": len(observed["Tmin"]),
                "Tmax": len(observed["Tmax"]),
                "missing_PREC": sorted(months - observed["PREC"]),
                "missing_Tmin": sorted(months - observed["Tmin"]),
                "missing_Tmax": sorted(months - observed["Tmax"]),
                "unparsed_tifs": unparsed,
            },
        )
    ]

    try:
        import rasterio

        crs_values: set[str | None] = set()
        step = max(1, len(files) // 12)
        for path in files[::step]:
            with rasterio.open(path) as dataset:
                crs_values.add(None if dataset.crs is None else dataset.crs.to_string())
        checks.append(
            bool_check(
                "official_climate.crs",
                crs_values == {str(spec["expected_crs"])},
                f"Representative climate rasters use {spec['expected_crs']}.",
                "Climate CRS differs across representative rasters.",
                details={"observed_crs": sorted(str(value) for value in crs_values)},
            )
        )
    except ImportError:
        checks.append(
            check("official_climate.crs", "warn", "Rasterio unavailable; CRS not inspected.")
        )
    return checks


def audit_official_topography(
    root: Path,
    contract: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Audit official topography raster count and CRS."""
    spec = contract["official"]["topography"]
    directory = root / spec["root"]
    if not directory.is_dir():
        return [check("official_topography.exists", "fail", f"Missing {portable(directory, root)}")]

    files = sorted(directory.rglob("*.tif"))
    checks = [
        bool_check(
            "official_topography.count",
            len(files) == int(spec["expected_tif_count"]),
            f"Official topography contains {len(files)} GeoTIFFs.",
            f"Expected {spec['expected_tif_count']} topography GeoTIFFs, found {len(files)}.",
            details={"files": [portable(path, root) for path in files]},
        )
    ]
    try:
        import rasterio

        observed = {}
        for path in files:
            with rasterio.open(path) as dataset:
                observed[path.name] = None if dataset.crs is None else dataset.crs.to_string()
        checks.append(
            bool_check(
                "official_topography.crs",
                bool(observed) and set(observed.values()) == {str(spec["expected_crs"])},
                f"Official topography CRS is {spec['expected_crs']}.",
                "Official topography CRS differs from the contract.",
                details={"files": observed},
            )
        )
    except ImportError:
        checks.append(
            check("official_topography.crs", "warn", "Rasterio unavailable; CRS not inspected.")
        )
    return checks


def field_lookup(columns: Iterable[str]) -> dict[str, str]:
    """Map normalized field names to source spelling."""
    return {normalize_text(column): str(column) for column in columns}


def audit_municipalities(
    root: Path,
    contract: Mapping[str, Any],
    parcels: Any | None,
) -> list[dict[str, Any]]:
    """Audit INEGI municipal layers and the parcel-to-municipality overlap join."""
    spec = contract["external"]["inegi_municipios"]
    directory = root / spec["root"]
    try:
        import geopandas as gpd
    except ImportError:
        return [check("external_municipios.dependencies", "fail", "GeoPandas is required.")]

    checks: list[dict[str, Any]] = []
    frames = []
    for state_code, state_spec in spec["files"].items():
        path = directory / state_spec["path"]
        name = f"external_municipios.{state_code}"
        if not path.is_file():
            checks.append(check(name, "fail", f"Missing {portable(path, root)}"))
            continue
        frame = gpd.read_file(path)
        lookup = field_lookup(frame.columns)
        required = [normalize_text(field) for field in spec["required_fields"]]
        missing = [field for field in required if field not in lookup]
        count_ok = len(frame) == int(state_spec["expected_municipalities"])
        checks.append(
            bool_check(
                name,
                count_ok and not missing and frame.crs is not None,
                f"{state_spec['state']}: {len(frame)} municipalities with required keys.",
                f"{state_spec['state']} municipality layer violates the contract.",
                details={
                    "features": len(frame),
                    "expected": int(state_spec["expected_municipalities"]),
                    "missing_fields": missing,
                    "crs": None if frame.crs is None else str(frame.crs),
                },
            )
        )
        if not missing and frame.crs is not None:
            renamed = frame.rename(
                columns={
                    lookup[normalize_text("cvegeo")]: "cvegeo",
                    lookup[normalize_text("cve_ent")]: "cve_ent",
                    lookup[normalize_text("cve_mun")]: "cve_mun",
                    lookup[normalize_text("nomgeo")]: "nomgeo",
                }
            )[["cvegeo", "cve_ent", "cve_mun", "nomgeo", "geometry"]].copy()
            renamed["state_code"] = str(state_code)
            renamed["state_name"] = state_spec["state"]
            frames.append(renamed)

    if parcels is None or len(frames) != 3:
        checks.append(
            check(
                "external_municipios.parcel_join",
                "skip",
                "Parcel overlap join skipped because complete inputs were unavailable.",
            )
        )
        return checks

    base_crs = frames[0].crs
    if any(frame.crs != base_crs for frame in frames[1:]):
        checks.append(
            check(
                "external_municipios.crs_consistency",
                "fail",
                "Municipality files use inconsistent CRS values.",
            )
        )
        return checks

    municipalities = gpd.GeoDataFrame(
        pd.concat(frames, ignore_index=True),
        geometry="geometry",
        crs=base_crs,
    )
    id_col = contract["identity"]["id_column"]
    area_crs = "EPSG:6372"
    left = parcels[[id_col, "geometry"]].to_crs(area_crs).copy()
    left["_parcel_area_m2"] = left.geometry.area
    right = municipalities.to_crs(area_crs)
    intersections = gpd.overlay(left, right, how="intersection", keep_geom_type=False)
    if intersections.empty:
        checks.append(
            check(
                "external_municipios.parcel_join",
                "fail",
                "No parcel/municipality intersections were found.",
            )
        )
        return checks

    intersections["_overlap_area_m2"] = intersections.geometry.area
    winners = intersections.loc[
        intersections.groupby(id_col)["_overlap_area_m2"].idxmax()
    ].copy()
    winners["_overlap_fraction"] = winners["_overlap_area_m2"] / winners["_parcel_area_m2"]
    missing_ids = sorted(set(parcels[id_col].astype(str)) - set(winners[id_col].astype(str)))
    minimum_overlap = float(winners["_overlap_fraction"].min())
    policy = spec.get("assignment_policy", {})
    minimum_dominant_overlap = float(policy.get("minimum_dominant_overlap", 0.80))

    low_overlap_details: list[dict[str, Any]] = []
    low_overlap_ids = set(
        winners.loc[winners["_overlap_fraction"] < minimum_dominant_overlap, id_col].astype(str)
    )
    for parcel_id in sorted(low_overlap_ids):
        candidates = intersections.loc[
            intersections[id_col].astype(str).eq(parcel_id)
        ].copy()
        candidates["_overlap_fraction"] = (
            candidates["_overlap_area_m2"] / candidates["_parcel_area_m2"]
        )
        candidates = candidates.sort_values("_overlap_fraction", ascending=False)
        low_overlap_details.append(
            {
                "ID_POLIGONO": parcel_id,
                "candidates": [
                    {
                        "cvegeo": str(row["cvegeo"]),
                        "municipio": str(row["nomgeo"]),
                        "state_code": str(row["state_code"]),
                        "overlap_fraction": round(float(row["_overlap_fraction"]), 6),
                    }
                    for _, row in candidates.head(3).iterrows()
                ],
            }
        )

    checks.append(
        bool_check(
            "external_municipios.parcel_join",
            not missing_ids and minimum_overlap >= minimum_dominant_overlap,
            (
                "Every parcel has a dominant spatial municipality above the "
                f"{minimum_dominant_overlap:.0%} QA threshold."
            ),
            (
                "Some parcels lack a dominant spatial municipality above the "
                f"{minimum_dominant_overlap:.0%} QA threshold."
            ),
            details={
                "assigned_parcels": len(winners),
                "missing_ids": missing_ids,
                "min_overlap_fraction": round(minimum_overlap, 6),
                "qa_threshold": minimum_dominant_overlap,
                "low_overlap_parcels": low_overlap_details,
                "administrative_join_policy": policy.get("administrative_join"),
                "spatial_qc_policy": policy.get("spatial_qc"),
            },
            failure_status="warn",
        )
    )

    name_mismatches: list[dict[str, str]] = []
    if "Municipio" in parcels.columns:
        official = parcels[[id_col, "Municipio"]].copy()
        compared = winners[[id_col, "nomgeo"]].merge(official, on=id_col, validate="1:1")
        mismatch = compared.apply(
            lambda row: normalize_text(row["nomgeo"]) != normalize_text(row["Municipio"]),
            axis=1,
        )
        for _, row in compared.loc[mismatch].head(20).iterrows():
            name_mismatches.append(
                {
                    "ID_POLIGONO": str(row[id_col]),
                    "parcel_Municipio": str(row["Municipio"]),
                    "inegi_nomgeo": str(row["nomgeo"]),
                }
            )
    known_discrepancies = policy.get("known_name_discrepancies", {})
    unresolved_name_mismatches = [
        item
        for item in name_mismatches
        if item["ID_POLIGONO"] not in known_discrepancies
    ]
    if not name_mismatches:
        name_status = "pass"
        name_summary = "Parcel Municipio labels agree with INEGI overlap assignments."
    elif unresolved_name_mismatches:
        name_status = "warn"
        name_summary = "Some parcel Municipio labels differ from INEGI without a frozen policy."
    else:
        name_status = "pass"
        name_summary = (
            "Known parcel/INEGI municipality name discrepancies are documented; "
            "official parcel labels remain primary for administrative joins."
        )

    checks.append(
        check(
            "external_municipios.name_crosscheck",
            name_status,
            name_summary,
            details={
                "examples": name_mismatches,
                "known_discrepancies": known_discrepancies,
                "unresolved": unresolved_name_mismatches,
                "administrative_join_policy": policy.get("administrative_join"),
            },
        )
    )
    return checks


def audit_chirps_daily(
    root: Path,
    contract: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Audit daily CHIRPS filename continuity and CRS."""
    spec = contract["external"]["chirps_daily"]
    directory = root / spec["root"]
    if not directory.is_dir():
        return [check("external_chirps.exists", "fail", f"Missing {portable(directory, root)}")]

    pattern = re.compile(spec["filename_regex"])
    observed: dict[date, Path] = {}
    unparsed = []
    for path in sorted(directory.glob("*.tif")):
        match = pattern.match(path.name)
        if match is None:
            unparsed.append(path.name)
            continue
        observed[date.fromisoformat(match.group(1))] = path

    start = date.fromisoformat(str(spec["start"]))
    end = date.fromisoformat(str(spec["end"]))
    expected = set(date_range(start, end))
    observed_dates = set(observed)
    checks = [
        bool_check(
            "external_chirps.continuity",
            len(observed) == int(spec["expected_count"])
            and observed_dates == expected
            and not unparsed,
            f"CHIRPS has all {len(observed)} daily files from {start} through {end}.",
            "CHIRPS daily coverage is incomplete or contains unexpected files.",
            details={
                "count": len(observed),
                "missing_dates": sorted(day.isoformat() for day in expected - observed_dates),
                "extra_dates": sorted(day.isoformat() for day in observed_dates - expected),
                "unparsed_files": unparsed,
            },
        )
    ]
    if observed:
        try:
            import rasterio

            dates = sorted(observed)
            representatives = [dates[0], dates[len(dates) // 2], dates[-1]]
            crs_values = set()
            for day in representatives:
                with rasterio.open(observed[day]) as dataset:
                    crs_values.add(None if dataset.crs is None else dataset.crs.to_string())
            checks.append(
                bool_check(
                    "external_chirps.crs",
                    crs_values == {str(spec["expected_crs"])},
                    f"Representative CHIRPS rasters use {spec['expected_crs']}.",
                    "CHIRPS CRS differs from the contract.",
                    details={"observed_crs": sorted(str(value) for value in crs_values)},
                )
            )
        except ImportError:
            checks.append(
                check("external_chirps.crs", "warn", "Rasterio unavailable; CRS not inspected.")
            )
    return checks


def audit_soilgrids(
    root: Path,
    contract: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Audit exact SoilGrids property/depth inventory and CRS-header behavior."""
    spec = contract["external"]["soilgrids"]
    directory = root / spec["root"]
    if not directory.is_dir():
        return [check("external_soilgrids.exists", "fail", f"Missing {portable(directory, root)}")]

    expected_names = {
        f"{prop}_{depth}_{spec['quantile']}.tif"
        for prop in spec["properties"]
        for depth in spec["depths"]
    }
    observed_paths = sorted(directory.glob("*.tif"))
    observed_names = {path.name for path in observed_paths}
    checks = [
        bool_check(
            "external_soilgrids.inventory",
            observed_names == expected_names
            and len(observed_paths) == int(spec["expected_count"]),
            "SoilGrids contains 9 properties × 4 depths = 36 rasters.",
            "SoilGrids inventory differs from the contract.",
            details={
                "count": len(observed_paths),
                "missing": sorted(expected_names - observed_names),
                "extra": sorted(observed_names - expected_names),
            },
        ),
        check(
            "external_soilgrids.semantics",
            "pass",
            "ISRIC conversion divisors/units are frozen in Data Contract v2.",
            details={
                "source": spec["semantics_source"],
                "properties": spec["properties"],
            },
        ),
    ]
    try:
        import rasterio

        observed_crs = {}
        for path in observed_paths:
            with rasterio.open(path) as dataset:
                observed_crs[path.name] = None if dataset.crs is None else dataset.crs.to_string()
        acceptable = all(
            value is None or value == str(spec["source_crs_when_header_missing"])
            for value in observed_crs.values()
        )
        distinct = sorted({str(value) for value in observed_crs.values()})
        checks.append(
            bool_check(
                "external_soilgrids.crs_headers",
                acceptable,
                "SoilGrids headers are compatible with the source-specific CRS fallback.",
                "Unexpected SoilGrids CRS metadata was found.",
                details={"distinct_header_crs": distinct},
            )
        )
    except ImportError:
        checks.append(
            check(
                "external_soilgrids.crs_headers",
                "warn",
                "Rasterio unavailable; SoilGrids headers were not inspected.",
            )
        )
    return checks


def siap_year_files(directory: Path, pattern: re.Pattern[str]) -> dict[int, list[Path]]:
    """Group SIAP yearly CSVs by year, retaining duplicate copies."""
    grouped: dict[int, list[Path]] = defaultdict(list)
    for path in sorted(directory.glob("*.csv")):
        match = pattern.match(path.name)
        if match:
            grouped[int(match.group(1))].append(path)
    return dict(grouped)


def audit_siap(
    root: Path,
    contract: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Audit SIAP year continuity, duplicate copies, schema and barley labels."""
    spec = contract["external"]["siap"]
    directory = root / spec["root"]
    if not directory.is_dir():
        return [check("external_siap.exists", "fail", f"Missing {portable(directory, root)}")]

    pattern = re.compile(spec["yearly_file_regex"])
    all_csvs = sorted(directory.glob("*.csv"))
    files_by_year = siap_year_files(directory, pattern)
    unmatched_csvs = [path.name for path in all_csvs if pattern.match(path.name) is None]
    yearlike_unmatched = [
        name for name in unmatched_csvs if re.search(r"20\\d{2}", name)
    ]
    expected_years = set(range(int(spec["year_min"]), int(spec["year_max"]) + 1))
    observed_years = set(files_by_year)
    missing_years = sorted(expected_years - observed_years)
    duplicate_years = {
        year: [path.name for path in paths]
        for year, paths in files_by_year.items()
        if len(paths) > 1
    }
    checks = [
        bool_check(
            "external_siap.years",
            observed_years == expected_years,
            f"SIAP contains every year {min(expected_years)}–{max(expected_years)}.",
            "SIAP year coverage is incomplete.",
            details={
                "observed_years": sorted(observed_years),
                "missing_years": missing_years,
                "duplicate_years": duplicate_years,
                "unmatched_csvs": unmatched_csvs,
                "yearlike_unmatched_csvs": yearlike_unmatched,
            },
        )
    ]

    duplicate_hashes: dict[str, Any] = {}
    conflicts = []
    canonical_files: dict[str, str] = {}
    for year, paths in files_by_year.items():
        canonical = next(
            (path for path in paths if re.fullmatch(rf"Cierre_agricola_mun_{year}\.csv", path.name)),
            paths[0],
        )
        canonical_files[str(year)] = canonical.name
        if len(paths) <= 1:
            continue
        hashes = {path.name: sha256_file(path) for path in paths}
        duplicate_hashes[str(year)] = hashes
        if len(set(hashes.values())) > 1:
            conflicts.append(year)

    duplicate_status = "warn" if conflicts else "pass"
    if conflicts:
        duplicate_summary = (
            "Some duplicate SIAP yearly copies differ by SHA256 and require manual resolution."
        )
    elif duplicate_years:
        duplicate_summary = (
            "Duplicate SIAP yearly copies are byte-identical; canonical unsuffixed files are safe to use."
        )
    else:
        duplicate_summary = "No duplicate SIAP yearly copies detected."

    checks.append(
        check(
            "external_siap.duplicates",
            duplicate_status,
            duplicate_summary,
            details={
                "duplicate_years": duplicate_years,
                "sha256": duplicate_hashes,
                "conflicting_duplicate_years": conflicts,
                "canonical_files": canonical_files,
            },
        )
    )

    required = list(spec["required_columns"])
    schema_issues: dict[str, Any] = {}
    barley_labels: set[str] = set()
    barley_cycles: set[str] = set()
    barley_modalities: set[str] = set()
    barley_units: set[str] = set()
    target_states = {int(value) for value in spec["target_state_codes"]}
    crop_regex = re.compile(spec["crop_discovery_regex"])

    for year in sorted(observed_years):
        path = files_by_year[year][0]
        header = read_csv_flexible(path, nrows=0)
        crop_aliases = list(spec.get("crop_name_aliases", ["Nomcultivo"]))
        crop_column = next((name for name in crop_aliases if name in header.columns), None)
        missing = sorted(set(required).difference(header.columns))
        if crop_column is None:
            missing.append("crop_name_alias")
        if missing:
            schema_issues[path.name] = {
                "missing": sorted(set(missing)),
                "observed_columns": list(map(str, header.columns)),
                "crop_like_columns": [
                    str(column)
                    for column in header.columns
                    if "cult" in normalize_text(column)
                    or "producto" in normalize_text(column)
                ],
            }
            continue

        useful = [
            "Anio",
            "Idestado",
            "Idmunicipio",
            "Nomcicloproductivo",
            "Nommodalidad",
            crop_column,
            "Rendimiento",
        ]
        if "Nomunidad" in header.columns:
            useful.append("Nomunidad")
        frame = read_csv_flexible(path, usecols=useful)
        if crop_column != "Nomcultivo":
            frame = frame.rename(columns={crop_column: "Nomcultivo"})
        states = pd.to_numeric(frame["Idestado"], errors="coerce")
        mask = states.isin(target_states) & frame["Nomcultivo"].astype(str).str.contains(
            crop_regex,
            na=False,
        )
        barley = frame.loc[mask]
        barley_labels.update(barley["Nomcultivo"].dropna().astype(str).unique())
        barley_cycles.update(barley["Nomcicloproductivo"].dropna().astype(str).unique())
        barley_modalities.update(barley["Nommodalidad"].dropna().astype(str).unique())
        if "Nomunidad" in barley.columns:
            barley_units.update(barley["Nomunidad"].dropna().astype(str).unique())

    checks.extend(
        [
            bool_check(
                "external_siap.schema",
                not schema_issues,
                "All SIAP yearly files expose the required columns.",
                "Some SIAP yearly files are missing required columns.",
                details={"schema_issues": schema_issues},
            ),
            bool_check(
                "external_siap.barley_discovery",
                bool(barley_labels),
                f"Discovered SIAP barley labels: {sorted(barley_labels)}.",
                "No barley records were discovered for target states.",
                details={
                    "labels": sorted(barley_labels),
                    "cycles": sorted(barley_cycles),
                    "modalities": sorted(barley_modalities),
                    "production_units": sorted(barley_units),
                },
            ),
        ]
    )
    return checks


def inspect_netcdf_profile(path: Path, root: Path) -> dict[str, Any]:
    """Run the isolated NetCDF metadata worker and return its profile."""
    worker = root / "tools" / "_netcdf_catalog_worker.py"
    completed = subprocess.run(
        [sys.executable, str(worker), str(path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(detail)
    return json.loads(completed.stdout)["profile"]


def audit_wapor(
    root: Path,
    contract: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Audit WaPOR products, dekadal band count and unit/date metadata."""
    spec = contract["external"]["wapor"]
    directory = root / spec["root"]
    if not directory.is_dir():
        return [check("external_wapor.exists", "fail", f"Missing {portable(directory, root)}")]

    expected_paths = [directory / name for name in spec["expected_files"]]
    missing = [path.name for path in expected_paths if not path.is_file()]
    checks = [
        bool_check(
            "external_wapor.files",
            not missing,
            "Both expected WaPOR NetCDF products are present.",
            f"Missing WaPOR files: {missing}",
            details={"missing": missing},
        )
    ]
    if missing:
        return checks

    product_details: dict[str, Any] = {}
    issues: list[str] = []
    for product_name, product_spec in spec["products"].items():
        path = next(
            item
            for item in expected_paths
            if product_spec["filename_contains"] in item.name
        )
        try:
            profile = inspect_netcdf_profile(path, root)
        except Exception as exc:
            issues.append(f"{path.name}: {type(exc).__name__}: {exc}")
            continue

        variables = profile.get("variables", {})
        bands = {
            name: record
            for name, record in variables.items()
            if re.fullmatch(r"Band\d+", str(name))
        }
        ordered = sorted(bands, key=lambda name: int(str(name).replace("Band", "")))
        attrs = [bands[name].get("attributes", {}) for name in ordered]
        units = sorted({str(item.get("units")) for item in attrs if item.get("units") is not None})
        long_names = sorted(
            {str(item.get("long_name")) for item in attrs if item.get("long_name") is not None}
        )
        starts = [str(item.get("start_date")) for item in attrs if item.get("start_date")]
        ends = [str(item.get("end_date")) for item in attrs if item.get("end_date")]
        days = [int(item["number_of_days"]) for item in attrs if item.get("number_of_days")]
        product_details[product_name] = {
            "file": path.name,
            "band_count": len(bands),
            "units": units,
            "long_names": long_names,
            "start": min(starts) if starts else None,
            "end": max(ends) if ends else None,
            "number_of_days_values": sorted(set(days)),
            "total_days": sum(days),
        }
        if len(bands) != int(spec["expected_dekads"]):
            issues.append(f"{product_name}: wrong band count {len(bands)}")
        if units != [str(product_spec["expected_units"])]:
            issues.append(f"{product_name}: units={units}")
        if long_names != [str(product_spec["expected_long_name"])]:
            issues.append(f"{product_name}: long_name={long_names}")

    checks.append(
        bool_check(
            "external_wapor.metadata",
            not issues and len(product_details) == len(spec["products"]),
            "WaPOR AETI/NPP have 21 dekads with expected units and metadata.",
            "WaPOR metadata differs from Data Contract v2.",
            details={"products": product_details, "issues": issues},
            failure_status="warn",
        )
    )
    return checks


def audit_cem(
    root: Path,
    contract: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Audit high-resolution INEGI CEM state rasters."""
    spec = contract["external"]["inegi_cem_15m"]
    directory = root / spec["root"]
    if not directory.is_dir():
        return [check("external_cem15.exists", "fail", f"Missing {portable(directory, root)}")]

    paths = sorted(directory.rglob("*.tif"))
    observed_codes = {
        match.group(1)
        for path in paths
        if (match := re.match(r"(\d{2})_", path.name))
    }
    expected_codes = set(map(str, spec["expected_state_codes"]))
    checks = [
        bool_check(
            "external_cem15.inventory",
            len(paths) == int(spec["expected_tif_count"]) and observed_codes == expected_codes,
            "CEM 15 m inventory contains one raster for each target state.",
            "CEM 15 m state inventory differs from the contract.",
            details={
                "count": len(paths),
                "state_codes": sorted(observed_codes),
                "files": [portable(path, root) for path in paths],
            },
        )
    ]
    try:
        import rasterio

        crs = {}
        for path in paths:
            with rasterio.open(path) as dataset:
                crs[path.name] = None if dataset.crs is None else dataset.crs.to_string()
        checks.append(
            bool_check(
                "external_cem15.crs",
                bool(crs) and set(crs.values()) == {str(spec["expected_crs"])},
                f"CEM 15 m rasters use {spec['expected_crs']}.",
                "CEM 15 m CRS differs from the contract.",
                details={"files": crs},
            )
        )
    except ImportError:
        checks.append(
            check("external_cem15.crs", "warn", "Rasterio unavailable; CEM CRS not inspected.")
        )
    return checks


def audit_era5(
    root: Path,
    contract: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Explicitly mark ERA5 optional/partial unless separately certified."""
    spec = contract["external"]["era5_land"]
    directory = root / spec["root"]
    files = sorted(directory.glob("*.nc")) if directory.is_dir() else []
    return [
        check(
            "external_era5.policy",
            "skip" if not files else "warn",
            (
                "ERA5-Land is absent and intentionally not required."
                if not files
                else (
                    f"Found {len(files)} ERA5-Land files, but this source remains "
                    "partial/untrusted until a dedicated completeness audit succeeds."
                )
            ),
            details={
                "policy": spec["policy"],
                "file_count": len(files),
                "files": [path.name for path in files[:20]],
            },
        )
    ]


def build_audit(
    root: Path,
    contract: Mapping[str, Any],
    *,
    official_only: bool = False,
) -> dict[str, Any]:
    """Run Data Contract v2 audit and return a machine-readable report."""
    checks: list[dict[str, Any]] = []

    split_checks, split = audit_split(root, contract)
    checks.extend(split_checks)
    id_col = contract["identity"]["id_column"]
    split_col = contract["identity"]["split_column"]
    split_ids = (
        set(split[id_col].dropna().astype(str))
        if split is not None and id_col in split
        else None
    )
    split_lookup = (
        dict(zip(split[id_col].astype(str), split[split_col], strict=True))
        if split is not None and {id_col, split_col}.issubset(split.columns)
        else None
    )

    for kind in ("basic", "pro"):
        checks.extend(audit_satellite(root, contract, kind, split_ids, split_lookup))

    parcel_checks, parcels = audit_parcels(root, contract, split_ids)
    checks.extend(parcel_checks)
    checks.extend(audit_official_climate(root, contract))
    checks.extend(audit_official_topography(root, contract))

    if official_only:
        checks.append(
            check("external_sources", "skip", "External-source audit skipped by --official-only.")
        )
    else:
        checks.extend(audit_municipalities(root, contract, parcels))
        checks.extend(audit_chirps_daily(root, contract))
        checks.extend(audit_wapor(root, contract))
        checks.extend(audit_soilgrids(root, contract))
        checks.extend(audit_siap(root, contract))
        checks.extend(audit_cem(root, contract))
        checks.extend(audit_era5(root, contract))

    counts = Counter(item["status"] for item in checks)
    overall = "fail" if counts["fail"] else ("warn" if counts["warn"] else "pass")
    return {
        "schema_version": 2,
        "generated_at": datetime.now(UTC).isoformat(),
        "contract": "configs/data_contract_v2.yaml",
        "official_only": official_only,
        "overall_status": overall,
        "summary": {status: int(counts[status]) for status in STATUSES},
        "checks": checks,
    }


def write_json(report: Mapping[str, Any], output: Path) -> None:
    """Write stable UTF-8 JSON audit output."""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def markdown_report(report: Mapping[str, Any]) -> str:
    """Render a compact human-readable audit report."""
    lines = [
        "# Data Audit v2",
        "",
        f"Generated: {report['generated_at']}",
        f"Overall status: **{str(report['overall_status']).upper()}**",
        "",
        "## Summary",
        "",
        "| Status | Checks |",
        "|---|---:|",
    ]
    for status in STATUSES:
        lines.append(f"| {status.upper()} | {report['summary'][status]} |")

    lines.extend(["", "## Checks", "", "| Status | Check | Summary |", "|---|---|---|"])
    for item in report["checks"]:
        summary = str(item["summary"]).replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {item['status'].upper()} | {item['name']} | {summary} |")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- FAIL blocks feature-table construction.",
            "- WARN requires an explicit documented decision before downstream use.",
            "- SKIP is intentional or dependency/source-limited; "
            "it is not evidence of completeness.",
            "- Presence of a source is not equivalent to satisfying the contract.",
            "",
            "Machine-readable details: reports/data_audit_v2.json",
            "",
        ]
    )
    return "\n".join(lines)


def write_markdown(report: Mapping[str, Any], output: Path) -> None:
    """Write human-readable Markdown audit output."""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(markdown_report(report), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--official-only", action="store_true")
    parser.add_argument("--fail-on-warn", action="store_true")
    return parser.parse_args()


def resolved_under_root(root: Path, value: Path) -> Path:
    """Resolve a CLI path relative to root unless already absolute."""
    return value.expanduser().resolve() if value.is_absolute() else (root / value).resolve()


def main() -> int:
    """Run audit, write JSON/Markdown outputs and print a compact summary."""
    args = parse_args()
    root = args.root.expanduser().resolve()
    contract_path = resolved_under_root(root, args.contract)
    json_output = resolved_under_root(root, args.json_output)
    markdown_output = resolved_under_root(root, args.markdown_output)

    contract = load_contract(contract_path)
    report = build_audit(root, contract, official_only=args.official_only)
    write_json(report, json_output)
    write_markdown(report, markdown_output)

    print(f"Data Contract v2 audit: {report['overall_status'].upper()}")
    print(
        "Checks: "
        + ", ".join(f"{status}={report['summary'][status]}" for status in STATUSES)
    )
    print(f"JSON: {portable(json_output, root)}")
    print(f"Markdown: {portable(markdown_output, root)}")

    if report["overall_status"] == "fail":
        return 1
    if args.fail_on_warn and report["summary"]["warn"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
