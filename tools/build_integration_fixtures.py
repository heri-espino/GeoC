#!/usr/bin/env python
"""Build joinable integration fixtures around the same representative parcels.

The generated fixtures are small, deterministic development artifacts. They are not a
training dataset and must never be used to estimate model performance.

The builder requires the complete local workstation data tree, including gitignored
external sources that already passed Data Contract v2.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import unicodedata
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = Path("configs/data_contract_v2.yaml")


def load_contract(path: Path) -> dict[str, Any]:
    """Load Data Contract v2."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 2:
        raise ValueError("Expected Data Contract schema_version=2.")
    return payload


def normalize_text(value: Any) -> str:
    """Normalize labels for accent/case-insensitive joins."""
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(char for char in text if not unicodedata.combining(char))
    return " ".join(text.casefold().split())


def evenly_spaced_indices(length: int, count: int) -> list[int]:
    """Return deterministic unique indices spanning a sequence."""
    if length <= 0 or count <= 0:
        return []
    if length <= count:
        return list(range(length))
    return sorted(set(np.linspace(0, length - 1, num=count, dtype=int).tolist()))


def read_csv_flexible(path: Path, **kwargs: Any) -> pd.DataFrame:
    """Read source CSVs using the encodings observed in this project."""
    last_error: Exception | None = None
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False, **kwargs)
        except (UnicodeDecodeError, pd.errors.ParserError) as exc:
            last_error = exc
    raise RuntimeError(f"Could not parse {path}: {last_error}")


def canonicalize_parcels(frame: Any) -> Any:
    """Normalize official parcel ID/area aliases."""
    renamed = frame.copy()
    if "ID_POLIGONO" not in renamed.columns and "ID_POLIGON" in renamed.columns:
        renamed = renamed.rename(columns={"ID_POLIGON": "ID_POLIGONO"})
    if "area_ha" not in renamed.columns:
        for alias in ("área_ha", "AREA_HA"):
            if alias in renamed.columns:
                renamed = renamed.rename(columns={alias: "area_ha"})
                break
    return renamed


def satellite_metrics(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Summarize cloud, missingness, sensors and row coverage per parcel."""
    metadata = {"ID_POLIGONO", "fecha_captura", "sensor", "porcentaje_nubosidad"}
    features = [column for column in frame.columns if column not in metadata]

    missing_fraction = (
        frame[features].isna().mean(axis=1)
        if features
        else pd.Series(0.0, index=frame.index)
    )
    working = frame[["ID_POLIGONO", "sensor", "porcentaje_nubosidad"]].copy()
    working["_missing_fraction"] = missing_fraction

    result = (
        working.groupby("ID_POLIGONO", as_index=False)
        .agg(
            cloud_mean=("porcentaje_nubosidad", "mean"),
            missing_mean=("_missing_fraction", "mean"),
            sensor_count=("sensor", "nunique"),
            row_count=("sensor", "size"),
        )
        .rename(
            columns={
                "cloud_mean": f"{prefix}_cloud_mean",
                "missing_mean": f"{prefix}_missing_mean",
                "sensor_count": f"{prefix}_sensor_count",
                "row_count": f"{prefix}_row_count",
            }
        )
    )
    return result


def selection_metrics(
    split: pd.DataFrame,
    parcels: Any,
    basic: pd.DataFrame,
    pro: pd.DataFrame,
) -> pd.DataFrame:
    """Build target-free metrics used only to choose representative fixtures."""
    parcel_columns = ["ID_POLIGONO", "Estado", "Municipio", "area_ha"]
    missing = [column for column in parcel_columns if column not in parcels.columns]
    if missing:
        raise ValueError(f"Parcel source missing selection columns: {missing}")

    base = parcels[parcel_columns].copy()
    base = base.merge(
        split[["ID_POLIGONO", "CONJUNTO", "AREA_HA"]],
        on="ID_POLIGONO",
        how="inner",
        validate="1:1",
    )
    base["area_ha"] = pd.to_numeric(base["area_ha"], errors="coerce")
    source_area = pd.to_numeric(base["AREA_HA"], errors="coerce")
    base["area_ha"] = base["area_ha"].fillna(source_area)

    result = (
        base.merge(satellite_metrics(basic, "basic"), on="ID_POLIGONO", validate="1:1")
        .merge(satellite_metrics(pro, "pro"), on="ID_POLIGONO", validate="1:1")
        .drop(columns=["AREA_HA"])
    )
    return result


def _rank_matrix(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """Return rank-scaled numeric columns with median filling."""
    ranked = []
    for column in columns:
        series = pd.to_numeric(frame[column], errors="coerce")
        if series.notna().any():
            series = series.fillna(series.median())
        else:
            series = pd.Series(0.0, index=frame.index)
        ranked.append(series.rank(method="average", pct=True).to_numpy(dtype=float))
    return np.column_stack(ranked)


def _farthest_index(
    frame: pd.DataFrame,
    candidates: list[int],
    anchors: list[int],
    matrix: np.ndarray,
) -> int:
    """Choose candidate farthest from existing anchors, deterministic on ties."""
    if not anchors:
        raise ValueError("At least one anchor is required.")
    scored: list[tuple[float, str, int]] = []
    for idx in candidates:
        distances = [
            float(np.linalg.norm(matrix[idx] - matrix[anchor])) for anchor in anchors
        ]
        scored.append((min(distances), str(frame.iloc[idx]["ID_POLIGONO"]), idx))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return scored[0][2]


def representative_ids(
    metrics: pd.DataFrame,
    *,
    per_stratum: int,
    forced_ids: Iterable[str] = (),
) -> list[str]:
    """Choose diverse parcels within each Estado × CONJUNTO stratum.

    Diversity is based only on predictor/coverage metadata, never on the yield target.
    Forced IDs preserve known spatial edge cases while keeping the same stratum quota.
    """
    forced = set(map(str, forced_ids))
    unknown_forced = sorted(forced - set(metrics["ID_POLIGONO"].astype(str)))
    if unknown_forced:
        raise ValueError(f"Forced integration IDs not found: {unknown_forced}")

    diversity_columns = [
        "area_ha",
        "basic_cloud_mean",
        "basic_missing_mean",
        "basic_sensor_count",
        "basic_row_count",
        "pro_cloud_mean",
        "pro_missing_mean",
        "pro_row_count",
    ]

    chosen: list[str] = []
    for (state, split_name), group in metrics.groupby(["Estado", "CONJUNTO"], sort=True):
        group = group.sort_values("ID_POLIGONO").reset_index(drop=True)
        if len(group) < per_stratum:
            raise ValueError(
                f"Stratum {(state, split_name)} has {len(group)} rows; "
                f"need {per_stratum}."
            )

        matrix = _rank_matrix(group, diversity_columns)
        forced_positions = [
            idx
            for idx, parcel_id in enumerate(group["ID_POLIGONO"].astype(str))
            if parcel_id in forced
        ]
        if len(forced_positions) > per_stratum:
            raise ValueError(
                f"Too many forced IDs in stratum {(state, split_name)}: "
                f"{len(forced_positions)} > {per_stratum}."
            )

        selected = list(forced_positions)
        available = [idx for idx in range(len(group)) if idx not in selected]

        if not selected and per_stratum >= 2:
            pairs: list[tuple[float, str, str, int, int]] = []
            for left in range(len(group)):
                for right in range(left + 1, len(group)):
                    distance = float(np.linalg.norm(matrix[left] - matrix[right]))
                    id_left = str(group.iloc[left]["ID_POLIGONO"])
                    id_right = str(group.iloc[right]["ID_POLIGONO"])
                    pairs.append((distance, id_left, id_right, left, right))
            pairs.sort(key=lambda item: (-item[0], item[1], item[2]))
            selected.extend([pairs[0][3], pairs[0][4]])
            available = [idx for idx in available if idx not in selected]

        if not selected and per_stratum == 1:
            selected.append(0)
            available = available[1:]

        while len(selected) < per_stratum:
            next_idx = _farthest_index(group, available, selected, matrix)
            selected.append(next_idx)
            available.remove(next_idx)

        chosen.extend(group.iloc[selected]["ID_POLIGONO"].astype(str).tolist())

    expected = metrics[["Estado", "CONJUNTO"]].drop_duplicates().shape[0] * per_stratum
    if len(chosen) != expected or len(set(chosen)) != expected:
        raise AssertionError("Representative parcel selection is not unique/complete.")
    if not forced.issubset(chosen):
        raise AssertionError("Not all forced integration IDs survived selection.")
    return sorted(chosen)


def sample_longitudinal(
    frame: pd.DataFrame,
    selected_ids: set[str],
    *,
    rows_per_sensor: int,
) -> pd.DataFrame:
    """Keep deterministic time-spanning rows for every selected parcel/sensor."""
    selected = frame.loc[frame["ID_POLIGONO"].astype(str).isin(selected_ids)].copy()
    selected["_date"] = pd.to_datetime(
        selected["fecha_captura"],
        format="%d/%m/%Y",
        errors="raise",
    )

    pieces: list[pd.DataFrame] = []
    for (_, _), part in selected.groupby(["ID_POLIGONO", "sensor"], sort=True):
        part = part.sort_values(["_date", "fecha_captura"])
        indices = evenly_spaced_indices(len(part), min(rows_per_sensor, len(part)))
        pieces.append(part.iloc[indices])

    result = pd.concat(pieces, ignore_index=True)
    return result.drop(columns="_date").sort_values(
        ["ID_POLIGONO", "sensor", "fecha_captura"]
    )


def municipality_frames(root: Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    """Load and normalize INEGI municipality files keyed by state code."""
    import geopandas as gpd

    spec = contract["external"]["inegi_municipios"]
    directory = root / spec["root"]
    result: dict[str, Any] = {}

    for state_code, state_spec in spec["files"].items():
        frame = gpd.read_file(directory / state_spec["path"])
        lookup = {normalize_text(column): str(column) for column in frame.columns}
        frame = frame.rename(
            columns={
                lookup["cvegeo"]: "cvegeo",
                lookup["cve_ent"]: "cve_ent",
                lookup["cve_mun"]: "cve_mun",
                lookup["nomgeo"]: "nomgeo",
            }
        )
        frame["state_code"] = str(state_code)
        frame["state_name"] = state_spec["state"]
        result[str(state_code)] = frame[
            ["cvegeo", "cve_ent", "cve_mun", "nomgeo", "state_code", "state_name", "geometry"]
        ].copy()
    return result


def build_admin_fixture(
    parcels: Any,
    municipalities: Mapping[str, Any],
    selected_ids: set[str],
) -> tuple[pd.DataFrame, Any]:
    """Build canonical parcel administration plus spatial-QA metadata."""
    import geopandas as gpd

    chosen = parcels.loc[parcels["ID_POLIGONO"].astype(str).isin(selected_ids)].copy()
    state_name_to_code = {
        normalize_text(frame["state_name"].iloc[0]): code
        for code, frame in municipalities.items()
    }

    all_municipalities = gpd.GeoDataFrame(
        pd.concat(list(municipalities.values()), ignore_index=True),
        geometry="geometry",
        crs=next(iter(municipalities.values())).crs,
    )

    projected_parcels = chosen[["ID_POLIGONO", "geometry"]].to_crs("EPSG:6372")
    projected_munis = all_municipalities.to_crs("EPSG:6372")
    overlaps = gpd.overlay(
        projected_parcels,
        projected_munis,
        how="intersection",
        keep_geom_type=False,
    )
    overlaps["_overlap_area"] = overlaps.geometry.area
    parcel_areas = projected_parcels.set_index("ID_POLIGONO").geometry.area.to_dict()
    winners = overlaps.loc[
        overlaps.groupby("ID_POLIGONO")["_overlap_area"].idxmax()
    ].copy()
    winners["_overlap_fraction"] = winners.apply(
        lambda row: row["_overlap_area"] / parcel_areas[row["ID_POLIGONO"]],
        axis=1,
    )
    spatial = winners.set_index("ID_POLIGONO")

    rows = []
    wanted_cvegeo: set[str] = set()
    for _, row in chosen.sort_values("ID_POLIGONO").iterrows():
        parcel_id = str(row["ID_POLIGONO"])
        state_code = state_name_to_code.get(normalize_text(row["Estado"]))
        official_match = None
        if state_code is not None:
            state_frame = municipalities[state_code]
            mask = state_frame["nomgeo"].map(normalize_text).eq(normalize_text(row["Municipio"]))
            if int(mask.sum()) == 1:
                official_match = state_frame.loc[mask].iloc[0]

        spatial_row = spatial.loc[parcel_id]
        if official_match is None:
            official_match = spatial_row
            join_source = "spatial_fallback"
        else:
            join_source = "official_name"

        cvegeo = str(official_match["cvegeo"]).zfill(5)
        wanted_cvegeo.add(cvegeo)
        rows.append(
            {
                "ID_POLIGONO": parcel_id,
                "Estado": row["Estado"],
                "Municipio": row["Municipio"],
                "cve_ent": str(official_match["cve_ent"]).zfill(2),
                "cve_mun": str(official_match["cve_mun"]).zfill(3),
                "cvegeo": cvegeo,
                "admin_join_source": join_source,
                "spatial_cvegeo": str(spatial_row["cvegeo"]).zfill(5),
                "spatial_municipio": str(spatial_row["nomgeo"]),
                "spatial_overlap_fraction": float(spatial_row["_overlap_fraction"]),
                "official_spatial_match": (
                    cvegeo == str(spatial_row["cvegeo"]).zfill(5)
                ),
            }
        )

    admin = pd.DataFrame(rows)

    selected_geometry = chosen.to_crs(all_municipalities.crs)
    spatial_union = selected_geometry.geometry.union_all()
    municipal_subset = all_municipalities.loc[
        all_municipalities["cvegeo"].astype(str).str.zfill(5).isin(wanted_cvegeo)
        | all_municipalities.geometry.intersects(spatial_union)
    ].copy()

    return admin, municipal_subset


def canonical_siap_files(directory: Path, contract: Mapping[str, Any]) -> list[Path]:
    """Return exactly one canonical SIAP CSV per expected year."""
    spec = contract["external"]["siap"]
    pattern = re.compile(spec["yearly_file_regex"])
    grouped: dict[int, list[Path]] = {}
    for path in sorted(directory.glob("*.csv")):
        match = pattern.match(path.name)
        if match:
            grouped.setdefault(int(match.group(1)), []).append(path)

    result = []
    for year in range(int(spec["year_min"]), int(spec["year_max"]) + 1):
        paths = grouped.get(year, [])
        if not paths:
            raise FileNotFoundError(f"Missing SIAP year {year}.")
        unsuffixed = directory / f"Cierre_agricola_mun_{year}.csv"
        result.append(unsuffixed if unsuffixed in paths else paths[0])
    return result


def canonicalize_siap_frame(
    frame: pd.DataFrame,
    aliases: Iterable[str],
) -> pd.DataFrame:
    """Normalize source-era SIAP crop name aliases and CVEGEO."""
    result = frame.copy()
    crop_column = next((alias for alias in aliases if alias in result.columns), None)
    if crop_column is None:
        raise ValueError("No declared SIAP crop-name alias found.")
    if crop_column != "Nomcultivo":
        result = result.rename(columns={crop_column: "Nomcultivo"})

    state = pd.to_numeric(result["Idestado"], errors="coerce").astype("Int64")
    municipality = pd.to_numeric(result["Idmunicipio"], errors="coerce").astype("Int64")
    result["cvegeo"] = (
        state.astype("string").str.zfill(2)
        + municipality.astype("string").str.zfill(3)
    )
    return result


def build_siap_fixture(
    root: Path,
    contract: Mapping[str, Any],
    cvegeo_values: set[str],
) -> pd.DataFrame:
    """Collect all historical barley rows for selected fixture municipalities."""
    spec = contract["external"]["siap"]
    directory = root / spec["root"]
    aliases = spec.get("crop_name_aliases", ["Nomcultivo"])
    pieces = []

    for path in canonical_siap_files(directory, contract):
        frame = canonicalize_siap_frame(read_csv_flexible(path), aliases)
        mask = (
            frame["cvegeo"].isin(cvegeo_values)
            & frame["Nomcultivo"].astype(str).str.contains(
                spec["crop_discovery_regex"],
                regex=True,
                na=False,
            )
        )
        if mask.any():
            pieces.append(frame.loc[mask].copy())

    if not pieces:
        raise ValueError("No SIAP barley rows matched selected fixture municipalities.")
    return pd.concat(pieces, ignore_index=True).sort_values(
        ["Anio", "cvegeo", "Nomcultivo", "Nomcicloproductivo", "Nommodalidad"]
    )


def raster_nodata(dtype: str, source_nodata: float | int | None) -> float | int:
    """Choose a stable fixture NoData value when a source header lacks one."""
    if source_nodata is not None:
        return source_nodata
    np_dtype = np.dtype(dtype)
    if np.issubdtype(np_dtype, np.floating):
        return float("nan")
    if np.issubdtype(np_dtype, np.signedinteger):
        return int(np.iinfo(np_dtype).min)
    return int(np.iinfo(np_dtype).max)


def write_masked_raster(
    source: Path,
    output: Path,
    parcels: Any,
    *,
    crs_override: str | None = None,
    pad_pixels: int = 2,
) -> None:
    """Write a compressed raster mask around the selected parcel geometries."""
    import rasterio
    from rasterio.mask import mask
    from shapely.geometry import mapping

    output.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(source) as dataset:
        target_crs = dataset.crs or crs_override
        if target_crs is None:
            raise ValueError(f"Raster has no CRS and no source-specific override: {source}")

        selected = parcels.to_crs(target_crs)
        pad_distance = pad_pixels * max(abs(dataset.res[0]), abs(dataset.res[1]))
        geometry = selected.geometry.union_all().buffer(pad_distance)
        nodata = raster_nodata(dataset.dtypes[0], dataset.nodata)
        array, transform = mask(
            dataset,
            [mapping(geometry)],
            crop=True,
            all_touched=True,
            filled=True,
            nodata=nodata,
        )
        profile = dataset.profile.copy()
        for key in ("blockxsize", "blockysize"):
            profile.pop(key, None)
        profile.update(
            height=array.shape[1],
            width=array.shape[2],
            transform=transform,
            crs=target_crs,
            nodata=nodata,
            compress="deflate",
            tiled=False,
        )
        with rasterio.open(output, "w", **profile) as target:
            target.write(array)


def find_by_name(root: Path, names: Iterable[str]) -> list[Path]:
    """Resolve exact filenames recursively and fail on missing/ambiguous names."""
    paths = list(root.rglob("*"))
    by_name: dict[str, list[Path]] = {}
    for path in paths:
        if path.is_file():
            by_name.setdefault(path.name, []).append(path)

    result = []
    for name in names:
        matches = by_name.get(str(name), [])
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected exactly one file named {name!r} under {root}, found {len(matches)}."
            )
        result.append(matches[0])
    return result


def clear_previous_fixture(output: Path) -> None:
    """Remove files recorded by the previous fixture manifest, preserving sibling samples."""
    manifest_path = output / "fixture_manifest.json"
    if not manifest_path.is_file():
        return

    try:
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        previous = {}

    for relative in previous.get("files", []):
        path = output / str(relative)
        if path.is_file():
            path.unlink()

    if manifest_path.is_file():
        manifest_path.unlink()

    for directory in sorted(
        (path for path in output.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        try:
            directory.rmdir()
        except OSError:
            pass


def relative_file_list(output: Path) -> list[str]:
    """Return deterministic fixture file listing relative to fixture root."""
    return sorted(
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file() and path.name != "fixture_manifest.json"
    )


def build_fixtures(root: Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    """Build the complete shared integration fixture set."""
    import geopandas as gpd

    settings = contract["integration_fixtures"]
    output = root / settings["output"]
    output.mkdir(parents=True, exist_ok=True)
    clear_previous_fixture(output)

    split = pd.read_csv(root / contract["official"]["split"]["path"], low_memory=False)
    basic = pd.read_csv(root / contract["official"]["basic"]["path"], low_memory=False)
    pro = pd.read_csv(root / contract["official"]["pro"]["path"], low_memory=False)

    parcel_path = root / contract["official"]["parcels"]["path"]
    parcel_source = f"zip://{parcel_path.resolve()}"
    parcels = canonicalize_parcels(gpd.read_file(parcel_source))

    metrics = selection_metrics(split, parcels, basic, pro)
    selected = representative_ids(
        metrics,
        per_stratum=int(settings["parcels_per_state_split"]),
        forced_ids=settings.get("forced_ids", []),
    )
    selected_ids = set(selected)
    if len(selected) != int(settings["expected_parcels"]):
        raise AssertionError(
            f"Expected {settings['expected_parcels']} fixture parcels, got {len(selected)}."
        )

    selected_metrics = metrics.loc[
        metrics["ID_POLIGONO"].astype(str).isin(selected_ids)
    ].sort_values("ID_POLIGONO")
    selected_metrics.to_csv(output / "selected_parcels.csv", index=False)

    split_fixture = split.loc[split["ID_POLIGONO"].astype(str).isin(selected_ids)].copy()
    split_fixture.sort_values("ID_POLIGONO").to_csv(output / "split.csv", index=False)

    parcel_fixture = parcels.loc[
        parcels["ID_POLIGONO"].astype(str).isin(selected_ids)
    ].sort_values("ID_POLIGONO")
    parcel_fixture.to_file(output / "parcels.geojson", driver="GeoJSON")

    row_limit = int(settings["satellite_rows_per_sensor"])
    basic_fixture = sample_longitudinal(
        basic,
        selected_ids,
        rows_per_sensor=row_limit,
    )
    pro_fixture = sample_longitudinal(
        pro,
        selected_ids,
        rows_per_sensor=row_limit,
    )
    basic_fixture.to_csv(output / "basic.csv", index=False)
    pro_fixture.to_csv(output / "pro.csv", index=False)

    municipalities = municipality_frames(root, contract)
    admin, municipal_subset = build_admin_fixture(
        parcels,
        municipalities,
        selected_ids,
    )
    admin.to_csv(output / "admin.csv", index=False)
    municipal_subset.to_file(output / "municipalities.geojson", driver="GeoJSON")

    siap = build_siap_fixture(
        root,
        contract,
        set(admin["cvegeo"].astype(str)),
    )
    siap.to_csv(output / "siap.csv", index=False)

    pad = int(settings["raster_pad_pixels"])

    climate_root = root / contract["official"]["climate"]["root"]
    for source in find_by_name(climate_root, settings["climate_representatives"]):
        write_masked_raster(
            source,
            output / "climate" / source.name,
            parcel_fixture,
            pad_pixels=pad,
        )

    chirps_root = root / contract["external"]["chirps_daily"]["root"]
    for source in find_by_name(chirps_root, settings["chirps_representatives"]):
        write_masked_raster(
            source,
            output / "chirps_daily" / source.name,
            parcel_fixture,
            pad_pixels=pad,
        )

    topo_root = root / contract["official"]["topography"]["root"]
    for source in sorted(topo_root.rglob("*.tif")):
        write_masked_raster(
            source,
            output / "topography" / source.name,
            parcel_fixture,
            pad_pixels=pad,
        )

    soil_spec = contract["external"]["soilgrids"]
    soil_root = root / soil_spec["root"]
    for source in sorted(soil_root.glob("*.tif")):
        write_masked_raster(
            source,
            output / "soilgrids" / source.name,
            parcel_fixture,
            crs_override=soil_spec["source_crs_when_header_missing"],
            pad_pixels=pad,
        )

    cem_root = root / contract["external"]["inegi_cem_15m"]["root"]
    parcel_state_norm = parcel_fixture["Estado"].map(normalize_text)
    for source in sorted(cem_root.rglob("*.tif")):
        match = re.match(r"(\d{2})_", source.name)
        if match is None:
            continue
        state_spec = contract["external"]["inegi_municipios"]["files"].get(match.group(1))
        if state_spec is None:
            continue
        state_parcels = parcel_fixture.loc[
            parcel_state_norm.eq(normalize_text(state_spec["state"]))
        ]
        if state_parcels.empty:
            continue
        write_masked_raster(
            source,
            output / "cem" / source.name,
            state_parcels,
            pad_pixels=pad,
        )

    wapor_root = root / contract["external"]["wapor"]["root"]
    for name in contract["external"]["wapor"]["expected_files"]:
        source = wapor_root / name
        target = output / "wapor" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    files = relative_file_list(output)
    total_bytes = sum((output / path).stat().st_size for path in files)

    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "purpose": "Joinable real-data integration fixtures; never model training/evaluation data.",
        "selected_ids": selected,
        "forced_ids": list(settings.get("forced_ids", [])),
        "selection_rule": (
            "Two diverse parcels per Estado×CONJUNTO stratum using target-free "
            "area/cloud/missingness/coverage ranks; known spatial edge cases forced."
        ),
        "counts": {
            "parcels": len(selected),
            "basic_rows": len(basic_fixture),
            "pro_rows": len(pro_fixture),
            "siap_rows": len(siap),
            "files": len(files),
        },
        "total_bytes": total_bytes,
        "files": files,
    }
    (output / "fixture_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def validate_fixture(root: Path, contract: Mapping[str, Any]) -> None:
    """Validate key-level fixture invariants after generation."""
    import geopandas as gpd

    output = root / contract["integration_fixtures"]["output"]
    expected = set(
        json.loads((output / "fixture_manifest.json").read_text(encoding="utf-8"))[
            "selected_ids"
        ]
    )

    split = pd.read_csv(output / "split.csv")
    basic = pd.read_csv(output / "basic.csv")
    pro = pd.read_csv(output / "pro.csv")
    admin = pd.read_csv(output / "admin.csv", dtype={"cvegeo": "string"})
    parcels = gpd.read_file(output / "parcels.geojson")

    for name, frame in {
        "split": split,
        "basic": basic,
        "pro": pro,
        "admin": admin,
        "parcels": parcels,
    }.items():
        observed = set(frame["ID_POLIGONO"].astype(str))
        if observed != expected:
            raise AssertionError(
                f"{name} IDs differ from fixture selection: "
                f"missing={sorted(expected-observed)}, extra={sorted(observed-expected)}"
            )

    if not split["ID_POLIGONO"].is_unique:
        raise AssertionError("Fixture split must have one row per parcel.")
    if not parcels["ID_POLIGONO"].is_unique:
        raise AssertionError("Fixture geometry must have one row per parcel.")
    if not admin["ID_POLIGONO"].is_unique:
        raise AssertionError("Fixture admin table must have one row per parcel.")

    identity = contract["identity"]
    prediction = split[identity["split_column"]].eq(identity["prediction_value"])
    if split.loc[prediction, identity["target_column"]].notna().any():
        raise AssertionError("Prediction fixture rows unexpectedly expose hidden targets.")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    return parser.parse_args()


def main() -> int:
    """Build and validate the shared integration fixture set."""
    args = parse_args()
    root = args.root.expanduser().resolve()
    contract_path = args.contract
    if not contract_path.is_absolute():
        contract_path = root / contract_path

    contract = load_contract(contract_path)
    manifest = build_fixtures(root, contract)
    validate_fixture(root, contract)

    output = root / contract["integration_fixtures"]["output"]
    size_mb = manifest["total_bytes"] / (1024 * 1024)
    print(f"Integration fixtures: {output}")
    print(f"Selected parcels ({len(manifest['selected_ids'])}): {manifest['selected_ids']}")
    print(
        f"Rows: BASIC={manifest['counts']['basic_rows']}, "
        f"PRO={manifest['counts']['pro_rows']}, SIAP={manifest['counts']['siap_rows']}"
    )
    print(f"Files: {manifest['counts']['files']}; total={size_mb:.2f} MiB")
    print("Validation: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
