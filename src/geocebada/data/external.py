"""Load and normalize public external sources used by parcel feature engineering."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from geocebada.features.parcel import ID_COLUMN, canonicalize_parcels
from geocebada.paths import find_project_root


def _normalize_label(value: object) -> str:
    """Normalize text for accent/case-insensitive administrative matching."""

    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(char for char in text if not unicodedata.combining(char))
    return " ".join(text.casefold().split())


def _read_csv_flexible(
    path: str | Path,
    *,
    usecols: list[str] | None = None,
    nrows: int | None = None,
) -> pd.DataFrame:
    """Read SIAP-style CSVs with conservative encoding fallback."""

    last_error: Exception | None = None
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
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


def load_inegi_municipalities(
    contract: Mapping[str, Any],
    *,
    root: str | Path | None = None,
) -> Any:
    """Load and normalize the three INEGI municipality layers from Data Contract v2."""

    try:
        import geopandas as gpd
    except ImportError as exc:
        raise ImportError("Municipality loading requires the geo optional dependency.") from exc

    project_root = Path(root).resolve() if root is not None else find_project_root()
    spec = contract["external"]["inegi_municipios"]
    directory = project_root / spec["root"]
    frames = []

    for state_code, state_spec in spec["files"].items():
        frame = gpd.read_file(directory / state_spec["path"])
        lookup = {_normalize_label(column): str(column) for column in frame.columns}
        required = ["cvegeo", "cve_ent", "cve_mun", "nomgeo"]
        missing = [field for field in required if field not in lookup]
        if missing:
            raise ValueError(
                f"Municipality layer {state_spec['path']} missing fields: {missing}"
            )
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
        frames.append(
            frame[
                [
                    "cvegeo",
                    "cve_ent",
                    "cve_mun",
                    "nomgeo",
                    "state_code",
                    "state_name",
                    "geometry",
                ]
            ].copy()
        )

    if not frames:
        raise ValueError("No municipality layers were loaded.")
    base_crs = frames[0].crs
    if base_crs is None or any(frame.crs != base_crs for frame in frames[1:]):
        raise ValueError("Municipality layers must share a known CRS.")
    return gpd.GeoDataFrame(
        pd.concat(frames, ignore_index=True),
        geometry="geometry",
        crs=base_crs,
    )


def build_admin_mapping(
    parcels: Any,
    municipalities: Any,
    *,
    area_crs: str = "EPSG:6372",
) -> pd.DataFrame:
    """Resolve official administrative keys and spatial-overlap QA per parcel."""

    try:
        import geopandas as gpd
    except ImportError as exc:
        raise ImportError("Administrative mapping requires the geo optional dependency.") from exc

    parcels = canonicalize_parcels(parcels)
    required = {ID_COLUMN, "Estado", "Municipio", "geometry"}
    missing = required.difference(parcels.columns)
    if missing:
        raise KeyError(f"Parcel data missing column(s): {sorted(missing)}")

    projected_parcels = parcels[[ID_COLUMN, "geometry"]].to_crs(area_crs).copy()
    projected_parcels["_parcel_area_m2"] = projected_parcels.geometry.area
    projected_munis = municipalities.to_crs(area_crs)
    intersections = gpd.overlay(
        projected_parcels,
        projected_munis,
        how="intersection",
        keep_geom_type=False,
    )
    intersections["_overlap_area_m2"] = intersections.geometry.area
    winners = intersections.loc[
        intersections.groupby(ID_COLUMN)["_overlap_area_m2"].idxmax()
    ].copy()
    winners["spatial_overlap_fraction"] = (
        winners["_overlap_area_m2"] / winners["_parcel_area_m2"]
    )
    spatial = winners.set_index(ID_COLUMN)

    rows = []
    for _, parcel in parcels.sort_values(ID_COLUMN).iterrows():
        parcel_id = str(parcel[ID_COLUMN])
        official = municipalities.loc[
            municipalities["state_name"].map(_normalize_label).eq(_normalize_label(parcel["Estado"]))
            & municipalities["nomgeo"].map(_normalize_label).eq(
                _normalize_label(parcel["Municipio"])
            )
        ]
        spatial_row = spatial.loc[parcel_id]

        if len(official) == 1:
            admin_row = official.iloc[0]
            source = "official_name"
        elif len(official) == 0:
            admin_row = spatial_row
            source = "spatial_fallback"
        else:
            raise ValueError(f"Ambiguous official municipality match for {parcel_id}.")

        cvegeo = str(admin_row["cvegeo"]).zfill(5)
        rows.append(
            {
                ID_COLUMN: parcel_id,
                "cve_ent": str(admin_row["cve_ent"]).zfill(2),
                "cve_mun": str(admin_row["cve_mun"]).zfill(3),
                "cvegeo": cvegeo,
                "admin_join_source": source,
                "spatial_cvegeo": str(spatial_row["cvegeo"]).zfill(5),
                "spatial_municipio": str(spatial_row["nomgeo"]),
                "spatial_overlap_fraction": float(spatial_row["spatial_overlap_fraction"]),
                "official_spatial_match": (
                    cvegeo == str(spatial_row["cvegeo"]).zfill(5)
                ),
            }
        )

    result = pd.DataFrame(rows)
    if len(result) != len(parcels) or not result[ID_COLUMN].is_unique:
        raise ValueError("Administrative mapping failed one-row-per-parcel invariant.")
    return result


def _canonical_siap_files(directory: Path, spec: Mapping[str, Any]) -> list[Path]:
    pattern = re.compile(spec["yearly_file_regex"])
    grouped: dict[int, list[Path]] = {}
    for path in sorted(directory.glob("*.csv")):
        match = pattern.match(path.name)
        if match:
            grouped.setdefault(int(match.group(1)), []).append(path)

    files = []
    for year in range(int(spec["year_min"]), int(spec["year_max"]) + 1):
        candidates = grouped.get(year, [])
        if not candidates:
            raise FileNotFoundError(f"Missing SIAP year {year}.")
        canonical = directory / f"Cierre_agricola_mun_{year}.csv"
        files.append(canonical if canonical in candidates else candidates[0])
    return files


def load_siap_barley_history(
    contract: Mapping[str, Any],
    *,
    root: str | Path | None = None,
) -> pd.DataFrame:
    """Load only target-state barley rows from the canonical SIAP yearly files."""

    project_root = Path(root).resolve() if root is not None else find_project_root()
    spec = contract["external"]["siap"]
    directory = project_root / spec["root"]
    target_states = {int(value) for value in spec["target_state_codes"]}
    aliases = list(spec.get("crop_name_aliases", ["Nomcultivo"]))
    pieces = []

    for path in _canonical_siap_files(directory, spec):
        header = _read_csv_flexible(path, nrows=0)
        crop_column = next((alias for alias in aliases if alias in header.columns), None)
        if crop_column is None:
            raise ValueError(f"No declared SIAP crop-name alias in {path.name}.")

        columns = [
            "Anio",
            "Idestado",
            "Idmunicipio",
            crop_column,
            "Sembrada",
            "Cosechada",
            "Siniestrada",
            "Volumenproduccion",
        ]
        frame = _read_csv_flexible(path, usecols=columns)
        if crop_column != "Nomcultivo":
            frame = frame.rename(columns={crop_column: "Nomcultivo"})

        state = pd.to_numeric(frame["Idestado"], errors="coerce")
        crop_mask = frame["Nomcultivo"].astype(str).str.contains(
            spec["crop_discovery_regex"],
            regex=True,
            na=False,
        )
        frame = frame.loc[state.isin(target_states) & crop_mask].copy()
        municipality = pd.to_numeric(frame["Idmunicipio"], errors="coerce").astype("Int64")
        state_int = pd.to_numeric(frame["Idestado"], errors="coerce").astype("Int64")
        frame["cvegeo"] = (
            state_int.astype("string").str.zfill(2)
            + municipality.astype("string").str.zfill(3)
        )
        pieces.append(frame)

    if not pieces:
        raise ValueError("No SIAP barley rows found in target states.")
    return pd.concat(pieces, ignore_index=True)
