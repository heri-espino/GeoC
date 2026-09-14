"""Parcel geometry loading and aggregation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from geocebada.paths import source_path

DEFAULT_PARCELS_ARCHIVE = "Parcelas_Reto_AGC_CONJUNTO_70_30.zip"


def load_parcels(
    path: str | Path | None = None,
    *,
    root: str | Path | None = None,
) -> Any:
    """Load the official parcel geometries as a GeoDataFrame.

    GeoPandas is imported lazily so non-geospatial users of ``geocebada`` do not
    need the optional ``geo`` dependency. The function never assigns or guesses a
    CRS: if the source geometry has no CRS, downstream mapping code must stop and
    ask for a source-backed CRS decision.
    """

    try:
        import geopandas as gpd
    except ImportError as exc:  # pragma: no cover - depends on optional dependency
        raise ImportError(
            "Parcel loading requires the optional geospatial dependencies. "
            "Install GeoCebada with `pip install -e '.[geo]'`."
        ) from exc

    parcel_path = (
        Path(path).expanduser()
        if path is not None
        else source_path(
            "geospatial",
            DEFAULT_PARCELS_ARCHIVE,
            root=root,
            must_exist=True,
        )
    )
    if path is not None and not parcel_path.is_absolute():
        from geocebada.paths import project_path

        parcel_path = project_path(parcel_path, root=root, must_exist=True)

    if parcel_path.suffix.lower() == ".zip":
        uri = f"zip://{parcel_path.resolve()}"
    else:
        uri = str(parcel_path)

    frame = gpd.read_file(uri)
    if frame.empty:
        raise ValueError(f"No parcel geometries were found in {parcel_path}.")
    if frame.geometry.isna().all():
        raise ValueError("Parcel source contains no usable geometries.")
    return frame


def aggregate_to_parcels(
    frame: pd.DataFrame,
    id_column: str,
    *,
    columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Collapse repeated observations to one row per parcel for mapping.

    Numeric variables are averaged. Non-numeric variables use the first non-null
    value. This helper is intended for exploratory maps only; scientifically
    meaningful temporal aggregation should be implemented as explicit feature
    engineering rather than relying on this generic summary.
    """

    if id_column not in frame.columns:
        raise KeyError(id_column)

    selected = list(columns) if columns is not None else [c for c in frame.columns if c != id_column]
    selected = [column for column in selected if column != id_column]
    missing = [column for column in selected if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing column(s): {missing}")

    working = frame[[id_column, *selected]].copy()
    aggregations: dict[str, str] = {}
    for column in selected:
        if pd.api.types.is_numeric_dtype(working[column]):
            aggregations[column] = "mean"
        else:
            aggregations[column] = "first"

    if not aggregations:
        return working[[id_column]].drop_duplicates().reset_index(drop=True)

    return working.groupby(id_column, as_index=False, dropna=False).agg(aggregations)
