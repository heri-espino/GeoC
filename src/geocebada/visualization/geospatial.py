"""Interactive geospatial visualizations for GeoCebada."""

from __future__ import annotations

import json
from typing import Any, Iterable

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from geocebada.geo import aggregate_to_parcels


def parcel_map_figure(
    parcels: Any,
    data: pd.DataFrame,
    *,
    parcel_id: str,
    data_id: str | None = None,
    color: str,
    hover_columns: Iterable[str] | None = None,
    opacity: float = 0.65,
) -> go.Figure:
    """Build an interactive choropleth map after a source-backed CRS reprojection.

    The parcel GeoDataFrame must already carry its source CRS. GeoCebada deliberately
    refuses to guess a missing CRS. Repeated data rows are reduced to parcel level
    using :func:`geocebada.geo.aggregate_to_parcels` for exploratory display only.
    """

    data_id = data_id or parcel_id
    if parcel_id not in parcels.columns:
        raise KeyError(parcel_id)
    if data_id not in data.columns:
        raise KeyError(data_id)
    if color not in data.columns:
        raise KeyError(color)
    if getattr(parcels, "crs", None) is None:
        raise ValueError(
            "Parcel geometries have no CRS metadata. GeoCebada will not guess a CRS; "
            "resolve it from the source documentation before mapping."
        )

    hover = [column for column in (hover_columns or []) if column in data.columns]
    summary_columns = list(dict.fromkeys([color, *hover]))
    summary = aggregate_to_parcels(data, data_id, columns=summary_columns)

    geometry = parcels.copy()
    geometry[parcel_id] = geometry[parcel_id].astype("string")
    summary[data_id] = summary[data_id].astype("string")
    merged = geometry.merge(summary, left_on=parcel_id, right_on=data_id, how="inner")
    if merged.empty:
        raise ValueError("No parcel IDs matched between geometry and the filtered data.")

    mapped = merged.to_crs("EPSG:4326").reset_index(drop=True)
    mapped["__map_id__"] = mapped.index.astype(str)
    geojson = json.loads(mapped.to_json())

    minx, miny, maxx, maxy = mapped.total_bounds
    center = {"lon": float((minx + maxx) / 2), "lat": float((miny + maxy) / 2)}

    figure = px.choropleth_map(
        mapped,
        geojson=geojson,
        locations="__map_id__",
        featureidkey="id",
        color=color,
        hover_name=parcel_id,
        hover_data={column: True for column in hover},
        map_style="carto-positron",
        center=center,
        zoom=6,
        opacity=opacity,
    )
    figure.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0})
    return figure
