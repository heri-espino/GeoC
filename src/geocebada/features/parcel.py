"""Reusable one-row-per-parcel feature extraction for GeoCebada.

These transforms are deterministic and target-free. They preserve source provenance
and classify each feature as clean or competition for downstream modeling.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ID_COLUMN = "ID_POLIGONO"
TARGET_COLUMN = "RENDIMIENTO_T_HA"
SPLIT_COLUMN = "CONJUNTO"

MetadataRecord = dict[str, str]


def _slug(value: object) -> str:
    """Return a stable lowercase ASCII feature-name token."""

    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()


def _provenance_records(
    columns: Iterable[str],
    *,
    source: str,
    mode: str,
    description: str,
) -> list[MetadataRecord]:
    """Build feature-provenance records."""

    return [
        {
            "column": column,
            "source": source,
            "mode": mode,
            "description": description,
        }
        for column in columns
    ]


def canonicalize_parcels(parcels: Any) -> Any:
    """Normalize official parcel ID/area aliases without editing source evidence."""

    result = parcels.copy()
    if ID_COLUMN not in result.columns and "ID_POLIGON" in result.columns:
        result = result.rename(columns={"ID_POLIGON": ID_COLUMN})
    if "area_ha" not in result.columns:
        for alias in ("área_ha", "AREA_HA"):
            if alias in result.columns:
                result = result.rename(columns={alias: "area_ha"})
                break
    return result


def build_base_features(
    split: pd.DataFrame,
    parcels: Any,
    *,
    admin: pd.DataFrame | None = None,
    projected_area_crs: str = "EPSG:6372",
) -> tuple[pd.DataFrame, list[MetadataRecord]]:
    """Build identity, geometry and administrative parcel features."""

    required = {ID_COLUMN, TARGET_COLUMN, SPLIT_COLUMN}
    missing = required.difference(split.columns)
    if missing:
        raise KeyError(f"Split missing column(s): {sorted(missing)}")

    parcels = canonicalize_parcels(parcels)
    if parcels.crs is None:
        raise ValueError("Parcel CRS is required.")

    base = split[[ID_COLUMN, SPLIT_COLUMN, TARGET_COLUMN]].copy()
    if "AREA_HA" in split.columns:
        base["base_area_ha"] = pd.to_numeric(split["AREA_HA"], errors="coerce")

    geo = parcels[[ID_COLUMN, "geometry"]].copy()
    projected = geo.to_crs(projected_area_crs)
    centroids = projected.copy()
    centroids.geometry = projected.geometry.centroid
    centroids = centroids.to_crs("EPSG:4326")
    geometry = pd.DataFrame(
        {
            ID_COLUMN: geo[ID_COLUMN].astype(str).to_numpy(),
            "base_geom_area_ha": (projected.geometry.area / 10_000.0).to_numpy(),
            "base_centroid_lon": centroids.geometry.x.to_numpy(),
            "base_centroid_lat": centroids.geometry.y.to_numpy(),
        }
    )
    base = base.merge(geometry, on=ID_COLUMN, how="left", validate="1:1")

    if "base_area_ha" not in base.columns and "area_ha" in parcels.columns:
        area = parcels[[ID_COLUMN, "area_ha"]].rename(columns={"area_ha": "base_area_ha"})
        base = base.merge(area, on=ID_COLUMN, how="left", validate="1:1")

    if "base_area_ha" in base.columns:
        base["base_area_delta_ha"] = base["base_geom_area_ha"] - base["base_area_ha"]
        base["base_area_ratio"] = (
            base["base_geom_area_ha"] / base["base_area_ha"].replace(0, np.nan)
        )

    for source_name, output_name in (("Estado", "meta_estado"), ("Municipio", "meta_municipio")):
        if source_name in parcels.columns:
            labels = parcels[[ID_COLUMN, source_name]].rename(columns={source_name: output_name})
            base = base.merge(labels, on=ID_COLUMN, how="left", validate="1:1")

    if admin is not None:
        admin_block = admin.copy()
        rename = {
            column: f"admin_{_slug(column)}"
            for column in admin_block.columns
            if column != ID_COLUMN
        }
        admin_block = admin_block.rename(columns=rename)
        base = base.merge(admin_block, on=ID_COLUMN, how="left", validate="1:1")

    excluded = {ID_COLUMN, SPLIT_COLUMN, TARGET_COLUMN, "meta_estado", "meta_municipio"}
    columns = [column for column in base.columns if column not in excluded]
    return base, _provenance_records(
        columns,
        source="official_split+official_parcels+admin",
        mode="clean",
        description="Static parcel geometry or administrative feature.",
    )


def _flatten_sensor_aggregation(
    grouped: pd.DataFrame,
    *,
    prefix: str,
    sensor_column: str,
) -> pd.DataFrame:
    if grouped.empty:
        return pd.DataFrame(columns=[ID_COLUMN])
    wide = grouped.unstack(sensor_column)
    labels: list[str] = []
    for column in wide.columns:
        if len(column) == 3:
            variable, aggregation, sensor = column
            labels.append(
                f"{prefix}__{_slug(sensor)}__{_slug(variable)}__{_slug(aggregation)}"
            )
        elif len(column) == 2:
            variable, sensor = column
            labels.append(f"{prefix}__{_slug(sensor)}__{_slug(variable)}")
        else:
            raise ValueError(f"Unexpected grouped column shape: {column!r}")
    wide.columns = labels
    return wide.reset_index()


def _period_satellite_summary(
    frame: pd.DataFrame,
    *,
    prefix: str,
    aggregations: Sequence[str],
    sensor_column: str,
) -> pd.DataFrame:
    metadata = {ID_COLUMN, "fecha_captura", sensor_column, "porcentaje_nubosidad"}
    numeric = [
        column
        for column in frame.columns
        if column not in metadata
        and pd.api.types.is_numeric_dtype(frame[column])
        and frame[column].notna().any()
    ]
    if not numeric:
        return pd.DataFrame({ID_COLUMN: sorted(frame[ID_COLUMN].astype(str).unique())})
    grouped = frame.groupby([ID_COLUMN, sensor_column], observed=True)[numeric].agg(
        list(aggregations)
    )
    return _flatten_sensor_aggregation(
        grouped,
        prefix=prefix,
        sensor_column=sensor_column,
    )


def _satellite_coverage(
    frame: pd.DataFrame,
    *,
    prefix: str,
    sensor_column: str,
) -> pd.DataFrame:
    metadata = {ID_COLUMN, "fecha_captura", sensor_column, "porcentaje_nubosidad"}
    feature_columns = [column for column in frame.columns if column not in metadata]
    work = frame[[ID_COLUMN, "fecha_captura", sensor_column, "porcentaje_nubosidad"]].copy()
    work["_valid_fraction"] = (
        1.0 - frame[feature_columns].isna().mean(axis=1)
        if feature_columns
        else np.nan
    )
    grouped = work.groupby([ID_COLUMN, sensor_column], observed=True).agg(
        n_records=("fecha_captura", "size"),
        n_dates=("fecha_captura", "nunique"),
        cloud_mean=("porcentaje_nubosidad", "mean"),
        cloud_median=("porcentaje_nubosidad", "median"),
        valid_fraction_mean=("_valid_fraction", "mean"),
    )
    return _flatten_sensor_aggregation(
        grouped,
        prefix=prefix,
        sensor_column=sensor_column,
    )


def _monthly_promedio_features(
    frame: pd.DataFrame,
    *,
    prefix: str,
    months: Sequence[int],
    sensor_column: str,
) -> pd.DataFrame:
    value_columns = [
        column
        for column in frame.columns
        if column.endswith("_promedio")
        and pd.api.types.is_numeric_dtype(frame[column])
        and frame[column].notna().any()
    ]
    result = pd.DataFrame({ID_COLUMN: sorted(frame[ID_COLUMN].astype(str).unique())})
    if not value_columns:
        return result

    work = frame[[ID_COLUMN, "fecha_captura", sensor_column, *value_columns]].copy()
    work["_month"] = work["fecha_captura"].dt.month
    for month in months:
        month_frame = work.loc[work["_month"].eq(int(month))]
        if month_frame.empty:
            continue
        grouped = month_frame.groupby([ID_COLUMN, sensor_column], observed=True)[
            value_columns
        ].mean()
        wide = grouped.unstack(sensor_column)
        wide.columns = [
            f"{prefix}__{_slug(sensor)}__{_slug(variable)}__m{month:02d}"
            for variable, sensor in wide.columns
        ]
        result = result.merge(wide.reset_index(), on=ID_COLUMN, how="left", validate="1:1")
    return result


def build_satellite_features(
    frame: pd.DataFrame,
    *,
    source_prefix: str,
    target_start: str,
    target_end: str,
    historical_start: str,
    historical_end: str,
    historical_months: Sequence[int],
    target_months: Sequence[int],
    aggregations: Sequence[str] = ("mean", "std", "min", "max"),
    sensor_column: str = "sensor",
) -> tuple[pd.DataFrame, list[MetadataRecord]]:
    """Build historical-clean and full-season-2025 satellite features."""

    required = {ID_COLUMN, "fecha_captura", sensor_column, "porcentaje_nubosidad"}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Satellite table missing column(s): {sorted(missing)}")

    work = frame.copy()
    work["fecha_captura"] = pd.to_datetime(
        work["fecha_captura"],
        format="%d/%m/%Y",
        errors="raise",
    )
    result = pd.DataFrame({ID_COLUMN: sorted(work[ID_COLUMN].astype(str).unique())})
    records: list[MetadataRecord] = []

    historical = work.loc[
        work["fecha_captura"].between(pd.Timestamp(historical_start), pd.Timestamp(historical_end))
        & work["fecha_captura"].dt.month.isin(list(historical_months))
    ].copy()
    target = work.loc[
        work["fecha_captura"].between(pd.Timestamp(target_start), pd.Timestamp(target_end))
        & work["fecha_captura"].dt.month.isin(list(target_months))
    ].copy()

    blocks = [
        (
            _period_satellite_summary(
                historical,
                prefix=f"{source_prefix}_hist",
                aggregations=aggregations,
                sensor_column=sensor_column,
            ),
            "clean",
            "Historical Apr-Oct satellite summary through 2024.",
        ),
        (
            _satellite_coverage(
                historical,
                prefix=f"{source_prefix}_hist_coverage",
                sensor_column=sensor_column,
            ),
            "clean",
            "Historical satellite capture/cloud/missingness coverage.",
        ),
        (
            _monthly_promedio_features(
                historical,
                prefix=f"{source_prefix}_hist_monthly",
                months=historical_months,
                sensor_column=sensor_column,
            ),
            "clean",
            "Historical Apr-Oct monthly climatology of promedio satellite variables.",
        ),
        (
            _period_satellite_summary(
                target,
                prefix=f"{source_prefix}_2025",
                aggregations=aggregations,
                sensor_column=sensor_column,
            ),
            "competition",
            "Full target-season 2025 satellite summary.",
        ),
        (
            _satellite_coverage(
                target,
                prefix=f"{source_prefix}_2025_coverage",
                sensor_column=sensor_column,
            ),
            "competition",
            "Full target-season 2025 satellite coverage.",
        ),
        (
            _monthly_promedio_features(
                target,
                prefix=f"{source_prefix}_2025_monthly",
                months=target_months,
                sensor_column=sensor_column,
            ),
            "competition",
            "Full target-season 2025 monthly promedio satellite means.",
        ),
    ]
    for block, mode, description in blocks:
        columns = [column for column in block.columns if column != ID_COLUMN]
        result = result.merge(block, on=ID_COLUMN, how="left", validate="1:1")
        records.extend(
            _provenance_records(
                columns,
                source=source_prefix,
                mode=mode,
                description=description,
            )
        )
    return result, records


def zonal_stats_for_raster(
    parcels: Any,
    raster_path: str | Path,
    *,
    stats: Sequence[str] = ("mean",),
    crs_override: str | None = None,
    divisor: float = 1.0,
) -> pd.DataFrame:
    """Compute parcel zonal statistics for one raster."""

    try:
        import rasterio
        from rasterio.mask import mask
        from shapely.geometry import mapping
    except ImportError as exc:
        raise ImportError("Raster extraction requires the geo optional dependency.") from exc

    allowed = {"mean", "std", "min", "max", "median", "count", "sum"}
    unknown = set(stats).difference(allowed)
    if unknown:
        raise ValueError(f"Unsupported zonal stat(s): {sorted(unknown)}")
    if divisor == 0:
        raise ValueError("divisor must be non-zero.")

    rows: list[dict[str, float | str]] = []
    with rasterio.open(Path(raster_path)) as dataset:
        target_crs = dataset.crs or crs_override
        if target_crs is None:
            raise ValueError(f"Raster CRS missing with no approved override: {raster_path}")
        projected = canonicalize_parcels(parcels)[[ID_COLUMN, "geometry"]].to_crs(target_crs)

        for _, parcel in projected.iterrows():
            record: dict[str, float | str] = {ID_COLUMN: str(parcel[ID_COLUMN])}
            try:
                array, _ = mask(
                    dataset,
                    [mapping(parcel.geometry)],
                    crop=True,
                    all_touched=True,
                    filled=False,
                    indexes=1,
                )
                values = np.asarray(array.compressed(), dtype=float)
            except ValueError:
                values = np.array([], dtype=float)
            values = values[np.isfinite(values)] / float(divisor)

            for stat in stats:
                if values.size == 0:
                    value = np.nan
                elif stat == "mean":
                    value = float(np.mean(values))
                elif stat == "std":
                    value = float(np.std(values, ddof=0))
                elif stat == "min":
                    value = float(np.min(values))
                elif stat == "max":
                    value = float(np.max(values))
                elif stat == "median":
                    value = float(np.median(values))
                elif stat == "count":
                    value = float(values.size)
                else:
                    value = float(np.sum(values))
                record[stat] = value
            rows.append(record)
    return pd.DataFrame(rows)


def build_static_raster_features(
    parcels: Any,
    raster_specs: Sequence[Mapping[str, Any]],
    *,
    source: str,
    mode: str = "clean",
) -> tuple[pd.DataFrame, list[MetadataRecord]]:
    """Build named static-raster zonal-stat feature blocks."""

    result = pd.DataFrame(
        {ID_COLUMN: sorted(canonicalize_parcels(parcels)[ID_COLUMN].astype(str).unique())}
    )
    records: list[MetadataRecord] = []
    for spec in raster_specs:
        stats = tuple(spec.get("stats", ("mean",)))
        block = zonal_stats_for_raster(
            parcels,
            spec["path"],
            stats=stats,
            crs_override=spec.get("crs_override"),
            divisor=float(spec.get("divisor", 1.0)),
        )
        rename = {
            stat: f"{spec['name']}__{_slug(stat)}"
            for stat in stats
        }
        block = block.rename(columns=rename)
        result = result.merge(block, on=ID_COLUMN, how="left", validate="1:1")
        records.extend(
            _provenance_records(
                rename.values(),
                source=source,
                mode=mode,
                description=f"Parcel zonal statistic from {Path(spec['path']).name}.",
            )
        )
    return result, records


def build_official_climate_features(
    parcels: Any,
    raster_paths: Sequence[str | Path],
    *,
    historical_years: Sequence[int],
    target_year: int,
    months: Sequence[int],
) -> tuple[pd.DataFrame, list[MetadataRecord]]:
    """Build 2022-2024 climatology and full-season 2025 climate features."""

    pattern = re.compile(r"^(PREC|Tmin|Tmax)_(\d{4})_(\d{2})\.tif$", re.I)
    ids = sorted(canonicalize_parcels(parcels)[ID_COLUMN].astype(str).unique())
    index = pd.Index(ids, name=ID_COLUMN)
    values: dict[tuple[str, int, int], pd.Series] = {}

    for item in raster_paths:
        path = Path(item)
        match = pattern.match(path.name)
        if match is None:
            continue
        variable = match.group(1).lower()
        year = int(match.group(2))
        month = int(match.group(3))
        if month not in months or year not in {*historical_years, target_year}:
            continue
        block = zonal_stats_for_raster(parcels, path, stats=("mean",))
        values[(variable, year, month)] = block.set_index(ID_COLUMN)["mean"].reindex(index)

    result = pd.DataFrame({ID_COLUMN: ids})
    records: list[MetadataRecord] = []

    for variable in ("prec", "tmin", "tmax"):
        for month in months:
            history = [
                values[(variable, int(year), int(month))]
                for year in historical_years
                if (variable, int(year), int(month)) in values
            ]
            hist_column = f"clim_official_hist__{variable}__m{month:02d}_mean"
            if history:
                result[hist_column] = pd.concat(history, axis=1).mean(axis=1).to_numpy()
                records.extend(
                    _provenance_records(
                        [hist_column],
                        source="official_climate",
                        mode="clean",
                        description="2022-2024 monthly parcel climate climatology.",
                    )
                )

            key = (variable, int(target_year), int(month))
            if key in values:
                target_column = f"clim_official_2025__{variable}__m{month:02d}"
                result[target_column] = values[key].to_numpy()
                records.extend(
                    _provenance_records(
                        [target_column],
                        source="official_climate",
                        mode="competition",
                        description="Full target-season 2025 monthly climate value.",
                    )
                )
                if hist_column in result.columns:
                    anomaly = f"clim_official_2025__{variable}__m{month:02d}_anomaly"
                    result[anomaly] = result[target_column] - result[hist_column]
                    records.extend(
                        _provenance_records(
                            [anomaly],
                            source="official_climate",
                            mode="competition",
                            description="2025 monthly anomaly versus 2022-2024 climatology.",
                        )
                    )

    for prefix, variable, operation, mode in [
        ("clim_official_hist", "prec", "sum", "clean"),
        ("clim_official_hist", "tmin", "mean", "clean"),
        ("clim_official_hist", "tmax", "mean", "clean"),
        ("clim_official_2025", "prec", "sum", "competition"),
        ("clim_official_2025", "tmin", "mean", "competition"),
        ("clim_official_2025", "tmax", "mean", "competition"),
    ]:
        if prefix.endswith("hist"):
            columns = [
                f"{prefix}__{variable}__m{month:02d}_mean"
                for month in months
                if f"{prefix}__{variable}__m{month:02d}_mean" in result
            ]
        else:
            columns = [
                f"{prefix}__{variable}__m{month:02d}"
                for month in months
                if f"{prefix}__{variable}__m{month:02d}" in result
            ]
        if not columns:
            continue
        suffix = "season_sum" if operation == "sum" else "season_mean"
        output = f"{prefix}__{variable}__{suffix}"
        if operation == "sum":
            result[output] = result[columns].sum(axis=1, min_count=1)
        else:
            result[output] = result[columns].mean(axis=1)
        records.extend(
            _provenance_records(
                [output],
                source="official_climate",
                mode=mode,
                description="April-Oct aggregate derived from monthly climate features.",
            )
        )
    return result, records


def _longest_true_run(values: np.ndarray) -> int:
    """Return the longest consecutive True run."""

    best = 0
    current = 0
    for value in values:
        if bool(value):
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def build_daily_chirps_features(
    parcels: Any,
    dated_rasters: Sequence[tuple[pd.Timestamp, str | Path]],
    *,
    rain_day_threshold_mm: float = 1.0,
    heavy_rain_threshold_mm: float = 10.0,
    rolling_window_days: int = 5,
) -> tuple[pd.DataFrame, list[MetadataRecord]]:
    """Build full-season 2025 daily precipitation/extreme features."""

    ids = sorted(canonicalize_parcels(parcels)[ID_COLUMN].astype(str).unique())
    daily = pd.DataFrame(index=pd.Index(ids, name=ID_COLUMN))
    for day, path in sorted(dated_rasters, key=lambda item: item[0]):
        block = zonal_stats_for_raster(parcels, path, stats=("mean",))
        daily[pd.Timestamp(day)] = block.set_index(ID_COLUMN)["mean"].reindex(daily.index)

    result = pd.DataFrame({ID_COLUMN: ids})
    if daily.empty:
        return result, []

    result["chirps_daily_2025__season_sum"] = daily.sum(axis=1, min_count=1)
    result["chirps_daily_2025__daily_mean"] = daily.mean(axis=1)
    result["chirps_daily_2025__max_1d"] = daily.max(axis=1)
    rolling = daily.T.rolling(int(rolling_window_days), min_periods=1).sum().T
    result[f"chirps_daily_2025__max_{rolling_window_days}d"] = rolling.max(axis=1)
    result["chirps_daily_2025__rain_days"] = (daily >= rain_day_threshold_mm).sum(axis=1)
    result["chirps_daily_2025__heavy_rain_days"] = (
        daily >= heavy_rain_threshold_mm
    ).sum(axis=1)
    result["chirps_daily_2025__dry_days"] = (daily < rain_day_threshold_mm).sum(axis=1)

    dry_spell = []
    for _, row in daily.iterrows():
        valid = row.notna().to_numpy()
        dry = (row.to_numpy(dtype=float) < rain_day_threshold_mm) & valid
        dry_spell.append(float(_longest_true_run(dry)))
    result["chirps_daily_2025__max_dry_spell_days"] = dry_spell

    for month in sorted({timestamp.month for timestamp in daily.columns}):
        columns = [column for column in daily.columns if column.month == month]
        result[f"chirps_daily_2025__m{month:02d}_sum"] = daily[columns].sum(
            axis=1,
            min_count=1,
        )

    columns = [column for column in result.columns if column != ID_COLUMN]
    return result, _provenance_records(
        columns,
        source="external_chirps_daily",
        mode="competition",
        description="Full April-Oct 2025 daily CHIRPS precipitation aggregate/extreme.",
    )


def build_soilgrids_features(
    parcels: Any,
    raster_specs: Sequence[Mapping[str, Any]],
    *,
    depth_weights_cm: Mapping[str, float],
) -> tuple[pd.DataFrame, list[MetadataRecord]]:
    """Build scaled depth-specific and depth-weighted SoilGrids features."""

    result = pd.DataFrame(
        {ID_COLUMN: sorted(canonicalize_parcels(parcels)[ID_COLUMN].astype(str).unique())}
    )
    records: list[MetadataRecord] = []
    mean_columns: dict[tuple[str, str], str] = {}

    for spec in raster_specs:
        prop = str(spec["property"])
        depth = str(spec["depth"])
        stats = tuple(spec.get("stats", ("mean", "std")))
        block = zonal_stats_for_raster(
            parcels,
            spec["path"],
            stats=stats,
            crs_override=spec.get("crs_override"),
            divisor=float(spec.get("divisor", 1.0)),
        )
        rename = {
            stat: f"soilgrids__{_slug(prop)}__{_slug(depth)}__{_slug(stat)}"
            for stat in stats
        }
        block = block.rename(columns=rename)
        result = result.merge(block, on=ID_COLUMN, how="left", validate="1:1")
        if "mean" in rename:
            mean_columns[(prop, depth)] = rename["mean"]
        records.extend(
            _provenance_records(
                rename.values(),
                source="external_soilgrids",
                mode="clean",
                description="Scaled SoilGrids parcel zonal statistic.",
            )
        )

    intervals = {
        "0_30cm": ["0-5cm", "5-15cm", "15-30cm"],
        "0_60cm": ["0-5cm", "5-15cm", "15-30cm", "30-60cm"],
    }
    for prop in sorted({key[0] for key in mean_columns}):
        for label, depths in intervals.items():
            columns = [mean_columns.get((prop, depth)) for depth in depths]
            if any(column is None for column in columns):
                continue
            weights = np.array([float(depth_weights_cm[depth]) for depth in depths])
            matrix = result[list(columns)].to_numpy(dtype=float)
            complete = np.isfinite(matrix).all(axis=1)
            values = np.full(len(result), np.nan)
            values[complete] = np.average(matrix[complete], axis=1, weights=weights)
            output = f"soilgrids__{_slug(prop)}__depth_weighted_{label}"
            result[output] = values
            records.extend(
                _provenance_records(
                    [output],
                    source="external_soilgrids",
                    mode="clean",
                    description="Thickness-weighted SoilGrids property mean.",
                )
            )
    return result, records


def _safe_trend(years: pd.Series, values: pd.Series) -> float:
    """Return a linear trend per year for finite pairs."""

    x = pd.to_numeric(years, errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2 or np.ptp(x[mask]) == 0:
        return float("nan")
    return float(np.polyfit(x[mask], y[mask], 1)[0])


def build_siap_features(
    admin: pd.DataFrame,
    siap: pd.DataFrame,
    *,
    crop: str = "Cebada grano",
    historical_end_year: int = 2024,
    recent_years: Sequence[int] = (2020, 2021, 2022, 2023, 2024),
    competition_year: int = 2025,
) -> tuple[pd.DataFrame, list[MetadataRecord]]:
    """Build municipal grain-barley history and 2025 proxy features."""

    if not {ID_COLUMN, "cvegeo"}.issubset(admin.columns):
        raise KeyError("Admin table must contain ID_POLIGONO and cvegeo.")
    required = {
        "Anio",
        "cvegeo",
        "Nomcultivo",
        "Sembrada",
        "Cosechada",
        "Siniestrada",
        "Volumenproduccion",
    }
    missing = required.difference(siap.columns)
    if missing:
        raise KeyError(f"SIAP table missing column(s): {sorted(missing)}")

    work = siap.copy()
    work["cvegeo"] = work["cvegeo"].astype("string").str.zfill(5)
    work = work.loc[work["Nomcultivo"].astype(str).str.casefold().eq(crop.casefold())].copy()
    for column in ("Sembrada", "Cosechada", "Siniestrada", "Volumenproduccion"):
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work["Anio"] = pd.to_numeric(work["Anio"], errors="coerce")

    annual = (
        work.groupby(["cvegeo", "Anio"], as_index=False)
        .agg(
            sembrada=("Sembrada", "sum"),
            cosechada=("Cosechada", "sum"),
            siniestrada=("Siniestrada", "sum"),
            produccion=("Volumenproduccion", "sum"),
        )
        .sort_values(["cvegeo", "Anio"])
    )
    annual["yield_t_ha"] = annual["produccion"] / annual["cosechada"].replace(0, np.nan)
    annual["damage_rate"] = annual["siniestrada"] / annual["sembrada"].replace(0, np.nan)

    rows = []
    for cvegeo, group in annual.groupby("cvegeo", sort=True):
        hist = group.loc[group["Anio"].le(historical_end_year)]
        recent = hist.loc[hist["Anio"].isin(list(recent_years))]
        latest = hist.loc[hist["Anio"].eq(historical_end_year)]
        comp = group.loc[group["Anio"].eq(competition_year)]
        row: dict[str, float | str] = {"cvegeo": str(cvegeo).zfill(5)}
        row["siap_hist__yield_mean_all"] = float(hist["yield_t_ha"].mean())
        row["siap_hist__yield_std_all"] = float(hist["yield_t_ha"].std(ddof=0))
        row["siap_hist__yield_trend_all"] = _safe_trend(hist["Anio"], hist["yield_t_ha"])
        row["siap_hist__yield_mean_recent"] = float(recent["yield_t_ha"].mean())
        row["siap_hist__yield_std_recent"] = float(recent["yield_t_ha"].std(ddof=0))
        row["siap_hist__yield_trend_recent"] = _safe_trend(
            recent["Anio"],
            recent["yield_t_ha"],
        )
        for year in recent_years:
            year_row = hist.loc[hist["Anio"].eq(int(year))]
            row[f"siap_hist__yield_{year}"] = (
                float(year_row["yield_t_ha"].iloc[0]) if len(year_row) else np.nan
            )
        for name, source_column in [
            ("yield", "yield_t_ha"),
            ("harvested_ha", "cosechada"),
            ("production_t", "produccion"),
            ("damage_rate", "damage_rate"),
        ]:
            row[f"siap_hist__{name}_{historical_end_year}"] = (
                float(latest[source_column].iloc[0]) if len(latest) else np.nan
            )
            row[f"siap_{competition_year}__{name}"] = (
                float(comp[source_column].iloc[0]) if len(comp) else np.nan
            )
        row[f"siap_{competition_year}__yield_anomaly_recent"] = (
            row[f"siap_{competition_year}__yield"] - row["siap_hist__yield_mean_recent"]
        )
        rows.append(row)

    municipal = pd.DataFrame(rows)
    mapping = admin[[ID_COLUMN, "cvegeo"]].copy()
    mapping["cvegeo"] = mapping["cvegeo"].astype("string").str.zfill(5)
    result = mapping.merge(municipal, on="cvegeo", how="left", validate="m:1").drop(
        columns="cvegeo"
    )
    clean = [column for column in result if column.startswith("siap_hist__")]
    competition = [
        column for column in result if column.startswith(f"siap_{competition_year}__")
    ]
    records = _provenance_records(
        clean,
        source="external_siap",
        mode="clean",
        description="Historical municipal Cebada grano feature through 2024.",
    )
    records.extend(
        _provenance_records(
            competition,
            source="external_siap",
            mode="competition",
            description="Contemporaneous SIAP 2025 municipal outcome proxy.",
        )
    )
    return result, records


def build_cem15_features(
    parcels: Any,
    raster_paths: Sequence[str | Path],
    *,
    state_column: str = "Estado",
    stats: Sequence[str] = ("mean", "std", "min", "max"),
) -> tuple[pd.DataFrame, list[MetadataRecord]]:
    """Build one state-appropriate high-resolution CEM elevation block."""

    parcels = canonicalize_parcels(parcels)
    code_to_state = {"13": "Hidalgo", "21": "Puebla", "29": "Tlaxcala"}
    pieces = []
    for item in raster_paths:
        path = Path(item)
        match = re.match(r"(13|21|29)_", path.name)
        if match is None:
            continue
        state = code_to_state[match.group(1)]
        subset = parcels.loc[
            parcels[state_column].astype(str).str.casefold().eq(state.casefold())
        ]
        if not subset.empty:
            pieces.append(zonal_stats_for_raster(subset, path, stats=stats))

    combined = pd.concat(pieces, ignore_index=True)
    if combined[ID_COLUMN].duplicated().any():
        raise ValueError("CEM extraction produced duplicate parcel IDs.")
    rename = {stat: f"cem15__elevation__{_slug(stat)}" for stat in stats}
    combined = combined.rename(columns=rename)
    combined["cem15__elevation__range"] = combined[rename["max"]] - combined[rename["min"]]
    columns = [*rename.values(), "cem15__elevation__range"]
    return combined[[ID_COLUMN, *columns]], _provenance_records(
        columns,
        source="external_inegi_cem",
        mode="clean",
        description="High-resolution state CEM elevation zonal statistic.",
    )


def build_wapor_features(
    parcels: Any,
    netcdf_paths: Sequence[str | Path],
) -> tuple[pd.DataFrame, list[MetadataRecord]]:
    """Build duration-weighted WaPOR AETI/NPP parcel features."""

    try:
        import xarray as xr
        from rasterio.features import geometry_mask
        from rasterio.transform import from_origin
        from shapely.geometry import mapping
    except ImportError as exc:
        raise ImportError(
            "WaPOR extraction requires geo and external optional dependencies."
        ) from exc

    parcels = canonicalize_parcels(parcels)
    result = pd.DataFrame({ID_COLUMN: sorted(parcels[ID_COLUMN].astype(str).unique())})
    records: list[MetadataRecord] = []

    for item in netcdf_paths:
        path = Path(item)
        product = "aeti" if "AETI" in path.name.upper() else "npp"
        with xr.open_dataset(path, decode_times=False, engine="h5netcdf") as dataset:
            lat = np.asarray(dataset["lat"].values, dtype=float)
            lon = np.asarray(dataset["lon"].values, dtype=float)
            flip_y = lat[0] < lat[-1]
            flip_x = lon[0] > lon[-1]
            lat_grid = lat[::-1] if flip_y else lat
            lon_grid = lon[::-1] if flip_x else lon
            xres = float(np.median(np.abs(np.diff(lon_grid))))
            yres = float(np.median(np.abs(np.diff(lat_grid))))
            transform = from_origin(
                float(lon_grid[0] - xres / 2.0),
                float(lat_grid[0] + yres / 2.0),
                xres,
                yres,
            )

            bands = sorted(
                [name for name in dataset.data_vars if re.fullmatch(r"Band\d+", name)],
                key=lambda name: int(name.replace("Band", "")),
            )
            arrays = []
            days = []
            months = []
            for band in bands:
                array = np.asarray(dataset[band].values, dtype=float)
                if flip_y:
                    array = array[::-1, :]
                if flip_x:
                    array = array[:, ::-1]
                arrays.append(array)
                attrs = dataset[band].attrs
                days.append(int(attrs.get("number_of_days", 1)))
                months.append(pd.Timestamp(str(attrs.get("start_date"))).month)

            parcel_rows = []
            for _, parcel in parcels[[ID_COLUMN, "geometry"]].to_crs("EPSG:4326").iterrows():
                inside = geometry_mask(
                    [mapping(parcel.geometry)],
                    out_shape=arrays[0].shape,
                    transform=transform,
                    invert=True,
                    all_touched=True,
                )
                rates = np.array(
                    [
                        float(np.nanmean(array[inside])) if np.any(inside) else np.nan
                        for array in arrays
                    ]
                )
                day_values = np.asarray(days, dtype=float)
                valid = np.isfinite(rates)
                row: dict[str, float | str] = {ID_COLUMN: str(parcel[ID_COLUMN])}
                if valid.any():
                    row[f"wapor_2025__{product}__season_total"] = float(
                        np.nansum(rates * day_values)
                    )
                    row[f"wapor_2025__{product}__mean_rate"] = float(
                        np.nansum(rates * day_values) / np.sum(day_values[valid])
                    )
                    row[f"wapor_2025__{product}__peak_rate"] = float(np.nanmax(rates))
                    row[f"wapor_2025__{product}__peak_dekad"] = float(
                        int(np.nanargmax(rates)) + 1
                    )
                for month in sorted(set(months)):
                    mask_month = np.array([value == month for value in months])
                    month_rates = rates[mask_month]
                    month_days = day_values[mask_month]
                    row[f"wapor_2025__{product}__m{month:02d}_total"] = (
                        float(np.nansum(month_rates * month_days))
                        if np.isfinite(month_rates).any()
                        else np.nan
                    )
                parcel_rows.append(row)

            block = pd.DataFrame(parcel_rows)
            columns = [column for column in block if column != ID_COLUMN]
            result = result.merge(block, on=ID_COLUMN, how="left", validate="1:1")
            records.extend(
                _provenance_records(
                    columns,
                    source="external_wapor",
                    mode="competition",
                    description="Duration-weighted full-season 2025 WaPOR parcel feature.",
                )
            )
    return result, records


def merge_feature_blocks(base: pd.DataFrame, *blocks: pd.DataFrame) -> pd.DataFrame:
    """Left-join unique one-row-per-parcel feature blocks."""

    if not base[ID_COLUMN].is_unique:
        raise ValueError("Base table must contain unique parcel IDs.")
    result = base.copy()
    for block in blocks:
        if ID_COLUMN not in block.columns or not block[ID_COLUMN].is_unique:
            raise ValueError("Every feature block must contain unique ID_POLIGONO values.")
        overlap = (set(result.columns) & set(block.columns)) - {ID_COLUMN}
        if overlap:
            raise ValueError(f"Duplicate feature columns: {sorted(overlap)}")
        result = result.merge(block, on=ID_COLUMN, how="left", validate="1:1")
    return result


def validate_parcel_feature_table(
    frame: pd.DataFrame,
    *,
    expected_rows: int | None = 197,
    expected_train: int | None = 138,
    expected_prediction: int | None = 59,
) -> None:
    """Validate one-row-per-parcel model-table invariants."""

    required = {ID_COLUMN, SPLIT_COLUMN, TARGET_COLUMN}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Feature table missing required column(s): {sorted(missing)}")
    if expected_rows is not None and len(frame) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, found {len(frame)}.")
    if frame[ID_COLUMN].isna().any() or not frame[ID_COLUMN].is_unique:
        raise ValueError("ID_POLIGONO must be non-null and unique.")

    train = frame[SPLIT_COLUMN].eq("ENTRENAMIENTO")
    prediction = frame[SPLIT_COLUMN].eq("PREDICCION")
    if expected_train is not None and int(train.sum()) != expected_train:
        raise ValueError(f"Expected {expected_train} training rows, found {int(train.sum())}.")
    if expected_prediction is not None and int(prediction.sum()) != expected_prediction:
        raise ValueError(
            f"Expected {expected_prediction} prediction rows, found {int(prediction.sum())}."
        )
    if frame.loc[train, TARGET_COLUMN].isna().any():
        raise ValueError("Training rows contain missing targets.")
    if frame.loc[prediction, TARGET_COLUMN].notna().any():
        raise ValueError("Prediction rows unexpectedly expose hidden targets.")

    numeric = frame.select_dtypes(include=[np.number])
    if np.isinf(numeric.to_numpy(dtype=float, copy=True)).any():
        raise ValueError("Feature table contains infinite numeric values.")
