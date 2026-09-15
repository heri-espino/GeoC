"""Temporal alignment utilities for longitudinal satellite observations.

The official BASIC/PRO tables contain repeated observations per parcel and date.
This module aligns those irregular observations to a common temporal grid without
using the yield target. Sensor streams remain separate by default.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Literal

import numpy as np
import pandas as pd

Backend = Literal["auto", "cpu", "gpu"]


def make_temporal_grid(start: str, end: str, *, freq: str = "14D") -> pd.DatetimeIndex:
    """Return an inclusive regular temporal grid between ``start`` and ``end``."""

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if end_ts < start_ts:
        raise ValueError("end must be on or after start.")
    grid = pd.date_range(start=start_ts, end=end_ts, freq=freq)
    if len(grid) == 0:
        raise ValueError("The requested temporal grid is empty.")
    return grid


def temporal_coverage_summary(
    frame: pd.DataFrame,
    *,
    id_column: str = "ID_POLIGONO",
    date_column: str = "fecha_captura",
    sensor_column: str | None = "sensor",
    cloud_column: str | None = "porcentaje_nubosidad",
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Summarize date coverage, temporal gaps and cloudiness by parcel/sensor.

    Monthly capture-count columns are added for months present inside the chosen
    date window. This function is descriptive only and does not use the target.
    """

    required = {id_column, date_column}
    if sensor_column is not None:
        required.add(sensor_column)
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Missing required column(s): {sorted(missing)}")

    columns = [id_column, date_column]
    if sensor_column is not None:
        columns.append(sensor_column)
    if cloud_column is not None and cloud_column in frame.columns:
        columns.append(cloud_column)

    work = frame[columns].copy()
    work[date_column] = pd.to_datetime(work[date_column], errors="coerce")
    work = work.dropna(subset=[id_column, date_column])
    if start is not None:
        work = work.loc[work[date_column] >= pd.Timestamp(start)]
    if end is not None:
        work = work.loc[work[date_column] <= pd.Timestamp(end)]

    group_columns = [id_column]
    if sensor_column is not None:
        group_columns.append(sensor_column)

    rows: list[dict[str, object]] = []
    for key, group in work.groupby(group_columns, dropna=False, sort=True):
        key_tuple = key if isinstance(key, tuple) else (key,)
        dates = pd.Series(group[date_column].dropna().unique()).sort_values()
        gaps = dates.diff().dt.total_seconds().dropna() / 86400.0

        row: dict[str, object] = dict(zip(group_columns, key_tuple, strict=True))
        row.update(
            {
                "n_records": int(len(group)),
                "n_unique_dates": int(len(dates)),
                "first_date": dates.iloc[0] if len(dates) else pd.NaT,
                "last_date": dates.iloc[-1] if len(dates) else pd.NaT,
                "span_days": (
                    float((dates.iloc[-1] - dates.iloc[0]).days) if len(dates) > 1 else 0.0
                ),
                "median_gap_days": float(gaps.median()) if len(gaps) else np.nan,
                "max_gap_days": float(gaps.max()) if len(gaps) else np.nan,
            }
        )

        if cloud_column is not None and cloud_column in group.columns:
            cloud = pd.to_numeric(group[cloud_column], errors="coerce")
            row["cloud_mean"] = float(cloud.mean()) if cloud.notna().any() else np.nan
            row["cloud_median"] = float(cloud.median()) if cloud.notna().any() else np.nan

        month_counts = group[date_column].dt.to_period("M").value_counts()
        for month, count in month_counts.items():
            row[f"n_{month.strftime('%Y_%m')}"] = int(count)
        rows.append(row)

    result = pd.DataFrame(rows)
    month_columns = sorted(column for column in result.columns if column.startswith("n_20"))
    if month_columns:
        result[month_columns] = result[month_columns].fillna(0).astype(int)
    return result


def temporal_backend_available(backend: Backend = "auto") -> str:
    """Resolve ``backend`` to ``"cpu"`` or ``"gpu"`` without running alignment."""

    if backend not in {"auto", "cpu", "gpu"}:
        raise ValueError("backend must be 'auto', 'cpu' or 'gpu'.")
    if backend == "cpu":
        return "cpu"

    try:
        import cupy as cp

        if cp.cuda.runtime.getDeviceCount() > 0:
            return "gpu"
    except Exception:
        if backend == "gpu":
            raise RuntimeError(
                "GPU backend requested but CuPy/CUDA is unavailable. Install the optional "
                "GPU dependencies with `python -m pip install -e \".[gpu]\"` on a compatible "
                "NVIDIA/CUDA machine."
            ) from None

    if backend == "gpu":
        raise RuntimeError("GPU backend requested but no CUDA device is available.")
    return "cpu"


def align_temporal_knn(
    frame: pd.DataFrame,
    value_columns: Sequence[str],
    *,
    grid: Iterable[pd.Timestamp] | None = None,
    start: str | None = None,
    end: str | None = None,
    freq: str = "14D",
    id_column: str = "ID_POLIGONO",
    date_column: str = "fecha_captura",
    sensor_column: str | None = "sensor",
    cloud_column: str | None = "porcentaje_nubosidad",
    k: int = 3,
    bandwidth_days: float = 14.0,
    max_distance_days: float | None = 30.0,
    cloud_weighting: bool = True,
    backend: Backend = "auto",
    collapse_same_day: bool = True,
) -> pd.DataFrame:
    """Align irregular parcel observations to a common grid with temporal k-NN.

    For every parcel/sensor/grid-date combination, each variable independently
    selects its ``k`` nearest non-missing capture dates. Values are averaged with
    exponential temporal weights, optionally multiplied by a cloud-quality
    weight ``1 - cloud_pct / 100``.

    Selecting neighbors per variable matters because BASIC/PRO missingness is
    index-dependent: a missing NDVI on a nearby row should not prevent NDVI from
    using the next-nearest valid capture inside ``max_distance_days``.

    The GPU backend uses CuPy for distance/weighted-reduction operations. Pandas
    grouping and date parsing remain on CPU, so GPU acceleration is most useful
    when many value columns are aligned at once. For the current ~100k-row
    official tables, a strong multicore CPU may already be very fast.
    """

    if not value_columns:
        raise ValueError("value_columns must contain at least one variable.")
    if k < 1:
        raise ValueError("k must be at least 1.")
    if bandwidth_days <= 0:
        raise ValueError("bandwidth_days must be positive.")
    if max_distance_days is not None and max_distance_days < 0:
        raise ValueError("max_distance_days must be non-negative or None.")

    required = {id_column, date_column, *value_columns}
    if sensor_column is not None:
        required.add(sensor_column)
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Missing required column(s): {sorted(missing)}")

    if grid is None:
        if start is None or end is None:
            raise ValueError("Provide either grid or both start and end.")
        grid_index = make_temporal_grid(start, end, freq=freq)
    else:
        grid_index = pd.DatetimeIndex(pd.to_datetime(list(grid), errors="raise"))
        if len(grid_index) == 0:
            raise ValueError("grid must contain at least one date.")

    resolved_backend = temporal_backend_available(backend)
    xp = np
    cp = None
    if resolved_backend == "gpu":
        import cupy as cp_module

        cp = cp_module
        xp = cp_module

    columns = [id_column, date_column, *value_columns]
    if sensor_column is not None:
        columns.append(sensor_column)
    use_cloud = cloud_column is not None and cloud_column in frame.columns
    if use_cloud and cloud_column not in columns:
        columns.append(cloud_column)

    work = frame[columns].copy()
    work[date_column] = pd.to_datetime(work[date_column], errors="coerce")
    work = work.dropna(subset=[id_column, date_column])
    for column in value_columns:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    if use_cloud:
        work[cloud_column] = pd.to_numeric(work[cloud_column], errors="coerce")

    if start is not None:
        work = work.loc[work[date_column] >= pd.Timestamp(start)]
    if end is not None:
        work = work.loc[work[date_column] <= pd.Timestamp(end)]

    group_columns = [id_column]
    if sensor_column is not None:
        group_columns.append(sensor_column)

    if collapse_same_day:
        aggregations = {column: "mean" for column in value_columns}
        if use_cloud:
            aggregations[cloud_column] = "mean"
        work = (
            work.groupby([*group_columns, date_column], as_index=False, dropna=False)
            .agg(aggregations)
            .sort_values([*group_columns, date_column])
        )

    epoch = pd.Timestamp("1970-01-01")
    target_days_cpu = ((grid_index - epoch) / pd.Timedelta(days=1)).to_numpy(dtype=float)
    target_days = xp.asarray(target_days_cpu)

    rows: list[dict[str, object]] = []
    for key, group in work.groupby(group_columns, dropna=False, sort=True):
        key_tuple = key if isinstance(key, tuple) else (key,)
        group = group.sort_values(date_column)
        capture_days_cpu = (
            (group[date_column] - epoch) / pd.Timedelta(days=1)
        ).to_numpy(dtype=float)
        values_cpu = group[list(value_columns)].to_numpy(dtype=float)

        capture_days = xp.asarray(capture_days_cpu)
        values = xp.asarray(values_cpu)

        if use_cloud and cloud_weighting:
            cloud_cpu = pd.to_numeric(group[cloud_column], errors="coerce").to_numpy(dtype=float)
            cloud_quality_cpu = 1.0 - np.clip(cloud_cpu, 0.0, 100.0) / 100.0
            cloud_quality_cpu = np.where(np.isfinite(cloud_quality_cpu), cloud_quality_cpu, 1.0)
            cloud_quality = xp.asarray(cloud_quality_cpu)
        else:
            cloud_quality = xp.ones(len(group), dtype=float)

        for target_idx, target_day in enumerate(target_days):
            distances = xp.abs(capture_days - target_day)
            if max_distance_days is None:
                candidate_mask = xp.ones(distances.shape, dtype=bool)
            else:
                candidate_mask = distances <= max_distance_days

            if cp is not None:
                candidate_count = int(cp.asnumpy(xp.sum(candidate_mask)))
            else:
                candidate_count = int(np.sum(candidate_mask))

            row: dict[str, object] = dict(zip(group_columns, key_tuple, strict=True))
            row["grid_date"] = grid_index[target_idx]
            row["backend"] = resolved_backend
            row["neighbor_count"] = min(k, candidate_count)

            if candidate_count == 0:
                row["nearest_gap_days"] = np.nan
                for column in value_columns:
                    row[column] = np.nan
                rows.append(row)
                continue

            candidate_distances = distances[candidate_mask]
            if cp is not None:
                nearest_gap = float(cp.asnumpy(xp.min(candidate_distances)))
            else:
                nearest_gap = float(np.min(np.asarray(candidate_distances)))
            row["nearest_gap_days"] = nearest_gap

            valid_values = xp.isfinite(values)
            allowed = valid_values & candidate_mask[:, None]
            masked_distances = xp.where(allowed, distances[:, None], xp.inf)

            n_take = min(k, candidate_count)
            neighbor_idx = xp.argpartition(masked_distances, n_take - 1, axis=0)[:n_take, :]
            selected_distances = xp.take_along_axis(masked_distances, neighbor_idx, axis=0)
            selected_values = xp.take_along_axis(values, neighbor_idx, axis=0)
            selected_cloud = cloud_quality[neighbor_idx]

            usable = xp.isfinite(selected_distances) & xp.isfinite(selected_values)
            weights = xp.exp(-selected_distances / bandwidth_days) * selected_cloud
            weights = xp.where(usable, weights, 0.0)
            numerators = xp.sum(xp.where(usable, selected_values, 0.0) * weights, axis=0)
            denominators = xp.sum(weights, axis=0)
            aligned = xp.full(len(value_columns), xp.nan, dtype=float)
            xp.divide(numerators, denominators, out=aligned, where=denominators > 0)

            if cp is not None:
                aligned_cpu = cp.asnumpy(aligned)
            else:
                aligned_cpu = np.asarray(aligned)

            row.update(dict(zip(value_columns, aligned_cpu.tolist(), strict=True)))
            rows.append(row)

    return pd.DataFrame(rows)


def aligned_to_wide(
    aligned: pd.DataFrame,
    *,
    value_columns: Sequence[str],
    id_column: str = "ID_POLIGONO",
    date_column: str = "grid_date",
    sensor_column: str | None = "sensor",
    date_format: str = "%Y_%m_%d",
) -> pd.DataFrame:
    """Pivot aligned temporal values to one modeling row per parcel.

    Sensor names are embedded in output column names when ``sensor_column`` is
    provided, preventing silent mixing of same-named indices across sensors.
    """

    required = {id_column, date_column, *value_columns}
    if sensor_column is not None:
        required.add(sensor_column)
    missing = required.difference(aligned.columns)
    if missing:
        raise KeyError(f"Missing required column(s): {sorted(missing)}")

    work = aligned[[*required]].copy()
    work[date_column] = pd.to_datetime(work[date_column], errors="raise")
    work["_date_label"] = work[date_column].dt.strftime(date_format)

    key_columns = ["_date_label"]
    if sensor_column is not None:
        key_columns.insert(0, sensor_column)

    wide = work.pivot_table(
        index=id_column,
        columns=key_columns,
        values=list(value_columns),
        aggfunc="first",
        observed=False,
    )

    labels: list[str] = []
    for column in wide.columns:
        if sensor_column is not None:
            variable, sensor, date_label = column
            labels.append(f"{sensor}__{variable}__{date_label}")
        else:
            variable, date_label = column
            labels.append(f"{variable}__{date_label}")
    wide.columns = labels
    return wide.reset_index()
