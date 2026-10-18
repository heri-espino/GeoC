"""X-only phenology and weather features from existing compact 2025 data.

Peak month is an *optical proxy* for crop development, not an observed
sowing, heading, or harvest date. No yield labels enter this module.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

ID = "ID_POLIGONO"
MONTHS = tuple(range(4, 11))
DAYS = np.array([30, 31, 30, 31, 31, 30, 31], dtype=float)


def _check_ids(frame: pd.DataFrame, ids: list[str], name: str) -> pd.DataFrame:
    if ID not in frame or frame[ID].isna().any() or frame[ID].duplicated().any():
        raise ValueError(f"{name} requires unique non-null parcel IDs.")
    data = frame.copy()
    data[ID] = data[ID].astype(str)
    if not set(data[ID]).issubset(ids):
        raise ValueError(f"{name} has unknown parcel IDs.")
    return data


def _weighted_mean(x: np.ndarray, w: np.ndarray) -> float:
    good = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if not good.any():
        return float("nan")
    return float(np.average(x[good], weights=w[good]))


def _hls_summary(group: pd.DataFrame) -> dict[str, float]:
    """Summarize quality-filtered daily NDVI without interpolating absent days."""
    if group.empty:
        return {
            "ndvi_mean": np.nan, "ndvi_peak": np.nan, "ndvi_auc": np.nan,
            "ndvi_change": np.nan, "peak_month": np.nan,
            "peak_day": np.nan, "ndvi_n_days": 0.0,
            "ndvi_n_months": 0.0, "ndvi_max_gap": np.nan,
        }
    g = group.copy()
    g["month"] = g["date"].dt.month
    by_month = {}
    for month, sub in g.groupby("month"):
        by_month[int(month)] = _weighted_mean(
            sub["ndvi_mean"].to_numpy(float),
            sub["valid_fraction"].to_numpy(float),
        )
    months = sorted(m for m, v in by_month.items() if np.isfinite(v))
    daily = []
    for day, sub in g.groupby("date"):
        daily.append((
            day,
            _weighted_mean(
                sub["ndvi_mean"].to_numpy(float),
                sub["valid_fraction"].to_numpy(float),
            ),
        ))
    daily = [(t, v) for t, v in daily if np.isfinite(v)]
    daily.sort()
    values = np.array([x[1] for x in daily], dtype=float)
    ordinals = np.array([x[0].toordinal() for x in daily], dtype=float)
    gaps = np.diff(ordinals)
    integral = np.nan
    if len(values) >= 2:
        valid_gaps = gaps <= 35
        if valid_gaps.any():
            integral = float(np.sum(
                0.5 * (values[:-1] + values[1:])[valid_gaps] * gaps[valid_gaps]
            ))
    peak_month = max(months, key=lambda m: by_month[m]) if len(months) >= 3 else np.nan
    peak_day = (
        pd.Timestamp(2025, int(peak_month), 15).dayofyear
        if np.isfinite(peak_month) else np.nan
    )
    return {
        "ndvi_mean": float(np.mean(values)) if len(values) else np.nan,
        "ndvi_peak": float(np.max(values)) if len(values) else np.nan,
        "ndvi_auc": integral,
        "ndvi_change": float(values[-1] - values[0]) if len(values) >= 2 else np.nan,
        "peak_month": float(peak_month),
        "peak_day": float(peak_day),
        "ndvi_n_days": float(len(values)),
        "ndvi_n_months": float(len(months)),
        "ndvi_max_gap": float(gaps.max()) if len(gaps) else np.nan,
    }


def _smap_summary(group: pd.DataFrame, peak_month: float) -> dict[str, float]:
    """Summarize valid regional SMAP measurements and 30-day peak windows."""
    if group.empty:
        return {"smap_mean": np.nan, "smap_std": np.nan,
                "smap_before_peak": np.nan, "smap_after_peak": np.nan,
                "smap_n_days": 0.0}
    # AM and PM readings from one grid cell are not independent parcels.
    by_day = group.groupby("date")["soil_moisture_m3_m3"].mean().sort_index()
    values = by_day.to_numpy(float)
    before = after = np.nan
    if np.isfinite(peak_month):
        peak = pd.Timestamp(2025, int(peak_month), 15)
        before_values = by_day.loc[(by_day.index >= peak - pd.Timedelta(days=30))
                                   & (by_day.index < peak)]
        after_values = by_day.loc[(by_day.index >= peak)
                                  & (by_day.index <= peak + pd.Timedelta(days=30))]
        if not before_values.empty:
            before = float(before_values.mean())
        if not after_values.empty:
            after = float(after_values.mean())
    return {
        "smap_mean": float(np.mean(values)),
        "smap_std": float(np.std(values)),
        "smap_before_peak": before,
        "smap_after_peak": after,
        "smap_n_days": float(len(values)),
    }


def build_phenology_weather_features(
    weighted_monthly: pd.DataFrame,
    hls: pd.DataFrame,
    smap: pd.DataFrame,
    ids: Sequence[str],
) -> pd.DataFrame:
    """Construct one X-only parcel row with phenology-conditioned 2025 weather.

    Input schemas match the existing area-weighted climate, HLS and SMAP
    extractors. Missing environmental measurements remain NaN until fold-local
    imputation. No true agronomic stage or subpixel weather is inferred.
    """
    ordered_ids = list(map(str, ids))
    if len(ordered_ids) != len(set(ordered_ids)):
        raise ValueError("Requested parcel IDs must be unique.")
    climate = _check_ids(weighted_monthly, ordered_ids, "Climate").set_index(ID)
    if set(climate.index) != set(ordered_ids):
        raise ValueError("Climate missing parcels.")
    h = _check_ids(hls, ordered_ids, "HLS")
    s = _check_ids(smap, ordered_ids, "SMAP")
    h_required = {"date", "valid_fraction", "ndvi_mean"}
    s_required = {"date", "soil_moisture_m3_m3", "recommended_quality"}
    if not h_required.issubset(h) or not s_required.issubset(s):
        raise ValueError("Missing required HLS or SMAP columns.")
    required = [
        f"area_climate__{name}__2025_m{m:02d}"
        for name in ("prec", "tmin", "tmax") for m in MONTHS
    ]
    if not set(required).issubset(climate):
        raise ValueError("Missing 2025 area-weighted monthly climate columns.")

    for frame in (h, s):
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        if frame["date"].isna().any():
            raise ValueError("Invalid acquisition dates.")
        frame.drop(frame.loc[
            (frame["date"] < "2025-04-01") | (frame["date"] > "2025-10-31")
        ].index, inplace=True)

    h["valid_fraction"] = pd.to_numeric(h["valid_fraction"], errors="coerce")
    h["ndvi_mean"] = pd.to_numeric(h["ndvi_mean"], errors="coerce")
    h = h.loc[
        h["valid_fraction"].ge(0.05)
        & h["ndvi_mean"].between(-1.0, 1.0)
    ]
    s["recommended_quality"] = pd.to_numeric(s["recommended_quality"], errors="coerce")
    s["soil_moisture_m3_m3"] = pd.to_numeric(
        s["soil_moisture_m3_m3"], errors="coerce"
    )
    s = s.loc[
        s["recommended_quality"].eq(1)
        & s["soil_moisture_m3_m3"].between(0, 0.7)
    ]
    hgroups = dict(tuple(h.groupby(ID)))
    sgroups = dict(tuple(s.groupby(ID)))
    rows = []
    for pid in ordered_ids:
        row: dict[str, str | float] = {ID: pid}
        row.update(_hls_summary(hgroups.get(pid, h.iloc[:0])))
        row.update(_smap_summary(sgroups.get(pid, s.iloc[:0]), row["peak_month"]))
        rain = np.array([
            climate.at[pid, f"area_climate__prec__2025_m{m:02d}"] for m in MONTHS
        ], dtype=float)
        lo = np.array([
            climate.at[pid, f"area_climate__tmin__2025_m{m:02d}"] for m in MONTHS
        ], dtype=float)
        hi = np.array([
            climate.at[pid, f"area_climate__tmax__2025_m{m:02d}"] for m in MONTHS
        ], dtype=float)
        if np.isfinite(rain).any() and (rain[np.isfinite(rain)] < 0).any():
            raise ValueError(f"Negative precipitation for {pid}")
        if np.any(hi[np.isfinite(lo) & np.isfinite(hi)]
                  < lo[np.isfinite(lo) & np.isfinite(hi)]):
            raise ValueError(f"Tmax below Tmin for {pid}")
        mean_temp = 0.5 * (lo + hi)
        row["prec_total"] = float(np.sum(rain)) if np.isfinite(rain).all() else np.nan
        row["prec_wettest_month"] = (
            float(np.max(rain)) if np.isfinite(rain).all() else np.nan
        )
        row["temp_mean"] = (
            float(np.mean(mean_temp)) if np.isfinite(mean_temp).all() else np.nan
        )
        row["diurnal_range"] = (
            float(np.mean(hi - lo))
            if np.isfinite(hi - lo).all() else np.nan
        )
        row["gdd5_monthly_proxy"] = (
            float(np.dot(DAYS, np.maximum(mean_temp - 5.0, 0)))
            if np.isfinite(mean_temp).all() else np.nan
        )
        peak = row["peak_month"]
        if np.isfinite(peak):
            early = np.array([m <= int(peak) for m in MONTHS])
            for tag, mask in (("before_peak", early), ("after_peak", ~early)):
                selected_rain = rain[mask]
                selected_temp = mean_temp[mask]
                row[f"prec_{tag}"] = (
                    float(selected_rain.sum()) if selected_rain.size
                    and np.isfinite(selected_rain).all() else np.nan
                )
                row[f"gdd5_{tag}_proxy"] = (
                    float(np.dot(DAYS[mask], np.maximum(selected_temp - 5, 0)))
                    if selected_temp.size and np.isfinite(selected_temp).all()
                    else np.nan
                )
        else:
            for tag in ("before_peak", "after_peak"):
                row[f"prec_{tag}"] = np.nan
                row[f"gdd5_{tag}_proxy"] = np.nan
        rows.append(row)
    result = pd.DataFrame(rows)
    if not result[ID].tolist() == ordered_ids:
        raise AssertionError("Feature order changed.")
    if len(result.columns) < 20:
        raise AssertionError("Unexpectedly small physical feature set.")
    return result
