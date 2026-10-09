"""X-only high-dimensional 07B feature expansion from dated HLS and SMAP panels.

This deliberately constructs many candidate metrics, nonlinear transforms and
cross-family products without inspecting any yield. PCA/scalers are fitted
separately inside each supervised validation split downstream.
"""

from __future__ import annotations

import itertools
import re
from collections import defaultdict
from collections.abc import Sequence

import numpy as np
import pandas as pd

ID = "ID_POLIGONO"
BANDS = ("blue", "green", "red", "nir", "swir1", "swir2", "ndvi", "ndwi_nir_swir1")
STATS = ("mean", "std", "p10", "p50", "p90")
PERIODS = (
    ("early", 4, 5),
    ("development", 6, 7),
    ("late", 8, 9),
    ("october", 10, 10),
)


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", value.lower()).strip("_")


def _correlation(x: np.ndarray, y: np.ndarray) -> float:
    good = np.isfinite(x) & np.isfinite(y)
    if good.sum() < 3:
        return np.nan
    xx, yy = x[good], y[good]
    if np.std(xx) < 1e-10 or np.std(yy) < 1e-10:
        return np.nan
    return float(np.corrcoef(xx, yy)[0, 1])


def irregular_series_features(
    days: Sequence[object],
    values: Sequence[float],
    *,
    weights: Sequence[float] | None = None,
    max_gap_days: float = 35,
) -> dict[str, float]:
    """Describe irregular observations without inventing values in long gaps."""
    dates = pd.to_datetime(list(days), errors="coerce")
    val = np.asarray(values, dtype=float)
    w = np.ones(val.shape) if weights is None else np.asarray(weights, dtype=float)
    if len(val) != len(dates) or len(w) != len(val):
        raise ValueError("Dates, values and weights must have the same length.")
    valid = np.asarray(~pd.isna(dates)) & np.isfinite(val) & np.isfinite(w) & (w > 0)
    out = {"n_observations": float(valid.sum())}
    if not valid.any():
        return out
    times = np.asarray(dates[valid].asi8, dtype=np.float64) / 86400e9
    v, weights_valid = val[valid], w[valid]
    order = np.argsort(times, kind="stable")
    times, v, weights_valid = times[order], v[order], weights_valid[order]
    # Merge repeated dates from different satellite sensors.
    unique, inverse = np.unique(times, return_inverse=True)
    weight_by_day = np.bincount(inverse, weights=weights_valid)
    v = np.bincount(inverse, weights=v * weights_valid) / weight_by_day
    t = unique - unique[0]
    n = len(v)
    q = np.quantile(v, [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
    avg = float(np.mean(v))
    std = float(np.std(v))
    out.update({
        "n_days": float(n),
        "span_days": float(t[-1]),
        "mean": avg,
        "weighted_mean": float(np.average(v, weights=weight_by_day)),
        "std": std,
        "median": float(q[3]),
        "min": float(v.min()), "max": float(v.max()),
        "range": float(np.ptp(v)),
        "q05": float(q[0]), "q10": float(q[1]), "q25": float(q[2]),
        "q75": float(q[4]), "q90": float(q[5]), "q95": float(q[6]),
        "iqr": float(q[4] - q[2]),
        "mad": float(np.median(np.abs(v - q[3]))),
        "rms": float(np.sqrt(np.mean(v * v))),
        "cv": float(std / (abs(avg) + 1e-6)),
        "initial": float(v[0]), "final": float(v[-1]),
        "final_minus_initial": float(v[-1] - v[0]),
        "day_of_peak": float(t[np.argmax(v)]),
        "day_of_minimum": float(t[np.argmin(v)]),
        "fraction_above_median": float(np.mean(v > q[3])),
        "fraction_below_q25": float(np.mean(v < q[2])),
    })
    if n > 2 and std > 1e-12:
        z = (v - avg) / std
        out["skewness"] = float(np.mean(z**3))
        out["excess_kurtosis"] = float(np.mean(z**4) - 3)
    if n < 2:
        return out
    gap = np.diff(t)
    delta = np.diff(v)
    rate = delta / np.maximum(gap, 1.0)
    good_gap = gap <= max_gap_days
    out.update({
        "median_gap_days": float(np.median(gap)),
        "maximum_gap_days": float(np.max(gap)),
        "slope_day": float(np.polyfit(t, v, 1)[0]),
        "mean_absolute_velocity": float(np.mean(np.abs(rate))),
        "maximum_growth_velocity": float(np.max(rate)),
        "maximum_decline_velocity": float(np.min(rate)),
        "velocity_variability": float(np.std(rate)),
        "total_variation": float(np.sum(np.abs(delta))),
        "positive_variation": float(np.maximum(delta, 0).sum()),
        "negative_variation": float(np.maximum(-delta, 0).sum()),
        "direction_changes": float(np.sum(np.diff(np.sign(rate)) != 0)),
        "lag1_observation_autocorrelation": _correlation(v[:-1], v[1:]),
        "observed_auc": float(
            np.sum(((v[:-1] + v[1:]) * gap / 2)[good_gap])
        ),
        "covered_days": float(np.sum(gap[good_gap])),
        "n_contiguous_intervals": float(np.sum(good_gap)),
    })
    if n >= 3:
        out["quadratic_curvature"] = float(
            np.polyfit(t - t.mean(), v, 2)[0]
        )
    if n >= 4:
        half = n // 2
        out["late_minus_early"] = float(v[half:].mean() - v[:half].mean())
        out["acceleration_proxy"] = float(
            rate[len(rate)//2:].mean() - rate[:len(rate)//2].mean()
        )
    return out


def _add_series(
    record: dict[str, float | str],
    prefix: str,
    source: pd.DataFrame,
    column: str,
    *,
    weight: str | None = None,
) -> None:
    values = pd.to_numeric(source[column], errors="coerce")
    weights = None if weight is None else pd.to_numeric(source[weight], errors="coerce")
    for key, value in irregular_series_features(
        source["date"], values, weights=weights
    ).items():
        record[f"{prefix}__{key}"] = value


def build_hls_time_features(
    hls: pd.DataFrame,
    ids: Sequence[str],
    *,
    min_valid_fraction: float = 0.05,
) -> pd.DataFrame:
    """High-dimensional per-parcel HLS date, spectral, QA and sensor features."""
    needed = {ID, "date", "sensor", "valid_fraction"}
    if not needed.issubset(hls):
        raise ValueError(f"HLS missing columns: {sorted(needed - set(hls))}")
    frame = hls.copy()
    frame[ID] = frame[ID].astype(str)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["valid_fraction"] = pd.to_numeric(frame["valid_fraction"], errors="coerce")
    frame = frame.loc[
        frame[ID].isin(set(ids))
        & frame["date"].notna()
        & frame["valid_fraction"].ge(min_valid_fraction)
    ]
    columns = [
        f"{band}_{stat}" for band in BANDS for stat in STATS
        if f"{band}_{stat}" in frame
    ]
    grouped = {str(pid): g.sort_values("date") for pid, g in frame.groupby(ID)}
    records: list[dict[str, float | str]] = []
    for pid in ids:
        group = grouped.get(pid, frame.iloc[:0])
        row: dict[str, float | str] = {
            ID: pid,
            "hls__n_scenes": float(len(group)),
            "hls__n_days": float(group["date"].nunique()),
            "hls__n_sensors": float(group["sensor"].nunique()),
        }
        for label, data in [("all", group)] + [
            (sensor, group[group["sensor"].eq(sensor)]) for sensor in ("L30", "S30")
        ]:
            row[f"hls_{label}__n_scenes"] = float(len(data))
            if data.empty:
                continue
            for col in columns:
                _add_series(
                    row, f"hls_{label}__{_slug(col)}", data, col,
                    weight="valid_fraction",
                )
            for name, first, last in PERIODS:
                sub = data[data["date"].dt.month.between(first, last)]
                row[f"hls_{label}__{name}__n"] = float(len(sub))
                for col in columns:
                    val = pd.to_numeric(sub[col], errors="coerce").dropna()
                    if val.empty:
                        continue
                    label_col = f"hls_{label}__{name}__{_slug(col)}"
                    row[f"{label_col}__mean"] = float(val.mean())
                    row[f"{label_col}__max"] = float(val.max())
                    row[f"{label_col}__min"] = float(val.min())
            for col in (
                "qa_clear_fraction", "valid_fraction",
                "n_pixels_polygon", "n_pixels_all_bands_valid",
            ):
                if col in data:
                    v = pd.to_numeric(data[col], errors="coerce").dropna()
                    if len(v):
                        row[f"hls_{label}__qa__{col}__mean"] = float(v.mean())
                        row[f"hls_{label}__qa__{col}__minimum"] = float(v.min())
        for threshold in (0.2, 0.4, 0.6):
            if "ndvi_mean" in group:
                valid_ndvi = pd.to_numeric(
                    group["ndvi_mean"], errors="coerce"
                ).dropna()
                if not valid_ndvi.empty:
                    row[f"hls__ndvi_above_{threshold:.1f}"] = float(
                        (valid_ndvi > threshold).mean()
                    )
        records.append(row)
    return pd.DataFrame.from_records(records)


def build_smap_time_features(smap: pd.DataFrame, ids: Sequence[str]) -> pd.DataFrame:
    """Per-parcel daily regional soil-moisture dynamics and AM/PM contrasts."""
    needed = {ID, "date", "overpass", "soil_moisture_m3_m3"}
    if not needed.issubset(smap):
        raise ValueError(f"SMAP missing columns: {sorted(needed - set(smap))}")
    df = smap.copy()
    df[ID] = df[ID].astype(str)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["soil_moisture_m3_m3"] = pd.to_numeric(
        df["soil_moisture_m3_m3"], errors="coerce"
    )
    df = df.loc[df[ID].isin(set(ids)) & df["date"].notna()]
    grouped = {str(pid): g.sort_values("date") for pid, g in df.groupby(ID)}
    records: list[dict[str, float | str]] = []
    for pid in ids:
        group = grouped.get(pid, df.iloc[:0])
        record: dict[str, float | str] = {ID: pid}
        for label, values in [("all", group)] + [
            (pass_name, group[group["overpass"].eq(pass_name)])
            for pass_name in ("AM", "PM")
        ]:
            valid = values.dropna(subset=["soil_moisture_m3_m3"])
            record[f"smap_{label}__n_quality_days"] = float(valid["date"].nunique())
            if valid.empty:
                continue
            _add_series(
                record, f"smap_{label}__moisture",
                valid, "soil_moisture_m3_m3",
            )
            for name, first, last in PERIODS:
                subset = valid.loc[valid["date"].dt.month.between(first, last)]
                s = subset["soil_moisture_m3_m3"]
                if len(s):
                    record[f"smap_{label}__{name}__mean"] = float(s.mean())
                    record[f"smap_{label}__{name}__minimum"] = float(s.min())
                    record[f"smap_{label}__{name}__std"] = float(s.std(ddof=0))
            for threshold in (0.1, 0.15, 0.2, 0.25, 0.3):
                record[f"smap_{label}__fraction_below_{threshold:.2f}"] = float(
                    (valid["soil_moisture_m3_m3"] < threshold).mean()
                )
        for col in ("distance_to_cell_center_km", "smap_grid_row", "smap_grid_col"):
            if col in group and len(group):
                s = pd.to_numeric(group[col], errors="coerce").dropna()
                if len(s):
                    record[f"smap__{col}"] = float(s.median())
        records.append(record)
    return pd.DataFrame.from_records(records)


def build_hls_smap_lag_features(
    hls: pd.DataFrame,
    smap: pd.DataFrame,
    ids: Sequence[str],
) -> pd.DataFrame:
    """Combine HLS greenness with *preceding* 7/14/30-day SMAP moisture."""
    if not {"date", ID, "ndvi_mean"}.issubset(hls) or not {
        "date", ID, "soil_moisture_m3_m3"
    }.issubset(smap):
        return pd.DataFrame({ID: list(ids)})
    h = hls.copy()
    s = smap.copy()
    for frame in (h, s):
        frame[ID] = frame[ID].astype(str)
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    h["ndvi_mean"] = pd.to_numeric(h["ndvi_mean"], errors="coerce")
    s["soil_moisture_m3_m3"] = pd.to_numeric(
        s["soil_moisture_m3_m3"], errors="coerce"
    )
    smap_groups = {
        pid: group.groupby("date")["soil_moisture_m3_m3"].mean().sort_index()
        for pid, group in s.groupby(ID)
    }
    hls_groups = {pid: g for pid, g in h.groupby(ID)}
    records: list[dict[str, float | str]] = []
    for pid in ids:
        row: dict[str, float | str] = {ID: pid}
        hg = hls_groups.get(pid)
        series = smap_groups.get(pid)
        if hg is None or series is None:
            records.append(row)
            continue
        for horizon in (7, 14, 30):
            xs, ys = [], []
            for date, ndvi in zip(hg["date"], hg["ndvi_mean"], strict=True):
                if pd.isna(date) or not np.isfinite(ndvi):
                    continue
                # No future moisture is included in the antecedent window.
                previous = series.loc[
                    (series.index < date)
                    & (series.index >= date - pd.Timedelta(days=horizon))
                ].dropna()
                if len(previous):
                    xs.append(float(ndvi))
                    ys.append(float(previous.mean()))
            row[f"lag{horizon}__n"] = float(len(xs))
            row[f"lag{horizon}__ndvi_smap_correlation"] = _correlation(
                np.asarray(xs), np.asarray(ys)
            )
            if ys:
                row[f"lag{horizon}__mean_antecedent_moisture"] = float(np.mean(ys))
                row[f"lag{horizon}__ndvi_times_moisture"] = float(
                    np.mean(np.asarray(xs) * np.asarray(ys))
                )
        records.append(row)
    return pd.DataFrame.from_records(records)


def _feature_family(column: str) -> str:
    if column.startswith("agro_"):
        return "_".join(column.split("__")[0].split("_")[:2])
    if column.startswith("hls_"):
        return column.split("__")[0]
    if column.startswith("smap_"):
        return column.split("__")[0]
    if column.startswith("soilgrids_"):
        return "soilgrids"
    if column.startswith("clim_"):
        return "climate"
    if column.startswith("siap_"):
        return "siap"
    return column.split("_")[0]


def select_x_only_interaction_seeds(
    frame: pd.DataFrame,
    *,
    max_seeds: int = 180,
) -> list[str]:
    """Select balanced, well-observed seed columns without using the target."""
    if max_seeds < 0:
        raise ValueError("max_seeds cannot be negative")
    prohibited = {
        ID, "CONJUNTO", "RENDIMIENTO_T_HA", "RENDIMIENTO",
        "PRODUCCION_T", "target", "yield",
    }
    candidates: dict[str, list[str]] = defaultdict(list)
    for name in sorted(frame.select_dtypes(include=[np.number]).columns):
        if name in prohibited or name.lower().startswith(("target_", "yield_")):
            continue
        values = pd.to_numeric(frame[name], errors="coerce")
        good = values.replace([np.inf, -np.inf], np.nan).dropna()
        if len(good) < max(3, int(len(frame) * 0.6)):
            continue
        if good.nunique() < 3:
            continue
        candidates[_feature_family(name)].append(name)
    families = sorted(candidates)
    chosen = []
    for batch in itertools.count():
        if len(chosen) >= max_seeds:
            break
        picked = False
        for family in families:
            group = candidates[family]
            if batch < len(group):
                chosen.append(group[batch])
                picked = True
                if len(chosen) >= max_seeds:
                    break
        if not picked:
            break
    return chosen


def build_nonlinear_interactions(
    frame: pd.DataFrame,
    *,
    max_seeds: int = 180,
    max_products: int = 12000,
    block_size: int = 128,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Generate bounded pointwise transforms and cross-family pair products.

    A signed log transform makes large magnitudes manageable without any
    target-dependent fitted scaling. Full fold-local scaling occurs *after*
    these transforms. Products are X-only, with no held-out yield access.
    """
    if max_products < 0 or block_size < 1:
        raise ValueError("Invalid interaction cap or block size")
    seeds = select_x_only_interaction_seeds(frame, max_seeds=max_seeds)
    columns: dict[str, np.ndarray] = {}
    transformed: dict[str, np.ndarray] = {}
    for name in seeds:
        values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
        values[~np.isfinite(values)] = np.nan
        signed_log = np.sign(values) * np.log1p(np.abs(values))
        transformed[name] = signed_log
        columns[f"int__signed_log__{name}"] = signed_log
        columns[f"int__signed_square__{name}"] = (
            np.sign(signed_log) * signed_log**2
        )
    names = list(seeds)
    pairs = [
        (x, y) for x, y in itertools.combinations(names, 2)
        if _feature_family(x) != _feature_family(y)
    ]
    # Deterministic broad coverage across pairs; no outcome-driven ranking.
    if max_products and len(pairs) > max_products:
        selected = np.linspace(0, len(pairs) - 1, max_products, dtype=int)
        pairs = [pairs[i] for i in selected]
    elif max_products == 0:
        pairs = []
    for start in range(0, len(pairs), block_size):
        for left, right in pairs[start:start + block_size]:
            columns[f"int__product__{left}__X__{right}"] = (
                transformed[left] * transformed[right]
            )
    result = pd.DataFrame(columns, index=frame.index)
    return result, {
        "n_seeds": len(seeds),
        "n_products": len(pairs),
        "n_expanded_features": result.shape[1],
        "seed_names": seeds,
    }


def merge_x_only_blocks(
    base: pd.DataFrame,
    *blocks: pd.DataFrame,
) -> pd.DataFrame:
    """Combine 197-row X tables with ID integrity and no supervised target."""
    if ID not in base:
        raise ValueError(f"Missing {ID} in base table")
    output = base.copy()
    output[ID] = output[ID].astype(str)
    if output[ID].duplicated().any():
        raise ValueError("Duplicate parcel IDs in base.")
    forbidden = {"RENDIMIENTO_T_HA", "CONJUNTO", "RENDIMIENTO", "PRODUCCION_T"}
    output = output.drop(columns=[c for c in forbidden if c in output], errors="ignore")
    for block in blocks:
        if ID not in block:
            raise ValueError("Feature block missing parcel ID.")
        part = block.copy()
        part[ID] = part[ID].astype(str)
        if part[ID].duplicated().any():
            raise ValueError("Duplicate parcel IDs in feature block.")
        overlap = (set(output.columns) & set(part.columns)) - {ID}
        if overlap:
            raise ValueError(f"Duplicate feature names: {sorted(overlap)[:5]}")
        if set(part[ID]) != set(output[ID]):
            raise ValueError("Feature block parcel IDs differ from base.")
        output = output.merge(part, on=ID, how="left", validate="one_to_one", sort=False)
    return output
