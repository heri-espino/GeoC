"""X-only imputation alternatives for the 138-row teammate Reference26 bundle.

Methods:
* spatial: region-prioritized inverse-distance neighbors on the *saved*
  normalized 138x138 geographic distance matrix;
* temporal: same-parcel observations from original BASIC/PRO satellite dates;
* spatiotemporal: date-weighted nearby raw satellite measurements, then
  region-prioritized spatial fallback for still-missing numeric fields.

Only missing input cells are filled. Existing values, geology shares, parcel
coordinates, categorical fields and all targets are left untouched. Any
remaining missing inputs are handled by the original teammate's train-only
median preprocessing. All X transformations are target-free; the original
LOOCV held-out labels remain masked separately by the modeling runner.

Do not confuse spatial X imputation with kriging the yield y.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ID = "ID_POLIGONO"
TARGET = "RENDIMIENTO_T_HA"
SKIP_PREFIXES = ("geology_physiography::",)
SKIP_COLUMNS = {
    ID, TARGET, "base_centroid_lat", "base_centroid_lon",
}
BASIC = "Conjunto_datos_BASICO_AgroCebada2026.csv"
PRO = "Conjunto_datos_PRO_AgroCebada.csv"
# The target crop cycle is April--October 2025. Historical S2 is 2022--2024.
SATELLITE_FEATURES = {
    "S2_ndvi_month_05": ("basic", "sentinel-2", "ndvi_promedio", "may2025", "value"),
    "sat_basic_hist__sentinel_2__ndvi_promedio__mean": (
        "basic", "sentinel-2", "ndvi_promedio", "hist", "mean",
    ),
    "sat_basic_hist__sentinel_2__ndvi_promedio__std": (
        "basic", "sentinel-2", "ndvi_promedio", "hist", "std",
    ),
    "sat_basic_2025__sentinel_2__ndwi_promedio__mean": (
        "basic", "sentinel-2", "ndwi_promedio", "crop2025", "mean",
    ),
    "sat_basic_2025__sentinel_2__ndwi_promedio__std": (
        "basic", "sentinel-2", "ndwi_promedio", "crop2025", "std",
    ),
    "agro_pheno__y2025_planet_ndvi__jun_to_aug_delta": (
        "pro", "planet", "ndvi_promedio", "junaug2025", "delta",
    ),
}


@dataclass(frozen=True)
class Reconstruction:
    frames: dict[str, pd.DataFrame]
    audit: pd.DataFrame
    original_missing: int
    raw_observations: dict[str, int]


def donor_columns(frame: pd.DataFrame, features: list[str]) -> list[str]:
    """Only numeric, nongraph, noncomposition feature fields can be X-imputed."""
    return [
        column for column in features
        if column in frame.columns
        and column not in SKIP_COLUMNS
        and not column.startswith(SKIP_PREFIXES)
        and pd.api.types.is_numeric_dtype(frame[column])
    ]


def _distance_checks(distance: np.ndarray, n: int):
    distance = np.asarray(distance, dtype=float)
    if distance.shape != (n, n) or not np.isfinite(distance).all():
        raise ValueError(f"Expected finite {n}x{n} geographic distance matrix.")
    if (distance < -1e-12).any() or not np.allclose(distance, distance.T, atol=1e-7):
        raise ValueError("Invalid normalized geographic distances.")
    if not np.allclose(np.diag(distance), 0, atol=1e-8):
        raise ValueError("Geographic distance diagonal is not zero.")
    return distance


def candidate_neighbors(
    i: int, frame: pd.DataFrame, distance: np.ndarray, *, k: int = 8,
    include_self: bool = False,
) -> np.ndarray:
    """Same municipality first, then same state, then nearest other parcels."""
    n = len(frame)
    state = frame["Estado"].fillna("").astype(str).to_numpy()
    municipality = frame["Municipio"].fillna("").astype(str).to_numpy()
    indices = [j for j in range(n) if j != i]
    ordered = sorted(indices, key=lambda j: (
        0 if (
            state[i] and municipality[i]
            and state[i] == state[j] and municipality[i] == municipality[j]
        ) else 1 if state[i] and state[i] == state[j] else 2,
        distance[i, j], j,
    ))
    selected = ordered[:k]
    return np.asarray(([i] if include_self else []) + selected, dtype=int)


def spatial_fill(
    frame: pd.DataFrame, distance: np.ndarray, columns: list[str], *, k: int = 8,
) -> pd.DataFrame:
    """Donors have observed X; do not let newly imputed values propagate."""
    original = frame.copy(deep=True)
    filled = frame.copy(deep=True)
    n = len(frame)
    order = [candidate_neighbors(i, frame, distance, k=k) for i in range(n)]
    distance_scale = np.median(
        np.sort(distance + np.eye(n) * 1e12, axis=1)[:, :min(k, n-1)]
    )
    epsilon = max(float(distance_scale) * 0.1, 1e-8)
    for col in columns:
        values = pd.to_numeric(original[col], errors="coerce").to_numpy(float)
        missing = np.flatnonzero(~np.isfinite(values))
        for i in missing:
            neighbors = order[i]
            valid = neighbors[np.isfinite(values[neighbors])]
            if not len(valid):
                # Expand beyond the preferred 8 when nearest parcels have no X.
                valid = np.asarray(
                    [j for j in range(n) if j != i and np.isfinite(values[j])],
                    dtype=int,
                )
                valid = valid[np.lexsort((valid, distance[i, valid]))][:k]
            if not len(valid):
                continue
            weight = 1.0 / (distance[i, valid] + epsilon)
            filled.at[filled.index[i], col] = float(
                np.dot(weight, values[valid]) / weight.sum()
            )
    return filled


def read_satellite_source(
    path: Path, source: str, parcel_index: dict[str, int],
) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing original {source.upper()} dated satellite data: {path}. "
            "Supply --basic or --pro; temporal reconstruction needs raw dates."
        )
    cols = [ID, "fecha_captura", "sensor", "porcentaje_nubosidad", "ndvi_promedio"]
    if source == "basic":
        cols.append("ndwi_promedio")
    data = pd.read_csv(
        path, usecols=cols,
        dtype={ID: str, "sensor": str},
        low_memory=False,
    )
    data["position"] = data[ID].map(parcel_index)
    data = data.dropna(subset=["position"]).copy()
    data["position"] = data["position"].astype(int)
    data["date"] = pd.to_datetime(
        data["fecha_captura"], format="%d/%m/%Y", errors="coerce"
    )
    if data["date"].isna().any():
        raise ValueError(f"{source} contains an invalid source acquisition date.")
    data["sensor_key"] = (
        data["sensor"].astype(str)
        .str.lower().str.strip().str.replace("_", "-", regex=False)
        .str.replace(" ", "", regex=False)
    )
    data["quality"] = 1.0 - (
        pd.to_numeric(data["porcentaje_nubosidad"], errors="coerce")
        .fillna(50.0).clip(0, 100) / 100.0
    )
    data["quality"] = data["quality"].clip(lower=0.05)
    for var in ("ndvi_promedio", "ndwi_promedio"):
        if var in data.columns:
            data[var] = pd.to_numeric(data[var], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
    return data


def _time_grid(season: str) -> list[pd.Timestamp]:
    if season == "may2025":
        return [pd.Timestamp("2025-05-16")]
    if season == "junaug2025":
        return [pd.Timestamp("2025-06-15"), pd.Timestamp("2025-08-15")]
    if season == "crop2025":
        return list(pd.date_range("2025-04-01", "2025-10-31", freq="14D"))
    if season == "hist":
        return [
            ts for year in (2022, 2023, 2024)
            for ts in pd.date_range(f"{year}-04-01", f"{year}-10-31", freq="14D")
        ]
    raise ValueError(f"Unrecognized temporal season: {season}")


def _observation_library(
    basic: pd.DataFrame, pro: pd.DataFrame,
) -> dict[tuple[str, str, str], dict[int, np.ndarray]]:
    result = {}
    for source, data in (("basic", basic), ("pro", pro)):
        for sensor, variable in (
            ("sentinel-2", "ndvi_promedio"),
            ("sentinel-2", "ndwi_promedio"),
            ("planet", "ndvi_promedio"),
        ):
            if (source == "basic" and sensor == "planet") or (
                source == "pro" and sensor != "planet"
            ):
                continue
            sub = data[data["sensor_key"].eq(sensor)].copy()
            sub = sub[np.isfinite(sub[variable].to_numpy(float))]
            groups = {}
            for pos, group in sub.groupby("position"):
                # days since 1970-01-01 (constant offsets are irrelevant).
                days = (group["date"] - pd.Timestamp("1970-01-01")).dt.days
                groups[int(pos)] = np.column_stack([
                    days.to_numpy(float),
                    group[variable].to_numpy(float),
                    group["quality"].to_numpy(float),
                ])
            result[(source, sensor, variable)] = groups
    return result


def _estimate_at(
    when: pd.Timestamp, position: int, library: dict[int, np.ndarray],
    neighbors: np.ndarray, distance: np.ndarray, *,
    time_days: float = 14.0, max_lag: int = 35,
    radius: float = 0.05,
    season: str = "crop2025",
) -> float:
    t = float((when - pd.Timestamp("1970-01-01")).days)
    year = when.year if season == "hist" else 2025
    start = float((pd.Timestamp(f"{year}-04-01") - pd.Timestamp("1970-01-01")).days)
    end = float((pd.Timestamp(f"{year}-10-31") - pd.Timestamp("1970-01-01")).days)
    numerator = 0.
    denominator = 0.
    for j in neighbors:
        obs = library.get(int(j))
        if obs is None or not len(obs):
            continue
        lag = abs(obs[:, 0] - t)
        valid = (lag <= max_lag) & (obs[:, 0] >= start) & (obs[:, 0] <= end)
        if not valid.any():
            continue
        local = obs[valid]
        dt = lag[valid]
        # X-only spatiotemporal kernel. Spatial radius is in the normalized
        # private distance units, NOT kilometers.
        dweight = np.exp(-0.5 * (distance[position, j] / radius)**2)
        w = local[:, 2] * np.exp(-dt / time_days) * dweight
        numerator += float(w @ local[:, 1])
        denominator += float(w.sum())
    return numerator / denominator if denominator > 1e-10 else np.nan


def temporal_fill(
    frame: pd.DataFrame,
    distance: np.ndarray,
    basic: pd.DataFrame,
    pro: pd.DataFrame,
    *,
    spatiotemporal: bool,
    k: int = 8,
) -> pd.DataFrame:
    """Fill only supported missing features, using actual dated observations."""
    filled = frame.copy(deep=True)
    library = _observation_library(basic, pro)
    n = len(frame)
    positive = np.sort(distance + np.eye(n) * 1e12, axis=1)
    # Median 5th-nearest distance in frozen normalized distance units.
    radius = max(float(np.median(positive[:, min(4, n - 2)])), 1e-6)
    available = set(filled.columns)
    for feature, (source, sensor, variable, season, summary) in SATELLITE_FEATURES.items():
        if feature not in available:
            continue
        missing = np.flatnonzero(
            ~np.isfinite(pd.to_numeric(frame[feature], errors="coerce").to_numpy(float))
        )
        raw = library[(source, sensor, variable)]
        times = _time_grid(season)
        for i in missing:
            neigh = (
                candidate_neighbors(i, frame, distance, include_self=True, k=k)
                if spatiotemporal else np.asarray([i], int)
            )
            values = np.asarray([
                _estimate_at(
                    when, i, raw, neigh, distance, radius=radius, season=season
                )
                for when in times
            ], dtype=float)
            finite = values[np.isfinite(values)]
            if summary == "value":
                if len(finite) != 1:
                    continue
                answer = float(finite[0])
            elif summary == "delta":
                if len(finite) != 2:
                    continue
                answer = float(values[1] - values[0])
            elif summary == "mean":
                if len(finite) < max(3, int(len(times) * 0.35)):
                    continue
                answer = float(np.mean(finite))
            elif summary == "std":
                if len(finite) < max(4, int(len(times) * 0.35)):
                    continue
                answer = float(np.std(finite, ddof=1))
            else:
                raise ValueError(f"Unsupported statistic {summary}.")
            if np.isfinite(answer):
                filled.at[filled.index[i], feature] = answer
    return filled


def audit_imputations(
    original: pd.DataFrame,
    methods: dict[str, pd.DataFrame],
    features: list[str],
) -> pd.DataFrame:
    rows = []
    for method, candidate in methods.items():
        if candidate.columns.tolist() != original.columns.tolist():
            raise ValueError("Imputation changed the original feature table schema.")
        if not candidate[ID].astype(str).equals(original[ID].astype(str)):
            raise ValueError("Imputation changed parcel IDs.")
        if not candidate[TARGET].equals(original[TARGET]):
            raise ValueError("Imputation touched observed or held-out target yields.")
        for feature in features:
            src = pd.to_numeric(original[feature], errors="coerce")
            dst = pd.to_numeric(candidate[feature], errors="coerce")
            before = src.to_numpy(float)
            after = dst.to_numpy(float)
            observed = np.isfinite(before)
            if not np.array_equal(before[observed], after[observed]):
                raise ValueError(f"{method} modified a known feature: {feature}")
            recovered = int((~observed & np.isfinite(after)).sum())
            total = int((~observed).sum())
            rows.append({
                "method": method,
                "feature": feature,
                "missing_before": total,
                "newly_filled": recovered,
                "still_missing": total - recovered,
            })
    return pd.DataFrame(rows)


def prepare(
    frame: pd.DataFrame,
    distance: np.ndarray,
    features: list[str],
    basic_file: Path,
    pro_file: Path,
    *,
    spatial_k: int = 8,
) -> Reconstruction:
    if len(frame) != 138 or not frame[ID].astype(str).is_unique:
        raise ValueError("Expected 138 unique teammate training parcels.")
    if not {"Estado", "Municipio"}.issubset(frame.columns):
        raise ValueError("Need original teammate Estado and Municipio for spatial priors.")
    if not 1 <= spatial_k < len(frame):
        raise ValueError("spatial_k must be between 1 and 137.")
    distance = _distance_checks(distance, len(frame))
    fields = donor_columns(frame, features)
    if not fields:
        raise ValueError("No eligible numeric predictors for this comparison.")
    index = {pid: i for i, pid in enumerate(frame[ID].astype(str))}
    basic = read_satellite_source(basic_file, "basic", index)
    pro = read_satellite_source(pro_file, "pro", index)
    raw_counts = {"BASIC": len(basic), "PRO": len(pro)}
    if not raw_counts["BASIC"] or not raw_counts["PRO"]:
        raise ValueError("No original BASIC or PRO rows match the 138 parcel IDs.")
    spatial = spatial_fill(frame, distance, fields, k=spatial_k)
    temporal = temporal_fill(frame, distance, basic, pro, spatiotemporal=False, k=spatial_k)
    joined = temporal_fill(frame, distance, basic, pro, spatiotemporal=True, k=spatial_k)
    # If no valid raw time series exists, fall back to regional spatial X
    # rather than inventing a season trajectory.
    for feature in fields:
        value = pd.to_numeric(joined[feature], errors="coerce").to_numpy(float)
        other = pd.to_numeric(spatial[feature], errors="coerce").to_numpy(float)
        missing = ~np.isfinite(value)
        if missing.any():
            joined.loc[missing, feature] = other[missing]
    methods = {
        "spatial": spatial,
        "temporal": temporal,
        "spatiotemporal": joined,
    }
    audits = audit_imputations(frame, methods, features)
    initial = int(sum(
        (~np.isfinite(pd.to_numeric(frame[c], errors="coerce").to_numpy(float))).sum()
        for c in fields
    ))
    return Reconstruction(methods, audits, initial, raw_counts)
