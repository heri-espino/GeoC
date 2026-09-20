"""Checkpoint 04A target-set topology, similarity and support diagnostics.

This module implements the first active transductive diagnostic stage.  It never
uses the 59 hidden FIRA yields.  X-only representations may use all 197 parcel
covariates because the final task is fixed-target transductive reconstruction.

Pseudo-target validation belongs to Checkpoint 04B; 04A characterizes geometry,
support, temporal similarity, adversarial train/target shift and spatial
autocorrelation so later models can decide when local/transductive information is
credible.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import pairwise_distances, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

ID_COLUMN = "ID_POLIGONO"
TARGET_COLUMN = "RENDIMIENTO_T_HA"
SPLIT_COLUMN = "CONJUNTO"
TRAIN_VALUE = "ENTRENAMIENTO"
PREDICTION_VALUE = "PREDICCION"


@dataclass(frozen=True)
class EmbeddingResult:
    """One transductive PCA representation and its metadata."""

    name: str
    frame: pd.DataFrame
    features_used: tuple[str, ...]
    n_components: int
    explained_variance_ratio_sum: float


def _as_float_array(frame: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    return frame.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce").to_numpy(float)


def resolve_numeric_representation_features(
    *,
    joined: pd.DataFrame,
    manifests: Mapping[str, Mapping[str, Any]],
    layers: Sequence[str],
    allowed_modes: Sequence[str] = ("clean", "competition"),
    exclude_sources: Sequence[str] = (),
    exclude_columns: Sequence[str] = (),
) -> tuple[str, ...]:
    """Resolve one deterministic numeric competition representation.

    This is intentionally X-only.  It does not include fold-local discovered
    expressions from Checkpoint 03 because those are target-aware.
    """

    layer_set = set(map(str, layers))
    mode_set = set(map(str, allowed_modes))
    source_exclusions = set(map(str, exclude_sources))
    column_exclusions = set(map(str, exclude_columns))
    selected: list[str] = []

    for layer, manifest in manifests.items():
        if layer not in layer_set:
            continue
        records = manifest.get("features", [])
        if not isinstance(records, list):
            raise ValueError(f"Manifest for {layer!r} has no valid 'features' list.")
        for record in records:
            column = str(record["column"])
            if column in column_exclusions or column not in joined.columns:
                continue
            if str(record.get("mode", "")) not in mode_set:
                continue
            if str(record.get("source", "")) in source_exclusions:
                continue
            if not pd.api.types.is_numeric_dtype(joined[column]):
                continue
            selected.append(column)

    selected = list(dict.fromkeys(selected))
    if not selected:
        raise ValueError(f"No numeric features resolved for layers={list(layers)!r}.")
    return tuple(selected)


def build_transductive_pca_embedding(
    frame: pd.DataFrame,
    *,
    name: str,
    features: Sequence[str],
    n_components: int = 20,
) -> EmbeddingResult:
    """Fit median imputation, scaling and PCA jointly on all available X rows.

    The transform is deliberately transductive: all 197 parcel covariates may
    determine the X-only coordinate system.  Hidden y is never inspected.
    """

    feature_list = [column for column in features if column in frame.columns]
    if not feature_list:
        raise ValueError(f"{name}: no requested features exist in the frame.")

    numeric = frame.loc[:, feature_list].apply(pd.to_numeric, errors="coerce")
    usable = [
        column
        for column in numeric.columns
        if numeric[column].notna().any() and numeric[column].nunique(dropna=True) > 1
    ]
    if not usable:
        raise ValueError(f"{name}: all requested features are empty or constant.")

    imputer = SimpleImputer(strategy="median", keep_empty_features=False)
    scaler = StandardScaler()
    x_imputed = imputer.fit_transform(numeric.loc[:, usable])
    x_scaled = scaler.fit_transform(x_imputed)

    max_components = min(x_scaled.shape[0] - 1, x_scaled.shape[1])
    n_pc = max(1, min(int(n_components), int(max_components)))
    pca = PCA(n_components=n_pc, svd_solver="full")
    embedded = pca.fit_transform(x_scaled)

    result = frame.loc[:, [ID_COLUMN, SPLIT_COLUMN]].copy()
    for index in range(n_pc):
        result[f"pc{index + 1:02d}"] = embedded[:, index]

    return EmbeddingResult(
        name=str(name),
        frame=result,
        features_used=tuple(usable),
        n_components=n_pc,
        explained_variance_ratio_sum=float(np.sum(pca.explained_variance_ratio_)),
    )


def haversine_distance_matrix(
    latitude_deg: Sequence[float],
    longitude_deg: Sequence[float],
) -> np.ndarray:
    """Return the full great-circle distance matrix in kilometres."""

    lat = np.radians(np.asarray(latitude_deg, dtype=float))
    lon = np.radians(np.asarray(longitude_deg, dtype=float))
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2.0) ** 2
    )
    a = np.clip(a, 0.0, 1.0)
    return 6371.0088 * 2.0 * np.arcsin(np.sqrt(a))


def _neighbor_rows(
    *,
    distance_matrix: np.ndarray,
    metadata: pd.DataFrame,
    query_positions: Sequence[int],
    reference_positions: Sequence[int],
    k: int,
    metric: str,
    representation: str,
    exclude_self: bool,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    reference_positions = list(map(int, reference_positions))
    ids = metadata[ID_COLUMN].astype(str).to_numpy()
    states = metadata.get("meta_estado", pd.Series([""] * len(metadata))).astype(str).to_numpy()
    municipalities = metadata.get(
        "meta_municipio", pd.Series([""] * len(metadata))
    ).astype(str).to_numpy()
    yields = pd.to_numeric(
        metadata.get(TARGET_COLUMN, pd.Series([np.nan] * len(metadata))),
        errors="coerce",
    ).to_numpy(float)

    for q_pos in map(int, query_positions):
        candidates = [
            r_pos for r_pos in reference_positions if not (exclude_self and r_pos == q_pos)
        ]
        if not candidates:
            continue
        ranked = sorted(candidates, key=lambda r_pos: distance_matrix[q_pos, r_pos])
        for rank, r_pos in enumerate(ranked[: int(k)], start=1):
            rows.append(
                {
                    "metric": metric,
                    "representation": representation,
                    "query_id": ids[q_pos],
                    "query_split": str(metadata.iloc[q_pos][SPLIT_COLUMN]),
                    "neighbor_id": ids[r_pos],
                    "neighbor_split": str(metadata.iloc[r_pos][SPLIT_COLUMN]),
                    "rank": rank,
                    "distance": float(distance_matrix[q_pos, r_pos]),
                    "same_state": bool(states[q_pos] == states[r_pos]),
                    "same_municipality": bool(
                        states[q_pos] == states[r_pos]
                        and municipalities[q_pos] == municipalities[r_pos]
                    ),
                    "neighbor_y": float(yields[r_pos]) if np.isfinite(yields[r_pos]) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def geographic_neighbor_tables(
    frame: pd.DataFrame,
    *,
    latitude_column: str = "base_centroid_lat",
    longitude_column: str = "base_centroid_lon",
    k: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Return target-to-train and train leave-one-out geographic neighbours."""

    required = {ID_COLUMN, SPLIT_COLUMN, latitude_column, longitude_column}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Missing geographic columns: {sorted(missing)}")

    coordinates = frame[[latitude_column, longitude_column]].apply(
        pd.to_numeric, errors="coerce"
    )
    if coordinates.isna().any().any():
        raise ValueError("Geographic diagnostics require finite centroid coordinates.")

    distances = haversine_distance_matrix(
        coordinates[latitude_column].to_numpy(float),
        coordinates[longitude_column].to_numpy(float),
    )
    train_pos = np.flatnonzero(frame[SPLIT_COLUMN].eq(TRAIN_VALUE).to_numpy())
    target_pos = np.flatnonzero(frame[SPLIT_COLUMN].eq(PREDICTION_VALUE).to_numpy())

    target_neighbors = _neighbor_rows(
        distance_matrix=distances,
        metadata=frame,
        query_positions=target_pos,
        reference_positions=train_pos,
        k=k,
        metric="haversine_km",
        representation="geographic",
        exclude_self=False,
    )
    train_neighbors = _neighbor_rows(
        distance_matrix=distances,
        metadata=frame,
        query_positions=train_pos,
        reference_positions=train_pos,
        k=k,
        metric="haversine_km",
        representation="geographic",
        exclude_self=True,
    )
    return target_neighbors, train_neighbors, distances


def feature_neighbor_tables(
    frame: pd.DataFrame,
    embedding: EmbeddingResult,
    *,
    k: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Compute Euclidean neighbour tables in one transductive PCA space."""

    pc_columns = [column for column in embedding.frame.columns if column.startswith("pc")]
    values = embedding.frame.loc[:, pc_columns].to_numpy(float)
    distances = pairwise_distances(values, metric="euclidean")

    train_pos = np.flatnonzero(frame[SPLIT_COLUMN].eq(TRAIN_VALUE).to_numpy())
    target_pos = np.flatnonzero(frame[SPLIT_COLUMN].eq(PREDICTION_VALUE).to_numpy())
    target_neighbors = _neighbor_rows(
        distance_matrix=distances,
        metadata=frame,
        query_positions=target_pos,
        reference_positions=train_pos,
        k=k,
        metric="euclidean_pca",
        representation=embedding.name,
        exclude_self=False,
    )
    train_neighbors = _neighbor_rows(
        distance_matrix=distances,
        metadata=frame,
        query_positions=train_pos,
        reference_positions=train_pos,
        k=k,
        metric="euclidean_pca",
        representation=embedding.name,
        exclude_self=True,
    )
    return target_neighbors, train_neighbors, distances


def _interpolated_z_curve(values: np.ndarray, *, min_points: int) -> np.ndarray | None:
    series = pd.Series(np.asarray(values, dtype=float))
    if int(series.notna().sum()) < int(min_points):
        return None
    series = series.interpolate(limit_direction="both")
    array = series.to_numpy(float)
    if not np.isfinite(array).all():
        return None
    std = float(np.std(array))
    if std <= 1.0e-12:
        return None
    return (array - float(np.mean(array))) / std


def _pearson_valid(a: np.ndarray, b: np.ndarray, *, min_points: int) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < int(min_points):
        return np.nan
    x = a[mask]
    y = b[mask]
    if float(np.std(x)) <= 1.0e-12 or float(np.std(y)) <= 1.0e-12:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def _spearman_valid(a: np.ndarray, b: np.ndarray, *, min_points: int) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < int(min_points):
        return np.nan
    x = a[mask]
    y = b[mask]
    if len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return np.nan
    return float(spearmanr(x, y).statistic)


def _best_lag_correlation(
    a: np.ndarray,
    b: np.ndarray,
    *,
    max_lag: int,
    min_points: int,
) -> tuple[float, int]:
    best = np.nan
    best_lag = 0
    for lag in range(-int(max_lag), int(max_lag) + 1):
        if lag < 0:
            x = a[-lag:]
            y = b[: len(b) + lag]
        elif lag > 0:
            x = a[: len(a) - lag]
            y = b[lag:]
        else:
            x = a
            y = b
        corr = _pearson_valid(x, y, min_points=min_points)
        if np.isfinite(corr) and (not np.isfinite(best) or corr > best):
            best = float(corr)
            best_lag = int(lag)
    return float(best) if np.isfinite(best) else np.nan, best_lag


def _dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    n = len(a)
    m = len(b)
    matrix = np.full((n + 1, m + 1), np.inf, dtype=float)
    matrix[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(float(a[i - 1] - b[j - 1]))
            matrix[i, j] = cost + min(
                matrix[i - 1, j],
                matrix[i, j - 1],
                matrix[i - 1, j - 1],
            )
    return float(matrix[n, m] / max(n, m))


def build_monthly_temporal_arrays(
    frame: pd.DataFrame,
    *,
    series: Sequence[Mapping[str, Any]],
    months: Sequence[int],
) -> dict[str, np.ndarray]:
    """Extract configured monthly parcel trajectories from Feature Table v1."""

    arrays: dict[str, np.ndarray] = {}
    for spec in series:
        name = str(spec["name"])
        prefix = str(spec["prefix"])
        columns = [f"{prefix}{int(month):02d}" for month in months]
        if not all(column in frame.columns for column in columns):
            continue
        arrays[name] = _as_float_array(frame, columns)
    if not arrays:
        raise ValueError("No configured monthly temporal series were found in the feature table.")
    return arrays


def temporal_pair_metrics(
    frame: pd.DataFrame,
    *,
    series_arrays: Mapping[str, np.ndarray],
    max_lag: int = 2,
    min_points: int = 4,
) -> pd.DataFrame:
    """Compute aggregate target-train and unique train-train temporal similarities."""

    ids = frame[ID_COLUMN].astype(str).to_numpy()
    splits = frame[SPLIT_COLUMN].astype(str).to_numpy()
    train_pos = np.flatnonzero(splits == TRAIN_VALUE)
    target_pos = np.flatnonzero(splits == PREDICTION_VALUE)

    pairs: list[tuple[int, int, str]] = []
    for q_pos in target_pos:
        for r_pos in train_pos:
            pairs.append((int(q_pos), int(r_pos), "target_train"))
    for left_index, q_pos in enumerate(train_pos):
        for r_pos in train_pos[left_index + 1 :]:
            pairs.append((int(q_pos), int(r_pos), "train_train"))

    rows: list[dict[str, Any]] = []
    for q_pos, r_pos, pair_type in pairs:
        pearsons: list[float] = []
        spearmans: list[float] = []
        lag_corrs: list[float] = []
        lags: list[int] = []
        dtws: list[float] = []

        for array in series_arrays.values():
            a = array[q_pos]
            b = array[r_pos]
            pearson = _pearson_valid(a, b, min_points=min_points)
            spearman = _spearman_valid(a, b, min_points=min_points)
            lag_corr, lag = _best_lag_correlation(
                a,
                b,
                max_lag=max_lag,
                min_points=max(3, min_points - 1),
            )
            za = _interpolated_z_curve(a, min_points=min_points)
            zb = _interpolated_z_curve(b, min_points=min_points)
            dtw = _dtw_distance(za, zb) if za is not None and zb is not None else np.nan

            if np.isfinite(pearson):
                pearsons.append(float(pearson))
            if np.isfinite(spearman):
                spearmans.append(float(spearman))
            if np.isfinite(lag_corr):
                lag_corrs.append(float(lag_corr))
                lags.append(int(lag))
            if np.isfinite(dtw):
                dtws.append(float(dtw))

        rows.append(
            {
                "pair_type": pair_type,
                "query_id": ids[q_pos],
                "neighbor_id": ids[r_pos],
                "n_series": len(lag_corrs),
                "pearson_mean": float(np.mean(pearsons)) if pearsons else np.nan,
                "spearman_mean": float(np.mean(spearmans)) if spearmans else np.nan,
                "best_lag_corr_mean": float(np.mean(lag_corrs)) if lag_corrs else np.nan,
                "best_lag_median_months": float(np.median(lags)) if lags else np.nan,
                "dtw_mean": float(np.mean(dtws)) if dtws else np.nan,
            }
        )
    return pd.DataFrame(rows)


def temporal_neighbor_tables(
    temporal_pairs: pd.DataFrame,
    *,
    k: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rank temporal neighbours by high lagged correlation then low DTW."""

    rows: list[dict[str, Any]] = []
    for pair_type in ["target_train", "train_train"]:
        subset = temporal_pairs.loc[temporal_pairs["pair_type"].eq(pair_type)].copy()
        if pair_type == "train_train":
            reverse = subset.rename(
                columns={"query_id": "neighbor_id", "neighbor_id": "query_id"}
            )
            subset = pd.concat([subset, reverse], ignore_index=True)

        for query_id, group in subset.groupby("query_id", sort=False):
            ranked = group.sort_values(
                ["best_lag_corr_mean", "dtw_mean"],
                ascending=[False, True],
                na_position="last",
            ).head(int(k))
            for rank, row in enumerate(ranked.itertuples(index=False), start=1):
                rows.append(
                    {
                        "pair_type": pair_type,
                        "query_id": query_id,
                        "neighbor_id": row.neighbor_id,
                        "rank": rank,
                        "n_series": int(row.n_series),
                        "pearson_mean": row.pearson_mean,
                        "spearman_mean": row.spearman_mean,
                        "best_lag_corr_mean": row.best_lag_corr_mean,
                        "best_lag_median_months": row.best_lag_median_months,
                        "dtw_mean": row.dtw_mean,
                    }
                )

    ranked = pd.DataFrame(rows)
    return (
        ranked.loc[ranked["pair_type"].eq("target_train")].reset_index(drop=True),
        ranked.loc[ranked["pair_type"].eq("train_train")].reset_index(drop=True),
    )


def build_train_pair_validation(
    frame: pd.DataFrame,
    *,
    geographic_distances: np.ndarray,
    feature_distances: Mapping[str, np.ndarray],
    temporal_pairs: pd.DataFrame,
) -> pd.DataFrame:
    """Build one row per labeled-labeled pair with distance and |delta y|."""

    train_pos = np.flatnonzero(frame[SPLIT_COLUMN].eq(TRAIN_VALUE).to_numpy())
    ids = frame[ID_COLUMN].astype(str).to_numpy()
    y = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce").to_numpy(float)

    temporal_lookup = {
        tuple(sorted((str(row.query_id), str(row.neighbor_id)))): row
        for row in temporal_pairs.loc[temporal_pairs["pair_type"].eq("train_train")].itertuples()
    }

    rows: list[dict[str, Any]] = []
    for left_index, i in enumerate(train_pos):
        for j in train_pos[left_index + 1 :]:
            key = tuple(sorted((ids[i], ids[j])))
            row: dict[str, Any] = {
                "id_a": ids[i],
                "id_b": ids[j],
                "yield_a": float(y[i]),
                "yield_b": float(y[j]),
                "abs_yield_difference": float(abs(y[i] - y[j])),
                "geo_distance_km": float(geographic_distances[i, j]),
            }
            for name, matrix in feature_distances.items():
                row[f"{name}__distance"] = float(matrix[i, j])
            temporal = temporal_lookup.get(key)
            if temporal is not None:
                row["temporal_pearson_mean"] = temporal.pearson_mean
                row["temporal_best_lag_corr_mean"] = temporal.best_lag_corr_mean
                row["temporal_dtw_mean"] = temporal.dtw_mean
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_similarity_yield_relationships(pair_table: pd.DataFrame) -> pd.DataFrame:
    """Summarize monotone association between pair similarity and |delta yield|."""

    target = pd.to_numeric(pair_table["abs_yield_difference"], errors="coerce")
    rows: list[dict[str, Any]] = []
    metric_columns = [
        column
        for column in pair_table.columns
        if column == "geo_distance_km"
        or column.endswith("__distance")
        or column.startswith("temporal_")
    ]
    for column in metric_columns:
        values = pd.to_numeric(pair_table[column], errors="coerce")
        mask = values.notna() & target.notna()
        if int(mask.sum()) < 10 or values.loc[mask].nunique() < 2:
            continue
        result = spearmanr(values.loc[mask], target.loc[mask])
        higher_means_more_similar = (
            ("corr" in column or "pearson" in column or "spearman" in column)
            and "distance" not in column
        )
        rows.append(
            {
                "metric": column,
                "n_pairs": int(mask.sum()),
                "spearman_rho_with_abs_yield_difference": float(result.statistic),
                "p_value": float(result.pvalue),
                "expected_direction_if_useful": (
                    "negative" if higher_means_more_similar else "positive"
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        "spearman_rho_with_abs_yield_difference",
        key=lambda series: series.abs(),
        ascending=False,
    ).reset_index(drop=True)


def adversarial_train_target_validation(
    frame: pd.DataFrame,
    embedding: EmbeddingResult,
    *,
    random_state: int = 42,
    folds: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Cross-validate a train-vs-target classifier in a transductive X embedding."""

    pcs = [column for column in embedding.frame.columns if column.startswith("pc")]
    x = embedding.frame.loc[:, pcs].to_numpy(float)
    z = frame[SPLIT_COLUMN].eq(PREDICTION_VALUE).astype(int).to_numpy()

    model = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=3000,
        random_state=int(random_state),
    )
    cv = StratifiedKFold(
        n_splits=int(folds),
        shuffle=True,
        random_state=int(random_state),
    )
    oof_probability = cross_val_predict(
        model,
        x,
        z,
        cv=cv,
        method="predict_proba",
    )[:, 1]
    auc = float(roc_auc_score(z, oof_probability))
    fpr, tpr, thresholds = roc_curve(z, oof_probability)

    model.fit(x, z)
    full_probability = model.predict_proba(x)[:, 1]
    scores = frame.loc[:, [ID_COLUMN, SPLIT_COLUMN]].copy()
    scores["adversarial_oof_target_probability"] = oof_probability
    scores["adversarial_full_target_probability"] = full_probability
    roc = pd.DataFrame(
        {
            "false_positive_rate": fpr,
            "true_positive_rate": tpr,
            "threshold": thresholds,
        }
    )
    summary = {
        "auc": auc,
        "n_rows": float(len(frame)),
        "n_train": float(np.sum(z == 0)),
        "n_target": float(np.sum(z == 1)),
    }
    return scores, roc, summary


def feature_shift_table(
    frame: pd.DataFrame,
    *,
    features: Sequence[str],
) -> pd.DataFrame:
    """Compute per-feature standardized mean and missingness shift."""

    train = frame.loc[frame[SPLIT_COLUMN].eq(TRAIN_VALUE)]
    target = frame.loc[frame[SPLIT_COLUMN].eq(PREDICTION_VALUE)]
    rows: list[dict[str, Any]] = []

    for column in features:
        if column not in frame.columns or not pd.api.types.is_numeric_dtype(frame[column]):
            continue
        a = pd.to_numeric(train[column], errors="coerce")
        b = pd.to_numeric(target[column], errors="coerce")
        mean_a = float(a.mean()) if a.notna().any() else np.nan
        mean_b = float(b.mean()) if b.notna().any() else np.nan
        var_a = float(a.var(ddof=1)) if int(a.notna().sum()) > 1 else np.nan
        var_b = float(b.var(ddof=1)) if int(b.notna().sum()) > 1 else np.nan
        pooled = np.nanmean([var_a, var_b])
        sd = float(np.sqrt(pooled)) if np.isfinite(pooled) and pooled > 0 else np.nan
        smd = (mean_b - mean_a) / sd if np.isfinite(sd) else np.nan
        rows.append(
            {
                "feature": column,
                "train_mean": mean_a,
                "target_mean": mean_b,
                "standardized_mean_difference": float(smd) if np.isfinite(smd) else np.nan,
                "abs_standardized_mean_difference": abs(float(smd))
                if np.isfinite(smd)
                else np.nan,
                "train_missing_fraction": float(a.isna().mean()),
                "target_missing_fraction": float(b.isna().mean()),
                "missing_fraction_difference": float(b.isna().mean() - a.isna().mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        "abs_standardized_mean_difference",
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)


def knn_weight_matrix(distance_matrix: np.ndarray, *, k: int = 5) -> np.ndarray:
    """Create a symmetric row-standardized kNN spatial weight matrix."""

    distances = np.asarray(distance_matrix, dtype=float)
    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be square.")
    n = distances.shape[0]
    if n < 3:
        raise ValueError("At least three observations are required.")
    k_eff = max(1, min(int(k), n - 1))

    weights = np.zeros_like(distances, dtype=float)
    for i in range(n):
        order = np.argsort(distances[i])
        neighbours = [j for j in order if j != i][:k_eff]
        weights[i, neighbours] = 1.0
    weights = np.maximum(weights, weights.T)
    row_sum = weights.sum(axis=1)
    nonzero = row_sum > 0
    weights[nonzero] = weights[nonzero] / row_sum[nonzero, None]
    return weights


def morans_i(
    values: Sequence[float],
    weights: np.ndarray,
    *,
    permutations: int = 999,
    random_state: int = 42,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    """Compute Moran's I and a two-sided permutation p-value."""

    x = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    if len(x) != w.shape[0] or w.shape[0] != w.shape[1]:
        raise ValueError("Values and weights have incompatible shapes.")
    if not np.isfinite(x).all():
        raise ValueError("Moran's I requires finite values.")

    z = x - float(np.mean(x))
    denominator = float(np.sum(z**2))
    s0 = float(np.sum(w))
    if denominator <= 0 or s0 <= 0:
        raise ValueError("Moran's I requires nonconstant values and nonzero weights.")

    def statistic(current: np.ndarray) -> float:
        centered = current - float(np.mean(current))
        return float(
            len(current)
            / s0
            * np.sum(w * np.outer(centered, centered))
            / np.sum(centered**2)
        )

    observed = statistic(x)
    rng = np.random.default_rng(int(random_state))
    permuted = np.asarray([statistic(rng.permutation(x)) for _ in range(int(permutations))])
    extreme = int(np.sum(np.abs(permuted) >= abs(observed)))
    p_value = float((extreme + 1) / (len(permuted) + 1))
    spatial_lag = w @ z

    summary = {
        "morans_i": float(observed),
        "expected_i": float(-1.0 / (len(x) - 1)),
        "permutation_p_value_two_sided": p_value,
        "permutations": float(permutations),
    }
    return summary, z, spatial_lag


def build_checkpoint03_ensemble_residuals(
    oof: pd.DataFrame,
    *,
    protocol: str,
) -> pd.DataFrame:
    """Reconstruct E13/E123 OOF residuals from the frozen 03C.2 finalists."""

    finalists = {
        "F1": ("C0_base", "PLS"),
        "F2": ("C3_base_agro_plus_discovery", "Ridge"),
        "F3": ("C1_agronomic", "CatBoost"),
    }
    selected: list[pd.DataFrame] = []
    for label, (representation, model) in finalists.items():
        part = oof.loc[
            oof["protocol"].eq(protocol)
            & oof["representation"].eq(representation)
            & oof["model"].eq(model),
            [ID_COLUMN, "observed", "predicted"],
        ].copy()
        if part.empty:
            raise ValueError(f"Missing {label} OOF rows for {protocol}.")
        part = part.rename(columns={"predicted": label})
        selected.append(part)

    merged = selected[0]
    for part in selected[1:]:
        merged = merged.merge(
            part.drop(columns=["observed"]),
            on=ID_COLUMN,
            how="inner",
            validate="1:1",
        )
    merged["E13_predicted"] = merged[["F1", "F3"]].mean(axis=1)
    merged["E123_predicted"] = merged[["F1", "F2", "F3"]].mean(axis=1)
    merged["E13_residual"] = merged["observed"] - merged["E13_predicted"]
    merged["E123_residual"] = merged["observed"] - merged["E123_predicted"]
    return merged


def _distance_support(value: float, reference: Sequence[float]) -> float:
    ref = np.asarray(reference, dtype=float)
    ref = ref[np.isfinite(ref)]
    if not np.isfinite(value) or len(ref) == 0:
        return np.nan
    return float(np.mean(ref >= float(value)))


def _similarity_support(value: float, reference: Sequence[float]) -> float:
    ref = np.asarray(reference, dtype=float)
    ref = ref[np.isfinite(ref)]
    if not np.isfinite(value) or len(ref) == 0:
        return np.nan
    return float(np.mean(ref <= float(value)))


def build_target_support_profile(
    frame: pd.DataFrame,
    *,
    geographic_target_neighbors: pd.DataFrame,
    geographic_train_neighbors: pd.DataFrame,
    feature_target_neighbors: pd.DataFrame,
    feature_train_neighbors: pd.DataFrame,
    temporal_target_neighbors: pd.DataFrame,
    temporal_train_neighbors: pd.DataFrame,
    adversarial_scores: pd.DataFrame,
    primary_representation: str,
    weights: Mapping[str, float] | None = None,
    high_threshold: float = 0.67,
    low_threshold: float = 0.33,
) -> pd.DataFrame:
    """Assemble one diagnostic support profile for each of the 59 targets."""

    if weights is None:
        weights = {
            "geographic": 1.0,
            "feature": 1.0,
            "temporal": 1.0,
            "adversarial": 1.0,
            "neighbor_consistency": 1.0,
        }

    targets = frame.loc[
        frame[SPLIT_COLUMN].eq(PREDICTION_VALUE),
        [ID_COLUMN, "meta_estado", "meta_municipio"],
    ].copy()

    geo_target = geographic_target_neighbors.loc[
        geographic_target_neighbors["rank"].eq(1)
    ].set_index("query_id")
    geo_train = geographic_train_neighbors.loc[
        geographic_train_neighbors["rank"].eq(1), "distance"
    ].to_numpy(float)

    ft = feature_target_neighbors.loc[
        feature_target_neighbors["representation"].eq(primary_representation)
    ].copy()
    fr = feature_train_neighbors.loc[
        feature_train_neighbors["representation"].eq(primary_representation)
    ].copy()
    ft1 = ft.loc[ft["rank"].eq(1)].set_index("query_id")
    fr1_distance = fr.loc[fr["rank"].eq(1), "distance"].to_numpy(float)

    temporal_target1 = temporal_target_neighbors.loc[
        temporal_target_neighbors["rank"].eq(1)
    ].set_index("query_id")
    temporal_train1_values = temporal_train_neighbors.loc[
        temporal_train_neighbors["rank"].eq(1), "best_lag_corr_mean"
    ].to_numpy(float)

    adversarial = adversarial_scores.set_index(ID_COLUMN)
    train_adv = adversarial.loc[
        frame.loc[frame[SPLIT_COLUMN].eq(TRAIN_VALUE), ID_COLUMN].astype(str),
        "adversarial_full_target_probability",
    ].to_numpy(float)

    train_neighbor_sd = (
        fr.groupby("query_id")["neighbor_y"].std(ddof=0).dropna().to_numpy(float)
    )

    rows: list[dict[str, Any]] = []
    for row in targets.itertuples(index=False):
        query_id = str(getattr(row, ID_COLUMN))
        geo = geo_target.loc[query_id] if query_id in geo_target.index else None
        feature = ft1.loc[query_id] if query_id in ft1.index else None
        temporal = (
            temporal_target1.loc[query_id]
            if query_id in temporal_target1.index
            else None
        )
        adv_probability = (
            float(adversarial.loc[query_id, "adversarial_full_target_probability"])
            if query_id in adversarial.index
            else np.nan
        )
        neighbor_sd = float(
            ft.loc[ft["query_id"].eq(query_id), "neighbor_y"].std(ddof=0)
        )

        components = {
            "geographic_support": _distance_support(
                float(geo["distance"]) if geo is not None else np.nan,
                geo_train,
            ),
            "feature_support": _distance_support(
                float(feature["distance"]) if feature is not None else np.nan,
                fr1_distance,
            ),
            "temporal_support": _similarity_support(
                float(temporal["best_lag_corr_mean"])
                if temporal is not None
                else np.nan,
                temporal_train1_values,
            ),
            "adversarial_support": _similarity_support(
                -adv_probability,
                -train_adv,
            ),
            "neighbor_consistency_support": _distance_support(
                neighbor_sd,
                train_neighbor_sd,
            ),
        }

        numerator = 0.0
        denominator = 0.0
        for key, value in components.items():
            component_name = key.removesuffix("_support")
            weight = float(weights.get(component_name, 1.0))
            if np.isfinite(value) and weight > 0:
                numerator += weight * float(value)
                denominator += weight
        overall = numerator / denominator if denominator > 0 else np.nan

        geo_support = components["geographic_support"]
        feature_support = components["feature_support"]
        if np.isfinite(overall) and overall >= high_threshold:
            tier = "A_high_support"
        elif np.isfinite(overall) and overall < low_threshold:
            tier = "D_extrapolation"
        elif (
            np.isfinite(feature_support)
            and np.isfinite(geo_support)
            and feature_support >= high_threshold
            and geo_support < low_threshold
        ):
            tier = "B_feature_supported"
        elif (
            np.isfinite(feature_support)
            and np.isfinite(geo_support)
            and geo_support >= high_threshold
            and feature_support < low_threshold
        ):
            tier = "C_geographic_supported"
        else:
            tier = "E_mixed_support"

        rows.append(
            {
                ID_COLUMN: query_id,
                "meta_estado": row.meta_estado,
                "meta_municipio": row.meta_municipio,
                "nearest_geo_train_id": geo["neighbor_id"] if geo is not None else "",
                "nearest_geo_km": float(geo["distance"]) if geo is not None else np.nan,
                "nearest_feature_train_id": (
                    feature["neighbor_id"] if feature is not None else ""
                ),
                "nearest_feature_distance": (
                    float(feature["distance"]) if feature is not None else np.nan
                ),
                "nearest_temporal_train_id": (
                    temporal["neighbor_id"] if temporal is not None else ""
                ),
                "nearest_temporal_similarity": (
                    float(temporal["best_lag_corr_mean"])
                    if temporal is not None
                    else np.nan
                ),
                "feature_neighbor_y_mean": float(
                    ft.loc[ft["query_id"].eq(query_id), "neighbor_y"].mean()
                ),
                "feature_neighbor_y_sd": neighbor_sd,
                "adversarial_target_probability": adv_probability,
                **components,
                "support_score": float(overall) if np.isfinite(overall) else np.nan,
                "support_tier": tier,
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["support_score", ID_COLUMN],
        ascending=[False, True],
        na_position="last",
    ).reset_index(drop=True)
