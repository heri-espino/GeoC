"""Checkpoint 04B transductive pseudo-competition validation utilities.

The real competition exposes X for all 197 parcels and y for only 138. 04B
simulates that information structure by hiding y for pseudo-target parcels while
keeping every X row available to X-only transformations, distances and graphs.

Split construction is strictly X-only. No yield is accepted by the target-matched
split generator.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from geocebada.evaluation.checkpoint04a import (
    ID_COLUMN,
    PREDICTION_VALUE,
    SPLIT_COLUMN,
    TARGET_COLUMN,
    TRAIN_VALUE,
)


@dataclass(frozen=True)
class PseudoSplit:
    """One pseudo-competition split over the 138 observed labels."""

    split_id: str
    family: str
    repeat: int
    pseudo_target_positions: tuple[int, ...]
    match_objective: float
    municipality_tv: float
    profile_mean_distance: float


def hamilton_apportion(counts: Mapping[str, int], total: int) -> dict[str, int]:
    """Apportion an integer total proportionally using largest remainders."""

    positive = {str(key): int(value) for key, value in counts.items() if int(value) > 0}
    if not positive:
        raise ValueError("hamilton_apportion requires at least one positive count.")
    if int(total) < 0:
        raise ValueError("total must be nonnegative.")
    if int(total) == 0:
        return {key: 0 for key in positive}

    weight_sum = float(sum(positive.values()))
    quotas = {key: total * value / weight_sum for key, value in positive.items()}
    result = {key: int(np.floor(value)) for key, value in quotas.items()}
    remaining = int(total) - sum(result.values())
    order = sorted(
        positive,
        key=lambda key: (quotas[key] - result[key], positive[key], key),
        reverse=True,
    )
    for key in order[:remaining]:
        result[key] += 1
    return result


def temporal_similarity_matrix(
    frame: pd.DataFrame,
    temporal_pairs: pd.DataFrame,
    *,
    value_column: str = "best_lag_corr_mean",
) -> np.ndarray:
    """Build the symmetric train-train/target-train temporal similarity matrix."""

    ids = frame[ID_COLUMN].astype(str).tolist()
    index = {parcel_id: position for position, parcel_id in enumerate(ids)}
    matrix = np.full((len(frame), len(frame)), np.nan, dtype=float)
    np.fill_diagonal(matrix, 1.0)

    for row in temporal_pairs.itertuples(index=False):
        left = index.get(str(row.query_id))
        right = index.get(str(row.neighbor_id))
        if left is None or right is None:
            continue
        value = float(getattr(row, value_column))
        if not np.isfinite(value):
            continue
        matrix[left, right] = value
        matrix[right, left] = value
    return matrix


def _nearest_distance(
    matrix: np.ndarray,
    query_position: int,
    reference_positions: Sequence[int],
    *,
    exclude_self: bool,
) -> float:
    refs = [
        int(position)
        for position in reference_positions
        if not (exclude_self and int(position) == int(query_position))
    ]
    if not refs:
        return np.nan
    values = np.asarray(matrix[int(query_position), refs], dtype=float)
    values = values[np.isfinite(values)]
    return float(np.min(values)) if len(values) else np.nan


def _best_similarity(
    matrix: np.ndarray,
    query_position: int,
    reference_positions: Sequence[int],
    *,
    exclude_self: bool,
) -> float:
    refs = [
        int(position)
        for position in reference_positions
        if not (exclude_self and int(position) == int(query_position))
    ]
    if not refs:
        return np.nan
    values = np.asarray(matrix[int(query_position), refs], dtype=float)
    values = values[np.isfinite(values)]
    return float(np.max(values)) if len(values) else np.nan


def build_static_x_profile(
    frame: pd.DataFrame,
    *,
    geographic_distances: np.ndarray,
    agronomic_distances: np.ndarray,
    temporal_similarities: np.ndarray,
    adversarial_scores: pd.DataFrame,
) -> pd.DataFrame:
    """Build X-only support descriptors relative to the full 138-label set."""

    train_positions = np.flatnonzero(frame[SPLIT_COLUMN].eq(TRAIN_VALUE).to_numpy())
    adv = adversarial_scores.set_index(ID_COLUMN)[
        "adversarial_full_target_probability"
    ].astype(float)
    ids = frame[ID_COLUMN].astype(str).to_numpy()

    rows: list[dict[str, Any]] = []
    for position in range(len(frame)):
        is_train = bool(frame.iloc[position][SPLIT_COLUMN] == TRAIN_VALUE)
        rows.append(
            {
                ID_COLUMN: ids[position],
                SPLIT_COLUMN: str(frame.iloc[position][SPLIT_COLUMN]),
                "meta_estado": str(frame.iloc[position]["meta_estado"]),
                "meta_municipio": str(frame.iloc[position]["meta_municipio"]),
                "nearest_geo_km": _nearest_distance(
                    geographic_distances,
                    position,
                    train_positions,
                    exclude_self=is_train,
                ),
                "nearest_agro_distance": _nearest_distance(
                    agronomic_distances,
                    position,
                    train_positions,
                    exclude_self=is_train,
                ),
                "best_temporal_similarity": _best_similarity(
                    temporal_similarities,
                    position,
                    train_positions,
                    exclude_self=is_train,
                ),
                "adversarial_target_probability": float(adv.loc[ids[position]]),
            }
        )
    return pd.DataFrame(rows)


def standardized_profile_coordinates(
    profile: pd.DataFrame,
    *,
    columns: Sequence[str],
) -> np.ndarray:
    """Standardize X-only target-matching descriptors over all 197 parcels."""

    values = profile.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce")
    if values.isna().any().any():
        medians = values.median(axis=0)
        values = values.fillna(medians)
    return StandardScaler().fit_transform(values.to_numpy(float))


def _total_variation(
    selected: Sequence[str],
    target: Sequence[str],
) -> float:
    selected_series = pd.Series(list(selected), dtype="string")
    target_series = pd.Series(list(target), dtype="string")
    categories = sorted(set(selected_series.dropna()) | set(target_series.dropna()))
    if not categories:
        return 0.0
    selected_freq = selected_series.value_counts(normalize=True)
    target_freq = target_series.value_counts(normalize=True)
    return float(
        0.5
        * sum(
            abs(float(selected_freq.get(key, 0.0)) - float(target_freq.get(key, 0.0)))
            for key in categories
        )
    )


def _profile_distance_to_target_cloud(
    profile_coordinates: np.ndarray,
    frame: pd.DataFrame,
    *,
    candidate_positions: Sequence[int],
    target_positions: Sequence[int],
) -> np.ndarray:
    """Distance each labeled candidate to the closest relevant real target profile."""

    result = np.full(len(candidate_positions), np.nan, dtype=float)
    for index, candidate in enumerate(map(int, candidate_positions)):
        state = str(frame.iloc[candidate]["meta_estado"])
        municipality = str(frame.iloc[candidate]["meta_municipio"])
        same_municipality = [
            int(position)
            for position in target_positions
            if str(frame.iloc[int(position)]["meta_estado"]) == state
            and str(frame.iloc[int(position)]["meta_municipio"]) == municipality
        ]
        same_state = [
            int(position)
            for position in target_positions
            if str(frame.iloc[int(position)]["meta_estado"]) == state
        ]
        references = same_municipality or same_state or list(map(int, target_positions))
        distances = cdist(
            profile_coordinates[[candidate]],
            profile_coordinates[references],
            metric="euclidean",
        )
        result[index] = float(np.min(distances))
    return result


def _sampling_weights(
    frame: pd.DataFrame,
    profile_coordinates: np.ndarray,
    *,
    candidates: Sequence[int],
    real_target_positions: Sequence[int],
    temperature: float,
    municipality_smoothing: float,
) -> np.ndarray:
    """Return X-only target-affinity sampling weights for labeled candidates."""

    candidate_positions = list(map(int, candidates))
    distances = _profile_distance_to_target_cloud(
        profile_coordinates,
        frame,
        candidate_positions=candidate_positions,
        target_positions=real_target_positions,
    )
    scale = max(float(temperature), 1.0e-6)
    base = np.exp(-distances / scale)

    target_municipalities = frame.iloc[list(real_target_positions)]["meta_municipio"].astype(str)
    candidate_municipalities = frame.iloc[candidate_positions]["meta_municipio"].astype(str)
    target_freq = target_municipalities.value_counts(normalize=True)
    candidate_freq = candidate_municipalities.value_counts(normalize=True)

    multipliers = np.ones(len(candidate_positions), dtype=float)
    smooth = max(float(municipality_smoothing), 1.0e-6)
    for index, municipality in enumerate(candidate_municipalities):
        target_share = float(target_freq.get(municipality, 0.0))
        candidate_share = float(candidate_freq.get(municipality, 0.0))
        ratio = (target_share + smooth) / (candidate_share + smooth)
        multipliers[index] = float(np.clip(ratio, 0.25, 4.0))

    weights = base * multipliers
    if not np.isfinite(weights).all() or float(np.sum(weights)) <= 0:
        return np.full(len(candidate_positions), 1.0 / len(candidate_positions))
    return weights / float(np.sum(weights))


def _split_match_quality(
    frame: pd.DataFrame,
    profile_coordinates: np.ndarray,
    *,
    selected_positions: Sequence[int],
    real_target_positions: Sequence[int],
) -> tuple[float, float, float]:
    selected = list(map(int, selected_positions))
    target = list(map(int, real_target_positions))
    profile_mean_distance = float(
        np.linalg.norm(
            np.mean(profile_coordinates[selected], axis=0)
            - np.mean(profile_coordinates[target], axis=0)
        )
    )
    municipality_tv = _total_variation(
        frame.iloc[selected]["meta_municipio"].astype(str),
        frame.iloc[target]["meta_municipio"].astype(str),
    )
    state_tv = _total_variation(
        frame.iloc[selected]["meta_estado"].astype(str),
        frame.iloc[target]["meta_estado"].astype(str),
    )
    objective = profile_mean_distance + municipality_tv + 0.5 * state_tv
    return float(objective), float(municipality_tv), float(profile_mean_distance)


def _max_jaccard(
    selected: set[int],
    previous: Sequence[set[int]],
) -> float:
    if not previous:
        return 0.0
    scores = []
    for other in previous:
        union = selected | other
        scores.append(len(selected & other) / len(union) if union else 0.0)
    return float(max(scores))


def generate_target_matched_splits(
    frame: pd.DataFrame,
    *,
    profile_coordinates: np.ndarray,
    n_pseudo_targets: int,
    repeats: int,
    candidates_per_repeat: int,
    temperature: float,
    municipality_smoothing: float,
    diversity_penalty: float,
    random_state: int,
) -> list[PseudoSplit]:
    """Generate repeated X-only pseudo-target masks matched to the real 59 targets."""

    train_positions = np.flatnonzero(frame[SPLIT_COLUMN].eq(TRAIN_VALUE).to_numpy())
    real_targets = np.flatnonzero(frame[SPLIT_COLUMN].eq(PREDICTION_VALUE).to_numpy())
    target_state_counts = (
        frame.iloc[real_targets]["meta_estado"].astype(str).value_counts().to_dict()
    )
    state_quotas = hamilton_apportion(target_state_counts, int(n_pseudo_targets))

    rng = np.random.default_rng(int(random_state))
    previous: list[set[int]] = []
    splits: list[PseudoSplit] = []

    for repeat in range(1, int(repeats) + 1):
        best_positions: list[int] | None = None
        best_quality: tuple[float, float, float] | None = None
        best_objective = np.inf

        for _ in range(int(candidates_per_repeat)):
            selected: list[int] = []
            for state, quota in state_quotas.items():
                candidates = [
                    int(position)
                    for position in train_positions
                    if str(frame.iloc[int(position)]["meta_estado"]) == state
                ]
                if quota > len(candidates):
                    raise ValueError(
                        f"Requested {quota} pseudo-targets in {state}, only {len(candidates)} "
                        "labeled parcels are available."
                    )
                weights = _sampling_weights(
                    frame,
                    profile_coordinates,
                    candidates=candidates,
                    real_target_positions=real_targets,
                    temperature=temperature,
                    municipality_smoothing=municipality_smoothing,
                )
                chosen = rng.choice(
                    candidates,
                    size=int(quota),
                    replace=False,
                    p=weights,
                )
                selected.extend(map(int, chosen))

            quality = _split_match_quality(
                frame,
                profile_coordinates,
                selected_positions=selected,
                real_target_positions=real_targets,
            )
            selected_set = set(selected)
            objective = quality[0] + float(diversity_penalty) * _max_jaccard(
                selected_set,
                previous,
            )
            if objective < best_objective:
                best_objective = objective
                best_positions = sorted(selected)
                best_quality = quality

        if best_positions is None or best_quality is None:
            raise RuntimeError("Failed to construct a target-matched pseudo split.")
        previous.append(set(best_positions))
        splits.append(
            PseudoSplit(
                split_id=f"target_matched_{repeat:02d}",
                family="target_matched",
                repeat=repeat,
                pseudo_target_positions=tuple(best_positions),
                match_objective=float(best_quality[0]),
                municipality_tv=float(best_quality[1]),
                profile_mean_distance=float(best_quality[2]),
            )
        )
    return splits


def generate_state_random_splits(
    frame: pd.DataFrame,
    *,
    profile_coordinates: np.ndarray,
    n_pseudo_targets: int,
    repeats: int,
    random_state: int,
) -> list[PseudoSplit]:
    """Generate state-matched random pseudo-target masks as a secondary protocol."""

    train_positions = np.flatnonzero(frame[SPLIT_COLUMN].eq(TRAIN_VALUE).to_numpy())
    real_targets = np.flatnonzero(frame[SPLIT_COLUMN].eq(PREDICTION_VALUE).to_numpy())
    target_state_counts = (
        frame.iloc[real_targets]["meta_estado"].astype(str).value_counts().to_dict()
    )
    state_quotas = hamilton_apportion(target_state_counts, int(n_pseudo_targets))
    rng = np.random.default_rng(int(random_state))

    splits: list[PseudoSplit] = []
    for repeat in range(1, int(repeats) + 1):
        selected: list[int] = []
        for state, quota in state_quotas.items():
            candidates = [
                int(position)
                for position in train_positions
                if str(frame.iloc[int(position)]["meta_estado"]) == state
            ]
            chosen = rng.choice(candidates, size=int(quota), replace=False)
            selected.extend(map(int, chosen))
        objective, municipality_tv, profile_distance = _split_match_quality(
            frame,
            profile_coordinates,
            selected_positions=selected,
            real_target_positions=real_targets,
        )
        splits.append(
            PseudoSplit(
                split_id=f"state_random_{repeat:02d}",
                family="state_random",
                repeat=repeat,
                pseudo_target_positions=tuple(sorted(selected)),
                match_objective=objective,
                municipality_tv=municipality_tv,
                profile_mean_distance=profile_distance,
            )
        )
    return splits


def build_legacy_stress_splits(
    frame: pd.DataFrame,
    folds: pd.DataFrame,
    *,
    protocols: Sequence[str],
    profile_coordinates: np.ndarray,
) -> list[PseudoSplit]:
    """Convert frozen Checkpoint 02 folds into transductive stress-test splits."""

    id_to_position = {
        parcel_id: position
        for position, parcel_id in enumerate(frame[ID_COLUMN].astype(str))
    }
    train_ids = set(
        frame.loc[frame[SPLIT_COLUMN].eq(TRAIN_VALUE), ID_COLUMN].astype(str)
    )
    fold_table = folds.loc[folds[ID_COLUMN].astype(str).isin(train_ids)].copy()
    real_targets = np.flatnonzero(frame[SPLIT_COLUMN].eq(PREDICTION_VALUE).to_numpy())

    result: list[PseudoSplit] = []
    for protocol in map(str, protocols):
        if protocol not in fold_table.columns:
            raise KeyError(f"Frozen fold table does not contain {protocol!r}.")
        values = sorted(pd.to_numeric(fold_table[protocol], errors="raise").astype(int).unique())
        for fold in values:
            ids = fold_table.loc[
                pd.to_numeric(fold_table[protocol]).astype(int).eq(int(fold)),
                ID_COLUMN,
            ].astype(str)
            positions = sorted(id_to_position[parcel_id] for parcel_id in ids)
            objective, municipality_tv, profile_distance = _split_match_quality(
                frame,
                profile_coordinates,
                selected_positions=positions,
                real_target_positions=real_targets,
            )
            result.append(
                PseudoSplit(
                    split_id=f"{protocol}_fold{fold}",
                    family=str(protocol),
                    repeat=int(fold),
                    pseudo_target_positions=tuple(positions),
                    match_objective=objective,
                    municipality_tv=municipality_tv,
                    profile_mean_distance=profile_distance,
                )
            )
    return result


def normalized_distance_matrix(
    matrix: np.ndarray,
    *,
    reference_positions: Sequence[int],
) -> tuple[np.ndarray, float]:
    """Scale a distance matrix by the median leave-one-out nearest reference distance."""

    reference = list(map(int, reference_positions))
    nearest: list[float] = []
    for position in reference:
        value = _nearest_distance(matrix, position, reference, exclude_self=True)
        if np.isfinite(value) and value > 0:
            nearest.append(float(value))
    if not nearest:
        raise ValueError("Cannot normalize a distance matrix without positive LOO distances.")
    scale = float(np.median(nearest))
    return np.asarray(matrix, dtype=float) / scale, scale


def temporal_distance_matrix(similarity_matrix: np.ndarray) -> np.ndarray:
    """Convert correlation-like similarity to a nonnegative distance."""

    similarity = np.asarray(similarity_matrix, dtype=float)
    distance = 1.0 - similarity
    distance[np.isfinite(distance)] = np.clip(distance[np.isfinite(distance)], 0.0, 2.0)
    return distance


def mixed_distance_matrix(
    components: Sequence[tuple[np.ndarray, float]],
) -> np.ndarray:
    """Combine normalized X-only distance matrices with nonnegative weights."""

    if not components:
        raise ValueError("At least one distance component is required.")
    weights = np.asarray([float(weight) for _, weight in components], dtype=float)
    if np.any(weights < 0) or float(np.sum(weights)) <= 0:
        raise ValueError("Mixed-distance weights must be nonnegative and not all zero.")
    weights = weights / float(np.sum(weights))

    result = np.zeros_like(np.asarray(components[0][0], dtype=float))
    finite_any = np.zeros_like(result, dtype=bool)
    for (matrix, _), weight in zip(components, weights, strict=True):
        values = np.asarray(matrix, dtype=float)
        finite = np.isfinite(values)
        result[finite] += weight * values[finite]
        finite_any |= finite
    result[~finite_any] = np.nan
    np.fill_diagonal(result, 0.0)
    return result


def _distance_support(value: float, reference: Sequence[float]) -> float:
    array = np.asarray(reference, dtype=float)
    array = array[np.isfinite(array)]
    if not np.isfinite(value) or len(array) == 0:
        return np.nan
    return float(np.mean(array >= value))


def _similarity_support(value: float, reference: Sequence[float]) -> float:
    array = np.asarray(reference, dtype=float)
    array = array[np.isfinite(array)]
    if not np.isfinite(value) or len(array) == 0:
        return np.nan
    return float(np.mean(array <= value))


def split_x_support(
    frame: pd.DataFrame,
    *,
    query_positions: Sequence[int],
    observed_positions: Sequence[int],
    geographic_distances: np.ndarray,
    agronomic_distances: np.ndarray,
    temporal_similarities: np.ndarray,
    adversarial_scores: pd.DataFrame,
    high_threshold: float = 0.67,
    low_threshold: float = 0.33,
) -> pd.DataFrame:
    """Compute dynamic X-only support against the labels visible in one split."""

    observed = list(map(int, observed_positions))
    queries = list(map(int, query_positions))
    ids = frame[ID_COLUMN].astype(str).to_numpy()
    adversarial = adversarial_scores.set_index(ID_COLUMN)[
        "adversarial_full_target_probability"
    ].astype(float)

    reference_geo = [
        _nearest_distance(geographic_distances, position, observed, exclude_self=True)
        for position in observed
    ]
    reference_agro = [
        _nearest_distance(agronomic_distances, position, observed, exclude_self=True)
        for position in observed
    ]
    reference_temporal = [
        _best_similarity(temporal_similarities, position, observed, exclude_self=True)
        for position in observed
    ]
    reference_adversarial = adversarial.reindex(ids[observed]).to_numpy(float)

    rows: list[dict[str, Any]] = []
    for position in queries:
        geo = _nearest_distance(
            geographic_distances,
            position,
            observed,
            exclude_self=False,
        )
        agro = _nearest_distance(
            agronomic_distances,
            position,
            observed,
            exclude_self=False,
        )
        temporal = _best_similarity(
            temporal_similarities,
            position,
            observed,
            exclude_self=False,
        )
        adv = float(adversarial.loc[ids[position]])
        components = {
            "x_geo_support": _distance_support(geo, reference_geo),
            "x_agro_support": _distance_support(agro, reference_agro),
            "x_temporal_support": _similarity_support(temporal, reference_temporal),
            "x_adversarial_support": _similarity_support(-adv, -reference_adversarial),
        }
        finite = [value for value in components.values() if np.isfinite(value)]
        score = float(np.mean(finite)) if finite else np.nan
        if np.isfinite(score) and score >= high_threshold:
            tier = "high"
        elif np.isfinite(score) and score < low_threshold:
            tier = "low"
        else:
            tier = "mid"
        rows.append(
            {
                "position": position,
                ID_COLUMN: ids[position],
                "nearest_geo_km": geo,
                "nearest_agro_distance": agro,
                "best_temporal_similarity": temporal,
                "adversarial_target_probability": adv,
                **components,
                "x_support_score": score,
                "x_support_tier": tier,
            }
        )
    return pd.DataFrame(rows)


def predict_global_mean(
    y: np.ndarray,
    observed_positions: Sequence[int],
    query_positions: Sequence[int],
) -> np.ndarray:
    """Predict the visible-label global mean."""

    value = float(np.mean(y[list(map(int, observed_positions))]))
    return np.full(len(query_positions), value, dtype=float)


def predict_state_mean(
    frame: pd.DataFrame,
    y: np.ndarray,
    observed_positions: Sequence[int],
    query_positions: Sequence[int],
) -> np.ndarray:
    """Predict state means with global fallback."""

    observed = list(map(int, observed_positions))
    global_mean = float(np.mean(y[observed]))
    result: list[float] = []
    for query in map(int, query_positions):
        state = str(frame.iloc[query]["meta_estado"])
        local = [
            position
            for position in observed
            if str(frame.iloc[position]["meta_estado"]) == state
        ]
        result.append(float(np.mean(y[local])) if local else global_mean)
    return np.asarray(result, dtype=float)


def predict_municipality_shrinkage(
    frame: pd.DataFrame,
    y: np.ndarray,
    observed_positions: Sequence[int],
    query_positions: Sequence[int],
    *,
    prior_strength: float,
) -> np.ndarray:
    """Shrink municipality means toward the visible same-state mean."""

    observed = list(map(int, observed_positions))
    global_mean = float(np.mean(y[observed]))
    result: list[float] = []

    for query in map(int, query_positions):
        state = str(frame.iloc[query]["meta_estado"])
        municipality = str(frame.iloc[query]["meta_municipio"])
        state_positions = [
            position
            for position in observed
            if str(frame.iloc[position]["meta_estado"]) == state
        ]
        state_mean = (
            float(np.mean(y[state_positions])) if state_positions else global_mean
        )
        municipality_positions = [
            position
            for position in state_positions
            if str(frame.iloc[position]["meta_municipio"]) == municipality
        ]
        if not municipality_positions:
            result.append(state_mean)
            continue
        n_local = len(municipality_positions)
        municipality_mean = float(np.mean(y[municipality_positions]))
        strength = max(float(prior_strength), 0.0)
        prediction = (
            n_local * municipality_mean + strength * state_mean
        ) / (n_local + strength)
        result.append(float(prediction))
    return np.asarray(result, dtype=float)


def predict_knn(
    distance_matrix: np.ndarray,
    y: np.ndarray,
    observed_positions: Sequence[int],
    query_positions: Sequence[int],
    *,
    k: int,
    power: float = 1.0,
) -> np.ndarray:
    """Inverse-distance weighted k-nearest-neighbor regression."""

    observed = list(map(int, observed_positions))
    result: list[float] = []
    for query in map(int, query_positions):
        distances = np.asarray(distance_matrix[query, observed], dtype=float)
        valid = np.isfinite(distances)
        candidate_positions = np.asarray(observed, dtype=int)[valid]
        candidate_distances = distances[valid]
        if len(candidate_positions) == 0:
            result.append(float(np.mean(y[observed])))
            continue
        order = np.argsort(candidate_distances)
        take = order[: max(1, min(int(k), len(order)))]
        selected_positions = candidate_positions[take]
        selected_distances = candidate_distances[take]
        exact = selected_distances <= 1.0e-12
        if exact.any():
            result.append(float(np.mean(y[selected_positions[exact]])))
            continue
        weights = 1.0 / np.power(selected_distances + 1.0e-8, float(power))
        prediction = np.average(y[selected_positions], weights=weights)
        result.append(float(prediction))
    return np.asarray(result, dtype=float)


def transductive_standardize(values: np.ndarray) -> np.ndarray:
    """Standardize an X-only representation using all visible covariate rows."""

    return StandardScaler().fit_transform(np.asarray(values, dtype=float))


def predict_global_ridge(
    x: np.ndarray,
    y: np.ndarray,
    observed_positions: Sequence[int],
    query_positions: Sequence[int],
    *,
    alpha: float,
) -> np.ndarray:
    """Fit Ridge on visible labels in an X-only transductive representation."""

    observed = list(map(int, observed_positions))
    query = list(map(int, query_positions))
    model = Ridge(alpha=float(alpha))
    model.fit(x[observed], y[observed])
    return np.asarray(model.predict(x[query]), dtype=float).reshape(-1)


def predict_global_pls(
    x: np.ndarray,
    y: np.ndarray,
    observed_positions: Sequence[int],
    query_positions: Sequence[int],
    *,
    n_components: int,
) -> np.ndarray:
    """Fit PLS using only visible y after X-only preprocessing."""

    observed = list(map(int, observed_positions))
    query = list(map(int, query_positions))
    max_components = max(1, min(len(observed) - 1, x.shape[1]))
    components = max(1, min(int(n_components), max_components))
    model = PLSRegression(n_components=components, scale=False, max_iter=1_000)
    model.fit(x[observed], y[observed])
    return np.asarray(model.predict(x[query]), dtype=float).reshape(-1)


def predict_local_ridge(
    x: np.ndarray,
    distance_matrix: np.ndarray,
    y: np.ndarray,
    observed_positions: Sequence[int],
    query_positions: Sequence[int],
    *,
    k: int,
    alpha: float,
) -> np.ndarray:
    """Fit one distance-weighted Ridge model per pseudo-target."""

    observed = np.asarray(list(map(int, observed_positions)), dtype=int)
    result: list[float] = []
    for query in map(int, query_positions):
        distances = np.asarray(distance_matrix[query, observed], dtype=float)
        valid = np.isfinite(distances)
        candidates = observed[valid]
        candidate_distances = distances[valid]
        if len(candidates) < 3:
            result.append(float(np.mean(y[observed])))
            continue
        order = np.argsort(candidate_distances)
        take = order[: max(3, min(int(k), len(order)))]
        local_positions = candidates[take]
        local_distances = candidate_distances[take]
        weights = 1.0 / (local_distances + 1.0e-6)
        model = Ridge(alpha=float(alpha))
        model.fit(
            x[local_positions],
            y[local_positions],
            sample_weight=weights,
        )
        result.append(float(model.predict(x[[query]])[0]))
    return np.asarray(result, dtype=float)


def build_knn_graph(
    distance_matrix: np.ndarray,
    *,
    k: int,
) -> np.ndarray:
    """Build a symmetric RBF-weighted kNN graph from X-only distances."""

    distances = np.asarray(distance_matrix, dtype=float)
    n = len(distances)
    adjacency = np.zeros((n, n), dtype=float)
    edge_distances: list[float] = []

    for row in range(n):
        values = distances[row].copy()
        values[row] = np.inf
        finite = np.isfinite(values)
        order = np.argsort(np.where(finite, values, np.inf))
        neighbours = [
            int(column)
            for column in order
            if np.isfinite(values[column])
        ][: max(1, min(int(k), n - 1))]
        for column in neighbours:
            edge_distances.append(float(values[column]))

    positive = [value for value in edge_distances if value > 0 and np.isfinite(value)]
    sigma = float(np.median(positive)) if positive else 1.0
    sigma = max(sigma, 1.0e-8)

    for row in range(n):
        values = distances[row].copy()
        values[row] = np.inf
        order = np.argsort(np.where(np.isfinite(values), values, np.inf))
        neighbours = [
            int(column)
            for column in order
            if np.isfinite(values[column])
        ][: max(1, min(int(k), n - 1))]
        for column in neighbours:
            weight = float(np.exp(-0.5 * (values[column] / sigma) ** 2))
            adjacency[row, column] = max(adjacency[row, column], weight)
            adjacency[column, row] = max(adjacency[column, row], weight)
    return adjacency


def predict_graph_laplacian(
    adjacency: np.ndarray,
    y: np.ndarray,
    observed_positions: Sequence[int],
    query_positions: Sequence[int],
    *,
    regularization: float,
) -> np.ndarray:
    """Graph-Laplacian regression with global-mean fallback on unlabeled nodes."""

    observed = np.asarray(list(map(int, observed_positions)), dtype=int)
    query = np.asarray(list(map(int, query_positions)), dtype=int)
    n = adjacency.shape[0]
    degree = np.diag(np.sum(adjacency, axis=1))
    laplacian = degree - adjacency

    mask = np.zeros(n, dtype=float)
    mask[observed] = 1.0
    mean_y = float(np.mean(y[observed]))
    centered = np.zeros(n, dtype=float)
    centered[observed] = y[observed] - mean_y

    system = np.diag(mask) + float(regularization) * laplacian
    system += np.eye(n) * 1.0e-8
    right = mask * centered
    solution = np.linalg.solve(system, right)
    return np.asarray(mean_y + solution[query], dtype=float)


def regression_metrics(observed: Sequence[float], predicted: Sequence[float]) -> dict[str, float]:
    """Return RMSE, MAE and R2."""

    y_true = np.asarray(observed, dtype=float)
    y_pred = np.asarray(predicted, dtype=float)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def summarize_method_results(split_metrics: pd.DataFrame) -> pd.DataFrame:
    """Aggregate repeated split-level metrics by family and method."""

    grouped = (
        split_metrics.groupby(["family", "method"], as_index=False)
        .agg(
            n_splits=("split_id", "nunique"),
            rmse_mean=("rmse", "mean"),
            rmse_median=("rmse", "median"),
            rmse_std=("rmse", "std"),
            rmse_worst=("rmse", "max"),
            mae_mean=("mae", "mean"),
            r2_mean=("r2", "mean"),
        )
        .sort_values(["family", "rmse_mean", "method"])
        .reset_index(drop=True)
    )
    return grouped


def summarize_support_tier_results(predictions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate pseudo-target errors by X-only support tier and method."""

    frame = predictions.copy()
    frame["squared_error"] = np.square(frame["observed"] - frame["predicted"])
    frame["absolute_error"] = np.abs(frame["observed"] - frame["predicted"])
    rows: list[dict[str, Any]] = []

    for keys, group in frame.groupby(["family", "method", "x_support_tier"]):
        family, method, tier = keys
        rows.append(
            {
                "family": family,
                "method": method,
                "x_support_tier": tier,
                "n_predictions": int(len(group)),
                "rmse": float(np.sqrt(group["squared_error"].mean())),
                "mae": float(group["absolute_error"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["family", "x_support_tier", "rmse", "method"]
    ).reset_index(drop=True)
