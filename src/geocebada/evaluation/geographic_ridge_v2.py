"""Fair exploratory ablation of geographically weighted PCA20 local Ridge.

Every arm uses the SAME C4 transductive PCA20, normalized haversine distance,
24 nearest labeled neighbors, and p=1. Only the local objective differs.
All supervised tuning is nested inside each frozen 97/41 split.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

from geocebada.evaluation.checkpoint04d1 import predict_weighted_local_ridge
from geocebada.evaluation.checkpoint08_local import ID
from geocebada.evaluation.checkpoint10_pca import FROZEN, PcaData

FROZEN_NAME = "GeoPCA20_Ridge_Frozen"
RAW = "GeoPCA20_Ridge_InnerCV"
NORMALIZED = "GeoPCA20_Ridge_NormalizedWeights"
ADAPTIVE = "GeoPCA20_Ridge_EffectiveNAlpha"
SHRINK = "GeoPCA20_Ridge_EffectiveNAlphaShrink"
ROBUST = "GeoPCA20_Ridge_RobustIRLS"
TUNED_ARMS = (RAW, NORMALIZED, ADAPTIVE, SHRINK, ROBUST)
ALL_ARMS = (FROZEN_NAME, *TUNED_ARMS)
ALPHAS = (0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0)
N_REPEATS = 2
N_FOLDS = 5
SEED = 20261016
K = 24
POWER = 1.0
HUBER_EPSILON = 1.35
HUBER_UPDATES = 3


@dataclass(frozen=True)
class _Weights:
    positions: np.ndarray
    raw: np.ndarray
    normalized: np.ndarray
    effective_n: float


def _weights(distances: np.ndarray, observed: np.ndarray, position: int) -> _Weights:
    """Neighbor geometry is independent of all y values."""
    distance = np.asarray(distances[position, observed], dtype=float)
    good = np.isfinite(distance) & (distance >= 0)
    if int(good.sum()) < K:
        raise ValueError("Fewer than 24 finite geographic training neighbors.")
    candidates = observed[good]
    local_distance = distance[good]
    order = np.argsort(local_distance, kind="stable")[:K]
    positions = candidates[order]
    d = local_distance[order]
    raw = 1.0 / np.power(d + 1.0e-6, POWER)
    total = float(raw.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Nonfinite or zero geographic weights.")
    normalized = raw * (K / total)
    effective_n = total**2 / float(np.dot(raw, raw))
    if not (1.0 <= effective_n <= K + 1e-8):
        raise RuntimeError("Impossible effective-neighbor count.")
    return _Weights(positions, raw, normalized, effective_n)


def _fit_ridge(x: np.ndarray, y: np.ndarray, weights: np.ndarray, alpha: float) -> Ridge:
    estimator = Ridge(alpha=float(alpha))
    estimator.fit(x, y, sample_weight=weights)
    return estimator


def _one_new_prediction(
    x: np.ndarray, y: np.ndarray, weights: _Weights, query: int,
    arm: str, alpha: float,
) -> float:
    fit_x = x[weights.positions]
    fit_y = y[weights.positions]
    local_w = weights.normalized.copy()
    effective = weights.effective_n
    local_alpha = float(alpha)
    if arm in (ADAPTIVE, SHRINK):
        local_alpha *= K / effective
    if arm == ROBUST:
        # Three fixed Huber-like IRLS updates. Re-normalizing the effective
        # weights maintains a comparable Ridge penalty throughout the updates.
        estimator = _fit_ridge(fit_x, fit_y, local_w, local_alpha)
        for _ in range(HUBER_UPDATES):
            residual = fit_y - estimator.predict(fit_x)
            centered = residual - np.median(residual)
            mad = 1.4826 * float(np.median(np.abs(centered)))
            scale = max(mad, .15 * float(np.std(fit_y)), 1.0e-6)
            influence = np.minimum(
                1.0, HUBER_EPSILON * scale / np.maximum(np.abs(residual), 1.0e-12)
            )
            robust_w = weights.normalized * influence
            robust_w *= K / float(robust_w.sum())
            estimator = _fit_ridge(fit_x, fit_y, robust_w, local_alpha)
    else:
        estimator = _fit_ridge(fit_x, fit_y, local_w, local_alpha)
    prediction = float(estimator.predict(x[[query]])[0])
    if arm == SHRINK:
        # When distance weights are concentrated on very few parcels, favor
        # their weighted mean rather than extrapolating the 20-PC coefficients.
        reliability = min(1.0, effective / K)
        local_mean = float(np.average(fit_y, weights=local_w))
        prediction = reliability * prediction + (1.0 - reliability) * local_mean
    if not np.isfinite(prediction):
        raise ValueError("A local prediction is not finite.")
    return prediction


def _predict(
    data: PcaData, fit_ids: list[str], query_ids: list[str],
    arm: str, alpha: float,
) -> np.ndarray:
    if arm not in ALL_ARMS:
        raise ValueError(f"Unknown Ridge arm {arm}.")
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("Ridge alpha must be finite and positive.")
    if set(fit_ids) & set(query_ids):
        raise ValueError("Training and query IDs overlap.")
    if len(fit_ids) < K:
        raise ValueError("Too few labeled training parcels.")
    observed = np.array([data.index[pid] for pid in fit_ids], dtype=int)
    query = np.array([data.index[pid] for pid in query_ids], dtype=int)
    yy = np.full(len(data.ids), np.nan, dtype=float)
    yy[observed] = data.y.loc[fit_ids].to_numpy(float)
    if not np.isfinite(yy[observed]).all():
        raise ValueError("Nonfinite training yields.")
    x = np.asarray(data.x[20], dtype=float)
    if arm in (FROZEN_NAME, RAW):
        # Exact original implementation, not a rewritten approximation.
        return predict_weighted_local_ridge(
            x, data.geographic, yy, observed, query,
            k=K, alpha=float(alpha), distance_power=POWER,
        )
    return np.array([
        _one_new_prediction(x, yy, _weights(data.geographic, observed, int(q)),
                            int(q), arm, alpha)
        for q in query
    ], dtype=float)


def _inner_folds(ids: list[str], split_id: str) -> list[tuple[list[str], list[str]]]:
    """Identical 2x5-fold partitions for every tunable family and alpha."""
    number = int(split_id.rsplit("_", 1)[-1])
    result = []
    for repeat in range(N_REPEATS):
        cv = KFold(
            n_splits=N_FOLDS, shuffle=True,
            random_state=SEED + 100 * number + repeat,
        )
        for train_idx, valid_idx in cv.split(ids):
            result.append((
                [ids[i] for i in train_idx],
                [ids[i] for i in valid_idx],
            ))
    return result


def _select_alphas(
    data: PcaData, train_ids: list[str], split_id: str,
    *, arms: tuple[str, ...] = TUNED_ARMS,
    alphas: tuple[float, ...] = ALPHAS,
) -> tuple[pd.DataFrame, dict[str, float]]:
    folds = _inner_folds(train_ids, split_id)
    records = []
    for arm in arms:
        for alpha in alphas:
            actual_parts, pred_parts = [], []
            for fit, held in folds:
                actual_parts.append(data.y.loc[held].to_numpy(float))
                pred_parts.append(_predict(data, fit, held, arm, alpha))
            actual = np.concatenate(actual_parts)
            prediction = np.concatenate(pred_parts)
            records.append({
                "split_id": split_id, "model": arm, "alpha": float(alpha),
                "inner_rmse": float(np.sqrt(np.mean((actual - prediction) ** 2))),
                "n_inner_predictions": len(actual),
                "inner_folds": len(folds),
            })
    table = pd.DataFrame(records)
    selected = {}
    for arm in arms:
        ordered = table.loc[table.model.eq(arm)].sort_values(
            ["inner_rmse", "alpha"], kind="stable",
        )
        selected[arm] = float(ordered.iloc[0]["alpha"])
    return table, selected


def _evaluate_split(
    data: PcaData, split_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, pd.DataFrame]:
    train, held = data.folds[split_id]
    if len(train) != 97 or len(held) != 41:
        raise ValueError("Expected frozen 97/41 split.")
    cv, selected = _select_alphas(data, train, split_id)
    truth = data.y.loc[held].to_numpy(float)
    result = []
    params = {FROZEN_NAME: float(FROZEN["alpha"]), **selected}
    for arm in ALL_ARMS:
        predicted = _predict(data, train, held, arm, params[arm])
        for pid, y, value in zip(held, truth, predicted, strict=True):
            result.append({
                "split_id": split_id, ID: pid, "model": arm,
                "observed": float(y), "predicted": float(value),
            })
    observed = np.array([data.index[pid] for pid in train], dtype=int)
    diagnostics = []
    for pid in held:
        w = _weights(data.geographic, observed, data.index[pid])
        diagnostics.append({
            "split_id": split_id, ID: pid, "effective_neighbors": w.effective_n,
            "distance_weight_concentration": float(w.raw.max() / w.raw.sum()),
            "unnormalized_weight_sum": float(w.raw.sum()),
        })
    return (
        pd.DataFrame(result), cv,
        {
            "split_id": split_id,
            "outer_training": 97, "outer_heldout": 41,
            "inner_repetitions": N_REPEATS, "inner_folds_per_repeat": N_FOLDS,
            "alpha_grid": list(ALPHAS), "selected_alpha": params,
            "selection_used_outer_y": False,
        },
        pd.DataFrame(diagnostics),
    )


def _summarize(
    predictions: pd.DataFrame, *, bootstraps: int = 1000,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Reject unpaired/partial outputs; group bootstrap by unique parcel."""
    necessary = {"split_id", ID, "model", "observed", "predicted"}
    if not necessary.issubset(predictions):
        raise ValueError("Missing prediction columns.")
    if predictions.duplicated(["split_id", ID, "model"]).any():
        raise ValueError("Duplicate paired predictions.")
    pairs = predictions.groupby(["split_id", ID])
    if (predictions["split_id"].nunique() != 16 or len(pairs) != 656
            or not pairs["model"].nunique().eq(len(ALL_ARMS)).all()
            or set(predictions["model"]) != set(ALL_ARMS)):
        raise ValueError("Incomplete 16-by-41 paired comparison.")
    if not pairs["observed"].nunique().eq(1).all():
        raise ValueError("Different ground truth for the same held-out parcel.")
    if not np.isfinite(predictions[["observed", "predicted"]].to_numpy(float)).all():
        raise ValueError("Nonfinite predictions or yields.")
    rows = []
    for model, frame in predictions.groupby("model"):
        sq = (frame["predicted"] - frame["observed"]) ** 2
        rmses = frame.groupby("split_id").apply(
            lambda g: float(np.sqrt(np.mean((g["observed"] - g["predicted"]) ** 2))),
            include_groups=False,
        )
        rows.append({
            "model": model, "n_splits": 16, "repeated_predictions": len(frame),
            "unique_parcels": frame[ID].nunique(),
            "rmse_pooled": float(np.sqrt(sq.mean())),
            "r2_pooled": float(r2_score(frame["observed"], frame["predicted"])),
            "rmse_mean_split": float(rmses.mean()),
            "rmse_worst_split": float(rmses.max()),
        })
    summary = pd.DataFrame(rows).sort_values("rmse_pooled").reset_index(drop=True)
    p = predictions.pivot(index=["split_id", ID], columns="model", values="predicted")
    y = predictions.drop_duplicates(["split_id", ID]).set_index(
        ["split_id", ID]
    )["observed"]
    sq = p.sub(y, axis=0).pow(2).sort_index()
    groups = {pid: block for pid, block in sq.groupby(level=ID)}
    parcels = sorted(groups)
    rng = np.random.default_rng(SEED)
    deltas = []
    for arm in TUNED_ARMS:
        difference = sq[arm] - sq[FROZEN_NAME]
        # Each sampled parcel contributes ALL of its repeated split scores.
        per_parcel = [
            (groups[pid][arm] - groups[pid][FROZEN_NAME]).to_numpy(float)
            for pid in parcels
        ]
        sampled = []
        for _ in range(bootstraps):
            chosen = rng.integers(0, len(parcels), size=len(parcels))
            sampled.append(float(np.concatenate([
                per_parcel[j] for j in chosen
            ]).mean()))
        deltas.append({
            "model": arm, "reference": FROZEN_NAME,
            "delta_mse": float(difference.mean()),
            "ci95_low": float(np.quantile(sampled, .025)),
            "ci95_high": float(np.quantile(sampled, .975)),
            "bootstrap_unit": "unique_parcel",
        })
    errors = predictions.assign(
        abs_error=lambda t: (t.predicted - t.observed).abs(),
    ).groupby([ID, "model"], as_index=False).agg(
        n_repetitions=("abs_error", "size"),
        mean_abs_error=("abs_error", "mean"),
    )
    return summary, pd.DataFrame(deltas), errors
