"""Small-sample phenology-conditioned Ridge plus spatial Matérn residual GP.

This is *not* Local04D: its environmental mean uses a fixed low-dimensional
2025 weather/vegetation/moisture representation and its residual has an
explicit spatial covariance. X-only PCA20 comes from the frozen C4 baseline.
All y-dependent preprocessing, Ridge and residual-GP fits are fold-local.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from geocebada.evaluation.checkpoint10_pca import PcaData, predict as predict_incumbent

ID = "ID_POLIGONO"
BASELINE = "GeoPCA20_GeographicWeightedRidge"
CANDIDATE = "PhenologyRidge_SpatialMaternGP"
SEED = 20261018


@dataclass(frozen=True)
class SpatialGPConfig:
    """Fixed regularization and residual covariance of one GP candidate."""

    alpha: float
    length_km: float
    spatial_share: float
    robust_nugget: bool = False


@dataclass
class SpatialGPData:
    """Aligned all-197 transductive X, physical features and spatial distances."""

    baseline: PcaData
    physical: pd.DataFrame
    spatial_km: np.ndarray

    def validate(self) -> None:
        """Require exact row alignment and finite geographic support."""
        n = len(self.baseline.ids)
        if n != 197 or len(self.baseline.y) != 138:
            raise ValueError("Expected 197 X rows and 138 observed labels.")
        if self.physical[ID].astype(str).tolist() != self.baseline.ids:
            raise ValueError("Physical features not aligned with C4 parcel IDs.")
        if self.physical.columns.duplicated().any():
            raise ValueError("Duplicated physical predictor.")
        if 20 not in self.baseline.x:
            raise ValueError("Frozen PCA20 representation missing.")
        if self.spatial_km.shape != (n, n):
            raise ValueError("Spatial distances must cover all parcels.")
        if not np.isfinite(self.spatial_km).all() or np.min(self.spatial_km) < 0:
            raise ValueError("Invalid spatial distance matrix.")
        if not np.allclose(self.spatial_km, self.spatial_km.T, atol=1e-8):
            raise ValueError("Spatial distance matrix must be symmetric.")
        if not np.allclose(np.diag(self.spatial_km), 0, atol=1e-6):
            raise ValueError("Spatial distance diagonal must be zero.")
        if any("RENDIMIENTO" in c.upper() or "SIAP_2025" in c.upper()
               for c in self.physical.columns):
            raise ValueError("Outcome-like column in physical X.")
        x = self.physical.drop(columns=ID).apply(pd.to_numeric, errors="raise")
        if np.isinf(x.to_numpy(dtype=float)).any():
            raise ValueError("Infinite physical predictor.")

    def raw_x(self) -> np.ndarray:
        """Return incumbent PCA20 concatenated with interpretable environment."""
        return np.column_stack([
            self.baseline.x[20],
            self.physical.drop(columns=ID).to_numpy(dtype=float),
        ])


def candidate_grid(*, smoke: bool = False) -> tuple[SpatialGPConfig, ...]:
    """Small predeclared, inner-CV-only grid; include Ridge-only ablations."""
    alphas = (30.0,) if smoke else (10.0, 100.0)
    lengths = (3.0,) if smoke else (0.75, 3.0, 12.0)
    shares = (0.0, 0.75) if smoke else (0.0, 0.5, 0.9)
    robust = (False,) if smoke else (False, True)
    candidates = []
    for a, ell, share, flag in product(alphas, lengths, shares, robust):
        if share == 0 and (ell != lengths[0] or flag):
            continue  # spatial length/robustness have no effect without a GP
        candidates.append(SpatialGPConfig(a, ell, share, flag))
    return tuple(candidates)


def _matern32(distance: np.ndarray, length_km: float) -> np.ndarray:
    if length_km <= 0:
        raise ValueError("GP length scale must be positive.")
    z = np.sqrt(3.0) * distance / length_km
    return (1.0 + z) * np.exp(-z)


def _nugget_inflation(residual: np.ndarray, robust: bool) -> np.ndarray:
    if not robust:
        return np.ones(len(residual))
    # Huber-like heteroscedastic diagonal: downweight unusually large
    # in-sample errors without removing them; *not* robust GP inference.
    med = float(np.median(residual))
    scale = max(1.4826 * float(np.median(np.abs(residual - med))), 0.05)
    z = np.abs(residual - med) / (2.0 * scale)
    return np.minimum(np.maximum(1.0, z) ** 2, 16.0)


def predict_spatial_gp(
    data: SpatialGPData,
    train_ids: list[str],
    query_ids: list[str],
    config: SpatialGPConfig,
) -> np.ndarray:
    """Predict query yields using strictly train-only y and fold-local scaling."""
    if (not train_ids or len(set(train_ids)) != len(train_ids)
            or len(set(query_ids)) != len(query_ids)
            or set(train_ids) & set(query_ids)):
        raise ValueError("Training and query ID sets must be unique and disjoint.")
    if not set(train_ids).issubset(data.baseline.y.index):
        raise ValueError("GP can only fit visible training labels.")
    if not set(query_ids).issubset(data.baseline.index):
        raise ValueError("Query parcel outside fixed transductive universe.")
    if not (config.alpha > 0 and config.length_km > 0
            and 0 <= config.spatial_share < 1):
        raise ValueError("Invalid GP regularization configuration.")
    pos = data.baseline.index
    tr = np.array([pos[pid] for pid in train_ids], dtype=int)
    te = np.array([pos[pid] for pid in query_ids], dtype=int)
    x = data.raw_x()
    # Inner-heldout and actual-target X are NEVER used for fitting these.
    imputer = SimpleImputer(strategy="median", keep_empty_features=True, add_indicator=True)
    xtr = imputer.fit_transform(x[tr])
    xte = imputer.transform(x[te])
    scaler = StandardScaler().fit(xtr)
    xtr = scaler.transform(xtr)
    xte = scaler.transform(xte)
    y = data.baseline.y.loc[train_ids].to_numpy(float)
    trend = Ridge(alpha=config.alpha).fit(xtr, y)
    prediction = trend.predict(xte)
    if config.spatial_share <= 0:
        return np.asarray(prediction, dtype=float)

    residual = y - trend.predict(xtr)
    share = config.spatial_share
    k = share * _matern32(data.spatial_km[np.ix_(tr, tr)], config.length_km)
    k[np.diag_indices_from(k)] += (
        (1.0 - share) * _nugget_inflation(residual, config.robust_nugget) + 1e-5
    )
    cross = share * _matern32(data.spatial_km[np.ix_(te, tr)], config.length_km)
    factor = cho_factor(k, lower=True, check_finite=True)
    correction = cross @ cho_solve(factor, residual, check_finite=True)
    return np.asarray(prediction + correction, dtype=float)


def inner_select(
    data: SpatialGPData,
    train_ids: list[str],
    *,
    seed: int = SEED,
    smoke: bool = False,
) -> tuple[SpatialGPConfig, pd.DataFrame]:
    """Select exactly one configuration without seeing any outer query y."""
    if len(train_ids) < 15:
        raise ValueError("Need enough labeled parcels for threefold selection.")
    kf = KFold(n_splits=2 if smoke else 3, shuffle=True, random_state=seed)
    pairs = [
        ([train_ids[j] for j in left], [train_ids[j] for j in right])
        for left, right in kf.split(train_ids)
    ]
    trials = []
    for config in candidate_grid(smoke=smoke):
        errors = []
        for fit_ids, validation_ids in pairs:
            forecast = predict_spatial_gp(data, fit_ids, validation_ids, config)
            y_validation = data.baseline.y.loc[validation_ids].to_numpy(float)
            errors.extend(np.square(forecast - y_validation).tolist())
        trials.append({
            "alpha": config.alpha,
            "length_km": config.length_km,
            "spatial_share": config.spatial_share,
            "robust_nugget": config.robust_nugget,
            "inner_rmse": float(np.sqrt(np.mean(errors))),
        })
    scores = pd.DataFrame(trials).sort_values(
        ["inner_rmse", "spatial_share", "alpha", "length_km", "robust_nugget"],
        kind="stable",
    ).reset_index(drop=True)
    best = scores.iloc[0]
    selected = SpatialGPConfig(
        alpha=float(best["alpha"]),
        length_km=float(best["length_km"]),
        spatial_share=float(best["spatial_share"]),
        robust_nugget=bool(best["robust_nugget"]),
    )
    return selected, scores


def evaluate_split(
    data: SpatialGPData,
    split_id: str,
    *,
    smoke: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Pair an inner-selected GP with the unchanged frozen Local04D baseline."""
    train, hold = data.baseline.folds[split_id]
    if set(train) & set(hold) or len(train) != 97 or len(hold) != 41:
        raise ValueError("Invalid matched pseudo-competition split.")
    selected, scores = inner_select(data, train, smoke=smoke)
    candidate = predict_spatial_gp(data, train, hold, selected)
    frozen = predict_incumbent(
        data.baseline, train, hold,
        components=20, k=24, alpha=30.0, power=1.0,
    )
    # Reveal pseudo-target labels only after the two procedures predict.
    observed = data.baseline.y.loc[hold].to_numpy(float)
    if not np.isfinite(candidate).all() or not np.isfinite(frozen).all():
        raise ValueError("Nonfinite matched predictions.")
    records = [
        {
            "split_id": split_id, ID: pid, "model": name,
            "observed": float(y), "predicted": float(p),
        }
        for name, preds in ((BASELINE, frozen), (CANDIDATE, candidate))
        for pid, y, p in zip(hold, observed, preds, strict=True)
    ]
    meta = {
        "split_id": split_id, "outer_train": len(train),
        "outer_holdout": len(hold), "selected": vars(selected),
        "selected_inner_rmse": float(scores.loc[0, "inner_rmse"]),
        "pseudo_target_y_used_for_selection": False,
    }
    return pd.DataFrame(records), scores, meta


def summarize_matched(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize paired repeated pseudo-target outcomes without IID claims."""
    needed = {"split_id", ID, "model", "observed", "predicted"}
    if not needed.issubset(predictions):
        raise ValueError("Missing paired evaluation columns.")
    if predictions.duplicated(["split_id", ID, "model"]).any():
        raise ValueError("Repeated identical split/parcel/model row.")
    models = set(predictions["model"])
    if models != {BASELINE, CANDIDATE}:
        raise ValueError("Require candidate and frozen control on same folds.")
    a = predictions.pivot(index=["split_id", ID], columns="model", values="predicted")
    truth = predictions.groupby(["split_id", ID])["observed"].first()
    if a.isna().any().any() or not a.index.equals(truth.index):
        raise ValueError("Unpaired or missing pseudo-targets.")
    per_pair = pd.DataFrame({
        "mse_candidate": np.square(a[CANDIDATE] - truth),
        "mse_baseline": np.square(a[BASELINE] - truth),
    }).reset_index()
    per_pair["delta_mse"] = per_pair["mse_candidate"] - per_pair["mse_baseline"]
    summary = []
    for name in (BASELINE, CANDIDATE):
        frame = predictions[predictions["model"].eq(name)]
        err = frame["predicted"].to_numpy(float) - frame["observed"].to_numpy(float)
        per_split = frame.assign(sq=np.square(err)).groupby("split_id")["sq"].mean()
        summary.append({
            "model": name, "n_predictions": len(frame),
            "n_unique_parcels": frame[ID].nunique(),
            "n_splits": frame["split_id"].nunique(),
            "pooled_rmse": float(np.sqrt(mean_squared_error(
                frame["observed"], frame["predicted"]
            ))),
            "mean_split_rmse": float(np.mean(np.sqrt(per_split))),
        })
    return pd.DataFrame(summary), per_pair
