"""Paired comparison of spatially aggregated climate features without label leakage.

All models use identical geographic neighbor weights, Ridge alpha and PCA20
incumbent features. New climate blocks derive exclusively from observed X
for the fixed 197 parcel geometries; no pseudo-target y enters feature creation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from geocebada.evaluation.checkpoint04d1 import predict_weighted_local_ridge
from geocebada.evaluation.checkpoint08_local import ID
from geocebada.evaluation.checkpoint10_pca import FROZEN, PcaData
from geocebada.features.area_weighted_climate import MONTHS

BASE = "GeoPCA20_GeographicWeightedRidge"
TOUCHED = "GeoPCA20_PlusTouchedPixelClimate"
WEIGHTED = "GeoPCA20_PlusAreaWeightedClimate"
TEMPORAL = "GeoPCA20_PlusAreaWeightedTemporalClimate"
FOUR = "FourCovariate_GlobalRidge"
ENSEMBLE = "GeoPCA20_FourCovariateOOFBlend"
SEED = 20261016
CLIMATE_PCS = 6
TEMPORAL_PCS = 6
BLEND_WEIGHTS = (0., .25, .5, .75, 1.)


@dataclass
class Features:
    baseline: PcaData
    spaces: dict[str, np.ndarray]
    peer: pd.DataFrame | None = None


def _scaled_pcs(frame: pd.DataFrame, ids: list[str], count: int) -> np.ndarray:
    """Label-blind all-parcel projection; no row is silently discarded."""
    if ID not in frame or frame[ID].astype(str).duplicated().any():
        raise ValueError("Climate block needs exactly one row per parcel.")
    cols = [c for c in frame.columns if c != ID]
    if len(cols) < count:
        raise ValueError("Not enough climate features for predeclared PCA rank.")
    ordered = frame.set_index(ID).reindex(ids)
    if ordered[cols].isna().all(axis=0).any():
        raise ValueError("Entirely missing climate feature; inspect raster coverage.")
    x = ordered[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if np.isinf(x).any():
        raise ValueError("Nonfinite infinite climate values.")
    x = SimpleImputer(strategy="median").fit_transform(x)
    x = StandardScaler().fit_transform(x)
    pcs = PCA(n_components=count, svd_solver="full").fit_transform(x)
    if not np.isfinite(pcs).all():
        raise ValueError("Nonfinite climate PCA scores.")
    return StandardScaler().fit_transform(pcs)


def make_features(
    baseline: PcaData,
    weighted_monthly: pd.DataFrame,
    touched_monthly: pd.DataFrame,
    temporal: pd.DataFrame,
    *,
    peer_features: pd.DataFrame | None = None,
) -> Features:
    """Freeze the incumbent 20 PCs and append controlled 6-PC blocks."""
    ids = baseline.ids
    if len(ids) != 197 or set(ids) != set(weighted_monthly[ID].astype(str)):
        raise ValueError("Climate extraction must cover exactly the canonical 197 parcels.")
    suffix = tuple(f"__2025_m{month:02d}" for month in MONTHS)
    w_now = weighted_monthly[[ID, *[
        c for c in weighted_monthly if c.endswith(suffix)
    ]]]
    t_now = touched_monthly[[ID, *[
        c for c in touched_monthly if c.endswith(suffix)
    ]]]
    if w_now.shape[1] != 22 or t_now.shape[1] != 22:
        raise ValueError("Expected 21 original 2025 monthly features in each block.")
    w = _scaled_pcs(w_now, ids, CLIMATE_PCS)
    t = _scaled_pcs(t_now, ids, CLIMATE_PCS)
    dynamics = _scaled_pcs(temporal, ids, TEMPORAL_PCS)
    incumbent = baseline.x[20]
    spaces = {
        BASE: incumbent,
        TOUCHED: np.column_stack([incumbent, t]),
        WEIGHTED: np.column_stack([incumbent, w]),
        TEMPORAL: np.column_stack([incumbent, w, dynamics]),
    }
    if peer_features is not None:
        if ID not in peer_features:
            raise ValueError("Peer feature CSV is missing ID_POLIGONO.")
        peer = peer_features.copy()
        if peer[ID].astype(str).duplicated().any() or set(peer[ID].astype(str)) != set(ids):
            raise ValueError("Peer feature CSV must have 197 unique matching parcels.")
        columns = [c for c in peer if c != ID]
        if len(columns) != 4:
            raise ValueError("Peer feature CSV must have exactly FOUR covariate columns.")
        if any(x in name.upper() for name in columns for x in ("RENDIMIENTO", "YIELD", "TARGET")):
            raise ValueError("Peer feature columns must not encode a yield/target label.")
        x = peer.set_index(ID).reindex(ids)[columns].apply(
            pd.to_numeric, errors="coerce"
        )
        if not np.isfinite(x.to_numpy(float)).all():
            raise ValueError("Peer four covariates must be numeric and fully observed.")
        peer = x
    else:
        peer = None
    return Features(baseline=baseline, spaces=spaces, peer=peer)


def predict_local(
    f: Features, train_ids: list[str], held_ids: list[str], model: str,
) -> np.ndarray:
    """Use identical k, alpha and geographic weights for every climate ablation."""
    base = f.baseline
    fit = [base.index[i] for i in train_ids]
    query = [base.index[i] for i in held_ids]
    y = np.full(len(base.ids), np.nan, dtype=float)
    y[fit] = base.y.loc[train_ids].to_numpy(float)
    return predict_weighted_local_ridge(
        f.spaces[model], base.geographic, y, fit, query,
        k=FROZEN["k"], alpha=FROZEN["alpha"], distance_power=FROZEN["power"],
    )


def predict_four_covariates(
    f: Features, train_ids: list[str], held_ids: list[str],
) -> np.ndarray:
    """Independent, explicit *proxy* Ridge model; not a claim to replicate peer code."""
    if f.peer is None:
        raise ValueError("Four-covariate model requires an external X-only file.")
    scaler = StandardScaler().fit(f.peer.loc[train_ids])
    model = Ridge(alpha=10.)
    model.fit(
        scaler.transform(f.peer.loc[train_ids]),
        f.baseline.y.loc[train_ids].to_numpy(float),
    )
    return np.asarray(model.predict(scaler.transform(f.peer.loc[held_ids])), dtype=float)


def inner_blend_weight(f: Features, train_ids: list[str], *, seed: int) -> tuple[float, float]:
    """Choose convex weight solely from 3fold out-of-fold errors on 97 labels."""
    if f.peer is None:
        raise ValueError("Missing four-covariate predictors.")
    cv = KFold(n_splits=3, shuffle=True, random_state=seed)
    errors = []
    observed, primary, auxiliary = [], [], []
    for a, b in cv.split(train_ids):
        fit = [train_ids[i] for i in a]
        valid = [train_ids[i] for i in b]
        observed.extend(f.baseline.y.loc[valid].to_numpy(float))
        primary.extend(predict_local(f, fit, valid, BASE))
        auxiliary.extend(predict_four_covariates(f, fit, valid))
    y = np.asarray(observed, dtype=float)
    p = np.asarray(primary, dtype=float)
    q = np.asarray(auxiliary, dtype=float)
    for w in BLEND_WEIGHTS:
        rmse = float(np.sqrt(np.mean(np.square(y - ((1. - w) * p + w * q)))))
        errors.append((rmse, w))
    best_rmse, best_weight = min(errors)
    return float(best_weight), float(best_rmse)


def split_predictions(f: Features, split_id: str) -> tuple[pd.DataFrame, dict]:
    """Matched 41 pseudo-target parcels; no y used for model or blend selection."""
    train, held = f.baseline.folds[split_id]
    truth = f.baseline.y.loc[held].to_numpy(float)
    predictions = {
        name: predict_local(f, train, held, name)
        for name in f.spaces
    }
    extra = {"split_id": split_id, "outer_train": len(train), "outer_heldout": len(held)}
    if f.peer is not None:
        auxiliary = predict_four_covariates(f, train, held)
        blend, oof_rmse = inner_blend_weight(
            f, train, seed=SEED + int(split_id[-2:])
        )
        predictions[FOUR] = auxiliary
        predictions[ENSEMBLE] = (
            (1. - blend) * predictions[BASE] + blend * auxiliary
        )
        extra.update({"oof_blend_peer_weight": blend, "oof_blend_rmse": oof_rmse})
    records = [
        {
            "split_id": split_id, ID: pid, "model": name,
            "observed": float(y), "predicted": float(estimate),
        }
        for name, yh in predictions.items()
        for pid, y, estimate in zip(held, truth, yh, strict=True)
    ]
    return pd.DataFrame(records), extra


def matched_summary(
    predictions: pd.DataFrame, *,
    n_splits: int = 16, rng_seed: int = SEED, bootstrap_count: int = 1000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pooled matched RMSE and parcel-cluster bootstrap; repeated IDs not IID."""
    expected = {BASE, TOUCHED, WEIGHTED, TEMPORAL}
    seen = set(predictions["model"])
    if seen != expected and seen != expected | {FOUR, ENSEMBLE}:
        raise ValueError("Required full set of paired models is missing.")
    if predictions.duplicated(["split_id", ID, "model"]).any():
        raise ValueError("Duplicate model–parcel pseudo-target.")
    groups = predictions.groupby(["split_id", ID])
    if len(groups) != 41 * n_splits or not groups["model"].nunique().eq(len(seen)).all():
        raise ValueError("Paired 41-target coverage incomplete.")
    if not groups["observed"].nunique().eq(1).all():
        raise ValueError("Ground truth differs between paired models.")
    rows = []
    for model, subset in predictions.groupby("model"):
        e = subset["predicted"] - subset["observed"]
        split_rmse = subset.groupby("split_id").apply(
            lambda g: float(np.sqrt(np.mean(np.square(g["predicted"] - g["observed"])))),
            include_groups=False,
        )
        rows.append({
            "model": model,
            "n_splits": n_splits, "repeated_predictions": len(subset),
            "unique_parcels": subset[ID].nunique(),
            "rmse_pooled": float(np.sqrt(np.mean(np.square(e)))),
            "r2_pooled": float(r2_score(subset["observed"], subset["predicted"])),
            "rmse_mean_split": float(split_rmse.mean()),
            "rmse_worst_split": float(split_rmse.max()),
        })
    scores = pd.DataFrame(rows).sort_values("rmse_pooled")
    pivot = predictions.pivot(
        index=["split_id", ID], columns="model", values="predicted"
    )
    truth = predictions.drop_duplicates(["split_id", ID]).set_index(
        ["split_id", ID]
    )["observed"]
    sq = pivot.sub(truth, axis=0).pow(2).sort_index()
    parcel_groups = {
        key: block for key, block in sq.groupby(level=ID, sort=True)
    }
    rng = np.random.default_rng(rng_seed)
    parcels = sorted(parcel_groups)
    delta_rows = []
    for model in sorted(seen - {BASE}):
        reference = sq[model] - sq[BASE]
        boot = np.empty(bootstrap_count)
        for k in range(bootstrap_count):
            picked = rng.choice(parcels, size=len(parcels), replace=True)
            boot[k] = float(np.concatenate([
                (parcel_groups[key][model] - parcel_groups[key][BASE]).to_numpy()
                for key in picked
            ]).mean())
        delta_rows.append({
            "model": model, "reference": BASE,
            "delta_mse": float(reference.mean()),
            "ci95_low": float(np.quantile(boot, .025)),
            "ci95_high": float(np.quantile(boot, .975)),
            "bootstrap_unit": "unique_parcel",
        })
    return scores, pd.DataFrame(delta_rows)
