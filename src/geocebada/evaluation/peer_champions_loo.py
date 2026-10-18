"""Paired 138-parcel LOOCV contracts for external champion comparisons.

Pure helpers: importing this module does not import peer code or read private data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

ID = "ID_POLIGONO"
Y = "RENDIMIENTO_T_HA"
BASELINE = "GeoC_Local04D"
PEER_MODELS = ("CT3", "Reference26", "Compact14", "B1")
MODELS = (BASELINE, *PEER_MODELS)
SEED = 20261018


def align_peer_targets(peer: pd.DataFrame, expected: pd.Series) -> list[str]:
    """Validate that both projects have the identical 138 labeled parcels/yields."""
    if ID not in peer or Y not in peer:
        raise ValueError("Peer bundle lacks canonical ID or yield.")
    ids = peer[ID].astype(str).tolist()
    if len(ids) != 138 or len(ids) != len(set(ids)):
        raise ValueError("Peer bundle must contain 138 unique training IDs.")
    truth = expected.copy()
    truth.index = truth.index.astype(str)
    if len(truth) != 138 or not truth.index.is_unique or set(ids) != set(truth.index):
        raise ValueError("GeoC and peer labeled parcel IDs differ.")
    observed = pd.Series(peer[Y].to_numpy(dtype=float), index=ids)
    if not np.isfinite(observed).all() or not np.isfinite(truth.to_numpy(float)).all():
        raise ValueError("Nonfinite reported yields.")
    np.testing.assert_allclose(
        observed.loc[truth.index].to_numpy(), truth.to_numpy(float),
        atol=1e-10, rtol=0, err_msg="GeoC and peer reported yields differ",
    )
    return ids


def outer_record(held_position: int, *, n: int = 138, seed: int = SEED) -> dict:
    """Build peer-native indices with exactly one excluded label.

    Inner folds are relative to the 137 outer-training positions, as required
    by peer src.current_champions.refit_fold. Every internal fit excludes the
    outer held-out target. Four inner folds mirror the peer's random protocol.
    """
    if n < 6 or not 0 <= held_position < n:
        raise ValueError("Invalid held-out position.")
    tr = np.array([j for j in range(n) if j != held_position], dtype=int)
    folds = list(KFold(n_splits=4, shuffle=True, random_state=seed).split(tr))
    inner = []
    seen = []
    for a, b in folds:
        if set(a) & set(b):
            raise AssertionError("Internal fold overlap.")
        inner.append({
            "train_relative_positions": a.tolist(),
            "validation_relative_positions": b.tolist(),
        })
        seen.extend(b.tolist())
    if sorted(seen) != list(range(n - 1)):
        raise AssertionError("Inner CV does not cover all outer training examples.")
    return {
        "protocol": "random",
        "repeat": 1,
        "fold": held_position + 1,
        "train_positions": tr.tolist(),
        "validation_positions": [held_position],
        "inner": inner,
    }


def validate_fold(frame: pd.DataFrame, *, expected_id: str, expected_y: float) -> None:
    """Refuse cached folds with shifted IDs, missing models or bad predictions."""
    needed = {ID, "observed", *MODELS}
    if len(frame) != 1 or set(frame.columns) != needed:
        raise ValueError("Unexpected peer LOOCV fold schema.")
    row = frame.iloc[0]
    if str(row[ID]) != expected_id or abs(float(row["observed"]) - expected_y) > 1e-10:
        raise ValueError("Cached peer LOOCV fold has wrong ID or yield.")
    if not np.isfinite(row[list(MODELS)].to_numpy(dtype=float)).all():
        raise ValueError("Cached peer LOOCV fold has nonfinite predictions.")


def summarize_complete(frame: pd.DataFrame, *, bootstraps: int = 3000) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute paired pooled scores and descriptive parcel bootstrap differences.

    This bootstrap does NOT adjust for spatial correlation or previous model
    selection, and must not be interpreted as independent confirmation.
    """
    if len(frame) != 138 or not frame[ID].is_unique or set(frame.columns) != {ID, "observed", *MODELS}:
        raise ValueError("Exactly 138 unique held-out predictions are required.")
    y = frame["observed"].to_numpy(float)
    a = frame[list(MODELS)].to_numpy(float)
    if not np.isfinite(a).all() or not np.isfinite(y).all():
        raise ValueError("Nonfinite LOOCV scores.")
    scores = []
    for j, name in enumerate(MODELS):
        e = a[:, j] - y
        scores.append({
            "model": name, "unique_heldout_parcels": 138,
            "training_labels_per_fit": 137,
            "rmse": float(np.sqrt(np.mean(e * e))),
            "mae": float(np.mean(np.abs(e))),
            "r2": float(r2_score(y, a[:, j])),
            "mean_error": float(np.mean(e)),
        })
    baseline = a[:, 0] - y
    draws = np.random.default_rng(SEED).integers(0, 138, size=(bootstraps, 138))
    paired = []
    for j, name in enumerate(MODELS[1:], 1):
        e = a[:, j] - y
        sq = e * e
        base = baseline * baseline
        delta = np.sqrt(np.mean(sq[draws], axis=1)) - np.sqrt(np.mean(base[draws], axis=1))
        paired.append({
            "model": name,
            "reference": BASELINE,
            "delta_rmse": float(np.sqrt(np.mean(sq)) - np.sqrt(np.mean(base))),
            "delta_mse": float(np.mean(sq - base)),
            "delta_rmse_ci95_low": float(np.quantile(delta, .025)),
            "delta_rmse_ci95_high": float(np.quantile(delta, .975)),
            "fraction_parcels_peer_better": float(np.mean(sq < base)),
            "bootstrap_unit": "parcel (spatial dependence uncorrected)",
        })
    return pd.DataFrame(scores).sort_values("rmse").reset_index(drop=True), pd.DataFrame(paired)
