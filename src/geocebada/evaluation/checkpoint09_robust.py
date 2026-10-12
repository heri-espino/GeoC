"""Checkpoint 09: fixed-geometry robust Local04D under possible label contamination.

This is a hypothesis test, NOT evidence that any agronomic yield was entered
incorrectly. All label-dependent operations are restricted to the training
partition of each outer split. The true 59 target labels remain unavailable.

Local04D_frozen is reproduced through Checkpoint 08's original implementation.
The robust candidates retain its C4 representation, geographic 24-nearest
neighbours, Ridge alpha=30 and inverse-distance weighting, changing only
the weights applied to potentially unreliable *training* observations.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from geocebada.evaluation.checkpoint04a import haversine_distance_matrix
from geocebada.evaluation.checkpoint04b import normalized_distance_matrix
from geocebada.evaluation.checkpoint08_local import (
    FROZEN,
    ID,
    Y,
    Inputs,
    predict_old,
    validate_memberships,
)

METHODS = (
    "Local04D_frozen",
    "Huber_c1p5",
    "Huber_c2p5",
    "LOOReliability_c2p0",
    "LOOReliability_c3p0",
)
ROBUST_METHODS = METHODS[1:]


def load_legacy_inputs(root: Path) -> Inputs:
    """Load original C4/Geo features without the 07B high-dimensional downloads."""
    meta = pd.read_csv(
        root / "data/processed/features_v1/parcel_features_all.csv",
        usecols=[ID, "CONJUNTO", "meta_estado", "meta_municipio",
                 "base_centroid_lat", "base_centroid_lon"],
    )
    meta[ID] = meta[ID].astype(str)
    if len(meta) != 197 or not meta[ID].is_unique:
        raise ValueError("Expected exactly 197 distinct parcel IDs.")
    original = pd.read_csv(
        root / "data/source/tabular/ID_area_rendimiento_70_30_Reto_AgroCebada.csv",
        usecols=[ID, "CONJUNTO", Y],
    )
    original[ID] = original[ID].astype(str)
    if len(original) != 197 or not original[ID].is_unique:
        raise ValueError("Original source must have 197 unique parcel IDs.")
    if set(original[ID]) != set(meta[ID]):
        raise ValueError("Parcel ID mismatch between source labels and X table.")
    aligned = original.set_index(ID).loc[meta[ID]]
    if not np.array_equal(aligned["CONJUNTO"].to_numpy(), meta["CONJUNTO"].to_numpy()):
        raise ValueError("Label metadata split disagrees with feature table.")
    train = meta["CONJUNTO"].eq("ENTRENAMIENTO").to_numpy()
    hidden = meta["CONJUNTO"].eq("PREDICCION").to_numpy()
    values = pd.to_numeric(aligned[Y], errors="coerce").to_numpy(float)
    if (int(train.sum()) != 138 or int(hidden.sum()) != 59
            or not np.isfinite(values[train]).all()
            or not np.isnan(values[hidden]).all()
            or np.any(values[train] <= 0)):
        raise ValueError("138 known positive yields/59 unknown yields contract violated.")
    emb = pd.read_csv(root / "reports/checkpoint_04a/transductive_embeddings.csv")
    pcs = sorted(
        [c for c in emb if c.startswith("pc") and c[2:].isdigit()],
        key=lambda x: int(x[2:]),
    )
    c4 = emb.loc[emb["representation"].eq("C4_all_deterministic")].copy()
    if not pcs or len(c4) != 197 or not c4[ID].is_unique:
        raise ValueError("Canonical C4 embedding must contain 197 unique rows.")
    c4[ID] = c4[ID].astype(str)
    ordered = c4.set_index(ID).reindex(meta[ID])[pcs]
    if ordered.isna().any().any():
        raise ValueError("Missing/invalid C4 embedding after ID alignment.")
    old = StandardScaler().fit_transform(ordered.to_numpy(float))
    geography = haversine_distance_matrix(
        meta["base_centroid_lat"], meta["base_centroid_lon"]
    )
    normal, _ = normalized_distance_matrix(
        geography, reference_positions=np.flatnonzero(train),
    )
    known = aligned.loc[train, Y].astype(float)
    known.index = meta.loc[train, ID].to_numpy()
    return Inputs(
        x07=pd.DataFrame(), old=old, geographic=normal,
        index={pid: j for j, pid in enumerate(meta[ID])},
        y=known, meta=meta, folds={},
    )


def _robust_scale(residuals: np.ndarray, *, floor: float = 0.20) -> float:
    finite = np.asarray(residuals, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) < 3:
        raise ValueError("Cannot estimate local residual scale.")
    mad = 1.4826 * np.median(np.abs(finite - np.median(finite)))
    return float(max(floor, mad))


def training_loo_residuals(data: Inputs, train_ids: list[str]) -> np.ndarray:
    """Calculate training-only LOO Local04D errors, never using external targets."""
    if len(set(train_ids)) != len(train_ids) or not set(train_ids) <= set(data.y.index):
        raise ValueError("LOO training IDs must be unique and labeled.")
    residuals = []
    for candidate in train_ids:
        observed = [pid for pid in train_ids if pid != candidate]
        pred = predict_old(data, observed, [candidate], FROZEN)[0]
        residuals.append(float(data.y.at[candidate]) - float(pred))
    return np.asarray(residuals, dtype=float)


def loo_reliability(
    residuals: np.ndarray, *,
    multiplier: float,
    minimum_weight: float = 0.35,
) -> np.ndarray:
    """Convert training-only LOO disagreement to conservative soft weights.

    Low weight is a *sensitivity scenario*, not a declaration of label error.
    Extreme but genuine yields can receive low weights too.
    """
    if multiplier <= 0 or not 0 < minimum_weight <= 1:
        raise ValueError("Invalid reliability weighting parameters.")
    errors = np.asarray(residuals, dtype=float)
    if not np.isfinite(errors).all() or errors.size < 3:
        raise ValueError("LOO residuals must be finite and nonempty.")
    threshold = multiplier * _robust_scale(errors)
    return np.clip(threshold / np.maximum(np.abs(errors), 1e-12),
                   minimum_weight, 1.0)


def predict_robust_local(
    data: Inputs,
    train_ids: list[str],
    query_ids: list[str],
    *,
    method: str,
    reliability: np.ndarray | None = None,
    max_iter: int = 10,
) -> np.ndarray:
    """Fit weighted-Ridge IRLS Huber or training-LOO reliability by target parcel."""
    if method not in ROBUST_METHODS:
        raise ValueError(f"Unexpected robust method: {method}")
    if len(set(train_ids)) != len(train_ids) or not set(train_ids) <= set(data.y.index):
        raise ValueError("Only known, distinct training yields can enter the fit.")
    if set(train_ids) & set(query_ids):
        raise ValueError("Query labels must be excluded from the local training set.")
    if len(train_ids) < 25:
        raise ValueError("At least 25 labeled training parcels are required.")
    obs = np.array([data.index[pid] for pid in train_ids], dtype=int)
    ys = data.y.loc[train_ids].to_numpy(float)
    if method.startswith("LOOReliability"):
        if reliability is None or len(reliability) != len(train_ids):
            raise ValueError("Need one TRAINING-ONLY quality weight for each training ID.")
        if not np.isfinite(reliability).all() or np.any(np.asarray(reliability) <= 0):
            raise ValueError("Reliability weights must be finite and positive.")
    result = []
    for pid in query_ids:
        j = data.index[pid]
        d = np.asarray(data.geographic[j, obs], dtype=float)
        valid = np.flatnonzero(np.isfinite(d))
        if len(valid) < 3:
            raise ValueError("Too few geographic neighbors: cannot compare models.")
        chosen = valid[np.argsort(d[valid], kind="stable")[:24]]
        x = data.old[obs[chosen]]
        y = ys[chosen]
        w = 1.0 / (d[chosen] + 1e-6)
        query_x = data.old[j:j + 1]
        fit = Ridge(alpha=30.0).fit(x, y, sample_weight=w)

        if method.startswith("LOOReliability"):
            assert reliability is not None
            fit = Ridge(alpha=30.0).fit(
                x, y, sample_weight=w * np.asarray(reliability)[chosen]
            )
        else:
            constant = 1.5 if method == "Huber_c1p5" else 2.5
            # Initial residual scale is frozen for this neighbourhood: the
            # robust candidate has no access to the query's held-out label.
            scale = _robust_scale(y - fit.predict(x))
            for _ in range(max_iter):
                abs_error = np.abs(y - fit.predict(x))
                huber_factor = np.minimum(
                    1.0, (constant * scale) / np.maximum(abs_error, 1e-12)
                )
                updated = Ridge(alpha=30.0).fit(
                    x, y, sample_weight=w * huber_factor
                )
                change = abs(float(updated.predict(query_x)[0] - fit.predict(query_x)[0]))
                fit = updated
                if change < 1e-7:
                    break
        result.append(float(fit.predict(query_x)[0]))
    return np.asarray(result, dtype=float)


def evaluate_bank(
    data: Inputs,
    memberships: pd.DataFrame,
    *,
    expected_splits: int = 16,
    selected_methods: tuple[str, ...] = METHODS,
) -> pd.DataFrame:
    """Score frozen candidates on held-out pseudo-targets; no tuning on test y."""
    folds = validate_memberships(data.meta, memberships, expected=expected_splits)
    if len(set(selected_methods)) != len(selected_methods):
        raise ValueError("Duplicated candidate method.")
    if not set(selected_methods) <= set(METHODS):
        raise ValueError("Unknown candidate method.")
    rows: list[dict[str, Any]] = []
    for split_id, (train, query) in sorted(folds.items()):
        low, high = np.quantile(data.y.loc[train], [0.15, 0.85])
        truth = data.y.loc[query].to_numpy(float)
        regime = np.where(truth < low, "low",
                          np.where(truth > high, "high", "center"))
        errors = (
            training_loo_residuals(data, train)
            if any(m.startswith("LOOReliability") for m in selected_methods)
            else None
        )
        for method in selected_methods:
            if method == "Local04D_frozen":
                pred = predict_old(data, train, query, FROZEN)
            elif method.startswith("LOOReliability"):
                assert errors is not None
                factor = 2.0 if method == "LOOReliability_c2p0" else 3.0
                pred = predict_robust_local(
                    data, train, query, method=method,
                    reliability=loo_reliability(errors, multiplier=factor),
                )
            else:
                pred = predict_robust_local(data, train, query, method=method)
            if not np.isfinite(pred).all() or len(pred) != len(query):
                raise RuntimeError(f"Invalid forecasts for {split_id}/{method}.")
            rows.extend({
                "split_id": split_id, ID: pid, "model": method,
                "observed": float(observed), "predicted": float(estimated),
                "regime": str(tail), "train_n": len(train), "holdout_n": len(query),
            } for pid, observed, estimated, tail in zip(
                query, truth, pred, regime, strict=True
            ))
    return pd.DataFrame(rows)


def summarize_bank(predictions: pd.DataFrame) -> pd.DataFrame:
    """Compute RMSE/MAE and train-defined tail metrics for paired forecasts."""
    if predictions.duplicated(["split_id", ID, "model"]).any():
        raise ValueError("Duplicated paired prediction.")
    counts = predictions.groupby(["split_id", ID])["model"].nunique()
    if not counts.eq(predictions["model"].nunique()).all():
        raise ValueError("Missing candidate for some pseudo-target.")
    rows = []
    for method, frame in predictions.groupby("model", sort=True):
        e = frame["predicted"].to_numpy(float) - frame["observed"].to_numpy(float)
        split_rmse = frame.groupby("split_id").apply(
            lambda f: float(np.sqrt(np.mean(
                (f["predicted"] - f["observed"]) ** 2
            ))), include_groups=False,
        )
        center = frame.loc[frame["regime"].eq("center")]
        tails = frame.loc[~frame["regime"].eq("center")]
        rows.append({
            "model": method,
            "n_splits": frame["split_id"].nunique(),
            "n_repeated_predictions": len(frame),
            "n_unique_parcels": frame[ID].nunique(),
            "rmse_pooled": float(np.sqrt(np.mean(e ** 2))),
            "rmse_mean_split": float(split_rmse.mean()),
            "rmse_worst_split": float(split_rmse.max()),
            "mae_pooled": float(np.mean(np.abs(e))),
            "center_rmse": float(np.sqrt(np.mean(
                (center["predicted"] - center["observed"]) ** 2
            ))) if len(center) else float("nan"),
            "tail_rmse": float(np.sqrt(np.mean(
                (tails["predicted"] - tails["observed"]) ** 2
            ))) if len(tails) else float("nan"),
        })
    return pd.DataFrame(rows).sort_values("rmse_pooled").reset_index(drop=True)


def leave_one_split_out_selection(
    predictions: pd.DataFrame, *,
    min_gain: float = 0.003,
    mae_tolerance: float = 0.002,
    tail_tolerance: float = 0.005,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """LOSO stability diagnostic (NOT statistically independent parcel folds).

    Selection is based only on OTHER outer masks, but those masks repeat parcel
    IDs; therefore an additional confirmation mask is required before promotion.
    """
    splits = sorted(predictions["split_id"].unique())
    if len(splits) != 16 or set(predictions["model"]) != set(METHODS):
        raise ValueError("Need all 16 split IDs and five fixed candidates for LOSO.")
    records = []
    chosen = []
    for holdout in splits:
        train_rows = predictions.loc[~predictions["split_id"].eq(holdout)]
        ranking = summarize_bank(train_rows)
        method = str(ranking.iloc[0]["model"])
        group = predictions.loc[
            predictions["split_id"].eq(holdout) & predictions["model"].eq(method)
        ].copy()
        if len(group) != 41:
            raise ValueError("Incomplete LOSO held-out candidate.")
        chosen.append(group)
        records.append({"split_id": holdout, "selected_on_other_15": method})
    selected = pd.concat(chosen, ignore_index=True)
    chosen_metrics = summarize_bank(selected).iloc[0].to_dict()
    base = summarize_bank(
        predictions.loc[predictions["model"].eq("Local04D_frozen")]
    ).iloc[0].to_dict()
    full = summarize_bank(predictions)
    rival = full.loc[full["model"].ne("Local04D_frozen")].iloc[0].to_dict()
    pass_gate = (
        chosen_metrics["rmse_pooled"] <= base["rmse_pooled"] - min_gain
        and chosen_metrics["mae_pooled"] <= base["mae_pooled"] + mae_tolerance
        and chosen_metrics["tail_rmse"] <= base["tail_rmse"] + tail_tolerance
        and rival["rmse_pooled"] < base["rmse_pooled"]
    )
    decision = {
        "LOSO_STABILITY_GATE": "PASS" if pass_gate else "FAIL",
        "development_best_challenger": rival["model"],
        "candidate_to_freeze_for_new_mask_confirmation": (
            rival["model"] if pass_gate else None
        ),
        "loso_rmse_pooled": chosen_metrics["rmse_pooled"],
        "baseline_rmse_pooled": base["rmse_pooled"],
        "loso_mae_pooled": chosen_metrics["mae_pooled"],
        "baseline_mae_pooled": base["mae_pooled"],
        "loso_tail_rmse": chosen_metrics["tail_rmse"],
        "baseline_tail_rmse": base["tail_rmse"],
        "min_gain_required": min_gain,
        "mae_tolerance": mae_tolerance,
        "tail_tolerance": tail_tolerance,
        "note": "Repeated parcels across masks make LOSO a stability check, not independent validation.",
    }
    return pd.DataFrame(records), selected, decision


def paired_cluster_interval(
    predictions: pd.DataFrame,
    challenger: str,
    *,
    seed: int = 20261013,
    bootstraps: int = 1000,
) -> dict[str, float | str]:
    """Parcel-block bootstrap over repeated paired errors (not spatial CI)."""
    paired = predictions.loc[
        predictions["model"].isin(["Local04D_frozen", challenger])
    ]
    if paired.duplicated(["split_id", ID, "model"]).any():
        raise ValueError("Duplicated row in paired confirmation.")
    table = paired.pivot(index=["split_id", ID], columns="model", values="predicted")
    if set(table.columns) != {"Local04D_frozen", challenger} or table.isna().any().any():
        raise ValueError("Incomplete paired predictions.")
    truth = paired.drop_duplicates(["split_id", ID]).set_index(
        ["split_id", ID]
    )["observed"]
    diff = ((table[challenger] - truth) ** 2
            - (table["Local04D_frozen"] - truth) ** 2)
    groups = {pid: grp.to_numpy(float) for pid, grp in diff.groupby(level=ID)}
    ids = sorted(groups)
    rng = np.random.default_rng(seed)
    replicates = np.empty(bootstraps)
    for j in range(bootstraps):
        chosen = rng.choice(ids, size=len(ids), replace=True)
        replicates[j] = float(np.mean(np.concatenate([groups[pid] for pid in chosen])))
    return {
        "challenger": challenger,
        "baseline": "Local04D_frozen",
        "delta_mse": float(diff.mean()),
        "ci95_low": float(np.quantile(replicates, 0.025)),
        "ci95_high": float(np.quantile(replicates, 0.975)),
        "bootstrap_unit": "unique_parcel",
        "independent_y_campaigns": False,
    }


def build_review_queue(data: Inputs) -> pd.DataFrame:
    """Rank training labels for HUMAN provenance review, never auto-delete any."""
    ids = data.meta.loc[
        data.meta["CONJUNTO"].eq("ENTRENAMIENTO"), ID
    ].astype(str).tolist()
    residual = training_loo_residuals(data, ids)
    view = data.meta.set_index(ID).loc[ids]
    scores = pd.DataFrame({
        ID: ids,
        "reported_yield_t_ha": data.y.loc[ids].to_numpy(float),
        "training_only_loo_local04d_residual": residual,
        "absolute_loo_disagreement": np.abs(residual),
        "state": view["meta_estado"].to_numpy(),
        "municipality": view["meta_municipio"].to_numpy(),
        "review_status": "unreviewed",
        "independent_source_verified": False,
        "verification_notes": "",
    })
    return scores.sort_values(
        ["absolute_loo_disagreement", ID], ascending=[False, True]
    ).reset_index(drop=True)
