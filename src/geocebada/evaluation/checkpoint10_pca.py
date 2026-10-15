"""Checkpoint 10: nested PCA-rank grid for the frozen geographic Local04D model.

The 20/40/80/160 dimensional C4 PCA spaces are computed using observable X
from all 197 fixed parcels, exactly as in Checkpoint 04A. No target y is used
in feature selection or PCA. Inner threefold CV chooses Local Ridge settings
from only the 97 outer pseudo-training yields. The 41 outer yields are scored
once, after a candidate is frozen; 59 actual target yields are never present.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from geocebada.evaluation.checkpoint04a import (
    build_transductive_pca_embedding,
    haversine_distance_matrix,
    resolve_numeric_representation_features,
)
from geocebada.evaluation.checkpoint04b import normalized_distance_matrix
from geocebada.evaluation.checkpoint04d1 import predict_weighted_local_ridge
from geocebada.evaluation.checkpoint08_local import (
    ID,
    Y,
    validate_memberships,
)
from geocebada.features import join_agronomic_features, join_empirical_features

DIMS = (20, 40, 80, 160)
K_GRID = (16, 24, 36)
ALPHA_GRID = (3.0, 10.0, 30.0, 100.0, 300.0)
POWER_GRID = (0.5, 1.0, 2.0)
FROZEN = {"k": 24, "alpha": 30.0, "power": 1.0}
EXPECTED_FEATURES = 2082


def inputs(root: Path, membership: Path) -> dict[str, Path]:
    return {
        "base": root / "data/processed/features_v1/parcel_features_all.csv",
        "base_manifest": root / "data/processed/features_v1/feature_manifest.json",
        "agronomic": root / "data/processed/agronomic_features_v1/parcel_agronomic_features.csv",
        "agronomic_manifest": root / "data/processed/agronomic_features_v1/feature_manifest.json",
        "empirical": root / "data/processed/empirical_features_v1/parcel_empirical_features.csv",
        "empirical_manifest": root / "data/processed/empirical_features_v1/feature_manifest.json",
        "original_embedding": root / "reports/checkpoint_04a/transductive_embeddings.csv",
        "original_embedding_manifest": root / "reports/checkpoint_04a/embedding_manifest.json",
        "config": root / "configs/checkpoint04a.yaml",
        "targets": root / "data/source/tabular/ID_area_rendimiento_70_30_Reto_AgroCebada.csv",
        "memberships": membership,
    }


def grid_parameters(*, smoke: bool = False) -> list[dict]:
    ks = (24,) if smoke else K_GRID
    alphas = (30.0, 100.0) if smoke else ALPHA_GRID
    powers = (1.0,) if smoke else POWER_GRID
    return [
        {"k": k, "alpha": alpha, "power": power}
        for k in ks for alpha in alphas for power in powers
    ]


@dataclass
class PcaData:
    ids: list[str]
    index: dict[str, int]
    y: pd.Series
    geographic: np.ndarray
    x: dict[int, np.ndarray]
    variance: dict[int, float]
    folds: dict[str, tuple[list[str], list[str]]]
    parity_max_diff: float


def load_pca_data(root: Path, membership: Path, *, max_components: int = 160) -> PcaData:
    paths = inputs(root, membership)
    config = yaml.safe_load(paths["config"].read_text(encoding="utf-8"))
    policy = config["feature_policy"]
    spec = config["representations"]["C4_all_deterministic"]
    if list(spec["layers"]) != ["base", "agronomic", "empirical"]:
        raise ValueError("The C4 representation's feature layers have changed.")
    base = pd.read_csv(paths["base"])
    agro = pd.read_csv(paths["agronomic"])
    empirical = pd.read_csv(paths["empirical"])
    joined = join_empirical_features(join_agronomic_features(base, agro), empirical)
    if len(joined) != 197 or not joined[ID].is_unique:
        raise ValueError("Expected exactly 197 distinct parcel records.")
    joined[ID] = joined[ID].astype(str)
    membership_frame = pd.read_csv(membership)
    folds = validate_memberships(joined[[ID, "CONJUNTO"]], membership_frame)
    target_table = pd.read_csv(paths["targets"], usecols=[ID, "CONJUNTO", Y])
    target_table[ID] = target_table[ID].astype(str)
    if (len(target_table) != 197 or not target_table[ID].is_unique
            or set(target_table[ID]) != set(joined[ID])):
        raise ValueError("Official target IDs differ from the 197 C4 feature rows.")
    train_y = target_table.loc[target_table["CONJUNTO"].eq("ENTRENAMIENTO"), [ID, Y]]
    hidden_y = target_table.loc[target_table["CONJUNTO"].eq("PREDICCION"), Y]
    if (len(train_y) != 138 or len(hidden_y) != 59
            or train_y[Y].isna().any() or hidden_y.notna().any()):
        raise ValueError("Expected 138 known and 59 strictly hidden yields.")
    y = train_y.set_index(ID)[Y].astype(float)
    if not np.isfinite(y).all():
        raise ValueError("Training labels contain nonfinite values.")
    ids = joined[ID].tolist()
    manifests = {
        name: json.loads(paths[f"{name}_manifest"].read_text(encoding="utf-8"))
        for name in ("base", "agronomic", "empirical")
    }
    features = resolve_numeric_representation_features(
        joined=joined, manifests=manifests, layers=spec["layers"],
        allowed_modes=policy["allowed_modes"],
        exclude_sources=policy.get("exclude_sources", []),
        exclude_columns=policy.get("exclude_columns", []),
    )
    embedded = build_transductive_pca_embedding(
        joined, name="C4_all_deterministic", features=features,
        n_components=max_components,
    )
    reference_manifest = json.loads(
        paths["original_embedding_manifest"].read_text(encoding="utf-8")
    )
    c4 = next(row for row in reference_manifest
              if row["representation"] == "C4_all_deterministic")
    if len(embedded.features_used) != EXPECTED_FEATURES or c4["n_features_used"] != EXPECTED_FEATURES:
        raise ValueError("C4 features no longer match the frozen 2,082-feature input.")
    if embedded.n_components != max_components:
        raise ValueError("PCA cannot retain all requested components.")

    matrix = (embedded.frame.set_index(ID).reindex(ids))
    pc_cols = [f"pc{i:02d}" for i in range(1, max_components + 1)]
    if matrix[pc_cols].isna().any().any():
        raise ValueError("PCA embedding has missing parcel/component values.")
    original = pd.read_csv(paths["original_embedding"])
    original = original.loc[original["representation"].eq("C4_all_deterministic")]
    original[ID] = original[ID].astype(str)
    old20 = original.set_index(ID).reindex(ids)[pc_cols[:20]]
    if old20.isna().any().any():
        raise ValueError("Historical 20-PC reference is missing parcels.")
    original20 = StandardScaler().fit_transform(old20.to_numpy(float))
    rebuilt20 = StandardScaler().fit_transform(matrix[pc_cols[:20]].to_numpy(float))
    parity = float(np.max(np.abs(original20 - rebuilt20)))
    if not np.isfinite(parity) or parity > 1e-6:
        raise RuntimeError(
            f"20-PC numerical parity failure ({parity:.4g}); aborting "
            "before reporting results. Check original data and PCA convention."
        )
    # 197 X rows are available transductively to every fold, but no hidden y.
    full_x = matrix[pc_cols].to_numpy(float)
    x = {n: StandardScaler().fit_transform(full_x[:, :n]) for n in DIMS if n <= max_components}
    # PCA explained variance for a truncated full SVD is tracked per prefix.
    # The original embedding contains 20-PC cumulative variance only.
    # Compute cumulative explained shares from each PC sample variance.
    # The PCA coordinates' sample variances are eigenvalues; source columns were standardized.
    usable = len(embedded.features_used)
    squared = np.var(full_x, axis=0, ddof=1)
    cumvar = np.cumsum(squared) / float(usable * 197.0 / 196.0)
    variance = {n: float(cumvar[n - 1]) for n in x}
    if abs(variance[20] - float(c4["explained_variance_ratio_sum"])) > 1e-6:
        raise RuntimeError("PCA explained variance does not match historical C4.")
    coords = joined[["base_centroid_lat", "base_centroid_lon"]]
    if coords.isna().any().any():
        raise ValueError("Missing parcel centroids.")
    distances = haversine_distance_matrix(
        coords["base_centroid_lat"], coords["base_centroid_lon"]
    )
    normalized, _ = normalized_distance_matrix(
        distances,
        reference_positions=np.flatnonzero(joined["CONJUNTO"].eq("ENTRENAMIENTO").to_numpy()),
    )
    return PcaData(ids=ids, index={pid: i for i, pid in enumerate(ids)},
                   y=y, geographic=normalized, x=x, variance=variance,
                   folds=folds, parity_max_diff=parity)


def predict(data: PcaData, train: list[str], query: list[str], *,
            components: int, k: int, alpha: float, power: float) -> np.ndarray:
    tr = [data.index[pid] for pid in train]
    te = [data.index[pid] for pid in query]
    observed = np.full(len(data.ids), np.nan, dtype=float)
    observed[tr] = data.y.loc[train].to_numpy(dtype=float)
    return predict_weighted_local_ridge(
        data.x[components], data.geographic, observed, tr, te,
        k=k, alpha=alpha, distance_power=power,
    )


def inner_scores(data: PcaData, train: list[str], *, components: int,
                 seed: int, smoke: bool = False) -> pd.DataFrame:
    kfold = KFold(n_splits=2 if smoke else 3, shuffle=True, random_state=seed)
    pairs = [([train[i] for i in ia], [train[i] for i in ib])
             for ia, ib in kfold.split(train)]
    rows = []
    for pars in grid_parameters(smoke=smoke):
        residuals = []
        for known, hidden in pairs:
            prediction = predict(data, known, hidden, components=components, **pars)
            residuals.extend((prediction - data.y.loc[hidden].to_numpy(float)) ** 2)
        rows.append({"components": components, **pars,
                     "inner_rmse": float(np.sqrt(np.mean(residuals)))})
    return pd.DataFrame(rows)


def assemble_split(data: PcaData, split_id: str,
                   grids: dict[int, pd.DataFrame]) -> tuple[pd.DataFrame, dict]:
    train, test = data.folds[split_id]
    actual = data.y.loc[test].to_numpy(float)
    records = []
    best_by_dim = {}
    for n in sorted(grids):
        options = grids[n].sort_values(
            ["inner_rmse", "k", "alpha", "power"], kind="mergesort",
        )
        best = options.iloc[0]
        params = {p: float(best[p]) for p in ("k", "alpha", "power")}
        params["k"] = int(params["k"])
        best_by_dim[n] = {**params, "inner_rmse": float(best["inner_rmse"])}
        for label, configuration in (
            (f"PCA{n}_fixed", FROZEN),
            (f"PCA{n}_tuned", params),
        ):
            yh = predict(data, train, test, components=n, **configuration)
            records.extend(
                {"split_id": split_id, ID: pid, "model": label,
                 "observed": float(y), "predicted": float(p)}
                for pid, y, p in zip(test, actual, yh, strict=True)
            )
    merged = pd.concat(grids.values(), ignore_index=True).sort_values(
        ["inner_rmse", "components", "k", "alpha", "power"], kind="mergesort"
    )
    joint = merged.iloc[0]
    n_best = int(joint["components"])
    jointpars = {"k": int(joint["k"]), "alpha": float(joint["alpha"]),
                 "power": float(joint["power"])}
    yh = predict(data, train, test, components=n_best, **jointpars)
    records.extend(
        {"split_id": split_id, ID: pid, "model": "PCA_joint_nested",
         "observed": float(y), "predicted": float(p)}
        for pid, y, p in zip(test, actual, yh, strict=True)
    )
    return pd.DataFrame(records), {
        "split_id": split_id, "best_by_dimension": best_by_dim,
        "joint_best": {"components": n_best, **jointpars,
                       "inner_rmse": float(joint["inner_rmse"])},
        "outer_train": len(train), "outer_holdout": len(test),
        "hidden_y_accessed": False,
    }


def summarize(predictions: pd.DataFrame, *, bootstraps: int = 1000,
              seed: int = 20261014) -> tuple[pd.DataFrame, pd.DataFrame]:
    expected = {*(f"PCA{n}_{suffix}" for n in DIMS
                  for suffix in ("fixed", "tuned")), "PCA_joint_nested"}
    if set(predictions["model"]) != expected:
        raise ValueError("Summary requires nine complete PCA grid models.")
    keys = [ "split_id", ID ]
    if predictions.duplicated(keys + ["model"]).any():
        raise ValueError("Duplicated per-parcel model results.")
    count = predictions.groupby(keys)["model"].nunique()
    if len(count) != 16 * 41 or not count.eq(len(expected)).all():
        raise ValueError("Summary requires 16 paired splits × 41 targets.")
    if not predictions.groupby(keys)["observed"].nunique().eq(1).all():
        raise ValueError("Paired models disagree about observed y.")
    summary = []
    for model, frame in predictions.groupby("model"):
        split_rmse = frame.groupby("split_id").apply(
            lambda a: float(np.sqrt(np.mean((a.predicted - a.observed) ** 2))),
            include_groups=False,
        )
        summary.append({
            "model": model, "n_splits": 16,
            "repeated_predictions": len(frame), "unique_parcels": frame[ID].nunique(),
            "rmse_pooled": float(np.sqrt(np.mean((frame.predicted - frame.observed) ** 2))),
            "r2_pooled": float(r2_score(frame.observed, frame.predicted)),
            "rmse_mean_split": float(split_rmse.mean()),
            "rmse_worst_split": float(split_rmse.max()),
        })
    result = pd.DataFrame(summary).sort_values("rmse_pooled")
    predictions_pivot = predictions.pivot(index=keys, columns="model", values="predicted")
    truth = predictions.drop_duplicates(keys).set_index(keys)["observed"]
    sq = predictions_pivot.sub(truth, axis=0).pow(2).sort_index()
    groups = {pid: part for pid, part in sq.groupby(level=ID, sort=True)}
    unique = sorted(groups)
    rng = np.random.default_rng(seed)
    ref = "PCA20_fixed"
    pairs = []
    for model in sorted(expected - {ref}):
        difference = sq[model] - sq[ref]
        trials = []
        for _ in range(bootstraps):
            sample = rng.choice(unique, len(unique), replace=True)
            trials.append(float(np.concatenate([
                (groups[pid][model] - groups[pid][ref]).to_numpy()
                for pid in sample
            ]).mean()))
        pairs.append({
            "model": model, "reference": ref, "delta_mse": float(difference.mean()),
            "ci95_low": float(np.quantile(trials, .025)),
            "ci95_high": float(np.quantile(trials, .975)),
            "bootstrap_unit": "unique_parcel",
        })
    return result, pd.DataFrame(pairs)
