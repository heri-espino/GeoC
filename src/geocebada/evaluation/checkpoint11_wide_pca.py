"""Checkpoint 11: wide C4 Local04D grid with repeated inner CV.

Exploratory continuation after Checkpoint 10. Frozen Checkpoint 08 outer masks
are reused, so they do NOT provide fresh independent confirmation. Every
hyperparameter decision is made on the 97 outer-training yields only.
"""
from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

from geocebada.evaluation.checkpoint08_local import ID
from geocebada.evaluation.checkpoint10_pca import (
    FROZEN,
    PcaData,
    predict,
)

DIMS = (10, 20, 30, 40, 60, 80, 120, 160)
NEIGHBORS = (8, 12, 16, 20, 24, 32, 48)
ALPHAS = (1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)
POWERS = (0.0, 0.5, 1.0, 1.5, 2.0)
TOP_PER_DIM = 4
FROZEN_MODEL = "PCA20_frozen"
JOINT_MODEL = "PCA_joint_nested"


def models(dimensions: tuple[int, ...] = DIMS) -> set[str]:
    """Fixed incumbent, per-dimension nested procedures, and joint selector."""
    return {FROZEN_MODEL, JOINT_MODEL} | {
        f"PCA{n}_nested" for n in dimensions
    }


def make_grid(components: int, *, smoke: bool = False) -> list[dict]:
    """Construct an exhaustive, ordered grid without learning from any labels."""
    ks = (20, 24) if smoke else NEIGHBORS
    alphas = (10.0, 30.0) if smoke else ALPHAS
    powers = (0.5, 1.0) if smoke else POWERS
    return [
        {
            "components": int(components), "k": int(k),
            "alpha": float(alpha), "power": float(power),
        }
        for k, alpha, power in product(ks, alphas, powers)
    ]


def enlarge_c4(data: PcaData, dimensions: tuple[int, ...] = DIMS) -> PcaData:
    """Use the standardized 160-PC coordinates from canonical Checkpoint 10.

    Each shorter prefix is numerically identical to separately standardizing
    that prefix, since StandardScaler processes columns independently.
    """
    if 160 not in data.x:
        raise ValueError("Checkpoint 10 must load canonical PCA160 first.")
    if any(d > 160 or d < 1 for d in dimensions):
        raise ValueError("Only valid PCA160 prefixes can be selected.")
    full = data.x[160]
    data.x = {n: full[:, :n].copy() for n in dimensions}
    return data


def _inner_folds(
    train: list[str], *,
    n_folds: int,
    n_repeats: int,
    seed: int,
) -> list[tuple[list[str], list[str]]]:
    if n_folds < 2 or n_repeats < 1 or len(train) < n_folds:
        raise ValueError("Invalid inner cross-validation parameters.")
    result = []
    for repeat in range(n_repeats):
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed + repeat)
        result.extend(
            ([train[i] for i in a], [train[i] for i in b])
            for a, b in kf.split(train)
        )
    return result


def score_grid(
    data: PcaData,
    train: list[str],
    configs: list[dict],
    *,
    n_folds: int = 3,
    n_repeats: int = 1,
    seed: int = 20261015,
) -> pd.DataFrame:
    """Score candidate configurations using ONLY the supplied training IDs.

    All models use identical folds. Across repeats each training ID appears
    exactly once in the validation side of each repetition.
    """
    folds = _inner_folds(
        train, n_folds=n_folds, n_repeats=n_repeats, seed=seed
    )
    total = len(train) * n_repeats
    if not configs or total < 1:
        raise ValueError("Empty candidate grid or training data.")
    truth = [
        data.y.loc[held_out].to_numpy(dtype=float) for _, held_out in folds
    ]
    records = []
    for pars in configs:
        square_error = 0.0
        for (fit_ids, check_ids), y in zip(folds, truth, strict=True):
            estimated = predict(data, fit_ids, check_ids, **pars)
            square_error += float(np.square(estimated - y).sum())
        records.append({
            **pars, "inner_rmse": float(np.sqrt(square_error / total)),
            "inner_folds": len(folds), "inner_predictions": total,
        })
    return pd.DataFrame(records)


def shortlist(coarse: dict[int, pd.DataFrame], *,
              top_per_dim: int = TOP_PER_DIM) -> list[dict]:
    """Promote strongest coarse configurations by PCA rank plus fixed anchor."""
    chosen = []
    for n in sorted(coarse):
        frame = coarse[n]
        required = {"components", "k", "alpha", "power", "inner_rmse"}
        if not required.issubset(frame):
            raise ValueError(f"Malformed stage-one grid for PCA{n}.")
        ordered = frame.sort_values(
            ["inner_rmse", "k", "alpha", "power"], kind="mergesort"
        )
        selected = ordered.head(top_per_dim)
        for row in selected.itertuples(index=False):
            chosen.append({
                "components": n,
                "k": int(row.k), "alpha": float(row.alpha),
                "power": float(row.power),
            })
        chosen.append({"components": n, **FROZEN})
    return list({(q["components"], q["k"], q["alpha"], q["power"]): q
                 for q in chosen}.values())


def choose_nested(
    data: PcaData,
    split_id: str,
    reranked: pd.DataFrame,
    *,
    dimensions: tuple[int, ...] = DIMS,
) -> tuple[pd.DataFrame, dict]:
    """Predict the 41 outer-held-out labels once after inner-CV selection."""
    train, held_out = data.folds[split_id]
    observed = data.y.loc[held_out].to_numpy(dtype=float)
    expected = set(dimensions)
    if set(reranked["components"].astype(int)) != expected:
        raise ValueError("Not all PCA dimensions have second-stage candidates.")
    ordered = reranked.sort_values(
        ["inner_rmse", "components", "k", "alpha", "power"],
        kind="mergesort",
    )
    rows = []
    selected = {}

    def add(name: str, config: dict) -> None:
        estimated = predict(data, train, held_out, **config)
        for pid, y, prediction in zip(held_out, observed, estimated, strict=True):
            rows.append({
                "split_id": split_id, ID: pid, "model": name,
                "observed": float(y), "predicted": float(prediction),
            })

    add(FROZEN_MODEL, {"components": 20, **FROZEN})
    for n in dimensions:
        best = ordered.loc[ordered["components"].eq(n)].iloc[0]
        selected[f"PCA{n}_nested"] = {
            "components": n, "k": int(best["k"]),
            "alpha": float(best["alpha"]), "power": float(best["power"]),
            "inner_rmse": float(best["inner_rmse"]),
        }
        add(f"PCA{n}_nested", {k: selected[f"PCA{n}_nested"][k]
                                for k in ("components", "k", "alpha", "power")})
    winner = ordered.iloc[0]
    joint = {
        "components": int(winner["components"]), "k": int(winner["k"]),
        "alpha": float(winner["alpha"]), "power": float(winner["power"]),
        "inner_rmse": float(winner["inner_rmse"]),
    }
    add(JOINT_MODEL, {k: joint[k] for k in ("components", "k", "alpha", "power")})
    frame = pd.DataFrame(rows)
    assert len(frame) == 41 * (len(dimensions) + 2)
    return frame, {
        "split_id": split_id, "best_by_dimension": selected,
        "joint_best": joint, "n_outer_train": len(train),
        "n_outer_holdout": len(held_out), "hidden_y_accessed": False,
    }


def summarize(
    predictions: pd.DataFrame, *,
    dimensions: tuple[int, ...] = DIMS,
    n_outer: int = 16,
    n_holdout: int = 41,
    bootstraps: int = 1000,
    seed: int = 20261015,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Paired outer-split RMSE/R2 and parcel-clustered delta-MSE intervals."""
    expected = models(dimensions)
    if set(predictions["model"]) != expected:
        raise ValueError("Missing model in paired Checkpoint 11 predictions.")
    keys = ["split_id", ID]
    if predictions.duplicated([*keys, "model"]).any():
        raise ValueError("Duplicated split/parcel/model prediction.")
    present = predictions.groupby(keys)["model"].nunique()
    if len(present) != n_outer * n_holdout or not present.eq(len(expected)).all():
        raise ValueError("Incomplete paired outer-validation bank.")
    truth = predictions.groupby(keys)["observed"].nunique()
    if not truth.eq(1).all():
        raise ValueError("Paired observed yields differ between models.")
    records = []
    for model, subset in predictions.groupby("model"):
        by_split = subset.groupby("split_id").apply(
            lambda d: float(np.sqrt(np.mean(np.square(d["predicted"] - d["observed"])))),
            include_groups=False,
        )
        records.append({
            "model": model,
            "n_splits": int(subset["split_id"].nunique()),
            "repeated_predictions": len(subset),
            "unique_parcels": int(subset[ID].nunique()),
            "rmse_pooled": float(np.sqrt(np.mean(
                np.square(subset["predicted"] - subset["observed"])
            ))),
            "r2_pooled": float(r2_score(subset["observed"], subset["predicted"])),
            "rmse_mean_split": float(by_split.mean()),
            "rmse_worst_split": float(by_split.max()),
        })
    result = pd.DataFrame(records).sort_values("rmse_pooled")
    pivot = predictions.pivot(index=keys, columns="model", values="predicted")
    observed = predictions.drop_duplicates(keys).set_index(keys)["observed"]
    squared = pivot.sub(observed, axis=0).pow(2).sort_index()
    by_parcel = {pid: rows for pid, rows in squared.groupby(level=ID, sort=True)}
    parcels = sorted(by_parcel)
    rng = np.random.default_rng(seed)
    comparison = []
    for model in sorted(expected - {FROZEN_MODEL}):
        direct = float((squared[model] - squared[FROZEN_MODEL]).mean())
        sampled = np.empty(bootstraps)
        for i in range(bootstraps):
            draw = rng.choice(parcels, len(parcels), replace=True)
            sampled[i] = float(np.concatenate([
                (by_parcel[pid][model] - by_parcel[pid][FROZEN_MODEL]).to_numpy()
                for pid in draw
            ]).mean())
        comparison.append({
            "model": model, "reference": FROZEN_MODEL,
            "delta_mse": direct,
            "ci95_low": float(np.quantile(sampled, .025)),
            "ci95_high": float(np.quantile(sampled, .975)),
            "bootstrap_unit": "unique_parcel",
        })
    return result, pd.DataFrame(comparison)
