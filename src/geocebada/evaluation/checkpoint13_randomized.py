"""Checkpoint 13: randomized 10,000-model search with nested, staged validation.

Six local regression / representation families compete. Exactly 10,000
seeded configurations are SCREENED PER OUTER SPLIT using only outer-training
y. Strongest candidates per family then receive two repeated 4-fold inner CVs
on the same 97 labels; the 41 outer labels are used only for final scoring.
X-only PCA, RP, ICA may use all 197 parcel X; PLS is fitted within each inner
fold from training y. No promotion of the 59 fixed forecasts occurs here.
"""
from __future__ import annotations

import warnings
from itertools import product

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import KFold

from geocebada.evaluation.checkpoint04d1 import predict_weighted_local_ridge
from geocebada.evaluation.checkpoint08_local import ID
from geocebada.evaluation.checkpoint10_pca import FROZEN, predict
from geocebada.evaluation.checkpoint12_representations import (
    RepresentationData,
    _matrix,
    summarize as summarize_checkpoint12,
)

SEED = 20261015
FAMILIES = (
    "PCA_Ridge",
    "ICA_Ridge",
    "RP_Ridge",
    "PLS_Ridge",
    "PCA_ElasticNet",
    "PLS_ElasticNet",
)
DIMENSIONS = {
    "PCA": (10, 20, 30, 40, 60, 80, 120, 160),
    "ICA": (8, 16, 32),
    "RP": (16, 32, 64),
    "PLS": (4, 8, 16),
}
NEIGHBORS = (8, 12, 16, 20, 24, 32, 48)
SCREEN_TARGETS = 12
SHORTLIST_PER_FAMILY = 12
CANDIDATES = 10000
BASELINE = "PCA20_frozen"
JOINT = "Random_joint_nested"


def model_names() -> set[str]:
    """Six family procedures, a joint selector, and the frozen Local04D."""
    return {BASELINE, JOINT, *FAMILIES}


def sample_candidates(*, count: int = CANDIDATES, seed: int = SEED) -> pd.DataFrame:
    """Draw a fixed, balanced, exactly count-configuration random search."""
    if count < len(FAMILIES):
        raise ValueError("At least one configuration per family is required.")
    rng = np.random.default_rng(seed)
    records = []
    for j, family in enumerate(FAMILIES):
        quota = count // len(FAMILIES) + (j < count % len(FAMILIES))
        rep, reg = family.split("_")
        for serial in range(quota):
            dims = DIMENSIONS[rep]
            component = int(rng.choice(dims))
            k = int(rng.choice(NEIGHBORS))
            power = float(rng.uniform(0.0, 2.5))
            if reg == "Ridge":
                alpha = float(10.0 ** rng.uniform(-1.0, 3.0))
                l1_ratio = np.nan
            else:
                alpha = float(10.0 ** rng.uniform(-3.0, 0.3))
                l1_ratio = float(rng.uniform(0.05, 0.95))
            records.append({
                "candidate_id": f"{family}_{serial:05d}",
                "family": family,
                "representation": rep,
                "components": component,
                "regressor": reg.lower(),
                "k": k,
                "alpha": alpha,
                "power": power,
                "l1_ratio": l1_ratio,
            })
    frame = pd.DataFrame(records)
    if len(frame) != count or not frame["candidate_id"].is_unique:
        raise RuntimeError("Random candidate set incomplete.")
    return frame


def extend_pca(data: RepresentationData) -> RepresentationData:
    """Add cheap PCA prefixes, leaving other target-free transforms unchanged."""
    base = data.baseline
    full = base.x[160]
    for n in DIMENSIONS["PCA"]:
        data.unsupervised[("PCA", n)] = full[:, :n]
    return data


def parse_candidate(row: dict | pd.Series) -> dict:
    """Normalize candidate floats and optional L1 from a CSV or pandas row."""
    rep = str(row["representation"])
    reg = str(row["regressor"])
    dimension = int(row["components"])
    if rep not in DIMENSIONS or dimension not in DIMENSIONS[rep]:
        raise ValueError(f"Invalid representation/dimension: {rep}/{dimension}")
    if reg not in ("ridge", "elasticnet"):
        raise ValueError(f"Unknown regression: {reg}")
    ratio = row.get("l1_ratio")
    return {
        "representation": rep,
        "components": dimension,
        "regressor": reg,
        "k": int(row["k"]),
        "alpha": float(row["alpha"]),
        "power": float(row["power"]),
        "l1_ratio": None if pd.isna(ratio) else float(ratio),
    }


def candidate_predict(
    data: RepresentationData,
    train_ids: list[str],
    targets: list[str],
    candidate: dict,
    *,
    cache: dict,
) -> np.ndarray:
    """Predict without using any held-out labels; fit PLS within train_ids."""
    base = data.baseline
    x = _matrix(
        data, candidate["representation"], candidate["components"],
        train_ids, cache=cache,
    )
    train = np.array([base.index[pid] for pid in train_ids], dtype=int)
    query = np.array([base.index[pid] for pid in targets], dtype=int)
    y = np.full(len(base.ids), np.nan, dtype=float)
    y[train] = base.y.loc[train_ids].to_numpy(dtype=float)
    k = candidate["k"]
    power = candidate["power"]
    alpha = candidate["alpha"]
    if candidate["regressor"] == "ridge":
        return predict_weighted_local_ridge(
            x, base.geographic, y, train, query, k=k,
            alpha=alpha, distance_power=power,
        )
    if candidate["l1_ratio"] is None:
        raise ValueError("ElasticNet requires an l1_ratio.")
    predictions = []
    for position in query:
        d = np.asarray(base.geographic[position, train], dtype=float)
        keep = np.isfinite(d)
        if int(keep.sum()) < 3:
            raise ValueError("Too few finite geographic neighbors.")
        near_positions = train[keep]
        near_distances = d[keep]
        order = np.argsort(near_distances)[:min(k, len(near_positions))]
        neighbors = near_positions[order]
        distances = near_distances[order]
        weights = (
            np.ones(len(neighbors), dtype=float) if power == 0.0
            else 1.0 / (distances + 1e-6) ** power
        )
        model = ElasticNet(
            alpha=alpha, l1_ratio=candidate["l1_ratio"],
            max_iter=2500, tol=1e-3, selection="cyclic",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(x[neighbors], y[neighbors], sample_weight=weights)
        predictions.append(float(model.predict(x[[position]])[0]))
    result = np.asarray(predictions, dtype=float)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite predictions in randomized trial.")
    return result


def screen_split(
    data: RepresentationData,
    train_ids: list[str],
    candidates: pd.DataFrame,
    *,
    seed: int,
    target_count: int = SCREEN_TARGETS,
    cache: dict | None = None,
) -> pd.DataFrame:
    """One common 12-target inner holdout, never the 41 outer y values."""
    if not 1 <= target_count < len(train_ids):
        raise ValueError("Invalid screening holdout count.")
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(len(train_ids))
    train = [train_ids[i] for i in shuffled[target_count:]]
    test = [train_ids[i] for i in shuffled[:target_count]]
    actual = data.baseline.y.loc[test].to_numpy(dtype=float)
    if cache is None:
        cache = {}
    rows = []
    for row in candidates.to_dict("records"):
        config = parse_candidate(row)
        yh = candidate_predict(data, train, test, config, cache=cache)
        rows.append({
            "candidate_id": row["candidate_id"],
            "family": row["family"],
            "screen_rmse": float(np.sqrt(np.mean(np.square(actual - yh)))),
        })
    return pd.DataFrame(rows)


def select_finalists(
    candidates: pd.DataFrame, screened: pd.DataFrame,
    *,
    per_family: int = SHORTLIST_PER_FAMILY,
) -> pd.DataFrame:
    """Shortlist every model family separately; avoid all-Ridge domination."""
    if screened["candidate_id"].duplicated().any():
        raise ValueError("Duplicate screening prediction rows.")
    merged = candidates.merge(
        screened, on=["candidate_id", "family"], validate="one_to_one",
    )
    if len(merged) != len(candidates) or not np.isfinite(merged["screen_rmse"]).all():
        raise ValueError("Incomplete random-model screen.")
    shortlist = (
        merged.sort_values(["screen_rmse", "candidate_id"], kind="mergesort")
        .groupby("family", sort=True).head(per_family)
        .sort_values(["family", "screen_rmse", "candidate_id"])
        .reset_index(drop=True)
    )
    if set(shortlist["family"]) != set(FAMILIES):
        raise ValueError("No finalists from some model families.")
    return shortlist


def repeated_cv_folds(
    train: list[str], *,
    seed: int,
    repeats: int = 2,
    folds: int = 4,
) -> list[tuple[list[str], list[str]]]:
    """Deterministic inner folds on the 97 IDs only."""
    if repeats < 1 or folds < 2 or folds > len(train):
        raise ValueError("Invalid repeated CV parameters.")
    pairs = []
    for repeat in range(repeats):
        cv = KFold(n_splits=folds, shuffle=True, random_state=seed + repeat)
        pairs.extend(
            ([train[i] for i in fit], [train[i] for i in held])
            for fit, held in cv.split(train)
        )
    return pairs


def rerank_finalists(
    data: RepresentationData,
    train_ids: list[str],
    finalists: pd.DataFrame,
    *,
    seed: int,
    repeats: int = 2,
    folds: int = 4,
    cache: dict | None = None,
) -> pd.DataFrame:
    """Repeated CV is trained/validated solely on the 97 outer training IDs."""
    if cache is None:
        cache = {}
    splits = repeated_cv_folds(
        train_ids, seed=seed, repeats=repeats, folds=folds
    )
    total = len(train_ids) * repeats
    actual = [data.baseline.y.loc[held].to_numpy(dtype=float) for _, held in splits]
    results = []
    for row in finalists.to_dict("records"):
        params = parse_candidate(row)
        sq = 0.0
        for (train, test), truth in zip(splits, actual, strict=True):
            predicted = candidate_predict(data, train, test, params, cache=cache)
            sq += float(np.square(predicted - truth).sum())
        results.append({
            "candidate_id": row["candidate_id"], "family": row["family"],
            "inner_rmse": float(np.sqrt(sq / total)),
            "n_inner_predictions": total,
        })
    return pd.DataFrame(results)


def choose_and_evaluate(
    data: RepresentationData,
    split_id: str,
    candidates: pd.DataFrame,
    ranking: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """Select only on inner-RMSE, then score external 41 parcels once."""
    train, held = data.baseline.folds[split_id]
    selected = candidates.merge(
        ranking, on=["candidate_id", "family"], validate="one_to_one"
    )
    if set(selected["candidate_id"]) != set(ranking["candidate_id"]):
        raise ValueError("Missing candidate metadata in finalist selection.")
    ordered = selected.sort_values(
        ["inner_rmse", "candidate_id"], kind="mergesort"
    )
    models = {
        family: ordered.loc[ordered["family"].eq(family)].iloc[0]
        for family in FAMILIES
    }
    models[JOINT] = ordered.iloc[0]
    observed = data.baseline.y.loc[held].to_numpy(float)
    records = []
    baseline = predict(data.baseline, train, held, components=20, **FROZEN)
    for model, config in [
        (BASELINE, None), *[(name, row) for name, row in models.items()]
    ]:
        if model == BASELINE:
            estimation = baseline
        else:
            estimation = candidate_predict(
                data, train, held, parse_candidate(config), cache={}
            )
        records.extend({
            "split_id": split_id, ID: pid, "model": model,
            "observed": float(y), "predicted": float(yh),
        } for pid, y, yh in zip(held, observed, estimation, strict=True))
    report = {
        "split_id": split_id,
        "outer_train": len(train), "outer_held_out": len(held),
        "inner_y_only": True, "hidden_59_y_used": False,
        "winners": {
            name: {
                "candidate_id": str(row["candidate_id"]),
                "family": str(row["family"]),
                "inner_rmse": float(row["inner_rmse"]),
                **parse_candidate(row),
            }
            for name, row in models.items()
        },
    }
    return pd.DataFrame(records), report


def summarize(predictions: pd.DataFrame):
    """Reuse parcel-cluster bootstrap, mapping arbitrary family names."""
    # Checkpoint12 summary hardcodes five models, so generalize only name checks
    # through an explicit copied, transparent procedure.
    from geocebada.evaluation.checkpoint11_wide_pca import summarize as common_summary

    # Common summary expects PCA-prefixed names, so delegate through a
    # reversible relabeling of all family names into eight stable names.
    from geocebada.evaluation.checkpoint11_wide_pca import DIMS as EIGHT_DIMS

    mapping = {BASELINE: "PCA20_frozen", JOINT: "PCA_joint_nested"}
    mapping.update({
        family: f"PCA{n}_nested"
        for family, n in zip(FAMILIES, EIGHT_DIMS, strict=False)
    })
    remapped = predictions.copy()
    remapped["model"] = remapped["model"].map(mapping)
    if remapped["model"].isna().any():
        raise ValueError("Unknown model label in evaluation.")
    scores, comparison = common_summary(
        remapped, dimensions=EIGHT_DIMS[:len(FAMILIES)]
    )
    reverse = {value: key for key, value in mapping.items()}
    scores["model"] = scores["model"].map(reverse)
    comparison["model"] = comparison["model"].map(reverse)
    comparison["reference"] = BASELINE
    return scores, comparison
