"""Checkpoint14: focused 1,000 candidate Local04D-centered full-inner-CV search.

* Exact historical Local04D (PCA20/k24/alpha30/p1) is the first candidate.
* Every candidate receives the SAME complete 3-fold CV on each outer 97.
* Top candidates per family and the incumbent receive 2x4-fold reranking.
* The 41 external pseudo-target labels never participate in inner selection.
* Previous Checkpoint08 outer masks have been repeatedly examined, so this
  is exploratory, not fresh independent confirmation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from geocebada.evaluation.checkpoint08_local import ID
from geocebada.evaluation.checkpoint10_pca import FROZEN, predict
from geocebada.evaluation.checkpoint12_representations import RepresentationData, prepare
from geocebada.evaluation.checkpoint13_randomized import (
    candidate_predict,
    parse_candidate,
    repeated_cv_folds,
)

SEED = 20261015
COUNT = 1000
BASELINE = "PCA20_frozen"
JOINT = "Focused_joint_nested"
FAMILIES = ("PCA_Ridge", "PLS_Ridge", "PCA_ElasticNet", "PLS_ElasticNet")
FAMILY_QUOTAS = {
    "PCA_Ridge": 700,
    "PLS_Ridge": 200,
    "PCA_ElasticNet": 60,
    "PLS_ElasticNet": 40,
}
SHORTLIST = {
    "PCA_Ridge": 20,
    "PLS_Ridge": 12,
    "PCA_ElasticNet": 8,
    "PLS_ElasticNet": 6,
}
PCA_COMPONENTS = (5, 8, 12, 16, 20, 24, 32, 40, 60, 80, 120, 160, 192)
PCA_PROBABILITIES = (
    .01, .02, .04, .08, .20, .10, .10, .16, .09, .12, .04, .03, .01
)
PLS_COMPONENTS = (4, 5, 8, 12, 16, 24, 32)
K_VALUES = (8, 12, 16, 20, 24, 30, 32, 40, 48)
K_PROBABILITIES = (.03, .06, .11, .17, .26, .16, .12, .07, .02)
ANCHOR_NAME = "PCA_Ridge_anchor_Local04D"


def model_names() -> set[str]:
    """Original frozen baseline, four family selectors and joint selector."""
    return {BASELINE, JOINT, *FAMILIES}


def focused_candidates(*, count: int = COUNT, seed: int = SEED) -> pd.DataFrame:
    """Reproducible broad-but-localized proposals and permanent incumbent.

    Anchors are selected from previous Local04D and PCA40/80 dimensionality
    studies, NOT from Checkpoint13 outer held-out labels. The exact Local04D
    candidate is always included in every screening and final selection.
    """
    if count != COUNT:
        raise ValueError("Checkpoint14 freezes exactly 1,000 candidate models.")
    rng = np.random.default_rng(seed)
    anchors = [
        (20, 24, 30., 1.), (20, 20, 30., 1.), (20, 30, 30., 1.),
        (20, 24, 10., 1.), (20, 24, 100., 1.),
        (40, 24, 30., 1.), (80, 24, 30., 1.),
        (16, 24, 30., 1.), (32, 24, 30., 1.),
        (20, 24, 30., .5), (20, 24, 30., 2.),
        (40, 20, 30., 1.), (80, 20, 30., 1.),
    ]
    records: list[dict] = []

    def record(family: str, serial: str, components: int, k: int,
               alpha: float, power: float, l1_ratio: float | None = None) -> None:
        rep, regressor = family.split("_", 1)
        records.append({
            "candidate_id": f"{family}_{serial}",
            "family": family,
            "representation": rep,
            "components": int(components),
            "regressor": regressor.lower(),
            "k": int(k), "alpha": float(alpha),
            "power": float(power),
            "l1_ratio": np.nan if l1_ratio is None else float(l1_ratio),
        })

    for i, (n, k, alpha, p) in enumerate(anchors):
        record(
            "PCA_Ridge", "anchor_Local04D" if i == 0 else f"anchor_{i:02d}",
            n, k, alpha, p,
        )
    for family, quota in FAMILY_QUOTAS.items():
        existing = sum(x["family"] == family for x in records)
        for i in range(quota - existing):
            rep, reg = family.split("_", 1)
            if rep == "PCA":
                n = int(rng.choice(PCA_COMPONENTS, p=PCA_PROBABILITIES))
            else:
                n = int(rng.choice(PLS_COMPONENTS))
            k = int(rng.choice(K_VALUES, p=K_PROBABILITIES))
            # Prefer historically useful scale while allowing broad deviations.
            power = float(np.clip(rng.normal(loc=1.0, scale=.42), 0., 2.5))
            if reg == "Ridge":
                alpha = float(10.0 ** rng.uniform(np.log10(3), np.log10(300)))
                ratio = None
            else:
                alpha = float(10.0 ** rng.uniform(-3, -0.2))
                ratio = float(rng.uniform(.1, .9))
            record(family, f"random_{i:04d}", n, k, alpha, power, ratio)
    frame = pd.DataFrame(records)
    if len(frame) != COUNT or frame["candidate_id"].duplicated().any():
        raise ValueError("Candidate set is incomplete or has duplicate IDs.")
    if frame.groupby("family").size().to_dict() != FAMILY_QUOTAS:
        raise RuntimeError("A model family exceeded its reserved search budget.")
    anchor = frame.loc[frame["candidate_id"].eq(ANCHOR_NAME)].iloc[0]
    if parse_candidate(anchor) != {
        "representation": "PCA", "components": 20, "regressor": "ridge",
        "k": int(FROZEN["k"]), "alpha": float(FROZEN["alpha"]),
        "power": float(FROZEN["power"]), "l1_ratio": None,
    }:
        raise RuntimeError("Frozen Local04D incumbent missing from candidates.")
    return frame


def prepare_focused(root, membership) -> RepresentationData:
    """Preserve original C4, reconstruct PCA192 X-only, and prepare PLS inputs."""
    data = prepare(root, membership, include_ica=False, include_rp=False)
    n = max(PCA_COMPONENTS)
    z = PCA(n_components=n, svd_solver="full").fit_transform(data.raw_z)
    x = StandardScaler().fit_transform(z)
    control = data.baseline.x[20]
    diff = float(np.max(np.abs(x[:, :20] - control)))
    if not np.isfinite(diff) or diff > 1e-6:
        raise RuntimeError(f"C4 PCA20 numerical parity FAILED: {diff:.4g}.")
    for components in PCA_COMPONENTS:
        data.unsupervised[("PCA", components)] = x[:, :components].copy()
    return data


def score_candidates(
    data: RepresentationData, train_ids: list[str], candidates: pd.DataFrame,
    *, seed: int, folds: int = 3, repeats: int = 1,
    cache: dict | None = None,
) -> pd.DataFrame:
    """Full inner CV for every candidate, not a 12-parcel holdout screen."""
    if cache is None:
        cache = {}
    pairs = repeated_cv_folds(train_ids, seed=seed, repeats=repeats, folds=folds)
    observations = [
        data.baseline.y.loc[held].to_numpy(dtype=float) for _, held in pairs
    ]
    results = []
    for item in candidates.to_dict("records"):
        parameters = parse_candidate(item)
        error_sum = 0.
        for (fit_ids, valid_ids), observed in zip(pairs, observations, strict=True):
            prediction = candidate_predict(
                data, fit_ids, valid_ids, parameters, cache=cache
            )
            error_sum += float(np.square(prediction - observed).sum())
        results.append({
            "candidate_id": item["candidate_id"],
            "family": item["family"],
            "inner_rmse": float(np.sqrt(error_sum / (len(train_ids) * repeats))),
            "inner_predictions": len(train_ids) * repeats,
        })
    return pd.DataFrame(results)


def shortlist_candidates(
    candidates: pd.DataFrame, initial_cv: pd.DataFrame,
) -> pd.DataFrame:
    """Shortlist per family, ALWAYS preserving Local04D as a finalist."""
    if initial_cv["candidate_id"].duplicated().any():
        raise ValueError("Duplicate initial candidate scores.")
    frame = candidates.merge(
        initial_cv, on=["candidate_id", "family"], validate="one_to_one"
    )
    if len(frame) != len(candidates) or not np.isfinite(frame["inner_rmse"]).all():
        raise ValueError("Initial inner CV missing some candidates.")
    parts = []
    for family in FAMILIES:
        group = frame.loc[frame["family"].eq(family)].sort_values(
            ["inner_rmse", "candidate_id"], kind="mergesort",
        )
        parts.append(group.head(SHORTLIST[family]))
    finalists = pd.concat(parts).drop_duplicates("candidate_id")
    incumbent = frame.loc[frame["candidate_id"].eq(ANCHOR_NAME)]
    if len(incumbent) != 1:
        raise ValueError("Frozen anchor is absent from 1,000 candidates.")
    finalists = pd.concat([finalists, incumbent]).drop_duplicates("candidate_id")
    return finalists.reset_index(drop=True)


def evaluate_outer(
    data: RepresentationData,
    split: str, candidates: pd.DataFrame, scored: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """Select nested hyperparameters on 97 labels and predict 41 just once."""
    full = candidates.merge(scored, on=["candidate_id", "family"], validate="one_to_one")
    if len(full) != len(scored) or full["inner_rmse"].isna().any():
        raise ValueError("Missing repeated-CV candidate evaluation.")
    if ANCHOR_NAME not in set(full["candidate_id"]):
        raise ValueError("The incumbent must receive the same reranking CV.")
    ordered = full.sort_values(
        ["inner_rmse", "candidate_id"], kind="mergesort"
    )
    winners = {
        name: ordered.loc[ordered["family"].eq(name)].iloc[0]
        for name in FAMILIES
    }
    winners[JOINT] = ordered.iloc[0]
    tr, held = data.baseline.folds[split]
    truth = data.baseline.y.loc[held].to_numpy(float)
    records = []
    baseline = predict(data.baseline, tr, held, components=20, **FROZEN)
    for name in [BASELINE, *FAMILIES, JOINT]:
        if name == BASELINE:
            prediction = baseline
        else:
            prediction = candidate_predict(
                data, tr, held, parse_candidate(winners[name]), cache={}
            )
        records.extend({
            "split_id": split, ID: pid, "model": name,
            "observed": float(value), "predicted": float(yhat),
        } for pid, value, yhat in zip(held, truth, prediction, strict=True))
    incumbent = full.loc[full["candidate_id"].eq(ANCHOR_NAME)].iloc[0]
    report = {
        "split_id": split,
        "outer_train": len(tr), "outer_holdout": len(held),
        "incumbent_initial_cv_rmse": None,
        "incumbent_repeated_cv_rmse": float(incumbent["inner_rmse"]),
        "incumbent_repeated_cv_rank_all": 1 + int(
            full["inner_rmse"].lt(float(incumbent["inner_rmse"])).sum()
        ),
        "incumbent_selected_by_joint": (
            str(winners[JOINT]["candidate_id"]) == ANCHOR_NAME
        ),
        "chosen": {
            name: {
                "candidate_id": str(row["candidate_id"]),
                "family": str(row["family"]),
                "inner_rmse": float(row["inner_rmse"]),
                **parse_candidate(row),
            } for name, row in winners.items()
        },
        "hidden_y_accessed": False,
    }
    if len(records) != 41 * len(model_names()):
        raise RuntimeError("Unpaired outer model predictions.")
    return pd.DataFrame(records), report


def summarize(
    predictions: pd.DataFrame, *,
    outer_splits: int = 16, n_held: int = 41,
    bootstraps: int = 1000, seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Paired pooled RMSE/R2 and parcel-cluster bootstrap MSE differences."""
    required = model_names()
    if set(predictions["model"]) != required:
        raise ValueError("Incomplete procedures in outer predictions.")
    keys = ["split_id", ID]
    if predictions.duplicated([*keys, "model"]).any():
        raise ValueError("Duplicate outer model scores.")
    coverage = predictions.groupby(keys)["model"].nunique()
    if len(coverage) != outer_splits * n_held or not coverage.eq(len(required)).all():
        raise ValueError("Incomplete paired outer split coverage.")
    if not predictions.groupby(keys)["observed"].nunique().eq(1).all():
        raise ValueError("Inconsistent observed yield between models.")
    results = []
    for name, block in predictions.groupby("model"):
        errors = block["predicted"] - block["observed"]
        per_split = block.groupby("split_id").apply(
            lambda f: float(np.sqrt(np.mean(
                np.square(f["predicted"] - f["observed"])
            ))),
            include_groups=False,
        )
        results.append({
            "model": name, "n_splits": outer_splits,
            "repeated_predictions": len(block),
            "unique_parcels": block[ID].nunique(),
            "rmse_pooled": float(np.sqrt(np.mean(np.square(errors)))),
            "r2_pooled": float(r2_score(block["observed"], block["predicted"])),
            "rmse_mean_split": float(per_split.mean()),
            "rmse_worst_split": float(per_split.max()),
        })
    summary = pd.DataFrame(results).sort_values("rmse_pooled")
    pivot = predictions.pivot(index=keys, columns="model", values="predicted")
    observed = predictions.drop_duplicates(keys).set_index(keys)["observed"]
    sq = pivot.sub(observed, axis=0).pow(2).sort_index()
    groups = {id_: f for id_, f in sq.groupby(level=ID, sort=True)}
    parcels = sorted(groups)
    rng = np.random.default_rng(seed)
    comparisons = []
    for name in sorted(required - {BASELINE}):
        delta = sq[name] - sq[BASELINE]
        sims = np.empty(bootstraps)
        for i in range(bootstraps):
            draws = rng.choice(parcels, size=len(parcels), replace=True)
            sims[i] = float(np.concatenate([
                (groups[pid][name] - groups[pid][BASELINE]).to_numpy()
                for pid in draws
            ]).mean())
        comparisons.append({
            "model": name, "reference": BASELINE,
            "delta_mse": float(delta.mean()),
            "ci95_low": float(np.quantile(sims, .025)),
            "ci95_high": float(np.quantile(sims, .975)),
            "bootstrap_unit": "unique_parcel",
        })
    return summary, pd.DataFrame(comparisons)
