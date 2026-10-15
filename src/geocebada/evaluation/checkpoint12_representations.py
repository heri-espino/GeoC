"""Checkpoint 12: inexpensive alternative C4 representations for Local04D.

This is EXPLORATORY: uses the previously examined 16 Checkpoint08 outer
masks. PCA/ICA/random projection only use the available parcel covariates X.
PLS is supervised and MUST be fitted on each inner-fold training subset, never
on held-out y. ElasticNet is a regression alternative, not an X projection.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import FastICA
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.random_projection import GaussianRandomProjection

from geocebada.evaluation.checkpoint04a import resolve_numeric_representation_features
from geocebada.evaluation.checkpoint04d1 import predict_weighted_local_ridge
from geocebada.evaluation.checkpoint08_local import ID
from geocebada.evaluation.checkpoint10_pca import (
    EXPECTED_FEATURES,
    FROZEN,
    PcaData,
    inputs,
    load_pca_data,
    predict,
)
from geocebada.features import join_agronomic_features, join_empirical_features

SEED = 20261015
PCA_DIMS = (20, 40)
ICA_DIMS = (8, 16, 32)
RP_DIMS = (16, 32, 64)
PLS_DIMS = (4, 8, 16)
RIDGE_ALPHAS = (10.0, 30.0, 100.0)
ENET_ALPHAS = (0.03, 0.3)
L1_RATIOS = (0.2, 0.8)
K = 24
POWER = 1.0
FAMILIES = ("ICA_Ridge", "RP_Ridge", "PLS_Ridge", "PCA_ElasticNet", "PLS_ElasticNet")
BASELINE = "PCA20_frozen"


@dataclass
class RepresentationData:
    baseline: PcaData
    raw_z: np.ndarray
    unsupervised: dict[tuple[str, int], np.ndarray]
    ica_warnings: list[str]


def load_c4_z(root: Path) -> tuple[list[str], np.ndarray]:
    """Rebuild the exact C4 X-only 2082-column standardized matrix."""
    p = inputs(root, root / "reports/checkpoint_08/confirmation_membership.csv")
    cfg = yaml.safe_load(p["config"].read_text(encoding="utf-8"))
    base = pd.read_csv(p["base"])
    joined = join_empirical_features(
        join_agronomic_features(base, pd.read_csv(p["agronomic"])),
        pd.read_csv(p["empirical"]),
    )
    if len(joined) != 197 or not joined[ID].is_unique:
        raise ValueError("Expected 197 unique C4 parcels.")
    manifests = {
        layer: json.loads(p[f"{layer}_manifest"].read_text(encoding="utf-8"))
        for layer in ("base", "agronomic", "empirical")
    }
    selected = resolve_numeric_representation_features(
        joined=joined, manifests=manifests,
        layers=cfg["representations"]["C4_all_deterministic"]["layers"],
        allowed_modes=cfg["feature_policy"]["allowed_modes"],
        exclude_sources=cfg["feature_policy"].get("exclude_sources", []),
        exclude_columns=cfg["feature_policy"].get("exclude_columns", []),
    )
    x = joined.loc[:, list(selected)].apply(pd.to_numeric, errors="coerce")
    usable = [
        name for name in x
        if x[name].notna().any() and x[name].nunique(dropna=True) > 1
    ]
    if len(usable) != EXPECTED_FEATURES:
        raise ValueError(f"C4 expected {EXPECTED_FEATURES} features; got {len(usable)}.")
    standardized = StandardScaler().fit_transform(
        SimpleImputer(strategy="median").fit_transform(x[usable])
    )
    if not np.isfinite(standardized).all():
        raise ValueError("Nonfinite standardized C4 features.")
    return joined[ID].astype(str).tolist(), standardized


def prepare(
    root: Path, membership: Path, *,
    include_ica: bool = True, include_rp: bool = True,
) -> RepresentationData:
    """Build target-free alternatives once, never using observed yields."""
    base = load_pca_data(root, membership, max_components=160)
    ids, z = load_c4_z(root)
    if ids != base.ids:
        raise ValueError("Raw C4 parcel IDs no longer align with PCA20 control.")
    representations = {}
    for n in PCA_DIMS:
        representations[("PCA", n)] = base.x[n]
    if include_rp:
        for n in RP_DIMS:
            projector = GaussianRandomProjection(n_components=n, random_state=SEED)
            x = projector.fit_transform(z)
            representations[("RP", n)] = StandardScaler().fit_transform(x)
    messages = []
    if include_ica:
        # FastICA after a PCA40 preprojection. This is PCA→ICA, not raw ICA
        # across all 2,082 variables. It keeps CPU cost modest.
        source = base.x[40]
        for n in ICA_DIMS:
            transformer = FastICA(
                n_components=n, whiten="unit-variance",
                random_state=SEED + n, max_iter=1400, tol=0.005,
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                scores = transformer.fit_transform(source)
            for warning in caught:
                if issubclass(warning.category, ConvergenceWarning):
                    messages.append(f"PCA→ICA({n}) did not fully converge.")
            if not np.isfinite(scores).all():
                raise ValueError(f"ICA {n} produced nonfinite coordinates.")
            representations[("ICA", n)] = StandardScaler().fit_transform(scores)
    return RepresentationData(base, z, representations, messages)


def configurations(family: str, *, smoke: bool = False) -> list[dict]:
    """Small predeclared dimensions/penalties for a fair compact benchmark."""
    if family not in FAMILIES:
        raise ValueError(f"Unknown family: {family}")
    if family.startswith("ICA_"):
        dimensions, transform, kind = ICA_DIMS, "ICA", "ridge"
    elif family.startswith("RP_"):
        dimensions, transform, kind = RP_DIMS, "RP", "ridge"
    elif family.startswith("PLS_"):
        dimensions, transform = PLS_DIMS, "PLS"
        kind = "elasticnet" if family.endswith("ElasticNet") else "ridge"
    else:
        dimensions, transform, kind = PCA_DIMS, "PCA", "elasticnet"
    if smoke:
        dimensions = dimensions[:1]
    if kind == "ridge":
        alphas = (30.0,) if smoke else RIDGE_ALPHAS
        return [
            {"representation": transform, "components": n, "regressor": kind,
             "alpha": alpha, "l1_ratio": None}
            for n in dimensions for alpha in alphas
        ]
    return [
        {"representation": transform, "components": n, "regressor": kind,
         "alpha": alpha, "l1_ratio": ratio}
        for n in dimensions
        for alpha in ((0.03,) if smoke else ENET_ALPHAS)
        for ratio in ((0.2,) if smoke else L1_RATIOS)
    ]


def _matrix(
    data: RepresentationData, representation: str, components: int,
    fitting: list[str], *,
    cache: dict[tuple, np.ndarray],
) -> np.ndarray:
    """Only PLS depends on y. Fit it with the supplied training labels."""
    if representation != "PLS":
        return data.unsupervised[(representation, components)]
    key = (tuple(fitting), components)
    if key not in cache:
        positions = [data.baseline.index[pid] for pid in fitting]
        x_train = data.raw_z[positions]
        y_train = data.baseline.y.loc[fitting].to_numpy(dtype=float)
        if components >= len(fitting):
            raise ValueError("PLS components exceed inner training sample rank.")
        pls = PLSRegression(n_components=components, scale=False, max_iter=500)
        pls.fit(x_train, y_train)
        projection = pls.transform(data.raw_z)
        # Only the fold's fitting data determine supervised PLS score scaling.
        scores = StandardScaler().fit(projection[positions]).transform(projection)
        if not np.isfinite(scores).all():
            raise ValueError("PLS produced nonfinite components.")
        cache[key] = scores
    return cache[key]


def _predict_elasticnet(
    data: RepresentationData,
    x: np.ndarray,
    fitting: list[str],
    query: list[str],
    *,
    alpha: float,
    l1_ratio: float,
) -> np.ndarray:
    """ElasticNet locally refit per query, with the same frozen 24 geo neighbors."""
    base = data.baseline
    observed = np.asarray([base.index[i] for i in fitting], dtype=int)
    actual_y = base.y.loc[fitting].to_numpy(dtype=float)
    predictions = []
    for pid in query:
        distances = np.asarray(base.geographic[base.index[pid], observed], dtype=float)
        nearest = np.argsort(distances)[:min(K, len(observed))]
        sample_weights = 1.0 / (distances[nearest] + 1.0e-6)
        reg = ElasticNet(
            alpha=float(alpha), l1_ratio=float(l1_ratio),
            max_iter=3000, tol=1e-3, selection="cyclic",
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            reg.fit(
                x[observed[nearest]], actual_y[nearest],
                sample_weight=sample_weights,
            )
        predictions.append(float(reg.predict(x[[base.index[pid]]])[0]))
    return np.asarray(predictions, dtype=float)


def evaluate(
    data: RepresentationData, fitting: list[str], query: list[str],
    config: dict, *,
    cache: dict[tuple, np.ndarray],
) -> np.ndarray:
    """Fit the designated fold-local model without external target labels."""
    x = _matrix(
        data, config["representation"], config["components"],
        fitting, cache=cache,
    )
    base = data.baseline
    if config["regressor"] == "ridge":
        train = [base.index[i] for i in fitting]
        held_out = [base.index[i] for i in query]
        y_arr = np.full(len(base.ids), np.nan)
        y_arr[train] = base.y.loc[fitting].to_numpy(dtype=float)
        return predict_weighted_local_ridge(
            x, base.geographic, y_arr, train, held_out,
            k=K, alpha=config["alpha"], distance_power=POWER,
        )
    return _predict_elasticnet(
        data, x, fitting, query,
        alpha=config["alpha"], l1_ratio=config["l1_ratio"],
    )


def inner_grid(data: RepresentationData, train: list[str], family: str,
               *, seed: int, smoke: bool = False) -> pd.DataFrame:
    """Calculate candidates' RMSE from only the outer 97 training yields."""
    folds = KFold(
        n_splits=2 if smoke else 3, shuffle=True, random_state=seed,
    )
    pairs = [
        ([train[i] for i in a], [train[i] for i in b])
        for a, b in folds.split(train)
    ]
    truths = [data.baseline.y.loc[b].to_numpy(dtype=float) for _, b in pairs]
    cache = {}
    records = []
    for config in configurations(family, smoke=smoke):
        mse_sum = 0.0
        for (fit_ids, test_ids), truth in zip(pairs, truths, strict=True):
            prediction = evaluate(
                data, fit_ids, test_ids, config, cache=cache,
            )
            mse_sum += float(np.square(prediction - truth).sum())
        records.append({
            "family": family, **config,
            "inner_rmse": float(np.sqrt(mse_sum / len(train))),
        })
    return pd.DataFrame(records)


def paired_split(
    data: RepresentationData, split: str,
    grids: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, dict]:
    """Run frozen Local04D plus five inner-selected alternatives."""
    train, test = data.baseline.folds[split]
    truth = data.baseline.y.loc[test].to_numpy(dtype=float)
    records = []
    selected = {}
    cache = {}

    def append(model: str, estimate: np.ndarray) -> None:
        records.extend({
            "split_id": split, ID: pid, "model": model,
            "observed": float(y), "predicted": float(y_hat),
        } for pid, y, y_hat in zip(test, truth, estimate, strict=True))

    append(BASELINE, predict(data.baseline, train, test, components=20, **FROZEN))
    for family in FAMILIES:
        frame = grids[family]
        best = frame.sort_values(
            ["inner_rmse", "components", "alpha", "l1_ratio"],
            kind="mergesort",
        ).iloc[0]
        config = {
            "representation": str(best["representation"]),
            "components": int(best["components"]),
            "regressor": str(best["regressor"]),
            "alpha": float(best["alpha"]),
            "l1_ratio": None if pd.isna(best["l1_ratio"]) else float(best["l1_ratio"]),
        }
        selected[family] = {
            **config, "inner_rmse": float(best["inner_rmse"]),
        }
        append(family, evaluate(data, train, test, config, cache=cache))
    return pd.DataFrame(records), {
        "split_id": split, "selected": selected,
        "outer_train": len(train), "outer_holdout": len(test),
        "hidden_y_accessed": False,
    }


def summarize(predictions: pd.DataFrame, *,
              n_splits: int = 16, n_holdout: int = 41,
              bootstraps: int = 1000) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute paired RMSE/R2 and bootstrap MSE intervals clustered by parcel."""
    expected = {BASELINE, *FAMILIES}
    if set(predictions["model"]) != expected:
        raise ValueError("Incomplete alternative-representation model bank.")
    keys = ["split_id", ID]
    if predictions.duplicated([*keys, "model"]).any():
        raise ValueError("Duplicate model/split/parcel predictions.")
    counts = predictions.groupby(keys)["model"].nunique()
    if len(counts) != n_splits * n_holdout or not counts.eq(len(expected)).all():
        raise ValueError("Incomplete paired split/parcel coverage.")
    if not predictions.groupby(keys)["observed"].nunique().eq(1).all():
        raise ValueError("Inconsistent paired observed yields.")
    scores = []
    for model, frame in predictions.groupby("model"):
        per_split = frame.groupby("split_id").apply(
            lambda f: float(np.sqrt(np.mean(
                np.square(f["predicted"] - f["observed"])
            ))), include_groups=False,
        )
        scores.append({
            "model": model, "n_splits": n_splits,
            "repeated_predictions": len(frame),
            "unique_parcels": frame[ID].nunique(),
            "rmse_pooled": float(np.sqrt(np.mean(
                np.square(frame["predicted"] - frame["observed"])
            ))),
            "r2_pooled": float(r2_score(frame["observed"], frame["predicted"])),
            "rmse_mean_split": float(per_split.mean()),
            "rmse_worst_split": float(per_split.max()),
        })
    scores_frame = pd.DataFrame(scores).sort_values("rmse_pooled")
    table = predictions.pivot(index=keys, columns="model", values="predicted")
    observed = predictions.drop_duplicates(keys).set_index(keys)["observed"]
    squares = table.sub(observed, axis=0).pow(2).sort_index()
    groups = {pid: f for pid, f in squares.groupby(level=ID, sort=True)}
    parcels = sorted(groups)
    generator = np.random.default_rng(SEED)
    delta = []
    for model in FAMILIES:
        difference = squares[model] - squares[BASELINE]
        sample = np.empty(bootstraps)
        for i in range(bootstraps):
            draw = generator.choice(parcels, size=len(parcels), replace=True)
            sample[i] = float(np.concatenate([
                (groups[pid][model] - groups[pid][BASELINE]).to_numpy()
                for pid in draw
            ]).mean())
        delta.append({
            "model": model, "reference": BASELINE,
            "delta_mse": float(difference.mean()),
            "ci95_low": float(np.quantile(sample, 0.025)),
            "ci95_high": float(np.quantile(sample, 0.975)),
            "bootstrap_unit": "unique_parcel",
        })
    return scores_frame, pd.DataFrame(delta)
