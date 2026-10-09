"""Checkpoint 07B PCA benchmark on frozen 04B target-matched parcel splits.

Input engineering is X-only; imputation, centering, scaling, PCA eigensystem,
quadratic PC expansion, and Ridge are fitted within EACH training split.
A high-dimensional representation may explain X variance without predicting y.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

ID = "ID_POLIGONO"
TARGET = "RENDIMIENTO_T_HA"


@dataclass
class DualPCAScores:
    """Out-of-fold PCA scores and explained X variance from training only."""

    train: np.ndarray
    test: np.ndarray
    explained_ratio: np.ndarray


def dual_pca_fold(train: np.ndarray, test: np.ndarray) -> DualPCAScores:
    """PCA through small training Gram matrix; no giant feature covariance."""
    if train.ndim != 2 or test.ndim != 2 or test.shape[1] != train.shape[1]:
        raise ValueError("Expected aligned 2D training and test design matrices.")
    if train.shape[0] < 3:
        raise ValueError("PCA requires at least three training observations.")
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    xtrain = imputer.fit_transform(train)
    xtest = imputer.transform(test)
    scaler = StandardScaler()
    xtrain = scaler.fit_transform(xtrain)
    xtest = scaler.transform(xtest)
    if not np.isfinite(xtrain).all() or not np.isfinite(xtest).all():
        raise ValueError("PCA data still contains nonfinite values after scaling.")
    # Equivalent to SVD PCA; avoids diagonalizing a 30,000 x 30,000 matrix.
    gram = xtrain @ xtrain.T
    eigenvalues, eigenvectors = eigh(gram, check_finite=True)
    eigenvalues = np.maximum(eigenvalues[::-1], 0.0)
    eigenvectors = eigenvectors[:, ::-1]
    active = eigenvalues > max(1e-9, float(eigenvalues[0]) * 1e-10)
    active[min(len(active) - 1, xtrain.shape[0] - 1):] = False
    eigenvalues = eigenvalues[active]
    eigenvectors = eigenvectors[:, active]
    if not len(eigenvalues):
        raise ValueError("The feature matrix has zero PCA variance.")
    singular = np.sqrt(eigenvalues)
    train_scores = eigenvectors * singular
    # (test @ training.T) @ U / singular values = ordinary PCA transform.
    test_scores = (xtest @ xtrain.T) @ (eigenvectors / singular)
    return DualPCAScores(
        train=train_scores,
        test=test_scores,
        explained_ratio=eigenvalues / eigenvalues.sum(),
    )


def retained_components(explained: np.ndarray, requested: float | int) -> int:
    """Choose smallest retained rank meeting a fraction, or fixed rank."""
    if isinstance(requested, float):
        if not (0 < requested <= 1):
            raise ValueError("PCA variance fraction must be in (0,1].")
        return min(
            len(explained),
            int(np.searchsorted(np.cumsum(explained), requested) + 1),
        )
    if requested < 1:
        raise ValueError("PCA fixed rank must be positive.")
    return min(int(requested), len(explained))


def ridge_from_pca(
    pca: DualPCAScores,
    ytrain: np.ndarray,
    *,
    components: float | int,
    alpha: float,
    pc_products: bool = False,
    quadratic_rank: int = 12,
) -> tuple[np.ndarray, int]:
    """Fit Ridge on PCA scores, optionally with fold-local quadratic PC terms."""
    n = retained_components(pca.explained_ratio, components)
    xtrain = pca.train[:, :n].copy()
    xtest = pca.test[:, :n].copy()
    if pc_products:
        count = min(quadratic_rank, n)
        poly = PolynomialFeatures(degree=2, include_bias=False)
        nonlinear_train = poly.fit_transform(xtrain[:, :count])[:, count:]
        nonlinear_test = poly.transform(xtest[:, :count])[:, count:]
        xtrain = np.column_stack([xtrain, nonlinear_train])
        xtest = np.column_stack([xtest, nonlinear_test])
    scaler = StandardScaler()
    xtrain = scaler.fit_transform(xtrain)
    xtest = scaler.transform(xtest)
    model = Ridge(alpha=alpha).fit(xtrain, np.asarray(ytrain, dtype=float))
    return model.predict(xtest), n


def mlp_from_pca(
    pca: DualPCAScores,
    ytrain: np.ndarray,
    *,
    components: float | int,
    hidden_layers: tuple[int, ...] = (16,),
    alpha: float = 1.0,
    max_iter: int = 600,
    seed: int = 7,
) -> tuple[np.ndarray, int]:
    """Train a small regularized neural net with internal training-only validation.

    This is NOT a large network: the 97 pseudo-train yield labels constrain
    capacity even when tens of thousands of X-only features are available.
    """
    n = retained_components(pca.explained_ratio, components)
    xtrain = pca.train[:, :n]
    xtest = pca.test[:, :n]
    scaler = StandardScaler()
    xtrain = scaler.fit_transform(xtrain)
    xtest = scaler.transform(xtest)
    y = np.asarray(ytrain, dtype=float)
    center = float(y.mean())
    spread = float(y.std())
    if spread < 1e-10:
        return np.full(len(xtest), center), n
    if len(y) < 20:
        raise ValueError("MLP early stopping requires at least 20 train labels.")
    learner = MLPRegressor(
        hidden_layer_sizes=hidden_layers,
        activation="relu",
        solver="adam",
        alpha=float(alpha),
        batch_size=min(32, len(y)),
        learning_rate_init=0.001,
        early_stopping=True,
        validation_fraction=0.20,
        n_iter_no_change=40,
        max_iter=max_iter,
        random_state=seed,
    )
    learner.fit(xtrain, (y - center) / spread)
    return learner.predict(xtest) * spread + center, n


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def evaluate_highdim_pca(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    membership: pd.DataFrame,
    *,
    variances: Sequence[float] = (0.8, 0.95, 0.99),
    fixed_ranks: Sequence[int] = (16, 32, 64),
    alphas: Sequence[float] = (10, 100, 1000),
    quadratic: bool = True,
    neural: bool = True,
    mlp_hidden: Sequence[tuple[int, ...]] = ((16,), (32, 16)),
    mlp_alphas: Sequence[float] = (1.0, 10.0),
    mlp_max_iter: int = 600,
    max_splits: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Benchmark PCA on held-out 41-parcel pseudo-targets; yield-only fit/train."""
    if ID not in features or features[ID].duplicated().any():
        raise ValueError("Unique parcel IDs required in high-dimensional feature table.")
    if not {ID, TARGET, "CONJUNTO"}.issubset(targets):
        raise ValueError("Missing official yield/split columns.")
    if not {ID, "split_id", "family", "role"}.issubset(membership):
        raise ValueError("Missing frozen pseudo-competition split columns.")
    if features.columns.duplicated().any():
        raise ValueError("Duplicate high-dimensional features.")
    forbidden = {TARGET, "CONJUNTO", "target", "yield", "RENDIMIENTO"}
    leak = forbidden.intersection(features.columns)
    if leak:
        raise ValueError(f"Target/split metadata in X: {sorted(leak)}")
    data = features.set_index(ID)
    if not data.index.is_unique:
        raise ValueError("Feature IDs are not unique.")
    numeric = data.select_dtypes(include=[np.number]).copy()
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    if numeric.shape[1] < 1:
        raise ValueError("No numeric predictors.")
    if not set(targets[ID].astype(str)).issubset(set(numeric.index)):
        raise ValueError("Missing official parcel features.")
    labeled = targets.loc[targets["CONJUNTO"].eq("ENTRENAMIENTO")].copy()
    labeled[ID] = labeled[ID].astype(str)
    known = labeled.set_index(ID)[TARGET].astype(float)
    split_df = membership.loc[membership["family"].eq("target_matched")].copy()
    split_df[ID] = split_df[ID].astype(str)
    ids = sorted(split_df["split_id"].unique())
    if max_splits:
        ids = ids[:max_splits]
    if not ids:
        raise ValueError("No target-matched development splits.")
    predictions = []
    metrics = []
    diagnostics = []
    for index, split_id in enumerate(ids, 1):
        rows = split_df.loc[split_df["split_id"].eq(split_id)]
        train_ids = rows.loc[rows["role"].eq("pseudo_train"), ID].tolist()
        test_ids = rows.loc[rows["role"].eq("pseudo_target"), ID].tolist()
        if not train_ids or not test_ids:
            raise ValueError(f"Incomplete pseudo-train/pseudo-target: {split_id}")
        if set(train_ids) & set(test_ids):
            raise ValueError("Train/test ID overlap.")
        if not set(train_ids + test_ids).issubset(set(known.index)):
            raise ValueError("Pseudo-split references missing/hidden yields.")
        ytrain = known.loc[train_ids].to_numpy(dtype=float)
        ytest = known.loc[test_ids].to_numpy(dtype=float)
        xtrain = numeric.loc[train_ids].to_numpy(dtype=float)
        xtest = numeric.loc[test_ids].to_numpy(dtype=float)
        pca = dual_pca_fold(xtrain, xtest)
        options = [*variances, *fixed_ranks]
        for request in options:
            n = retained_components(pca.explained_ratio, request)
            explained = float(pca.explained_ratio[:n].sum())
            for is_poly in ([False, True] if quadratic else [False]):
                for alpha in alphas:
                    guess, _ = ridge_from_pca(
                        pca, ytrain, components=request, alpha=alpha,
                        pc_products=is_poly,
                    )
                    label = (
                        f"PCA_{request:g}__"
                        f"{'quadratic' if is_poly else 'linear'}__a{alpha:g}"
                    )
                    metrics.append({
                        "split_id": split_id,
                        "model": label,
                        "rmse": _rmse(ytest, guess),
                        "n_components": n,
                        "explained_x_variance": explained,
                        "n_training": len(train_ids),
                        "n_test": len(test_ids),
                    })
                    for pid, truth, prediction in zip(
                        test_ids, ytest, guess, strict=True
                    ):
                        predictions.append({
                            "split_id": split_id, ID: pid, "model": label,
                            "observed": float(truth),
                            "predicted": float(prediction),
                        })
        if neural:
            for request in options:
                n = retained_components(pca.explained_ratio, request)
                explained = float(pca.explained_ratio[:n].sum())
                for hidden in mlp_hidden:
                    for alpha in mlp_alphas:
                        guess, _ = mlp_from_pca(
                            pca, ytrain, components=request,
                            hidden_layers=hidden, alpha=alpha,
                            max_iter=mlp_max_iter, seed=207,
                        )
                        architecture = "_".join(map(str, hidden))
                        label = f"PCA_{request:g}__MLP_{architecture}__a{alpha:g}"
                        metrics.append({
                            "split_id": split_id,
                            "model": label,
                            "rmse": _rmse(ytest, guess),
                            "n_components": n,
                            "explained_x_variance": explained,
                            "n_training": len(train_ids),
                            "n_test": len(test_ids),
                        })
                        for pid, truth, prediction in zip(
                            test_ids, ytest, guess, strict=True
                        ):
                            predictions.append({
                                "split_id": split_id, ID: pid, "model": label,
                                "observed": float(truth),
                                "predicted": float(prediction),
                            })
        diagnostics.append({
            "split_id": split_id,
            "n_features": numeric.shape[1],
            "n_pca_components": len(pca.explained_ratio),
            "n_training": len(train_ids),
            "n_test": len(test_ids),
        })
        print(
            f"[07B PCA {index}/{len(ids)}] {split_id} "
            f"features={numeric.shape[1]} PCs={len(pca.explained_ratio)}",
            flush=True,
        )
    return pd.DataFrame(predictions), pd.DataFrame(metrics), pd.DataFrame(diagnostics)


def summarize_pca_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate mean-split and pooled errors; repeated splits are not independent."""
    rows = []
    for model, group in predictions.groupby("model"):
        split_errors = group.groupby("split_id").apply(
            lambda g: _rmse(
                g["observed"].to_numpy(), g["predicted"].to_numpy()
            ),
            include_groups=False,
        )
        rows.append({
            "model": model,
            "n_splits": int(group["split_id"].nunique()),
            "n_rows_repeated": len(group),
            "rmse_pooled": _rmse(
                group["observed"].to_numpy(), group["predicted"].to_numpy()
            ),
            "rmse_split_mean": float(split_errors.mean()),
            "rmse_split_worst": float(split_errors.max()),
        })
    return pd.DataFrame(rows).sort_values(
        ["rmse_pooled", "rmse_split_mean"]
    ).reset_index(drop=True)
