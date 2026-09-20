"""Static figures for Checkpoint 04A target-set diagnostics."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from geocebada.evaluation.checkpoint04a import (
    ID_COLUMN,
    PREDICTION_VALUE,
    SPLIT_COLUMN,
    TRAIN_VALUE,
)


def _save(fig: plt.Figure, path: Path, *, dpi: int = 170) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_train_target_map(
    frame: pd.DataFrame,
    path: Path,
    *,
    latitude_column: str = "base_centroid_lat",
    longitude_column: str = "base_centroid_lon",
    dpi: int = 170,
) -> None:
    """Plot parcel centroids by official split."""

    fig, ax = plt.subplots(figsize=(8.5, 7.0))
    for split, label in [(TRAIN_VALUE, "138 labeled"), (PREDICTION_VALUE, "59 targets")]:
        subset = frame.loc[frame[SPLIT_COLUMN].eq(split)]
        ax.scatter(
            subset[longitude_column],
            subset[latitude_column],
            s=28,
            alpha=0.8,
            label=label,
        )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("GeoCebada parcel centroids: labeled vs fixed targets")
    ax.legend()
    _save(fig, path, dpi=dpi)


def plot_nearest_distance_distribution(
    target_neighbors: pd.DataFrame,
    train_neighbors: pd.DataFrame,
    path: Path,
    *,
    xlabel: str,
    title: str,
    dpi: int = 170,
) -> None:
    """Compare target-to-train nearest distances with train leave-one-out distances."""

    target = pd.to_numeric(
        target_neighbors.loc[target_neighbors["rank"].eq(1), "distance"],
        errors="coerce",
    ).dropna()
    train = pd.to_numeric(
        train_neighbors.loc[train_neighbors["rank"].eq(1), "distance"],
        errors="coerce",
    ).dropna()

    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    bins = max(10, min(30, int(np.sqrt(max(len(target), len(train))) * 2)))
    ax.hist(train, bins=bins, alpha=0.55, density=True, label="train LOO nearest")
    ax.hist(target, bins=bins, alpha=0.55, density=True, label="target → train nearest")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    _save(fig, path, dpi=dpi)


def plot_pca_embedding(
    embedding: pd.DataFrame,
    path: Path,
    *,
    title: str,
    dpi: int = 170,
) -> None:
    """Plot the first two transductive PCA coordinates."""

    if "pc02" not in embedding.columns:
        raise ValueError("At least two PCA components are required for the embedding figure.")

    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    for split, label in [(TRAIN_VALUE, "labeled"), (PREDICTION_VALUE, "targets")]:
        subset = embedding.loc[embedding[SPLIT_COLUMN].eq(split)]
        ax.scatter(subset["pc01"], subset["pc02"], s=30, alpha=0.8, label=label)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    ax.legend()
    _save(fig, path, dpi=dpi)


def plot_adversarial_roc(
    roc: pd.DataFrame,
    auc: float,
    path: Path,
    *,
    dpi: int = 170,
) -> None:
    """Plot the train-vs-target adversarial ROC curve."""

    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    ax.plot(
        roc["false_positive_rate"],
        roc["true_positive_rate"],
        label=f"Adversarial classifier AUC = {auc:.3f}",
    )
    ax.plot([0, 1], [0, 1], linestyle="--", label="No-separation reference")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Can X distinguish labeled parcels from the 59 targets?")
    ax.legend()
    _save(fig, path, dpi=dpi)


def plot_top_feature_shift(
    shift: pd.DataFrame,
    path: Path,
    *,
    top_n: int = 20,
    dpi: int = 170,
) -> None:
    """Plot features with the largest absolute standardized mean shift."""

    subset = shift.dropna(subset=["standardized_mean_difference"]).head(int(top_n)).copy()
    subset = subset.iloc[::-1]
    fig, ax = plt.subplots(figsize=(9.5, max(5.0, 0.32 * len(subset) + 1.8)))
    ax.barh(subset["feature"], subset["standardized_mean_difference"])
    ax.axvline(0.0, linewidth=1.0)
    ax.set_xlabel("Standardized target − train mean difference")
    ax.set_title(f"Top {len(subset)} covariate shifts")
    _save(fig, path, dpi=dpi)


def plot_temporal_similarity_distribution(
    target_neighbors: pd.DataFrame,
    train_neighbors: pd.DataFrame,
    path: Path,
    *,
    dpi: int = 170,
) -> None:
    """Compare nearest temporal similarity for targets and labeled LOO parcels."""

    target = pd.to_numeric(
        target_neighbors.loc[target_neighbors["rank"].eq(1), "best_lag_corr_mean"],
        errors="coerce",
    ).dropna()
    train = pd.to_numeric(
        train_neighbors.loc[train_neighbors["rank"].eq(1), "best_lag_corr_mean"],
        errors="coerce",
    ).dropna()

    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    bins = np.linspace(-1.0, 1.0, 25)
    ax.hist(train, bins=bins, alpha=0.55, density=True, label="train LOO best temporal")
    ax.hist(target, bins=bins, alpha=0.55, density=True, label="target → train best temporal")
    ax.set_xlabel("Mean best-lag correlation across configured series")
    ax.set_ylabel("Density")
    ax.set_title("Temporal support of the 59 fixed targets")
    ax.legend()
    _save(fig, path, dpi=dpi)


def plot_similarity_vs_yield_difference(
    pairs: pd.DataFrame,
    path: Path,
    *,
    metric: str,
    dpi: int = 170,
) -> None:
    """Plot one train-train distance/similarity metric against absolute yield difference."""

    x = pd.to_numeric(pairs[metric], errors="coerce")
    y = pd.to_numeric(pairs["abs_yield_difference"], errors="coerce")
    mask = x.notna() & y.notna()
    x = x.loc[mask]
    y = y.loc[mask]

    fig, ax = plt.subplots(figsize=(8.0, 5.7))
    ax.scatter(x, y, s=12, alpha=0.22)

    if len(x) >= 30 and x.nunique() >= 5:
        quantiles = pd.qcut(x, q=min(10, x.nunique()), duplicates="drop")
        summary = pd.DataFrame({"x": x, "y": y, "bin": quantiles}).groupby(
            "bin", observed=True
        ).agg(x=("x", "mean"), y=("y", "mean"))
        ax.plot(summary["x"], summary["y"], marker="o", linewidth=2.0, label="binned mean")
        ax.legend()

    ax.set_xlabel(metric)
    ax.set_ylabel("|Δ yield| (t/ha)")
    ax.set_title("Does parcel similarity transfer yield?")
    _save(fig, path, dpi=dpi)


def plot_moran_scatter(
    centered_values: np.ndarray,
    spatial_lag: np.ndarray,
    path: Path,
    *,
    title: str,
    morans_i: float,
    p_value: float,
    dpi: int = 170,
) -> None:
    """Plot centered values against their spatial lag."""

    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    ax.scatter(centered_values, spatial_lag, s=28, alpha=0.7)
    if np.std(centered_values) > 0:
        slope = np.polyfit(centered_values, spatial_lag, 1)
        grid = np.linspace(float(np.min(centered_values)), float(np.max(centered_values)), 100)
        ax.plot(grid, slope[0] * grid + slope[1])
    ax.axhline(0.0, linewidth=0.8)
    ax.axvline(0.0, linewidth=0.8)
    ax.set_xlabel("Centered value")
    ax.set_ylabel("Spatial lag")
    ax.set_title(f"{title}\nMoran's I={morans_i:.3f}, permutation p={p_value:.3f}")
    _save(fig, path, dpi=dpi)


def plot_support_ranking(
    profile: pd.DataFrame,
    path: Path,
    *,
    dpi: int = 170,
) -> None:
    """Plot all 59 targets ordered by composite support."""

    subset = profile.sort_values("support_score", ascending=True)
    fig, ax = plt.subplots(figsize=(9.0, max(9.0, 0.23 * len(subset) + 2.0)))
    ax.barh(subset[ID_COLUMN], subset["support_score"])
    ax.axvline(0.33, linestyle="--", linewidth=1.0)
    ax.axvline(0.67, linestyle="--", linewidth=1.0)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Composite support score (diagnostic, not a probability)")
    ax.set_title("59 target parcels ordered by transductive support")
    _save(fig, path, dpi=dpi)


def plot_primary_temporal_neighbor_panels(
    *,
    base: pd.DataFrame,
    temporal_neighbors: pd.DataFrame,
    series_name: str,
    prefix: str,
    months: list[int],
    output_dir: Path,
    dpi: int = 150,
) -> None:
    """Write one target-vs-best-neighbor curve figure for each target."""

    columns = [f"{prefix}{month:02d}" for month in months]
    if not all(column in base.columns for column in columns):
        return
    indexed = base.set_index(ID_COLUMN)

    best = temporal_neighbors.loc[temporal_neighbors["rank"].eq(1)]
    output_dir.mkdir(parents=True, exist_ok=True)
    for row in best.itertuples(index=False):
        query_id = str(row.query_id)
        neighbor_id = str(row.neighbor_id)
        if query_id not in indexed.index or neighbor_id not in indexed.index:
            continue

        query = pd.to_numeric(indexed.loc[query_id, columns], errors="coerce").to_numpy(float)
        neighbor = pd.to_numeric(
            indexed.loc[neighbor_id, columns], errors="coerce"
        ).to_numpy(float)

        fig, ax = plt.subplots(figsize=(7.2, 4.5))
        ax.plot(months, query, marker="o", label=f"target {query_id}")
        ax.plot(months, neighbor, marker="o", label=f"neighbor {neighbor_id}")
        ax.set_xlabel("Month in 2025")
        ax.set_ylabel(series_name)
        ax.set_title(
            f"{query_id}: nearest temporal labeled parcel\n"
            f"similarity={float(row.best_lag_corr_mean):.3f}"
        )
        ax.legend()
        _save(fig, output_dir / f"{query_id}.png", dpi=dpi)
