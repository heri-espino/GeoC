"""Static figures for Checkpoint 04B pseudo-competition validation."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from geocebada.evaluation.checkpoint04a import ID_COLUMN


def _save(fig: plt.Figure, path: Path, *, dpi: int = 170) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_primary_method_ranking(
    summary: pd.DataFrame,
    path: Path,
    *,
    family: str = "target_matched",
    top_n: int = 15,
    dpi: int = 170,
) -> None:
    """Plot mean pseudo-competition RMSE for the strongest methods."""

    subset = (
        summary.loc[summary["family"].eq(family)]
        .sort_values("rmse_mean")
        .head(int(top_n))
        .sort_values("rmse_mean", ascending=True)
    )
    fig, ax = plt.subplots(figsize=(10.0, max(5.0, 0.38 * len(subset) + 1.5)))
    ax.barh(subset["method"], subset["rmse_mean"])
    ax.set_xlabel("Mean RMSE across pseudo-competition splits")
    ax.set_title(f"Checkpoint 04B method ranking — {family}")
    _save(fig, path, dpi=dpi)


def plot_top_method_rmse_boxplot(
    split_metrics: pd.DataFrame,
    summary: pd.DataFrame,
    path: Path,
    *,
    family: str = "target_matched",
    top_n: int = 8,
    dpi: int = 170,
) -> None:
    """Plot split-to-split RMSE dispersion for top primary-family methods."""

    top_methods = (
        summary.loc[summary["family"].eq(family)]
        .sort_values("rmse_mean")
        .head(int(top_n))["method"]
        .tolist()
    )
    data = [
        split_metrics.loc[
            split_metrics["family"].eq(family)
            & split_metrics["method"].eq(method),
            "rmse",
        ].to_numpy(float)
        for method in top_methods
    ]
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ax.boxplot(data, tick_labels=top_methods, vert=True)
    ax.set_ylabel("RMSE")
    ax.set_title("Target-matched pseudo-competition RMSE dispersion")
    ax.tick_params(axis="x", rotation=45)
    _save(fig, path, dpi=dpi)


def plot_split_match_quality(
    split_quality: pd.DataFrame,
    path: Path,
    *,
    dpi: int = 170,
) -> None:
    """Plot X-profile and municipality matching quality of pseudo splits."""

    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    for family, group in split_quality.groupby("family"):
        if family not in {"target_matched", "state_random"}:
            continue
        ax.scatter(
            group["profile_mean_distance"],
            group["municipality_tv"],
            s=38,
            alpha=0.75,
            label=family,
        )
    ax.set_xlabel("Standardized X-profile mean distance to real 59")
    ax.set_ylabel("Municipality total-variation distance")
    ax.set_title("How closely do pseudo-target masks resemble the actual target set?")
    ax.legend()
    _save(fig, path, dpi=dpi)


def plot_support_tier_method_rmse(
    tier_summary: pd.DataFrame,
    overall_summary: pd.DataFrame,
    path: Path,
    *,
    family: str = "target_matched",
    top_n: int = 8,
    dpi: int = 170,
) -> None:
    """Plot RMSE by X-only support tier for top overall methods."""

    top_methods = (
        overall_summary.loc[overall_summary["family"].eq(family)]
        .sort_values("rmse_mean")
        .head(int(top_n))["method"]
        .tolist()
    )
    subset = tier_summary.loc[
        tier_summary["family"].eq(family)
        & tier_summary["method"].isin(top_methods)
    ].copy()
    tiers = ["low", "mid", "high"]
    x = np.arange(len(top_methods), dtype=float)
    width = 0.24

    fig, ax = plt.subplots(figsize=(11.5, 6.0))
    for offset, tier in enumerate(tiers):
        values = []
        for method in top_methods:
            match = subset.loc[
                subset["method"].eq(method) & subset["x_support_tier"].eq(tier),
                "rmse",
            ]
            values.append(float(match.iloc[0]) if len(match) else np.nan)
        ax.bar(x + (offset - 1) * width, values, width=width, label=tier)
    ax.set_xticks(x)
    ax.set_xticklabels(top_methods, rotation=45, ha="right")
    ax.set_ylabel("RMSE")
    ax.set_title("Target-matched performance by dynamic X-only support tier")
    ax.legend(title="X support")
    _save(fig, path, dpi=dpi)


def plot_actual_vs_pseudo_support(
    actual_support: pd.DataFrame,
    predictions: pd.DataFrame,
    path: Path,
    *,
    dpi: int = 170,
) -> None:
    """Compare real-target X support with target-matched pseudo-target support."""

    actual = pd.to_numeric(actual_support["x_support_score"], errors="coerce").dropna()
    pseudo = pd.to_numeric(
        predictions.loc[
            predictions["family"].eq("target_matched")
            & predictions["method"].eq(predictions["method"].iloc[0]),
            "x_support_score",
        ],
        errors="coerce",
    ).dropna()

    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    bins = np.linspace(0.0, 1.0, 21)
    ax.hist(pseudo, bins=bins, density=True, alpha=0.55, label="pseudo-targets")
    ax.hist(actual, bins=bins, density=True, alpha=0.55, label="actual 59 targets")
    ax.set_xlabel("Dynamic X-only support score")
    ax.set_ylabel("Density")
    ax.set_title("Support geometry: pseudo-targets versus actual fixed targets")
    ax.legend()
    _save(fig, path, dpi=dpi)


def plot_actual_method_routing(
    routing: pd.DataFrame,
    path: Path,
    *,
    dpi: int = 170,
) -> None:
    """Plot how many actual targets are routed to each pseudo-validated method."""

    counts = (
        routing["recommended_method"]
        .value_counts()
        .sort_values(ascending=True)
    )
    fig, ax = plt.subplots(figsize=(9.0, max(4.5, 0.45 * len(counts) + 1.5)))
    ax.barh(counts.index, counts.to_numpy())
    ax.set_xlabel("Number of the 59 actual targets")
    ax.set_title("04B routing proposal from pseudo-test support-tier winners")
    _save(fig, path, dpi=dpi)


def plot_target_support_routing_scatter(
    routing: pd.DataFrame,
    path: Path,
    *,
    dpi: int = 170,
) -> None:
    """Plot actual target X-support score by parcel and routed method."""

    ordered = routing.sort_values("x_support_score")
    fig, ax = plt.subplots(figsize=(10.0, max(8.0, 0.22 * len(ordered) + 2.0)))
    methods = ordered["recommended_method"].astype("category")
    codes = methods.cat.codes.to_numpy()
    scatter = ax.scatter(
        ordered["x_support_score"],
        np.arange(len(ordered)),
        c=codes,
        s=36,
    )
    ax.set_yticks(np.arange(len(ordered)))
    ax.set_yticklabels(ordered[ID_COLUMN])
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Actual-target X-only support score")
    ax.set_ylabel("Target parcel")
    ax.set_title("Actual 59 targets: support and pseudo-validated routing")
    handles, _ = scatter.legend_elements()
    labels = list(methods.cat.categories)
    ax.legend(handles, labels, title="Routed method", loc="best")
    _save(fig, path, dpi=dpi)
