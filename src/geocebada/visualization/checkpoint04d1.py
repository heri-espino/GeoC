"""Static figures for Checkpoint 04D.1 local/graph refinement."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from geocebada.evaluation.checkpoint04a import ID_COLUMN


def _save(fig: plt.Figure, path: Path, *, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_refinement_ranking(
    summary: pd.DataFrame,
    path: Path,
    *,
    top_n: int = 20,
    dpi: int = 170,
) -> None:
    """Plot top target-matched 04D.1 methods by mean split RMSE."""

    subset = summary.sort_values("rmse_mean").head(int(top_n)).sort_values("rmse_mean")
    fig, ax = plt.subplots(figsize=(11.0, max(6.0, 0.36 * len(subset) + 1.5)))
    ax.barh(subset["method"], subset["rmse_mean"])
    ax.set_xlabel("Mean target-matched RMSE")
    ax.set_title("Checkpoint 04D.1 local/graph refinement")
    _save(fig, path, dpi=dpi)


def plot_graph_k_profiles(
    summary: pd.DataFrame,
    manifest: pd.DataFrame,
    path: Path,
    *,
    dpi: int = 170,
) -> None:
    """Plot best graph RMSE by neighborhood size for each topology and graph mode."""

    required = {"method_family", "distance_name", "k"}
    merged = (
        summary.copy()
        if required.issubset(summary.columns)
        else summary.merge(manifest, on="method", how="left")
    )
    graph = merged.loc[
        merged["method_family"].isin(["graph_direct", "graph_residual"])
    ].copy()
    if graph.empty:
        return

    best = (
        graph.groupby(["method_family", "distance_name", "k"], as_index=False)["rmse_mean"]
        .min()
        .sort_values(["method_family", "distance_name", "k"])
    )
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    for keys, group in best.groupby(["method_family", "distance_name"]):
        family, distance_name = keys
        ax.plot(
            group["k"],
            group["rmse_mean"],
            marker="o",
            label=f"{family}:{distance_name}",
        )
    ax.set_xlabel("Graph k")
    ax.set_ylabel("Best mean target-matched RMSE over lambda")
    ax.set_title("Graph topology sensitivity")
    ax.legend(fontsize=8, ncol=2)
    _save(fig, path, dpi=dpi)


def plot_local_k_profile(
    summary: pd.DataFrame,
    manifest: pd.DataFrame,
    path: Path,
    *,
    dpi: int = 170,
) -> None:
    """Plot best local-Ridge RMSE by neighborhood size and representation."""

    required = {"method_family", "representation", "distance_name", "k"}
    merged = (
        summary.copy()
        if required.issubset(summary.columns)
        else summary.merge(manifest, on="method", how="left")
    )
    local = merged.loc[merged["method_family"].eq("local")].copy()
    if local.empty:
        return
    best = (
        local.groupby(["representation", "distance_name", "k"], as_index=False)[
            "rmse_mean"
        ]
        .min()
        .sort_values(["representation", "distance_name", "k"])
    )

    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    for keys, group in best.groupby(["representation", "distance_name"]):
        representation, distance_name = keys
        ax.plot(
            group["k"],
            group["rmse_mean"],
            marker="o",
            label=f"{representation}:{distance_name}",
        )
    ax.set_xlabel("Local neighborhood size k")
    ax.set_ylabel("Best mean target-matched RMSE over alpha/power")
    ax.set_title("Local Ridge neighborhood sensitivity")
    ax.legend(fontsize=8, ncol=2)
    _save(fig, path, dpi=dpi)


def plot_nested_validation_comparison(
    nested_summary: pd.DataFrame,
    path: Path,
    *,
    dpi: int = 170,
) -> None:
    """Compare leave-one-split-out selection and routing performance."""

    ordered = nested_summary.sort_values("rmse_mean")
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.bar(ordered["label"], ordered["rmse_mean"])
    ax.set_ylabel("Mean held-out split RMSE")
    ax.set_title("Nested target-matched validation")
    ax.tick_params(axis="x", rotation=20)
    _save(fig, path, dpi=dpi)


def plot_loso_selection_frequency(
    selection: pd.DataFrame,
    path: Path,
    *,
    top_n: int = 15,
    dpi: int = 170,
) -> None:
    """Plot how often each method is selected by leave-one-split-out search."""

    counts = selection["selected_method"].value_counts().head(int(top_n)).sort_values()
    fig, ax = plt.subplots(figsize=(10.0, max(4.5, 0.38 * len(counts) + 1.5)))
    ax.barh(counts.index, counts.to_numpy())
    ax.set_xlabel("Held-out splits selecting method")
    ax.set_title("LOSO hyperparameter-selection stability")
    _save(fig, path, dpi=dpi)


def plot_actual_candidate_spread(
    predictions: pd.DataFrame,
    path: Path,
    *,
    dpi: int = 170,
) -> None:
    """Plot finalist prediction spread for each of the 59 actual targets."""

    pivot = predictions.pivot(index=ID_COLUMN, columns="method", values="predicted")
    spread = pd.DataFrame(
        {
            "mean": pivot.mean(axis=1),
            "minimum": pivot.min(axis=1),
            "maximum": pivot.max(axis=1),
        }
    )
    spread["range"] = spread["maximum"] - spread["minimum"]
    spread = spread.sort_values("range", ascending=True)

    y = np.arange(len(spread))
    fig, ax = plt.subplots(figsize=(10.0, max(8.0, 0.22 * len(spread) + 2.0)))
    ax.hlines(y, spread["minimum"], spread["maximum"])
    ax.scatter(spread["mean"], y, s=22)
    ax.set_yticks(y)
    ax.set_yticklabels(spread.index)
    ax.set_xlabel("Candidate predicted yield (t/ha)")
    ax.set_ylabel("Target parcel")
    ax.set_title("Finalist disagreement on the 59 fixed targets")
    _save(fig, path, dpi=dpi)
