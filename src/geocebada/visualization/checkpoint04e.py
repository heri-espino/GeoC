"""Static figures for Checkpoint 04E.1 SIAP localization."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_external_ranking(
    summary: pd.DataFrame,
    path: str | Path,
    *,
    family: str,
    top_n: int = 12,
    dpi: int = 170,
) -> None:
    """Plot the strongest methods under the primary pseudo-competition family."""

    block = (
        summary.loc[summary["family"].eq(family)]
        .sort_values(["rmse_mean", "method"])
        .head(int(top_n))
        .sort_values("rmse_mean", ascending=True)
    )
    fig, ax = plt.subplots(figsize=(9, max(4, 0.42 * len(block))))
    ax.barh(block["method"], block["rmse_mean"])
    ax.set_xlabel("Mean RMSE")
    ax.set_ylabel("Method")
    ax.set_title("Checkpoint 04E.1 — target-matched method ranking")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_siap_scope_coverage(
    audit: pd.DataFrame,
    path: str | Path,
    *,
    dpi: int = 170,
) -> None:
    """Plot 2025 municipality coverage for the SIAP scope hierarchy."""

    block = audit.sort_values("municipalities_2025", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(block["scope"], block["municipalities_2025"])
    ax.set_xlabel("Municipalities with 2025 records")
    ax.set_ylabel("SIAP scope")
    ax.set_title("SIAP 2025 coverage by crop/cycle/modality specificity")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_siap_proxy_vs_yield(
    parcel_panel: pd.DataFrame,
    path: str | Path,
    *,
    dpi: int = 170,
) -> None:
    """Plot observed parcel yield against the attached public SIAP municipal prior."""

    block = parcel_panel.loc[
        parcel_panel["RENDIMIENTO_T_HA"].notna() & parcel_panel["siap_prior_yield"].notna()
    ].copy()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(block["siap_prior_yield"], block["RENDIMIENTO_T_HA"], alpha=0.75)
    if not block.empty:
        lo = min(block["siap_prior_yield"].min(), block["RENDIMIENTO_T_HA"].min())
        hi = max(block["siap_prior_yield"].max(), block["RENDIMIENTO_T_HA"].max())
        ax.plot([lo, hi], [lo, hi], linestyle="--")
    ax.set_xlabel("SIAP municipal prior (t/ha)")
    ax.set_ylabel("Observed parcel yield (t/ha)")
    ax.set_title("Observed parcel yield vs SIAP public municipal prior")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
