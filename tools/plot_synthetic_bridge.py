#!/usr/bin/env python3
"""Private PCA view of Brownian-bridge synthetic clouds and 197 real parcels.

  python tools/plot_synthetic_bridge.py --sigma 0.15
No model training, no publishing, no invented crop labels.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

import run_synthetic_bridge_three_regime as experiment


def plot_cloud(z, root, y, anchors, *, sigma, paths, steps, seed, outfile):
    """Visualize only X geometry: PCA is fitted on 197 real observable X."""
    a, b = anchors
    left_x, left_y, right_x, right_y = experiment.brownian_bridge_samples(
        z[a], z[b], y[a], y[b], root, sigma=sigma, paths=paths,
        steps=steps, seed=seed,
    )
    pca = PCA(n_components=2).fit(z)
    real = pca.transform(z)
    left = pca.transform(left_x)
    right = pca.transform(right_x)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(real[:138, 0], real[:138, 1], c="0.7", s=26,
               alpha=0.55, label="138 real labeled X")
    ax.scatter(real[138:, 0], real[138:, 1], marker="x", c="0.45",
               s=27, alpha=0.75, label="59 target X (no yields)")
    ax.scatter(left[:, 0], left[:, 1], s=8, alpha=0.17,
               label="Synthetic specialist A cloud")
    ax.scatter(right[:, 0], right[:, 1], s=8, alpha=0.17,
               label="Synthetic specialist B cloud")
    ax.plot(real[[a, b], 0], real[[a, b], 1], linestyle="--",
            color="black", linewidth=1, alpha=0.7)
    for label, index in (("Anchor A", a), ("Anchor B", b)):
        ax.scatter(*real[index, :], marker="*", s=240, edgecolor="black",
                   linewidth=0.8, zorder=5, label=label)
    ax.set_xlabel(f"PC1 ({100*pca.explained_variance_ratio_[0]:.1f}% real X variance)")
    ax.set_ylabel(f"PC2 ({100*pca.explained_variance_ratio_[1]:.1f}% real X variance)")
    ax.set_title(f"Original 197 X parcels and synthetic Gaussian-bridge clouds; sigma={sigma:g}")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=180)
    plt.close(fig)

    t = np.linspace(0, 1, 200)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t, 4*sigma*sigma*t*(1-t))
    ax.set_xlabel("Interpolation position t")
    ax.set_ylabel("Marginal noise covariance factor (relative to Sigma)")
    ax.set_title("Brownian bridge variance: zero at endpoints, highest in the middle")
    fig.tight_layout()
    fig.savefig(outfile.with_name("bridge_expected_variance.png"), dpi=180)
    plt.close(fig)
    return outfile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, default=experiment.native.BUNDLE)
    p.add_argument("--peer-repo", type=Path, default=experiment.native.PEER_REPO)
    p.add_argument("--config", type=Path, default=experiment.native.CONFIG)
    p.add_argument("--fresh-fold-dir", type=Path, default=experiment.native.OUTPUT)
    p.add_argument("--output", type=Path, default=experiment.OUT)
    p.add_argument("--sigma", type=float, default=0.15)
    p.add_argument("--paths", type=int, default=48)
    p.add_argument("--steps", type=int, default=17)
    p.add_argument("--seed", type=int, default=experiment.SEED)
    args = p.parse_args()
    if args.sigma < 0:
        p.error("sigma must be nonnegative.")
    _frame, _full, z, root, anchors, y, _loo, _selected = (
        experiment.verified_inputs(args)
    )
    outfile = args.output.resolve()/"bridge_cloud_pca_private.png"
    plot_cloud(z, root, y, anchors, sigma=args.sigma,
               paths=args.paths, steps=args.steps, seed=args.seed,
               outfile=outfile)
    print(f"[PLOT] X-only PCA cloud + variance curve: {outfile.parent}")
    print("[PRIVATE] These plots are not automatically committed or published.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
