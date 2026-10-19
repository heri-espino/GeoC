#!/usr/bin/env python3
"""PCA, UMAP and t-SNE of 197 real X parcels, signed LOOCV errors, top-10 X.

This is a read-only LOCAL diagnostic: no fabricated y and no model refits.
The 59 prediction parcels have X but no observed y; always display neutral.
Feature distances and native stored spatial distances are compared directly.
If actual lon/lat cannot be verified, render distance-based classical MDS,
explicitly NOT a latitude/longitude map.

  python tools/plot_reference26_error_geometry_197.py --check
  python tools/plot_reference26_error_geometry_197.py --run
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import LocalOutlierFactor

import run_global_synthetic_augmentation as existing

N = existing.N
ID = existing.ID
OUTPUT = existing.ROOT / "reports/intentionally_leaky_70_30/feature_geometry"
SEED = 20261019


def classical_mds(dist):
    """Approximate 2D coordinates from native spatial distance matrix."""
    n = len(dist)
    if dist.shape != (n, n) or not np.isfinite(dist).all():
        raise ValueError("Invalid native spatial distance matrix.")
    if not np.allclose(dist, dist.T, atol=1e-6):
        raise ValueError("Stored spatial distance is not symmetric.")
    sq = dist**2
    b = -0.5 * (sq - sq.mean(axis=0)[None, :]
                - sq.mean(axis=1)[:, None] + sq.mean())
    values, vectors = np.linalg.eigh(b)
    order = np.argsort(values)[::-1][:2]
    return vectors[:, order] * np.sqrt(np.maximum(values[order], 0))


def checked_coordinates(full, *, lat_col=None, lon_col=None):
    names = {str(v).lower(): v for v in full.columns}
    lat_names = [lat_col] if lat_col else [
        "latitude", "latitud", "centroid_lat", "centroid_latitude",
        "latitud_centroide", "lat", "centroid_lat_wgs84",
    ]
    lon_names = [lon_col] if lon_col else [
        "longitude", "longitud", "centroid_lon", "centroid_longitude",
        "longitud_centroide", "lon", "lng", "centroid_lon_wgs84",
    ]
    lat = next((names[str(c).lower()] for c in lat_names if c and str(c).lower() in names), None)
    lon = next((names[str(c).lower()] for c in lon_names if c and str(c).lower() in names), None)
    if lat is None or lon is None:
        if lat_col or lon_col:
            raise ValueError("Requested lon/lat columns do not exist.")
        return None
    coords = full[[lon, lat]].to_numpy(dtype=float)
    if (not np.isfinite(coords).all() or
        not np.all((coords[:, 0] > -119) & (coords[:, 0] < -85)) or
        not np.all((coords[:, 1] > 14) & (coords[:, 1] < 34))):
        if lat_col or lon_col:
            raise ValueError("Requested lon/lat columns outside valid Mexico bounds.")
        return None
    return coords


def neighbor_diagnostics(z, spatial, ids, error):
    """Compare X-neighbor to geographic-neighbor without inventing units."""
    n, d = z.shape
    dx = pairwise_distances(z) / np.sqrt(d)
    geo = spatial.astype(float).copy()
    if geo.shape != (n,n) or not np.isfinite(geo).all():
        raise ValueError("Native 197 geographic distances missing or malformed.")
    np.fill_diagonal(dx, np.inf)
    np.fill_diagonal(geo, np.inf)
    nx = np.argmin(dx, axis=1)
    ng = np.argmin(geo, axis=1)
    nx_known = np.argmin(dx[:N, :N], axis=1)
    ng_known = np.argmin(geo[:N, :N], axis=1)
    top = np.argsort(-abs(error), kind="stable")[:10]
    k = min(5, n-1)
    local_mean = np.sort(dx, axis=1)[:, :k].mean(axis=1)
    # Use the actual LOF factor rather than the {-1,1} classification.
    lof_object = LocalOutlierFactor(n_neighbors=min(20,n-1))
    lof_object.fit_predict(z)
    lof_scores = -lof_object.negative_outlier_factor_
    isolation_scores = -IsolationForest(
        n_estimators=200, random_state=SEED, n_jobs=1
    ).fit(z).score_samples(z)
    cov = LedoitWolf().fit(z)
    mahal = np.sqrt(np.maximum(cov.mahalanobis(z), 0))
    feature_rank_of_nearest_geo = np.empty(n, int)
    geo_rank_of_nearest_feature = np.empty(n, int)
    for i in range(n):
        feature_rank_of_nearest_geo[i] = 1 + int(np.sum(dx[i] < dx[i, ng[i]]))
        geo_rank_of_nearest_feature[i] = 1 + int(np.sum(geo[i] < geo[i, nx[i]]))
    out = pd.DataFrame({
        ID: list(ids),
        "is_labeled": np.arange(n) < N,
        "reference26_signed_error_private": np.r_[error, np.full(n-N, np.nan)],
        "reference26_absolute_error_private": np.r_[abs(error), np.full(n-N, np.nan)],
        "x_nn_id_private": np.asarray(ids)[nx],
        "spatial_nn_id_private": np.asarray(ids)[ng],
        "x_nn_labeled_id_private": np.r_[np.asarray(ids)[nx_known], np.full(n-N, "", dtype=object)],
        "spatial_nn_labeled_id_private": np.r_[np.asarray(ids)[ng_known], np.full(n-N, "", dtype=object)],
        "x_nn_distance_scaled": dx[np.arange(n), nx],
        "x_5nn_mean_distance_scaled": local_mean,
        "native_spatial_nn_distance_units_unspecified": geo[np.arange(n), ng],
        "feature_rank_of_spatial_nn": feature_rank_of_nearest_geo,
        "spatial_rank_of_feature_nn": geo_rank_of_nearest_feature,
        "mahalanobis_shrinkage": mahal,
        "local_outlier_factor": lof_scores,
        "isolation_anomaly_score": isolation_scores,
        "top10_reference26_error": np.isin(np.arange(n), top),
    })
    # Correlations over the 138 real labeled cases (exploratory, not independent).
    comparisons = []
    for var in ("x_nn_distance_scaled", "x_5nn_mean_distance_scaled",
                "native_spatial_nn_distance_units_unspecified",
                "feature_rank_of_spatial_nn", "mahalanobis_shrinkage",
                "local_outlier_factor", "isolation_anomaly_score"):
        cor, p = spearmanr(np.abs(error), out[var].to_numpy(float)[:N])
        comparisons.append({
            "feature_geometry_variable":var, "n_labeled":N,
            "spearman_rho_vs_absolute_reference26_error":float(cor),
            "nominal_p_unadjusted":float(p),
        })
    return out, pd.DataFrame(comparisons), top


def save_scatter(coords, errors, top, ids, output, *, method, mode):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9.5,7.5))
    ax.scatter(coords[N:,0], coords[N:,1], s=36, c="lightgray",
               marker="s", alpha=.8, label="59 without observed yield")
    if mode == "signed":
        lim = max(float(np.percentile(abs(errors), 98)), 0.05)
        points = ax.scatter(coords[:N,0], coords[:N,1], c=errors,
                            cmap="coolwarm", vmin=-lim, vmax=lim,
                            s=52, alpha=.85)
        colorname = "Signed error: predicted minus observed (t/ha)"
    else:
        points = ax.scatter(coords[:N,0], coords[:N,1], c=abs(errors),
                            cmap="viridis", s=52, alpha=.85)
        colorname = "Absolute Reference26 LOOCV error (t/ha)"
    ax.scatter(coords[top,0], coords[top,1], s=170, c="black",
               marker="x", linewidths=2.2, label="Top 10 absolute errors")
    for idx in top[:2]:
        ax.annotate(str(ids[idx]), coords[idx], xytext=(7, 5),
                    textcoords="offset points", fontsize=9, fontweight="bold")
    plt.colorbar(points, ax=ax, label=colorname, shrink=.8)
    ax.set_title(f"{method}: 197 original parcels, {mode} Reference26 LOOCV error")
    ax.set_xlabel("Axis 1 (projection units)")
    ax.set_ylabel("Axis 2 (projection units)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = output / f"{method.lower()}_{mode}_private.png"
    fig.savefig(path, dpi=175)
    plt.close(fig)


def run(args):
    frame, full, z, root, anchors, y, loo, features, ids138 = existing.loaded(args)
    ids = full[ID].astype(str).tolist()
    assert ids[:N] == ids138 and len(ids)==197 and len(set(ids))==197
    errors = loo["Reference26_mean"].to_numpy(float) - y
    if not np.isfinite(errors).all():
        raise ValueError("Genuine Reference26 LOOCV errors not finite.")
    file_dist = args.peer_repo.resolve()/"deployment/model_artifacts/distance197.npy"
    if not file_dist.is_file():
        raise FileNotFoundError(f"Need verified distance197 file: {file_dist}")
    spatial = np.load(file_dist, allow_pickle=False)
    if spatial.shape != (197,197) or not np.isfinite(spatial).all():
        raise ValueError("Bad 197x197 saved spatial distances.")
    diagnostics, correlations, top = neighbor_diagnostics(z, spatial, ids, errors)
    output = args.output.resolve()
    if args.check:
        print("[CHECK] 197 original feature vectors, 138 genuine signed OOF errors, 59 unlabeled.")
        print("[CHECK] Top-10 identified by |Reference26 mean OOF error|.")
        print("[CHECK] PCA, UMAP, t-SNE, signed/absolute plots, X/space nearest neighbors.")
        print("[CHECK] Native geographic distance units NOT assumed.")
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output.mkdir(parents=True, exist_ok=True)
    diagnostics.to_csv(output/"all_197_feature_outlier_private.csv", index=False)
    diagnostics.iloc[top].to_csv(output/"top10_feature_outlier_private.csv", index=False)
    correlations.to_csv(output/"error_distance_correlations.csv", index=False)
    projections = {"PCA": PCA(n_components=2).fit_transform(z)}
    projections["tSNE"] = TSNE(
        n_components=2, perplexity=30, random_state=args.seed,
        init="pca", learning_rate="auto", n_iter_without_progress=300,
    ).fit_transform(z)
    try:
        import umap
    except ImportError:
        (output/"umap_missing_dependency.txt").write_text(
            "Install umap-learn: conda install -c conda-forge umap-learn\n",
            encoding="utf-8",
        )
        print("[UMAP] Not installed: conda install -c conda-forge umap-learn")
    else:
        projections["UMAP"] = umap.UMAP(
            n_neighbors=15, min_dist=0.1, metric="euclidean",
            random_state=args.seed, n_components=2,
        ).fit_transform(z)
    coord = checked_coordinates(full, lat_col=args.lat_column,
                                lon_col=args.lon_column)
    if coord is None:
        projections["SpatialDistanceMDS"] = classical_mds(spatial)
        method = "Classical MDS of stored spatial distances (not a map)"
    else:
        projections["GeographicLonLat"] = coord
        method = "Geographic longitude and latitude"
    for name, points in projections.items():
        for mode in ("signed", "absolute"):
            save_scatter(points, errors, top, ids, output,
                         method=name, mode=mode)
    # Diagnostic: "near geographically but unusual in features" can be
    # quantified by feature rank of each parcel's nearest spatial neighbor.
    fig, ax = plt.subplots(figsize=(9,6))
    x = diagnostics["feature_rank_of_spatial_nn"].to_numpy(float)[:N]
    ax.scatter(x, abs(errors), s=42, alpha=.55)
    ax.scatter(x[top], abs(errors[top]), s=160, marker="x",
               c="black", label="Top 10 error")
    for i in top[:2]:
        ax.annotate(ids[i], (x[i], abs(errors[i])),
                    xytext=(5,5), textcoords="offset points")
    ax.set_xlabel("Feature-space rank of closest geographic neighbor (1=closest in X)")
    ax.set_ylabel("Absolute LOOCV error (t/ha)")
    ax.set_title("Are geographically near parcels far apart in their X?")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output/"error_vs_geographic_neighbor_feature_rank_private.png", dpi=175)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9,6))
    distances = diagnostics["x_5nn_mean_distance_scaled"].to_numpy(float)[:N]
    ax.scatter(distances, abs(errors), alpha=.5, s=40)
    ax.scatter(distances[top], abs(errors[top]), marker="x",
               c="black", s=155, label="Top 10 error")
    for i in top[:2]:
        ax.annotate(ids[i], (distances[i], abs(errors[i])),
                    xytext=(5,5), textcoords="offset points")
    ax.set_xlabel("Mean distance to five nearest parcels in robust-scaled X")
    ax.set_ylabel("Absolute LOOCV error (t/ha)")
    ax.set_title("Does feature-space isolation explain large model errors?")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output/"error_vs_x_5nn_private.png", dpi=175)
    plt.close(fig)
    manifest = {
        "n_observed":138, "n_unknown":59, "n_features":int(z.shape[1]),
        "coordinate_projection": method,
        "pca_tsne_umap_use_only_observable_197_X":True,
        "colored_138_only_from_authentic_reference26_mean_LOOCV":True,
        "top10_overlap_selected_by_highest_absolute_error":True,
        "spatial_distance_units":"unknown, do not report km",
        "umap_available":"UMAP" in projections,
        "outputs_are_private":True,
    }
    (output/"geometry_manifest.json").write_text(
        json.dumps(manifest, indent=2)+"\n", encoding="utf-8"
    )
    print(f"[GEOMETRY] {len(projections)} projections x2 error colors, "
          "two distance/error plots, 197 outlier-distance diagnostics.")
    print(correlations.to_string(index=False))
    print(f"[PRIVATE] Saved plots/tables: {output}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    option = p.add_mutually_exclusive_group(required=True)
    option.add_argument("--check", action="store_true")
    option.add_argument("--run", action="store_true")
    p.add_argument("--bundle", type=Path, default=existing.historic.native.BUNDLE)
    p.add_argument("--peer-repo", type=Path, default=existing.historic.native.PEER_REPO)
    p.add_argument("--config", type=Path, default=existing.historic.native.CONFIG)
    p.add_argument("--fresh-fold-dir", type=Path, default=existing.historic.native.OUTPUT)
    p.add_argument("--anchor-file", type=Path, default=existing.historic.ANCHOR_SOURCE)
    p.add_argument("--output", type=Path, default=OUTPUT)
    p.add_argument("--lat-column")
    p.add_argument("--lon-column")
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
