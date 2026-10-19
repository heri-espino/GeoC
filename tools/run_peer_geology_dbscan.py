#!/usr/bin/env python3
"""Pre-LOOCV sensitivity study: original geological KMeans versus DBSCAN.

All five original champions are trained per outer fold. The DBSCAN version
replaces ONLY Reference26's geological mixing gate (and hence B1's 26 branch).
Its min_samples and radius factor are tuned with the outer TRAIN labels only.

Run:
  python tools/run_peer_geology_dbscan.py --check
  python tools/run_peer_geology_dbscan.py --smoke
  python tools/run_peer_geology_dbscan.py --run --folds 3

This is a small, exploratory same-parcel cross-fitted comparison, NOT a
138-fold LOOCV or a winner-selection procedure. Needs trusted peer code and
the private 26-input bundle in GeoC/data_n/.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans

from geocebada.evaluation.checkpoint08_local import predict_old
from geocebada.evaluation.peer_geology_dbscan import (
    DBSCAN_GRID,
    adaptive_radius,
    dbscan_refit_mode,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from run_peer_champions_loo import (  # noqa: E402
    BASELINE,
    ID,
    Y,
    _git,
    _hash,
    _load,
    _lock,
    _spec,
    _write_fold,
    resolve_peer_bundle,
)

OUTPUT = ROOT / "reports/peer_geology_dbscan"
METHODS = (
    BASELINE, "CT3", "Compact14",
    "Reference26_KMeans", "Reference26_DBSCAN",
    "B1_KMeans", "B1_DBSCAN",
)


def _geo_space(frame: pd.DataFrame, geology: dict) -> np.ndarray:
    from src.signal_role_allocation import RoleSpace

    space = RoleSpace(
        ["geologic_composition"], [], {"geologic_composition": geology}
    ).fit(frame)
    z = np.asarray(space.transform(frame), dtype=float)
    if z.ndim != 2 or z.shape[1] != 10 or not np.isfinite(z).all():
        raise ValueError("Expected exactly ten finite, transformed geological features.")
    return z


def preview_clusters(peer: Path, bundle: Path, out: Path) -> pd.DataFrame:
    """Target-free KMeans/DBSCAN geometry comparison, no yield model refits."""
    sys.path.insert(0, str(peer))
    from src.current_champions import load_bundle

    frame, _groups, _distance, manifest = load_bundle(
        bundle, verify_saved_models=False
    )
    if len(frame) != 138:
        raise ValueError("Expected 138 rows in peer data bundle.")
    z = _geo_space(frame, manifest["geology"])
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for k in (4, 8):
        labels = KMeans(n_clusters=k, n_init=5, random_state=2026).fit_predict(z)
        sizes = np.bincount(labels)
        rows.append({
            "method": "KMeans", "parameter": f"k={k}",
            "n_clusters": int(len(sizes)),
            "noise_fraction": 0.0,
            "smallest_cluster": int(sizes.min()),
            "largest_cluster": int(sizes.max()),
            "radius_geology_units": np.nan,
        })
    for m, factor in DBSCAN_GRID:
        radius = adaptive_radius(z, m, factor)
        labels = DBSCAN(eps=radius, min_samples=m).fit_predict(z)
        positive = labels[labels >= 0]
        counts = np.unique(positive, return_counts=True)[1]
        rows.append({
            "method": "DBSCAN", "parameter": f"min_samples={m};factor={factor}",
            "n_clusters": int(len(counts)),
            "noise_fraction": float(np.mean(labels == -1)),
            "smallest_cluster": int(counts.min()) if len(counts) else 0,
            "largest_cluster": int(counts.max()) if len(counts) else 0,
            "radius_geology_units": radius,
        })
    summary = pd.DataFrame(rows)
    _write_fold(out / "geological_cluster_diagnostics.csv", summary)
    # Independent coordinate-based diagnostic, NOT part of the Reference26 gate.
    # Haversine BallTree requires radians, with epsilon given in angular units.
    latlon = frame[["base_centroid_lat", "base_centroid_lon"]].to_numpy(float)
    if not np.isfinite(latlon).all():
        raise ValueError("Missing coordinates required for optional geographic audit.")
    coordinates = np.radians(latlon)
    geo_rows = []
    earth_km = 6371.0088
    for min_samples in (3, 5):
        for radius_km in (0.5, 1.0, 2.0, 5.0, 10.0):
            cluster = DBSCAN(
                eps=radius_km / earth_km, min_samples=min_samples,
                metric="haversine", algorithm="ball_tree",
            ).fit_predict(coordinates)
            valid = cluster[cluster >= 0]
            groups = np.unique(valid, return_counts=True)[1]
            geo_rows.append({
                "min_samples": min_samples,
                "radius_km": radius_km,
                "n_clusters": int(len(groups)),
                "noise_fraction": float(np.mean(cluster == -1)),
                "smallest_cluster": int(groups.min()) if len(groups) else 0,
                "largest_cluster": int(groups.max()) if len(groups) else 0,
            })
    geo_summary = pd.DataFrame(geo_rows)
    _write_fold(out / "coordinate_cluster_diagnostics.csv", geo_summary)
    print("[GEOLOGY] Clusters from ten geological inputs only; no y used.")
    print(summary.to_string(index=False), flush=True)
    print(
        "[GEOGRAPHY] Coordinate-only DBSCAN diagnostics (haversine, km); "
        "not used by the geological gate.",
        flush=True,
    )
    print(geo_summary.to_string(index=False), flush=True)
    print(
        "[GEOLOGY] min_samples=3/5 and 4 relative radius factors are exploratory "
        "candidates; their effects on yield RMSE must be validated separately.",
        flush=True,
    )
    return summary


def _spec_experiment(peer: Path, bundle: Path) -> dict:
    spec = _spec(peer, bundle)
    spec.update({
        "geology_gate_experiment": "KMeans4or8 versus nested_DBSCAN_grid",
        "dbscan_grid": [
            {"min_samples": m, "radius_factor": factor}
            for m, factor in DBSCAN_GRID
        ],
        "dbscan_src_sha256": _hash(
            ROOT / "src/geocebada/evaluation/peer_geology_dbscan.py"
        ),
        "script_sha256": _hash(ROOT / "tools/run_peer_geology_dbscan.py"),
        "scoring": "exploratory 137+1 same-parcel folds, not final LOOCV",
    })
    return spec


def _fold_result(
    position, frame, groups, distance, manifest, ids, baseline,
    refit_fold, predict_saved, out: Path,
) -> tuple[pd.DataFrame, dict]:
    held_id = ids[position]
    observed = float(frame.iloc[position][Y])
    train = np.delete(np.arange(138), position)
    record = {
        "protocol": "random",
        "repeat": 1,
        "fold": position + 1,
    }
    from geocebada.evaluation.peer_champions_loo import outer_record

    record.update(outer_record(position))
    masked = frame.copy(deep=True)
    masked.loc[masked.index[position], Y] = np.nan
    assert np.isfinite(masked.iloc[train][Y].to_numpy(float)).all()
    assert np.isnan(float(masked.iloc[position][Y]))
    query = np.array([position])

    original = refit_fold(
        masked, groups, distance, manifest["geology"], record,
        out / f"cache/kmeans_{position + 1:03}",
    )
    with dbscan_refit_mode() as dbscan_refit:
        modified = dbscan_refit(
            masked, groups, distance, manifest["geology"], record,
            out / f"cache/dbscan_{position + 1:03}",
        )
    if set(original["train_positions"]) != set(train):
        raise AssertionError("Original pipeline used unexpected training IDs.")
    if set(modified["train_positions"]) != set(train):
        raise AssertionError("DBSCAN pipeline used unexpected training IDs.")

    km = predict_saved(original, query, masked, distance)
    db = predict_saved(modified, query, masked, distance)
    for unchanged in ("CT3", "Compact14"):
        if not np.allclose(km[unchanged], db[unchanged], atol=1e-8, rtol=0):
            raise RuntimeError(
                f"{unchanged} changed under geological gate substitution."
            )
    known = [pid for pid in baseline.y.index.astype(str) if pid != held_id]
    local04d = float(
        predict_old(baseline, known, [held_id], {
            "k": 24, "alpha": 30.0, "power": 1.0,
        })[0]
    )
    row = {
        ID: held_id,
        "observed": observed,
        BASELINE: local04d,
        "CT3": float(km["CT3"][0]),
        "Compact14": float(km["Compact14"][0]),
        "Reference26_KMeans": float(km["Reference26"][0]),
        "Reference26_DBSCAN": float(db["Reference26"][0]),
        "B1_KMeans": float(km["B1"][0]),
        "B1_DBSCAN": float(db["B1"][0]),
    }
    if not np.isfinite(np.asarray([row[x] for x in METHODS], float)).all():
        raise RuntimeError("Nonfinite candidate prediction.")
    share = modified["outer_experts"]["raw"]["shares"]
    gate_info = {
        "heldout_parcel": held_id, "fold": position + 1,
        "original_kmeans_groups": (
            original["outer_experts"]["raw"]["shares"]["k"]
            if original["outer_experts"]["raw"]["shares"] else None
        ),
        "dbscan_selection": (
            {
                "min_samples": share["min_samples"],
                "radius_factor": share["radius_factor"],
                "radius": share["eps"],
                "n_clusters": share["n_clusters"],
                "noise_fraction_training": share["noise_fraction"],
                "inner_selection_scores": share["candidates"],
                "query_weight": float(share["gate"].weights(
                    share["space"].transform(masked.iloc[query])
                )[0]),
            }
            if share is not None else None
        ),
    }
    return pd.DataFrame([row]), gate_info


def _preview_results(output: Path, n_folds: int) -> None:
    paths = [output / f"fold_{i:03}.csv" for i in range(1, n_folds + 1)]
    if any(not p.is_file() for p in paths):
        raise ValueError("The requested exploratory folds are incomplete.")
    data = pd.concat([pd.read_csv(p, dtype={ID: str}) for p in paths], ignore_index=True)
    if data[ID].duplicated().any() or set(data.columns) != {ID, "observed", *METHODS}:
        raise ValueError("Comparison contains repeated parcels or invalid columns.")
    rows = []
    for model in METHODS:
        error = data[model].to_numpy(float) - data["observed"].to_numpy(float)
        rows.append({
            "model": model,
            "n_heldout_parcels": n_folds,
            "exploratory_rmse": float(np.sqrt(np.mean(error**2))),
            "exploratory_mae": float(np.mean(np.abs(error))),
            "mean_error": float(np.mean(error)),
        })
    scores = pd.DataFrame(rows)
    _write_fold(output / "preliminary_errors.csv", scores)
    _write_fold(output / "preliminary_predictions.csv", data)
    print("\n[DBSCAN] Side-by-side predictions, SAME held-out parcels:")
    print(data.to_string(index=False), flush=True)
    print("\n[DBSCAN] DESCRIPTIVE errors on just these parcels:")
    print(scores.to_string(index=False), flush=True)
    print(
        "[DBSCAN] NOT 138-fold LOOCV; these data cannot establish an "
        "overall winner or independent significance.",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    for command in ("check", "smoke", "run"):
        mode.add_argument(f"--{command}", action="store_true")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--peer-repo", type=Path, default=ROOT.parent / "todoseboso")
    parser.add_argument("--bundle", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if not 1 <= args.folds <= 10:
        parser.error("Pre-LOOCV comparison permits 1 to 10 folds, not all 138.")

    peer = args.peer_repo.resolve()
    bundle = resolve_peer_bundle(ROOT, args.bundle, peer)
    if not (peer / "src/current_champions.py").is_file():
        raise FileNotFoundError("Peer modeling source missing: clone todoseboso beside GeoC.")
    if _git(peer, "status", "--porcelain", "--", "src", "configs"):
        raise RuntimeError("Commit or clean peer source changes before comparison.")
    out = (args.output / "smoke" if args.smoke else args.output).resolve()
    print(f"[DBSCAN] Private bundle: {bundle}", flush=True)
    if args.check:
        preview_clusters(peer, bundle, out / "geometry")
        return 0

    (
        frame, groups, distance, manifest, ids, baseline,
        refit_fold, predict_saved,
    ) = _load(peer, bundle)
    spec = _spec_experiment(peer, bundle)
    _lock(out, spec)
    n = 1 if args.smoke else args.folds
    for pos in range(n):
        path = out / f"fold_{pos + 1:03}.csv"
        if path.exists():
            cached = pd.read_csv(path, dtype={ID: str})
            if (
                len(cached) != 1
                or set(cached.columns) != {ID, "observed", *METHODS}
                or str(cached.iloc[0][ID]) != ids[pos]
                or abs(float(cached.iloc[0]["observed"]) - float(frame.iloc[pos][Y]))
                > 1e-10
                or not np.isfinite(cached[list(METHODS)].to_numpy(float)).all()
            ):
                raise RuntimeError(f"Stale or corrupt cached experiment fold: {path}")
            print(f"[DBSCAN] Fold {pos + 1}/{n} already completed.", flush=True)
            continue
        pred, gate_details = _fold_result(
            pos, frame, groups, distance, manifest, ids, baseline,
            refit_fold, predict_saved, out,
        )
        _write_fold(path, pred)
        details = out / f"gate_{pos + 1:03}.json"
        details.write_text(json.dumps(gate_details, indent=2), encoding="utf-8")
        print(f"[DBSCAN] Finished fold {pos + 1}/{n} ({ids[pos]}).", flush=True)
    _preview_results(out, n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
