#!/usr/bin/env python3
"""Two genuine synthetic Brownian-bridge GROUPS + unchanged Reference26 mean.

Group A: the 2 common worst-LOOCV parcels.
Group B: the OTHER 6 common top-10 parcels across 3 model families.
Group C: all other parcels, handled by the existing Reference26 baseline.

For group B, order real anchors along a deterministic first principal axis
and connect the 6 anchors via 5 consecutive Brownian-bridge segments.
All pseudo-y are interpolated FROM TRAINING LABELS ONLY. Each heldout is
removed from both bridge construction and routing-reference neighborhoods.

Requires existing private authentic 138x (137/1) Reference26 mean LOOCV folds.
The group selection used historical 138-fold errors: POST-HOC exploratory!

Usage:
  python tools/run_two_bridge_three_regime.py --check
  python tools/run_two_bridge_three_regime.py --smoke
  python tools/run_two_bridge_three_regime.py --run
  python tools/run_two_bridge_three_regime.py --summary
  python tools/run_two_bridge_three_regime.py --predict59 --sigma 0.05 --gate-ratio 0.8 --blend 0.25
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import pairwise_distances

import run_synthetic_bridge_three_regime as old

ROOT = old.ROOT
ID = old.ID
N = old.N
OUTPUT = ROOT / "reports/two_bridge_three_regime"
SEED = 20261019


def read_private_groups(path: Path, ids: list[str]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing private diagnostic manifest: {path}")
    record = json.loads(path.read_text(encoding="utf-8"))
    first = record.get("intersection_top2_across_families_private")
    hard10 = record.get("intersection_top10_across_families_private")
    if not isinstance(first, list) or not isinstance(hard10, list):
        raise ValueError("Private diagnostic must have exact top-2/top-10 intersections.")
    if (len(first) != 2 or len(set(first)) != 2 or
            len(hard10) != 8 or len(set(hard10)) != 8 or
            not set(first).issubset(hard10)):
        raise ValueError("Expected two common top-2 and eight common top-10 IDs.")
    second = sorted(set(hard10) - set(first))
    if not set(first + second).issubset(ids):
        raise ValueError("One or more private anchor IDs absent from 138 labels.")
    return tuple(ids.index(value) for value in first), tuple(
        ids.index(value) for value in second
    )


def chain_pairs(points: np.ndarray, positions: tuple[int, ...]) -> list[tuple[int, int]]:
    """A single piecewise bridge visiting every group B point once."""
    if len(positions) < 2 or len(set(positions)) != len(positions):
        return []
    block = points[np.asarray(positions)]
    centered = block-block.mean(axis=0, keepdims=True)
    if np.linalg.norm(centered) < 1e-10:
        ordered = sorted(positions)
    else:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        axis = vh[0]
        lead = int(np.argmax(np.abs(axis)))
        if axis[lead] < 0:
            axis = -axis
        coordinate = centered @ axis
        order = np.lexsort((np.asarray(positions), coordinate))
        ordered = [positions[int(k)] for k in order]
    return list(zip(ordered[:-1], ordered[1:]))


def unique_bridge(
    x1, x2, y1, y2, covariance_root, *,
    sigma: float, paths: int, steps: int, seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """One bridge with exactly paths*steps rows (no duplicated midpoints)."""
    left, left_y, right, right_y = old.brownian_bridge_samples(
        x1, x2, y1, y2, covariance_root,
        sigma=sigma, paths=paths, steps=steps, seed=seed,
    )
    half = steps//2 + 1
    right = right.reshape(paths, half, -1)[:, 1:, :].reshape(-1, len(x1))
    right_y = right_y.reshape(paths, half)[:, 1:].reshape(-1)
    return np.vstack([left, right]), np.r_[left_y, right_y]


def synthetic_group(
    z: np.ndarray, root: np.ndarray, y: np.ndarray,
    train: np.ndarray, anchors: tuple[int, ...], *,
    sigma: float, paths: int, steps: int, seed: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Pure augmentation of training-only anchor labels, no current test y."""
    allowed = set(map(int, train))
    active = tuple(i for i in anchors if i in allowed)
    links = chain_pairs(z, active)
    if not links:
        return np.empty((0, z.shape[1])), np.empty(0), 0
    clouds, labels = [], []
    for number, (a, b) in enumerate(links):
        x_syn, y_syn = unique_bridge(
            z[a], z[b], float(y[a]), float(y[b]), root,
            sigma=sigma, paths=paths, steps=steps, seed=seed+number,
        )
        clouds.append(x_syn)
        labels.append(y_syn)
    return np.vstack(clouds), np.concatenate(labels), len(links)


def fit_predict(
    z: np.ndarray, root: np.ndarray, y: np.ndarray, train: np.ndarray,
    query: np.ndarray, a: tuple[int, ...], b: tuple[int, ...], *,
    sigma: float, paths: int, steps: int, alpha: float, seed: int,
):
    """Two separate synthetic Ridge experts; X-only geometry selects a route."""
    train = np.asarray(train, int)
    query = np.asarray(query, int)
    if np.intersect1d(train, query).size or not len(train):
        raise ValueError("Invalid heldout/train overlap or empty training set.")
    if not np.isfinite(z).all() or not np.isfinite(y[train]).all():
        raise ValueError("Nonfinite X or training yields.")
    if alpha <= 0:
        raise ValueError("Ridge alpha must be positive.")
    ordinary = np.asarray([i for i in train if i not in set(a+b)], int)
    if len(ordinary) < 2:
        raise ValueError("Not enough ordinary real training support.")
    ds = pairwise_distances(z[ordinary]) / np.sqrt(z.shape[1])
    np.fill_diagonal(ds, np.inf)
    support = max(float(np.median(ds.min(axis=1))), 1e-7)
    ordinary_dist = pairwise_distances(
        z[query], z[ordinary]
    ).min(axis=1) / np.sqrt(z.shape[1])

    experts = []
    counts = []
    links = []
    for group, indices in enumerate((a, b), 1):
        cloud, pseudo_y, n_links = synthetic_group(
            z, root, y, train, indices, sigma=sigma, paths=paths,
            steps=steps, seed=seed+group*100000,
        )
        counts.append(len(pseudo_y))
        links.append(n_links)
        if len(pseudo_y):
            ridge = Ridge(alpha=alpha).fit(cloud, pseudo_y)
            dist = pairwise_distances(
                z[query], cloud
            ).min(axis=1) / np.sqrt(z.shape[1])
            prediction = ridge.predict(z[query])
        else:
            dist = np.full(len(query), np.inf)
            prediction = np.full(len(query), np.nan)
        experts.append((dist, prediction))

    distances = np.column_stack([experts[0][0], experts[1][0]])
    chosen = np.argmin(distances, axis=1)
    route = chosen + 1
    nearest = distances[np.arange(len(query)), chosen]
    prediction = np.column_stack([
        experts[0][1], experts[1][1]
    ])[np.arange(len(query)), chosen]
    return {
        "route": route, "expert": prediction, "synthetic_distance": nearest,
        "ordinary_distance": ordinary_dist, "support": support,
        "links_a": links[0], "links_b": links[1],
        "samples_a": counts[0], "samples_b": counts[1],
    }


def gated(base: np.ndarray, result: dict, *, ratio: float, radius: float,
          blend: float, max_change: float):
    if ratio <= 0 or radius <= 0 or max_change <= 0 or not (0 <= blend <= 1):
        raise ValueError("Invalid positive gate threshold, cap or blend.")
    base = np.asarray(base, float)
    active = (
        np.isfinite(result["synthetic_distance"])
        & np.isfinite(result["expert"])
        & (result["synthetic_distance"] <
           ratio * np.maximum(result["ordinary_distance"], 1e-12))
        & (result["synthetic_distance"] <= radius*result["support"])
    )
    correction = np.where(active, result["expert"]-base, 0.)
    correction = np.clip(correction*blend, -max_change, max_change)
    pred = base + correction
    routes = np.where(np.abs(correction) > 1e-12, result["route"], 0)
    return pred, routes


def scored(y, base, predictions, routes, groups, *, prefix=""):
    y, base, predictions, routes = [
        np.asarray(v) for v in (y, base, predictions, routes)
    ]
    errors = predictions-y
    old = base-y
    delta = predictions-base
    rows = {
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "rmse_reference": float(np.sqrt(np.mean(old**2))),
        "delta_mse": float(np.mean(errors**2-old**2)),
        "mae": float(np.mean(np.abs(errors))),
        "n_changed": int(np.count_nonzero(np.abs(delta)>1e-12)),
        "n_route_a": int(np.count_nonzero(routes==1)),
        "n_route_b": int(np.count_nonzero(routes==2)),
    }
    labels = np.asarray(groups)
    for label, name in ((1, "a2"), (2, "b6"), (0, "ordinary130")):
        mask = labels == label
        if not mask.any():
            rows[f"rmse_{name}"] = np.nan
            rows[f"baseline_rmse_{name}"] = np.nan
            rows[f"changed_{name}"] = 0
            rows[f"worse_{name}"] = 0
            continue
        rows[f"rmse_{name}"] = float(np.sqrt(np.mean(errors[mask]**2)))
        rows[f"baseline_rmse_{name}"] = float(np.sqrt(np.mean(old[mask]**2)))
        rows[f"changed_{name}"] = int(np.sum(np.abs(delta[mask])>1e-12))
        rows[f"worse_{name}"] = int(np.sum(
            errors[mask]**2 > old[mask]**2 + 1e-12
        ))
    return rows


def provenance(args, a, b):
    return {
        "schema": 1,
        "experiment": "two distinct Gaussian bridge rare-case groups + Reference26",
        "n_known_labels": N,
        "n_target": 59,
        "group_a_original_n": len(a),
        "group_b_original_n": len(b),
        "params": {
            "paths": args.paths, "steps": args.steps, "alpha": args.alpha,
            "sigmas": args.sigmas, "gate_ratios": args.gate_ratios,
            "blends": args.blends, "radius": args.radius,
            "max_change": args.max_change, "seed": args.seed,
        },
        "selection_is_posthoc": True,
        "inner_model_tuning_is_not_nested": True,
        "all_pseudo_y_from_available_outer_train_labels_only": True,
        "baseline_is_original_untouched_reference26_mean": True,
        "source_rare_group_private_sha256": old.sha(args.anchor_file.resolve()),
        "source_champion_bundle_sha256": old.sha(
            args.bundle.resolve()/"manifest.json"
        ),
        "source_reference26_fold_spec_sha256": old.sha(
            args.fresh_fold_dir.resolve()/"run_spec.json"
        ),
        "runner_sha256": old.sha(Path(__file__).resolve()),
    }


def verified_inputs(args):
    frame, full, z, root, _anchors, y, loo, features = old.verified_inputs(args)
    ids = frame[ID].astype(str).tolist()
    a, b = read_private_groups(args.anchor_file.resolve(), ids)
    return frame, full, z, root, y, loo, features, a, b


def run(args, *, smoke: bool):
    frame, full, z, root, y, loo, features, a, b = verified_inputs(args)
    sigmas = tuple(float(v) for v in args.sigmas.split(","))
    ratios = tuple(float(v) for v in args.gate_ratios.split(","))
    blends = tuple(float(v) for v in args.blends.split(","))
    if (not sigmas or not ratios or not blends
        or any(not np.isfinite(v) or v < 0 for v in sigmas)
        or any(not np.isfinite(v) or v <= 0 for v in ratios)
        or any(not np.isfinite(v) or v < 0 or v > 1 for v in blends)
        or args.paths <= 0 or args.steps < 3 or args.steps % 2 == 0
        or args.alpha <= 0 or args.radius <= 0 or args.max_change <= 0):
        raise ValueError("Invalid synthetic hyperparameters.")
    out = args.output.resolve()
    folds = (sorted(set(range(12)) | set(a) | set(b)) if smoke
             else list(range(N)))
    ys = y[folds]
    base = loo["Reference26_mean"].to_numpy(float)[folds]
    group_labels = np.asarray([
        1 if pos in a else (2 if pos in b else 0) for pos in folds
    ], int)
    specs = provenance(args, a, b)
    if not smoke:
        out.mkdir(parents=True, exist_ok=True)
        p = out/"run_spec.json"
        if p.is_file() and json.loads(p.read_text(encoding="utf-8")) != specs:
            raise ValueError("Changed fingerprint: use a new --output directory.")
        p.write_text(json.dumps(specs, indent=2)+"\n", encoding="utf-8")
    records = []
    private = []
    for sigma in sigmas:
        components = []
        for pos in folds:
            train = np.delete(np.arange(N), pos)
            component = fit_predict(
                z, root, y, train, np.asarray([pos]), a, b,
                sigma=sigma, paths=args.paths, steps=args.steps,
                alpha=args.alpha, seed=args.seed+pos,
            )
            components.append(component)
        for ratio in ratios:
            for blend in blends:
                forecast = base.copy()
                route = np.zeros(len(folds), int)
                for j, cmp in enumerate(components):
                    new, chosen = gated(
                        base[j:j+1], cmp, ratio=ratio, radius=args.radius,
                        blend=blend, max_change=args.max_change,
                    )
                    forecast[j], route[j] = new[0], chosen[0]
                record = {
                    "sigma": sigma, "gate_ratio": ratio, "blend": blend,
                    "radius": args.radius, "max_change": args.max_change,
                    "n_heldout": len(folds), "n_original_labels_per_fold": 137,
                    "samples_a_when_full": args.paths*args.steps,
                    "samples_b_when_full": 5*args.paths*args.steps,
                    **scored(ys, base, forecast, route, group_labels),
                }
                records.append(record)
                if not smoke:
                    private.extend({
                        ID: frame.iloc[pos][ID],
                        "observed": float(ys[j]),
                        "Reference26_mean": float(base[j]),
                        "TwoBridge": float(forecast[j]),
                        "route": int(route[j]),
                        "sigma": sigma, "gate_ratio": ratio, "blend": blend,
                    } for j, pos in enumerate(folds))
    aggregate = pd.DataFrame(records).sort_values(
        ["rmse", "changed_ordinary130", "sigma"], kind="stable"
    )
    if smoke:
        print(f"[SMOKE] {len(folds)} distinct heldouts, "
              "including every real specialist anchor; NOT full LOOCV.")
        print(aggregate.head(12).to_string(index=False))
        return
    aggregate.to_csv(out/"comparison_aggregate.csv", index=False)
    pd.DataFrame(private).to_csv(out/"parcel_predictions_private.csv", index=False)
    print(f"[FULL] {len(folds)} honest outer 137/1 reference predictions, "
          f"{len(aggregate)} grid configurations. Two original groups "
          f"({len(a)} and {len(b)}); {args.paths*args.steps}+"
          f"{5*args.paths*args.steps}={6*args.paths*args.steps} "
          "synthetic rows with all anchors present.")
    print(aggregate.head(20).to_string(index=False))
    print("[CAUTION] Groups were DISCOVERED from these same historical "
          "138 out-of-fold errors. This is POST-HOC comparison, "
          "not an independently selected unbiased model. "
          "All generated pseudo-labels are based on AVAILABLE TRAINING y.")
    print(f"[PRIVATE] {out}; do not commit per-parcel files.")


def predict59(args):
    frame, full, z, root, y, loo, features, a, b = verified_inputs(args)
    original = args.fresh_fold_dir.resolve()/"final_artifacts/predictions59_private.csv"
    manifest = args.fresh_fold_dir.resolve()/"final_artifacts/final_manifest.json"
    if not original.is_file() or not manifest.is_file():
        raise FileNotFoundError("Original verified 138-fitted mean Reference26 59 predictions missing.")
    record = json.loads(manifest.read_text(encoding="utf-8"))
    if record.get("n_train") != N or record.get("n_unknown") != 59:
        raise ValueError("Original 59 artifact lacks expected schema.")
    stored = pd.read_csv(original, dtype={ID: str})
    targetids = full.iloc[N:][ID].astype(str).to_numpy()
    if (len(stored) != 59 or
        not np.array_equal(stored[ID].astype(str).to_numpy(), targetids)):
        raise ValueError("Unknown 59 IDs/order mismatch.")
    base = stored["Reference26_mean_t_ha"].to_numpy(float)
    component = fit_predict(
        z, root, y, np.arange(N), np.arange(N, 197), a, b,
        sigma=args.sigma, paths=args.paths, steps=args.steps,
        alpha=args.alpha, seed=args.seed,
    )
    preds, routes = gated(
        base, component, ratio=args.gate_ratio, radius=args.radius,
        blend=args.blend, max_change=args.max_change,
    )
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    path = out/"predictions59_private.csv"
    if path.exists():
        raise FileExistsError(
            f"Existing private candidate: {path}. Supply a NEW --output."
        )
    pd.DataFrame({
        ID: targetids, "Reference26_mean_t_ha": base,
        "TwoBridge_t_ha": preds, "specialist_route": routes,
        "difference_t_ha": preds-base,
    }).to_csv(path, index=False)
    print(f"[59] Route A={int(np.sum(routes==1))}, "
          f"route B={int(np.sum(routes==2))}, "
          f"unchanged={int(np.sum(routes==0))}; "
          f"private candidate saved to {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    actions = p.add_mutually_exclusive_group(required=True)
    for name in ("check", "smoke", "run", "summary", "predict59"):
        actions.add_argument("--"+name, action="store_true")
    p.add_argument("--anchor-file", type=Path, default=old.ANCHOR_SOURCE)
    p.add_argument("--bundle", type=Path, default=old.native.BUNDLE)
    p.add_argument("--peer-repo", type=Path, default=old.native.PEER_REPO)
    p.add_argument("--config", type=Path, default=old.native.CONFIG)
    p.add_argument("--fresh-fold-dir", type=Path, default=old.native.OUTPUT)
    p.add_argument("--output", type=Path, default=OUTPUT)
    p.add_argument("--paths", type=int, default=16)
    p.add_argument("--steps", type=int, default=17)
    p.add_argument("--alpha", type=float, default=5.)
    p.add_argument("--sigmas", default="0,0.05,0.15")
    p.add_argument("--gate-ratios", default="0.5,0.8,1.0")
    p.add_argument("--blends", default="0.25,0.5")
    p.add_argument("--radius", type=float, default=1.5)
    p.add_argument("--max-change", type=float, default=0.35)
    p.add_argument("--sigma", type=float, default=0.05)
    p.add_argument("--gate-ratio", type=float, default=0.8)
    p.add_argument("--blend", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args()
    out = args.output.resolve()
    if args.summary:
        file = out/"comparison_aggregate.csv"
        if not file.is_file():
            raise FileNotFoundError(f"Run --run first: {file}")
        print(pd.read_csv(file).sort_values("rmse").to_string(index=False))
        return 0
    if args.check:
        frame, full, z, root, y, loo, features, a, b = verified_inputs(args)
        print(f"[CHECK] Original 138 genuine Reference26 mean heldouts and "
              f"197 X-only rows verified. Group A: {len(a)} original observations; "
              f"Group B: {len(b)} original observations; general: {N-len(a)-len(b)}.")
        print(f"[CHECK] Brownian bridge rows, when all anchors available: "
              f"A={args.paths*args.steps}, B={5*args.paths*args.steps}, "
              f"total={6*args.paths*args.steps}. "
              "Synthetic rows are not independent true yields.")
        return 0
    if args.predict59:
        predict59(args)
    else:
        run(args, smoke=args.smoke)


if __name__ == "__main__":
    main()
