#!/usr/bin/env python3
"""Two Gaussian-bridge specialists + original Reference26 mean, competition-only.

The 138 Reference26 outer-LOOCV baseline predictions are REUSED, NOT refitted.
Synthetic observations are drawn ONLY from the two rare endpoint labels still
available in each outer training fold. Thus, if an endpoint is held out, both
synthetic specialists are disabled (the honest baseline is retained).
All outputs are local and private; no publication/deployment is performed.

  python tools/run_synthetic_bridge_three_regime.py --check
  python tools/run_synthetic_bridge_three_regime.py --smoke
  python tools/run_synthetic_bridge_three_regime.py --run
  python tools/run_synthetic_bridge_three_regime.py --summary
  python tools/run_synthetic_bridge_three_regime.py --predict59 --sigma 0.15 --gate-ratio 1.0 --blend 1.0
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

import run_reference26_compact14_final as native

ROOT = native.ROOT
ID = native.ID
TARGET = native.TARGET
N = 138
OUT = ROOT / "reports/synthetic_bridge_three_regime"
ANCHOR_SOURCE = ROOT / "reports/loocv_hard_parcels/diagnostic_manifest_private.json"
SEED = 20261019


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def representation(frame: pd.DataFrame, feature_names: list[str]):
    """Fixed X-only robust representation. 197 X allowed; no hidden y used."""
    if len(frame) != 197 or not feature_names:
        raise ValueError("Expected 197 X rows and nonempty frozen feature list.")
    data = frame.loc[:, feature_names].apply(pd.to_numeric, errors="coerce")
    array = data.to_numpy(float)
    if np.any(np.isinf(array)):
        raise ValueError("Input features contain infinity.")
    missing = np.isnan(array)
    valid = (~np.all(missing, axis=0)) & (
        np.nan_to_num(np.nanstd(array, axis=0)) > 1e-9
    )
    if int(valid.sum()) < 2:
        raise ValueError("Fewer than two informative numeric features.")
    array = array[:, valid]
    med = np.nanmedian(array, axis=0)
    array = np.where(np.isnan(array), med, array)
    mad = np.median(np.abs(array-med), axis=0) * 1.4826
    std = np.std(array, axis=0)
    scale = np.where(mad > 1e-8, mad, np.maximum(std, 1e-8))
    z = np.clip((array-med) / scale, -8., 8.)
    # Empirical covariance regularized toward isotropic variation.
    covariance = np.cov(z.T)
    covariance = 0.7 * covariance + 0.3 * np.eye(z.shape[1])
    values, vectors = np.linalg.eigh(covariance)
    root = (vectors * np.sqrt(np.maximum(values, 1e-10))) @ vectors.T
    return z, root, [x for x, keep in zip(feature_names, valid) if keep]


def brownian_bridge_samples(
    a: np.ndarray, b: np.ndarray, ya: float, yb: float,
    covariance_root: np.ndarray, *, sigma: float,
    paths: int, steps: int, seed: int,
):
    """Correlated Gaussian Brownian-bridge paths with endpoint-zero noise.

    Cov(eps(s),eps(t)) = 4*sigma**2*(min(s,t)-s*t)*Sigma.
    Both feature clouds are split at t=0.5; labels remain linear pseudo-y.
    """
    if sigma < 0 or paths < 1 or steps < 3 or steps % 2 == 0:
        raise ValueError("sigma >= 0, paths >= 1, odd steps >= 3 required.")
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape or a.ndim != 1 or not np.isfinite([ya, yb]).all():
        raise ValueError("Invalid anchor shape or labels.")
    d = a.size
    if covariance_root.shape != (d, d):
        raise ValueError("Wrong covariance root geometry.")
    t = np.linspace(0., 1., steps)
    rng = np.random.default_rng(seed)
    left_x, left_y, right_x, right_y = [], [], [], []
    for _ in range(paths):
        increments = rng.standard_normal((steps-1, d)) / np.sqrt(steps-1)
        w = np.vstack([np.zeros((1, d)), np.cumsum(increments, axis=0)])
        bridge = w - t[:, None] * w[-1]
        cloud = (1-t[:, None])*a + t[:, None]*b
        cloud += (2*sigma*bridge) @ covariance_root.T
        pseudo_y = (1-t)*ya + t*yb
        low = t <= 0.5
        high = t >= 0.5
        left_x.append(cloud[low]); left_y.append(pseudo_y[low])
        right_x.append(cloud[high]); right_y.append(pseudo_y[high])
    return (
        np.vstack(left_x), np.concatenate(left_y),
        np.vstack(right_x), np.concatenate(right_y)
    )


def specialist_prediction(
    z: np.ndarray, covariance_root: np.ndarray, y: np.ndarray,
    train: np.ndarray, queries: np.ndarray, anchors: tuple[int, int],
    *, sigma: float, paths: int, steps: int, alpha: float, seed: int,
):
    """Predict specialist and return gate geometry WITHOUT using heldout y."""
    train = np.asarray(train, int)
    queries = np.asarray(queries, int)
    if np.intersect1d(train, queries).size:
        raise ValueError("A query parcel is also a supervised training parcel.")
    if any(p not in train for p in anchors):
        return None   # no pseudo-label from a missing anchor, ever
    if alpha <= 0:
        raise ValueError("alpha must be positive.")
    a, b = anchors
    left_x, left_y, right_x, right_y = brownian_bridge_samples(
        z[a], z[b], y[a], y[b], covariance_root,
        sigma=sigma, paths=paths, steps=steps, seed=seed,
    )
    models = (
        Ridge(alpha=alpha).fit(left_x, left_y),
        Ridge(alpha=alpha).fit(right_x, right_y),
    )
    q = z[queries]
    direction = z[b]-z[a]
    length_sq = float(direction @ direction)
    if length_sq < 1e-10:
        return None
    projection = ((q-z[a]) @ direction) / length_sq
    midpoint_split = projection > 0.5
    estimated = np.where(
        midpoint_split, models[1].predict(q), models[0].predict(q),
    )
    generated = np.vstack([left_x, right_x])
    dist_syn = pairwise_distances(q, generated).min(axis=1) / np.sqrt(z.shape[1])
    others = np.array([p for p in train if p not in anchors], dtype=int)
    if not len(others):
        return None
    dist_real = pairwise_distances(q, z[others]).min(axis=1) / np.sqrt(z.shape[1])
    if len(others) < 2:
        support = max(float(np.median(dist_real)), 1e-6)
    else:
        d = pairwise_distances(z[others]) / np.sqrt(z.shape[1])
        np.fill_diagonal(d, np.inf)
        support = max(float(np.median(d.min(axis=1))), 1e-6)
    return {
        "expert": estimated, "synthetic_distance": dist_syn,
        "ordinary_distance": dist_real, "support": support,
        "projection": projection,
        "expert_group": np.where(midpoint_split, 2, 1),
    }


def apply_gate(base, component, *, ratio: float, radius: float, blend: float):
    """Only replace baseline when the synthetic cloud has clear X-only support."""
    base = np.asarray(base, float)
    if component is None:
        return base.copy(), np.zeros(len(base), dtype=bool)
    if ratio <= 0 or radius <= 0 or not 0 <= blend <= 1:
        raise ValueError("Positive gate thresholds and blend in [0,1] required.")
    active = (
        (component["synthetic_distance"] <
         ratio * np.maximum(component["ordinary_distance"], 1e-10))
        & (component["synthetic_distance"] <= radius*component["support"])
        & (component["projection"] >= -0.1)
        & (component["projection"] <= 1.1)
    )
    result = base + blend * active * (component["expert"]-base)
    return result, active


def metrics(y, predictions, base, ordinary_mask):
    e = np.asarray(predictions)-y
    ref = base-y
    delta = np.asarray(predictions)-base
    return {
        "rmse_138": float(np.sqrt(np.mean(e*e))),
        "rmse_reference_138": float(np.sqrt(np.mean(ref*ref))),
        "delta_mse_138": float(np.mean(e*e-ref*ref)),
        "rmse_136_ordinary": float(np.sqrt(np.mean(e[ordinary_mask]**2))),
        "rmse_ref_136_ordinary": float(
            np.sqrt(np.mean(ref[ordinary_mask]**2))
        ),
        "mae_136_ordinary": float(np.mean(np.abs(e[ordinary_mask]))),
        "mae_ref_136_ordinary": float(np.mean(np.abs(ref[ordinary_mask]))),
        "changed_ordinary": int(np.sum(np.abs(delta[ordinary_mask]) > 1e-10)),
        "worsened_ordinary": int(np.sum(
            (e[ordinary_mask]**2-ref[ordinary_mask]**2) > 1e-10
        )),
        "mean_absolute_change_ordinary": float(
            np.mean(np.abs(delta[ordinary_mask]))
        ),
        "max_absolute_change_ordinary": float(
            np.max(np.abs(delta[ordinary_mask]))
        ),
    }


def verified_inputs(args):
    """Load authentic 197 X and historic 138-fold mean Reference26 predictions."""
    frame, groups, distance, manifest, ids, _, _ = native.load_inputs(
        args.bundle.resolve(), args.peer_repo.resolve()
    )
    full, _ = native.prepare_197(args.peer_repo.resolve(), frame, distance)
    anchor_file = args.anchor_file.resolve()
    if not anchor_file.is_file():
        raise FileNotFoundError(
            f"Missing private hard-parcel diagnostic: {anchor_file}; "
            "run tools/analyze_loocv_hard_parcels.py --run first."
        )
    anchor_data = json.loads(anchor_file.read_text(encoding="utf-8"))
    anchor_ids = anchor_data.get("two_highest_consensus_private", [])
    if (not isinstance(anchor_ids, list) or len(anchor_ids) != 2 or
        len(set(anchor_ids)) != 2 or not set(anchor_ids).issubset(ids)):
        raise ValueError("Private manifest must contain two distinct known parcel IDs.")
    y = frame[TARGET].to_numpy(float)
    weights = native.frozen_weights(args.config.resolve())
    loo = native.read_completed(args.fresh_fold_dir.resolve(), ids, y, weights)
    assert np.array_equal(loo[ID].astype(str).to_numpy(), np.asarray(ids))
    z, root, selected = representation(full, list(manifest["features"]))
    indices = tuple(ids.index(i) for i in anchor_ids)
    return frame, full, z, root, indices, y, loo, selected


def main():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    for key in ("check", "smoke", "run", "summary", "predict59"):
        mode.add_argument("--"+key, action="store_true")
    p.add_argument("--anchor-file", type=Path, default=ANCHOR_SOURCE)
    p.add_argument("--bundle", type=Path, default=native.BUNDLE)
    p.add_argument("--peer-repo", type=Path, default=native.PEER_REPO)
    p.add_argument("--config", type=Path, default=native.CONFIG)
    p.add_argument("--fresh-fold-dir", type=Path, default=native.OUTPUT)
    p.add_argument("--output", type=Path, default=OUT)
    p.add_argument("--paths", type=int, default=16)
    p.add_argument("--steps", type=int, default=17)
    p.add_argument("--alpha", type=float, default=5.)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--sigmas", default="0,0.05,0.15,0.30")
    p.add_argument("--gate-ratios", default="0.6,1.0,1.4")
    p.add_argument("--blends", default="0.5,1.0")
    p.add_argument("--radius", type=float, default=1.5)
    p.add_argument("--sigma", type=float, default=0.15)
    p.add_argument("--gate-ratio", type=float, default=1.0)
    p.add_argument("--blend", type=float, default=1.0)
    args = p.parse_args()
    out = args.output.resolve()
    if args.summary:
        file = out/"comparison_aggregate.csv"
        if not file.is_file():
            raise FileNotFoundError(f"Run --run first: {file}")
        print(pd.read_csv(file).sort_values("rmse_138").to_string(index=False))
        return 0
    frame, full, z, root, anchors, y, loo, selected = verified_inputs(args)
    base = loo["Reference26_mean"].to_numpy(float)
    if args.check:
        print(f"[CHECK] {len(selected)} numeric X-only variables; 197 rows, "
              "138 true 137/1 Reference26 predictions, two hard anchor IDs found.")
        print("[CHECK] No synthetic samples, folds or final predictions written.")
        return 0
    if args.predict59:
        file = args.fresh_fold_dir.resolve()/"final_artifacts/predictions59_private.csv"
        manifest = args.fresh_fold_dir.resolve()/"final_artifacts/final_manifest.json"
        if not file.is_file() or not manifest.is_file():
            raise FileNotFoundError("Verified original 138-fit private 59 predictions required.")
        saved = json.loads(manifest.read_text(encoding="utf-8"))
        if saved.get("n_train") != 138 or saved.get("n_unknown") != 59:
            raise ValueError("Wrong frozen final-model provenance.")
        original = pd.read_csv(file, dtype={ID: str})
        if (len(original) != 59 or
            not np.array_equal(original[ID].to_numpy(),
                               full.iloc[N:][ID].astype(str).to_numpy())):
            raise ValueError("59 baseline IDs do not match frozen feature order.")
        base59 = original["Reference26_mean_t_ha"].to_numpy(float)
        if not np.isfinite(base59).all():
            raise ValueError("Nonfinite base 59 predictions.")
        component = specialist_prediction(
            z, root, y, np.arange(N), np.arange(N,197), anchors,
            sigma=args.sigma, paths=args.paths, steps=args.steps,
            alpha=args.alpha, seed=args.seed,
        )
        result, active = apply_gate(
            base59, component, ratio=args.gate_ratio, radius=args.radius,
            blend=args.blend,
        )
        output = pd.DataFrame({
            ID: original[ID], "Reference26_mean_t_ha": base59,
            "Synthetic3_t_ha": result, "gate_active": active,
            "specialist_route": np.where(active, component["expert_group"], 0),
            "synthetic_distance": component["synthetic_distance"],
            "ordinary_distance": component["ordinary_distance"],
        })
        private = out/"predictions59_private.csv"
        private.parent.mkdir(parents=True, exist_ok=True)
        if private.exists():
            raise FileExistsError(f"Existing 59 predictions not overwritten: {private}")
        output.to_csv(private, index=False)
        print(f"[59] Private candidate predictions: {private}; "
              f"gate-active={int(active.sum())}/59; original baseline unchanged.")
        return 0
    sigmas = [float(v) for v in args.sigmas.split(",")]
    ratios = [float(v) for v in args.gate_ratios.split(",")]
    blends = [float(v) for v in args.blends.split(",")]
    if (not sigmas or not ratios or not blends
        or any(v < 0 for v in sigmas)
        or any(v <= 0 for v in ratios)
        or any(not (0 <= v <= 1) for v in blends)
        or args.radius <= 0 or args.paths < 1 or args.steps < 3):
        p.error("Invalid synthetic grid, gate or noise parameters.")
    count = 12 if args.smoke else N
    evaluation = np.arange(count)
    # Preserve the two anchors in a smoke experiment for missing-label checks.
    if args.smoke:
        evaluation = np.unique(np.r_[evaluation, anchors])
    else:
        out.mkdir(parents=True, exist_ok=True)
    spec = {
        "schema": 1,
        "mode": "existing Reference26 mean LOOCV + fold-safe synthetic gate",
        "seed": args.seed, "paths": args.paths, "steps": args.steps,
        "alpha": args.alpha, "sigmas": sigmas, "ratios": ratios,
        "blends": blends, "radius": args.radius,
        "training_labels_only_in_synthetic_bridge": True,
        "anchor_selection_from_prior_all_138_LOOCV_is_posthoc": True,
        "anchor_selection_manifest_sha": sha(args.anchor_file.resolve()),
        "input_bundle_sha": sha(args.bundle.resolve()/"manifest.json"),
        "original_LOOCV_run_spec_sha": sha(args.fresh_fold_dir.resolve()/"run_spec.json"),
        "synthetic_runner_sha": sha(Path(__file__).resolve()),
    }
    if not args.smoke:
        spec_file = out/"run_spec.json"
        if spec_file.is_file() and json.loads(spec_file.read_text()) != spec:
            raise ValueError("Run fingerprint changed; use a different --output.")
        spec_file.write_text(json.dumps(spec, indent=2)+"\n", encoding="utf-8")
    table = {}
    for sigma in sigmas:
        table[sigma] = {}
        for held in evaluation:
            train = np.delete(np.arange(N), held)
            component = specialist_prediction(
                z, root, y, train, np.asarray([held]), anchors,
                sigma=sigma, paths=args.paths, steps=args.steps,
                alpha=args.alpha, seed=args.seed+int(held),
            )
            table[sigma][int(held)] = component
    rows = []
    private = []
    known_y = y[evaluation]
    original = base[evaluation]
    ordinary = ~np.isin(evaluation, anchors)
    for sigma in sigmas:
        for ratio in ratios:
            for blend in blends:
                predictions = original.copy()
                selected_routes = np.zeros(len(evaluation), dtype=int)
                for k, held in enumerate(evaluation):
                    component = table[sigma][int(held)]
                    new, active = apply_gate(
                        np.asarray([original[k]]), component, ratio=ratio,
                        radius=args.radius, blend=blend,
                    )
                    predictions[k] = new[0]
                    if active[0]:
                        selected_routes[k] = int(component["expert_group"][0])
                record = {
                    "sigma": sigma, "gate_ratio": ratio,
                    "blend": blend, "radius": args.radius,
                    "n_heldouts": len(evaluation),
                    "n_real_labels_per_outer_fit": N-1,
                    **metrics(known_y, predictions, original, ordinary),
                }
                rows.append(record)
                if not args.smoke:
                    private.extend({
                        ID: frame.iloc[int(idx)][ID],
                        "observed": float(known_y[k]),
                        "Reference26_mean": float(original[k]),
                        "Synthetic3": float(predictions[k]),
                        "route": int(selected_routes[k]),
                        "sigma": sigma, "gate_ratio": ratio,
                        "blend": blend,
                    } for k, idx in enumerate(evaluation))
    comparison = pd.DataFrame(rows).sort_values("rmse_138")
    if args.smoke:
        print("[SMOKE] Small subset only; NOT a 138-fold validation.")
        print(comparison.head(10).to_string(index=False))
        return 0
    comparison.to_csv(out/"comparison_aggregate.csv", index=False)
    pd.DataFrame(private).to_csv(out/"parcel_predictions_private.csv", index=False)
    print(f"[FULL 138] {len(comparison)} configurations; "
          "existing Reference26 heldouts reused.")
    print(comparison.head(12).to_string(index=False))
    print("[CAUTION] Endpoint selection and grid comparison on these 138 labels "
          "are exploratory. Held-out endpoints revert to baseline. "
          "Synthetic samples are NOT independent labels.")
    print(f"[PRIVATE] Diagnostics saved in {out}; no final model promoted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
