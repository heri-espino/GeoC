#!/usr/bin/env python3
"""Global regression on real labels + MANY synthetic rare-case observations.

No specialist routing, no classifier. Compare each global model (Ridge or
ExtraTrees) fitted with 137 REAL fold-training labels alone vs the SAME model
fitted with 137 real labels + 256/1024 artificial observations.

Augmentation modes:
  repeat: real anchor oversampling, no synthetic information (control).
  local: local Gaussian jitter around each available rare anchor.
  bridge: correlated Gaussian Brownian-bridge between the two available
          rare anchors; when one is held out, fall back to local jitter.
  mixed: half local and half bridge.

Artificial yields are copied/interpolated from available TRAIN labels only.
Full 138-fold Reference26 mean OOF is a separate frozen comparison.
Endpoints are POST-HOC selected from previously inspected CV errors.
NO claims about independent accuracy, extra true labels, or 59-target y.

  python tools/run_global_synthetic_augmentation.py --check
  python tools/run_global_synthetic_augmentation.py --smoke
  python tools/run_global_synthetic_augmentation.py --run --jobs 4
  python tools/run_global_synthetic_augmentation.py --summary
  python tools/run_global_synthetic_augmentation.py --plot
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Avoid thread oversubscription on Windows/VM.
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import pairwise_distances

import run_synthetic_bridge_three_regime as historic

ROOT = historic.ROOT
ID = historic.ID
TARGET = historic.TARGET
N = historic.N
OUTPUT = ROOT / "reports/global_synthetic_augmentation"
SEED = 20261019
STATE = None


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_settings(args):
    methods = tuple(v.strip() for v in args.methods.split(",") if v.strip())
    sizes = tuple(int(v) for v in args.sizes.split(","))
    masses = tuple(v.strip() for v in args.masses.split(","))
    models = tuple(v.strip() for v in args.models.split(","))
    if (not methods or set(methods) - {"repeat", "local", "bridge", "mixed", "gaussian_hard"}
            or len(set(methods)) != len(methods)
            or not sizes or len(set(sizes)) != len(sizes)
            or min(sizes) < 1
            or not masses or len(set(masses)) != len(masses)
            or any(m != "full" and (not m.isdigit() or int(m) < 1) for m in masses)
            or not models or set(models) - {"ridge", "extra_trees"}
            or len(set(models)) != len(models)
            or args.sigma < 0 or args.local_neighbors < 3
            or args.ridge_alpha <= 0 or args.hard_centers < 2 or args.trees < 4
            or args.leaf < 1 or args.jobs < 1):
        raise ValueError("Invalid methods/sizes/masses/models/hyperparameters.")
    return methods, sizes, masses, models


def local_roots(z: np.ndarray, anchors: tuple[int, int], neighbors: int):
    """X-only local noise covariance estimated from all 197 observed X."""
    roots = {}
    for a in anchors:
        dist = np.sum((z-z[a])**2, axis=1)
        nearby = np.argsort(dist, kind="stable")[1:1+min(neighbors, len(z)-1)]
        centered = z[nearby] - z[nearby].mean(axis=0)
        cov = (centered.T@centered)/max(len(nearby)-1, 1)
        cov = 0.65*cov + 0.35*np.eye(z.shape[1])
        eigval, eigvec = np.linalg.eigh(cov)
        roots[a] = (eigvec*np.sqrt(np.maximum(eigval, 1e-10)))@eigvec.T
    return roots


def fold_hard_centers(
    z: np.ndarray, y: np.ndarray, train: np.ndarray, *,
    count: int, alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Select difficult TRAINING points by training-only Ridge PRESS residuals.

    A naive lookup of the saved 138 Reference26 LOOCV error ranking would
    contaminate each outer fold: those OTHER historic fits include its heldout
    label. Instead, PRESS computes exact leave-one-out Ridge residuals on
    only the 137 allowed training labels, with no outer heldout y access.
    This is a cheap surrogate hardness score, NOT Reference26's residual.
    """
    train = np.asarray(train, int)
    if (count < 2 or alpha <= 0 or len(train) < 3
            or np.unique(train).size != len(train)):
        raise ValueError("Require >=3 unique training indices, >=2 centers, alpha>0.")
    if not np.isfinite(y[train]).all():
        raise ValueError("Non-finite training labels.")
    x = z[train]
    x_c = x-x.mean(axis=0, keepdims=True)
    response = y[train]
    y_c = response-response.mean()
    matrix = x_c.T@x_c + alpha*np.eye(x.shape[1])
    cross = np.linalg.solve(matrix, x_c.T)
    estimate = response.mean() + x_c @ (cross@y_c)
    leverage = 1./len(train) + np.sum(x_c*cross.T, axis=1)
    denominator = 1.-leverage
    if not np.isfinite(denominator).all() or np.min(denominator) <= 1e-8:
        raise ValueError("Degenerate PRESS leverage values.")
    residual = (response-estimate)/denominator
    # Reproducible tie-breaking by global index.
    order = np.lexsort((train, -np.abs(residual)))
    selected = order[:min(count, len(train))]
    return train[selected], residual[selected]


def gaussian_hard_samples(
    z: np.ndarray, y: np.ndarray, train: np.ndarray,
    positions: np.ndarray, roots: dict[int, np.ndarray],
    *, count: int, sigma: float, seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian X clouds centered at fold-local high-PRESS-error REAL points."""
    train_set = set(np.asarray(train, int).tolist())
    centers = np.asarray(positions, int)
    if (not len(centers) or not set(centers).issubset(train_set)
            or len(np.unique(centers)) != len(centers)
            or count <= 0 or sigma < 0):
        raise ValueError("Gaussian centers must be unique training labels.")
    if any(int(i) not in roots for i in centers):
        raise ValueError("Missing X-only neighborhood covariance for center.")
    rng = np.random.default_rng(seed)
    assigned = rng.choice(centers, size=count, replace=True)
    noise = rng.normal(size=(count, z.shape[1]))
    synthetic = np.empty((count, z.shape[1]), float)
    for index in centers:
        mask = assigned == index
        synthetic[mask] = (
            z[index] + sigma*(noise[mask]@roots[int(index)].T)
        )
    synthetic = np.clip(synthetic, -9., 9.)
    return synthetic, y[assigned].copy()


def augmented(
    z: np.ndarray, y: np.ndarray, train: np.ndarray, anchors: tuple[int, int],
    root: np.ndarray, roots: dict, *, mode: str, count: int,
    sigma: float, seed: int,
    hard_positions: np.ndarray | None = None,
    hard_roots: dict | None = None,
):
    """Return exactly count synthetic X/pseudo-y; never read heldout y."""
    train = np.asarray(train, dtype=int)
    usable = [a for a in anchors if int(a) in set(train)]
    if len(usable) < 1 or count < 1:
        raise ValueError("At least one rare anchor must remain in training.")
    rng = np.random.default_rng(seed)
    d = z.shape[1]

    def repeat(n):
        idx = rng.choice(usable, size=n, replace=True)
        return z[idx].copy(), y[idx].copy()

    def local(n):
        idx = rng.choice(usable, size=n, replace=True)
        noise = rng.normal(size=(n, d))
        output = np.empty((n, d), float)
        for a in usable:
            take = idx == a
            if take.any():
                output[take] = z[a] + sigma*(noise[take]@roots[a].T)
        return np.clip(output, -9., 9.), y[idx].copy()

    def bridge(n):
        if len(usable) < 2:
            return local(n), "bridge_fallback_local_one_anchor"
        a, b = usable
        steps = 17
        paths = math.ceil(n/steps)
        time_grid = np.linspace(0., 1., steps)
        x_parts, y_parts = [], []
        for _ in range(paths):
            increments = rng.normal(size=(steps-1, d))/np.sqrt(steps-1)
            w = np.vstack([np.zeros((1, d)), np.cumsum(increments, axis=0)])
            brownian = w-time_grid[:, None]*w[-1]
            x = (1-time_grid[:, None])*z[a] + time_grid[:, None]*z[b]
            x += 2*sigma*(brownian@root.T)
            pseudo = (1-time_grid)*y[a] + time_grid*y[b]
            x_parts.append(x); y_parts.append(pseudo)
        x_all = np.concatenate(x_parts, axis=0)
        y_all = np.concatenate(y_parts)
        choice = rng.permutation(len(y_all))[:n]
        return (np.clip(x_all[choice], -9., 9.), y_all[choice]), "bridge"

    if mode == "gaussian_hard":
        if hard_positions is None or hard_roots is None:
            raise ValueError("gaussian_hard requires fold-local PRESS centers.")
        x, ys = gaussian_hard_samples(
            z, y, train, hard_positions, hard_roots,
            count=count, sigma=sigma, seed=seed,
        )
        recipe = "gaussian_around_fold_local_hard_training_points"
    elif mode == "repeat":
        x, ys = repeat(count)
        recipe = "repeated_rare_anchor_labels"
    elif mode == "local":
        x, ys = local(count)
        recipe = "local_gaussian_near_rare_anchors"
    elif mode == "bridge":
        (x, ys), recipe = bridge(count)
    elif mode == "mixed":
        local_n = count//2
        lx, ly = local(local_n)
        (bx, by), recipe = bridge(count-local_n)
        x = np.concatenate([lx, bx])
        ys = np.concatenate([ly, by])
        recipe = "mixed_local_and_" + recipe
    else:
        raise ValueError(f"Unknown mode {mode}")
    if (x.shape != (count, d) or ys.shape != (count,)
            or not np.isfinite(x).all() or not np.isfinite(ys).all()):
        raise AssertionError("Invalid generated sample shape/values.")
    return x, ys, recipe


def regressor(kind: str, *, alpha: float, trees: int, leaf: int, seed: int):
    if kind == "ridge":
        return Ridge(alpha=alpha, fit_intercept=True)
    if kind == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=trees, min_samples_leaf=leaf,
            max_features=0.8, n_jobs=1, random_state=seed
        )
    raise ValueError(f"Unsupported regressor {kind}")


def train_predict(
    z, y, train, query, *, kind, alpha, trees, leaf, seed,
    synthetic=None, mass="full",
):
    if np.intersect1d(train, query).size:
        raise ValueError("Heldout X in supervised train set.")
    x = z[train]
    labels = y[train]
    weights = np.ones(len(train), float)
    if synthetic is not None:
        sx, sy = synthetic
        count = len(sy)
        total_weight = float(count if mass == "full" else int(mass))
        if count <= 0 or total_weight <= 0:
            raise ValueError("Invalid synthetic weight.")
        x = np.vstack([x, sx])
        labels = np.r_[labels, sy]
        weights = np.r_[weights, np.full(count, total_weight/count)]
    learner = regressor(kind, alpha=alpha, trees=trees, leaf=leaf, seed=seed)
    learner.fit(x, labels, sample_weight=weights)
    pred = np.asarray(learner.predict(z[query]), float)
    if pred.shape != (len(query),) or not np.isfinite(pred).all():
        raise ValueError("Nonfinite or invalid regression predictions.")
    return pred


def keys(settings):
    methods, sizes, masses, models = settings
    answer = [f"plain__{model}" for model in models]
    for model in models:
        for mode in methods:
            for n in sizes:
                for mass in masses:
                    answer.append(f"aug__{model}__{mode}__n{n}__m{mass}")
    return answer


def init_worker(payload):
    global STATE
    STATE = payload


def one_fold(pos: int):
    if STATE is None:
        raise RuntimeError("Worker not initialized.")
    z, y, anchors, root, roots, args_dict, settings, ids, ref = STATE
    train = np.delete(np.arange(N), pos)
    query = np.array([pos], int)
    # The real held-out label is ONLY written to the evaluation record.
    record = {ID: ids[pos], "observed": float(y[pos]),
              "Reference26_mean": float(ref[pos])}
    methods, sizes, masses, models = settings
    for model in models:
        record[f"plain__{model}"] = float(train_predict(
            z, y, train, query, kind=model,
            alpha=args_dict["ridge_alpha"], trees=args_dict["trees"],
            leaf=args_dict["leaf"], seed=args_dict["seed"]+pos,
        )[0])
    for mode in methods:
        if mode == "gaussian_hard":
            positions, press_errors = fold_hard_centers(
                z, y, train, count=args_dict["hard_centers"],
                alpha=args_dict["ridge_alpha"],
            )
            hard_roots = local_roots(
                z, tuple(int(v) for v in positions),
                args_dict["local_neighbors"],
            )
        else:
            positions, hard_roots = None, None
        for count in sizes:
            x, pseudo, recipe = augmented(
                z, y, train, anchors, root, roots, mode=mode,
                count=count, sigma=args_dict["sigma"],
                seed=args_dict["seed"]+pos+count*10+methods.index(mode)*100000,
                hard_positions=positions, hard_roots=hard_roots,
            )
            for mass in masses:
                for model in models:
                    column = f"aug__{model}__{mode}__n{count}__m{mass}"
                    record[column] = float(train_predict(
                        z, y, train, query, kind=model,
                        alpha=args_dict["ridge_alpha"],
                        trees=args_dict["trees"], leaf=args_dict["leaf"],
                        seed=args_dict["seed"]+pos,
                        synthetic=(x, pseudo), mass=mass,
                    )[0])
    return pos, record


def metrics(table, anchors, settings):
    y = table["observed"].to_numpy(float)
    ref = table["Reference26_mean"].to_numpy(float)
    indices = table[ID].astype(str).to_numpy()
    anchor_ids = set(str(v) for v in anchors)
    hard = np.isin(indices, list(anchor_ids))
    out = []
    for model in keys(settings):
        pred = table[model].to_numpy(float)
        e, ref_e = pred-y, ref-y
        kind = model.split("__")[1]
        plain = table[f"plain__{kind}"].to_numpy(float)
        plain_error = plain-y
        row = {
            "model": model,
            "rmse": float(np.sqrt(np.mean(e*e))),
            "mae": float(np.mean(abs(e))),
            "rmse_reference26": float(np.sqrt(np.mean(ref_e*ref_e))),
            "delta_mse_vs_reference26": float(np.mean(e*e-ref_e*ref_e)),
            "rmse_plain_same_learner": float(
                np.sqrt(np.mean(plain_error*plain_error))
            ),
            "delta_mse_vs_plain_same_learner": float(
                np.mean(e*e-plain_error*plain_error)
            ),
            "rmse_common_hard2": float(np.sqrt(np.mean(e[hard]**2))),
            "rmse_other136": float(np.sqrt(np.mean(e[~hard]**2))),
            "mae_other136": float(np.mean(abs(e[~hard]))),
            "worse_than_reference26_other136": int(np.sum(
                e[~hard]**2 > ref_e[~hard]**2+1e-12
            )),
            "improved_over_reference26": int(np.sum(
                e*e < ref_e*ref_e-1e-12
            )),
            "max_prediction_change_vs_reference26": float(
                np.max(abs(pred-ref))
            ),
            "n_original_labelled": len(y)-1,
            "n_cv_holdouts": len(y),
            "posthoc_anchor_selection": True,
        }
        out.append(row)
    return pd.DataFrame(out).sort_values(
        ["rmse", "delta_mse_vs_plain_same_learner"], kind="stable"
    )


def loaded(args):
    frame, full, z, root, anchor_positions, y, loo, features = (
        historic.verified_inputs(args)
    )
    ids = frame[ID].astype(str).tolist()
    anchors = tuple(int(a) for a in anchor_positions)
    if len(anchors) != 2 or len(set(anchors)) != 2:
        raise ValueError("Need precisely two distinct originally hard anchors.")
    return frame, full, z, root, anchors, y, loo, features, ids


def spec_for(args, settings):
    return {
        "schema": 1,
        "algorithm": "one_global_regression_model_138_true_labels_plus_synthetic",
        "settings": {
            "methods": settings[0], "sizes": settings[1],
            "masses": settings[2], "models": settings[3],
            "sigma": args.sigma, "ridge_alpha": args.ridge_alpha,
            "trees": args.trees, "leaf": args.leaf,
            "local_neighbors": args.local_neighbors,
            "hard_centers": args.hard_centers,
            "hard_centers_selected_inside_each_outer_training_fold": True,
            "hardness_method": "Ridge_PRESS_137_training_labels_only",
            "seed": args.seed,
        },
        "anchor_source_sha256": digest(args.anchor_file.resolve()),
        "champion_bundle_sha256": digest(args.bundle.resolve()/"manifest.json"),
        "original_reference26_run_spec_sha256": digest(
            args.fresh_fold_dir.resolve()/"run_spec.json"
        ),
        "runner_sha256": digest(Path(__file__).resolve()),
        "same_fold_real_plain_controls": True,
        "posthoc_extreme_id_selection": True,
        "synthetic_y_only_from_fold_train": True,
        "original_reference26_not_refitted": True,
    }


def make_plot(frame, full, z, loo, anchors, folder, *, hard_centers=8, alpha=30.):
    """Private: 197 X in PCA, 138 authentic signed residuals; 59 X no y."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fold = PCA(n_components=2).fit(z)
    coords = fold.transform(z)
    y = frame[TARGET].to_numpy(float)
    residual = loo["Reference26_mean"].to_numpy(float)-y
    limit = max(float(np.quantile(abs(residual), .95)), 1e-6)
    hard_indices, press_residuals = fold_hard_centers(
        z, y, np.arange(N), count=hard_centers, alpha=alpha,
    )
    fig, ax = plt.subplots(figsize=(11, 7))
    pts = ax.scatter(
        coords[:N, 0], coords[:N, 1], s=37, c=residual,
        cmap="coolwarm", vmin=-limit, vmax=limit, edgecolor="none"
    )
    ax.scatter(coords[N:, 0], coords[N:, 1], marker="x",
               s=48, c="black", alpha=.55, label="59 unknown X")
    ax.scatter(coords[list(anchors), 0], coords[list(anchors), 1],
               s=270, facecolors="none", edgecolors="black",
               linewidths=2.2, label="2 historic Reference26 hard anchors")
    ax.scatter(coords[hard_indices, 0], coords[hard_indices, 1],
               s=160, marker="s", facecolors="none", edgecolors="darkgreen",
               linewidths=1.6, label=f"{len(hard_indices)} fold-local Ridge PRESS centers")
    plt.colorbar(pts, ax=ax, label="Reference26 OOF signed residual yhat - y (t/ha)")
    ax.set_xlabel(f"PC1 ({fold.explained_variance_ratio_[0]:.1%} X variance)")
    ax.set_ylabel(f"PC2 ({fold.explained_variance_ratio_[1]:.1%} X variance)")
    ax.legend()
    ax.set_title("Real parcel X geometry and signed Reference26 errors")
    fig.tight_layout()
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder/"signed_residual_pca_private.png", dpi=170)
    plt.close(fig)
    # Direct signed-residual versus actual-y plot helps identify "high actual,
    # predicted middle" and the converse without inventing groups.
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(y, residual, s=35, alpha=.65)
    ax.scatter(y[list(anchors)], residual[list(anchors)], s=150,
               facecolors="none", edgecolors="black", linewidths=2)
    ax.axhline(0, color="black", lw=1)
    ax.set_xlabel("Observed yield (t/ha)")
    ax.set_ylabel("Reference26 OOF signed residual (predicted - observed)")
    ax.set_title("Tail compression / directional model errors")
    fig.tight_layout()
    fig.savefig(folder/"signed_residual_vs_y_private.png", dpi=170)
    plt.close(fig)
    # This is a FINAL-138 exploratory visualization, not outer-CV
    # ranking. Per-outer-fold selections are independently recomputed.
    pd.DataFrame({
        ID: frame.iloc[hard_indices][ID].astype(str).to_numpy(),
        "observed": y[hard_indices],
        "ridge_press_residual_signed": press_residuals,
        "reference26_oof_residual_signed": residual[hard_indices],
    }).to_csv(folder/"hard_centers_full138_private.csv", index=False)
    print(f"[PLOT] Saved private diagnostics and {len(hard_indices)} "
          f"Gaussian-centre candidates in {folder}")


def predict59(args, frame, full, z, root, anchors, y, ids):
    """One explicitly chosen global 138-real-plus-synthetic candidate, private."""
    methods, sizes, masses, models = parse_settings(args)
    method = args.predict_method
    size = args.predict_size
    mass = args.predict_mass
    model = args.predict_model
    if (method not in methods or size not in sizes or mass not in masses
            or model not in models):
        raise ValueError(
            "The prediction recipe must be one of the candidate CV settings."
        )
    final = args.fresh_fold_dir.resolve()/"final_artifacts"
    path = final/"predictions59_private.csv"
    manifest = final/"final_manifest.json"
    if not path.is_file() or not manifest.is_file():
        raise FileNotFoundError(
            "Need existing original audited final Reference26 59 predictions."
        )
    provenance = json.loads(manifest.read_text(encoding="utf-8"))
    if provenance.get("n_train") != N or provenance.get("n_unknown") != 59:
        raise ValueError("Original Reference26 final fit record mismatches 138/59.")
    saved = pd.read_csv(path, dtype={ID:str})
    expected = full.iloc[N:][ID].astype(str).to_numpy()
    if (len(saved) != 59 or saved[ID].duplicated().any()
            or not np.array_equal(saved[ID].to_numpy(), expected)):
        raise ValueError("Original 59 target IDs/order do not match 197 features.")
    baseline = saved["Reference26_mean_t_ha"].to_numpy(float)
    if not np.isfinite(baseline).all():
        raise ValueError("Original Reference26 target predictions are not finite.")
    roots = local_roots(z, anchors, args.local_neighbors)
    hard_positions, hard_roots = None, None
    if method == "gaussian_hard":
        hard_positions, press_residuals = fold_hard_centers(
            z, y, np.arange(N), count=args.hard_centers,
            alpha=args.ridge_alpha,
        )
        hard_roots = local_roots(
            z, tuple(int(v) for v in hard_positions),
            args.local_neighbors,
        )
    synthetic = augmented(
        z, y, np.arange(N), anchors, root, roots,
        mode=method, count=size, sigma=args.sigma,
        seed=args.seed+size*10+methods.index(method)*100000,
        hard_positions=hard_positions, hard_roots=hard_roots,
    )
    estimates = train_predict(
        z, y, np.arange(N), np.arange(N,197),
        kind=model, alpha=args.ridge_alpha,
        trees=args.trees, leaf=args.leaf, seed=args.seed,
        synthetic=synthetic[:2], mass=mass,
    )
    output = args.output.resolve()/"candidate59_private.csv"
    if output.exists():
        raise FileExistsError(
            f"Refusing to replace previous private 59 predictions: {output}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        ID:expected,
        "Reference26_mean_t_ha":baseline,
        "GlobalSynthetic_t_ha":estimates,
        "difference_t_ha":estimates-baseline,
        "model":model,
        "method":method, "synthetic_count":size,
        "synthetic_total_weight":size if mass=="full" else int(mass),
        "sigma":args.sigma,
    }).to_csv(output, index=False)
    print(f"[59] Saved private candidate global predictions: {output}")
    print("[59] This does NOT alter Reference26/joblib/website/final submission.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    action = p.add_mutually_exclusive_group(required=True)
    for arg in ("check", "smoke", "run", "summary", "plot", "predict59"):
        action.add_argument("--"+arg, action="store_true")
    p.add_argument("--anchor-file", type=Path, default=historic.ANCHOR_SOURCE)
    p.add_argument("--bundle", type=Path, default=historic.native.BUNDLE)
    p.add_argument("--peer-repo", type=Path, default=historic.native.PEER_REPO)
    p.add_argument("--config", type=Path, default=historic.native.CONFIG)
    p.add_argument("--fresh-fold-dir", type=Path, default=historic.native.OUTPUT)
    p.add_argument("--output", type=Path, default=OUTPUT)
    p.add_argument("--methods", default="repeat,local,bridge,mixed,gaussian_hard")
    p.add_argument("--sizes", default="256,1024")
    p.add_argument("--masses", default="32,full")
    p.add_argument("--models", default="ridge,extra_trees")
    p.add_argument("--sigma", type=float, default=0.12)
    p.add_argument("--local-neighbors", type=int, default=20)
    p.add_argument("--hard-centers", type=int, default=8,
                   help="Number of hard real training centers, selected by fold-local Ridge PRESS")
    p.add_argument("--ridge-alpha", type=float, default=30.)
    p.add_argument("--trees", type=int, default=64)
    p.add_argument("--leaf", type=int, default=3)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--predict-method", default="gaussian_hard")
    p.add_argument("--predict-size", type=int, default=1024)
    p.add_argument("--predict-mass", default="32")
    p.add_argument("--predict-model", default="ridge")
    args = p.parse_args()
    settings = parse_settings(args)
    if args.summary:
        path = args.output.resolve()/"comparison_aggregate.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Run --run first: {path}")
        table = pd.read_csv(path)
        print(table.head(20).to_string(index=False))
        return
    frame, full, z, root, anchors, y, loo, features, ids = loaded(args)
    if args.check:
        print(f"[CHECK] 197 X-only, 138 measured yields, 59 unknown y; "
              f"features={len(features)}, rare anchors={len(anchors)}, "
              f"genuine Reference26 138-fold baseline present.")
        print(f"[CHECK] {len(keys(settings))} global learners/recipes, "
              f"including same-model no-augmentation controls; "
              f"n_synthetic={settings[1]}, mass={settings[2]}.")
        print("[CHECK] No 59 y used and no synthetic rows are real measurements.")
        return
    folder = args.output.resolve()
    if args.plot:
        make_plot(
            frame, full, z, loo, anchors, folder,
            hard_centers=args.hard_centers, alpha=args.ridge_alpha,
        )
        return
    if args.predict59:
        predict59(args, frame, full, z, root, anchors, y, ids)
        return
    spec = spec_for(args, settings)
    destination = folder/"smoke" if args.smoke else folder
    destination.mkdir(parents=True, exist_ok=True)
    info = destination/"run_spec.json"
    if info.is_file() and json.loads(info.read_text(encoding="utf-8")) != spec:
        raise ValueError("Changed recipe/data/source. Supply a different --output folder.")
    if not info.is_file():
        if list(destination.glob("fold_*.csv")):
            raise ValueError("Fold files exist without matching run specification.")
        info.write_text(json.dumps(spec, indent=2)+"\n", encoding="utf-8")
    positions = sorted(set([0, 1, *anchors])) if args.smoke else list(range(N))
    roots = local_roots(z, anchors, args.local_neighbors)
    payload = (
        z, y, anchors, root, roots, {
            "sigma": args.sigma, "ridge_alpha": args.ridge_alpha,
            "trees": args.trees, "leaf": args.leaf,
            "seed": args.seed,
        }, settings, ids, loo["Reference26_mean"].to_numpy(float)
    )
    required = {ID, "observed", "Reference26_mean", *keys(settings)}
    missing = []
    for pos in positions:
        path = destination/f"fold_{pos+1:03d}.csv"
        if path.is_file():
            saved = pd.read_csv(path, dtype={ID:str})
            if (len(saved) != 1 or set(saved.columns) != required
                    or saved.iloc[0][ID] != ids[pos]
                    or abs(float(saved.iloc[0]["observed"])-y[pos]) > 1e-9
                    or not np.isfinite(saved.drop(columns=ID).to_numpy(float)).all()):
                raise ValueError(f"Invalid cached outer fold {path}")
        else:
            missing.append(pos)
    print(f"[AUG] {len(missing)} new folds, {len(positions)} requested; "
          f"{len(keys(settings))} global recipes, jobs={args.jobs}.", flush=True)
    if missing:
        with ProcessPoolExecutor(
            max_workers=min(args.jobs, len(missing)),
            initializer=init_worker, initargs=(payload,),
        ) as pool:
            future_map = {pool.submit(one_fold, pos):pos for pos in missing}
            for completed, future in enumerate(as_completed(future_map), 1):
                pos, row = future.result()
                path = destination/f"fold_{pos+1:03d}.csv"
                pd.DataFrame([row]).to_csv(path, index=False,
                                           float_format="%.16g")
                if completed % 10 == 0 or completed == len(missing):
                    print(f"[AUG] {completed}/{len(missing)} new folds complete.",
                          flush=True)
    combined = pd.concat([
        pd.read_csv(destination/f"fold_{p+1:03d}.csv", dtype={ID: str})
        for p in positions
    ], ignore_index=True)
    if args.smoke:
        print("[SMOKE] Successfully trained/replayed "
              f"{len(positions)} folds. No full LOOCV metric claimed.")
        print(combined.drop(columns=[ID,"observed"]).head().to_string(index=False))
        return
    hard_ids = tuple(ids[a] for a in anchors)
    aggregate = metrics(combined, hard_ids, settings)
    aggregate.to_csv(destination/"comparison_aggregate.csv", index=False)
    combined.to_csv(destination/"private_fold_predictions.csv", index=False)
    print("[FULL] 138 real 137/1 heldouts, candidate models trained from "
          "scratch per fold, authentic Reference26 mean comparison.")
    print(aggregate.head(18).to_string(index=False))
    print("[NOTE] The source anchors were selected post-hoc from these same "
          "138 observed errors. This experiment is exploratory. "
          "Synthesis is a modeling assumption, not new y measurements.")
    print(f"[PRIVATE] Stored all outputs under {destination}")


if __name__ == "__main__":
    main()
