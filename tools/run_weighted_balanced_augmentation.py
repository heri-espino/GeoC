#!/usr/bin/env python3
"""Fold-safe weights vs balanced all-parcel Gaussian augmentation in GeoC.

One GLOBAL Ridge or ExtraTrees per candidate; NO three-class router.
Every outer LOOCV fit uses 137 actual labels, never 138. Artificial rows
are generated exclusively from those 137 allowed training parcels.

Comparison:
  - unweighted real-only controls
  - higher weights for training-only Ridge PRESS difficult parcels
  - higher weights for training-only X-space isolated parcels
  - equal representation of ALL 137 real parcels in Gaussian X clouds,
    with 2/10/50 or more replicas per real parcel and variable total
    synthetic mass, holding the original regularization mass constant.

Original Reference26 mean (0.468138) is read from genuine saved 138 LOOCV
results only; it is NOT retrained with the synthetic rows.
No 59 target labels are used or published.

  python tools/run_weighted_balanced_augmentation.py --check
  python tools/run_weighted_balanced_augmentation.py --smoke
  python tools/run_weighted_balanced_augmentation.py --run --jobs 4
  python tools/run_weighted_balanced_augmentation.py --summary
  python tools/run_weighted_balanced_augmentation.py --run --models ridge --replicas 50 --output reports/weighted_balanced_augmentation_large
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_key, "1")

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

import run_global_synthetic_augmentation as prior

ROOT = prior.ROOT
N = prior.N
ID = prior.ID
OUT = ROOT / "reports/weighted_balanced_augmentation"
SEED = 20261019
_STATE = None


def parse_grid(args):
    models = tuple(x.strip() for x in args.models.split(","))
    factors = tuple(float(x) for x in args.weight_factors.split(","))
    modes = tuple(x.strip() for x in args.weight_modes.split(","))
    replicas = tuple(int(x) for x in args.replicas.split(","))
    sigmas = tuple(float(x) for x in args.sigmas.split(","))
    masses = tuple(x.strip() for x in args.synth_masses.split(","))
    if (not models or set(models)-{"ridge", "extra_trees"}
        or not modes or set(modes)-{"press", "sparse"}
        or not factors or any((not np.isfinite(f) or f < 1) for f in factors)
        or not replicas or any(r < 1 for r in replicas)
        or not sigmas or any(not np.isfinite(s) or s < 0 for s in sigmas)
        or not masses or set(masses)-{"equal", "full"}
        or args.top_k < 1 or args.ridge_alpha <= 0 or args.local_neighbors < 3
        or args.trees < 4 or args.leaf < 1 or args.jobs < 1):
        raise ValueError("Invalid grid or non-positive hyperparameter.")
    if any(len(set(vals)) != len(vals) for vals in
           (models, factors, modes, replicas, sigmas, masses)):
        raise ValueError("Parameter lists contain duplicates.")
    return models, modes, factors, replicas, sigmas, masses


def score_training_only(z, y, train, *, criterion, ridge_alpha):
    """All ordering decisions depend only on the outer training fold."""
    train = np.asarray(train, dtype=int)
    if criterion == "press":
        positions, residuals = prior.fold_hard_centers(
            z, y, train, count=len(train), alpha=ridge_alpha
        )
        return positions, np.abs(residuals)
    if criterion == "sparse":
        distances = pairwise_distances(z[train])
        np.fill_diagonal(distances, np.inf)
        nearest = distances.min(axis=1)
        ordering = np.lexsort((train, -nearest))
        return train[ordering], nearest[ordering]
    raise ValueError("Unknown hardness criterion")


def weights_for_training(z, y, train, *, criterion, top_k, factor, alpha):
    positions, scores = score_training_only(
        z, y, train, criterion=criterion, ridge_alpha=alpha
    )
    selected = set(map(int, positions[:top_k]))
    weights = np.array([
        factor if int(i) in selected else 1. for i in train
    ], dtype=float)
    # Keep total sample-weight mass == number of genuine training rows,
    # otherwise Ridge's effective alpha also changes by global scale.
    return weights * (len(train) / float(weights.sum())), tuple(sorted(selected))


def balanced_gaussian(
    z, y, train, roots, *, replicas, sigma, seed
):
    """Same number of synthetic points per real training label, X jitter only.

    y_tilde == real center y; there is NO response noise and NO heldout y.
    """
    if replicas < 1 or sigma < 0:
        raise ValueError("Replicas >=1 and sigma>=0 required.")
    train = np.asarray(train, dtype=int)
    rng = np.random.default_rng(seed)
    d = z.shape[1]
    x_syn = np.empty((len(train)*replicas, d), dtype=float)
    y_syn = np.repeat(y[train], replicas)
    for j, i in enumerate(train):
        if int(i) not in roots:
            raise ValueError("Missing root covariance for training center")
        start = j*replicas
        noise = rng.normal(size=(replicas, d))
        x_syn[start:start+replicas] = (
            z[i] + sigma*(noise @ roots[int(i)].T)
        )
    # X was robust standardized/clipped when entering this procedure.
    x_syn = np.clip(x_syn, -9., 9.)
    if not np.isfinite(x_syn).all() or not np.isfinite(y_syn).all():
        raise ValueError("Nonfinite synthetic observations")
    return x_syn, y_syn


def combined_sample_weights(n_real, n_synthetic, mass):
    if n_real <= 0 or n_synthetic <= 0 or mass not in {"equal", "full"}:
        raise ValueError("Invalid augmented sample weight dimensions")
    # 'equal': synthetic block has the same total weight as all real rows.
    # 'full': synthetic row has initially weight 1 (aggressive treatment).
    syn_mass = n_real if mass == "equal" else n_synthetic
    sample_weights = np.r_[
        np.ones(n_real), np.full(n_synthetic, syn_mass/n_synthetic)
    ]
    return sample_weights * (n_real/sample_weights.sum())


def fit_predict(z, y, train, query, *, model, alpha, trees, leaf, seed,
                weights=None, synthetic=None, synthetic_mass="equal"):
    train = np.asarray(train, dtype=int)
    query = np.asarray(query, dtype=int)
    if np.intersect1d(train, query).size:
        raise ValueError("Heldout cannot be in supervised train rows")
    x_train = z[train]
    target = y[train]
    if not np.isfinite(target).all():
        raise ValueError("Nonfinite real training yields")
    if synthetic is not None:
        if weights is not None:
            raise ValueError("Do not mix row weights with augmentation in this ablation")
        x_syn, y_syn = synthetic
        mass_weights = combined_sample_weights(
            len(train), len(y_syn), synthetic_mass
        )
        x_train = np.vstack([x_train, x_syn])
        target = np.r_[target, y_syn]
        weights = mass_weights
    learner = prior.regressor(
        model, alpha=alpha, trees=trees, leaf=leaf, seed=seed
    )
    learner.fit(x_train, target, sample_weight=weights)
    prediction = np.asarray(learner.predict(z[query]), dtype=float)
    if not np.isfinite(prediction).all():
        raise ValueError("Nonfinite prediction")
    return prediction


def keys(grid):
    models, modes, factors, replicas, sigmas, masses = grid
    answer = [f"plain__{model}" for model in models]
    for model in models:
        for mode in modes:
            for factor in factors:
                answer.append(f"weighted__{model}__{mode}__top{SETTINGS_TOP_PLACEHOLDER}__w{factor:g}")
        for count in replicas:
            for sigma in sigmas:
                for mass in masses:
                    answer.append(
                        f"balanced__{model}__r{count}__sigma{sigma:g}__mass{mass}"
                    )
    return answer


def expected_columns(grid, top_k):
    return [key.replace("SETTINGS_TOP_PLACEHOLDER", str(top_k))
            for key in keys(grid)]


def init_worker(state):
    global _STATE
    _STATE = state


def one_fold(held):
    if _STATE is None:
        raise RuntimeError("Uninitialized fold worker")
    z, y, ids, baseline, roots, args, grid = _STATE
    train = np.delete(np.arange(N), held)
    query = np.array([held], dtype=int)
    models, modes, factors, replicas, sigmas, masses = grid
    result = {
        ID: ids[held], "observed": float(y[held]),
        "Reference26_mean": float(baseline[held]),
    }
    # Every candidate is trained from scratch on the SAME 137 real labels.
    for model in models:
        kw = dict(model=model, alpha=args["ridge_alpha"], trees=args["trees"],
                  leaf=args["leaf"], seed=args["seed"]+held)
        result[f"plain__{model}"] = float(fit_predict(
            z, y, train, query, **kw
        )[0])
        for mode in modes:
            for factor in factors:
                weight, chosen = weights_for_training(
                    z, y, train, criterion=mode,
                    top_k=args["top_k"], factor=factor,
                    alpha=args["ridge_alpha"],
                )
                key = f"weighted__{model}__{mode}__top{args['top_k']}__w{factor:g}"
                result[key] = float(fit_predict(
                    z, y, train, query, weights=weight, **kw
                )[0])
    # Generate ONE balanced cloud for each requested n and sigma,
    # reused across models and weight masses to ensure paired comparison.
    for count in replicas:
        for sigma in sigmas:
            syn = balanced_gaussian(
                z, y, train, roots, replicas=count, sigma=sigma,
                seed=args["seed"]+held+count*101+int(round(sigma*10000)),
            )
            for model in models:
                kw = dict(model=model, alpha=args["ridge_alpha"],
                          trees=args["trees"], leaf=args["leaf"],
                          seed=args["seed"]+held)
                for mass in masses:
                    key = f"balanced__{model}__r{count}__sigma{sigma:g}__mass{mass}"
                    result[key] = float(fit_predict(
                        z, y, train, query,
                        synthetic=syn, synthetic_mass=mass, **kw
                    )[0])
    return held, result


def summarize(table, grid, top_k, anchor_ids):
    observed = table["observed"].to_numpy(float)
    baseline = table["Reference26_mean"].to_numpy(float)
    hard = table[ID].astype(str).isin(anchor_ids).to_numpy()
    scores = []
    for key in expected_columns(grid, top_k):
        pred = table[key].to_numpy(float)
        e = pred-observed
        old = baseline-observed
        model = key.split("__")[1]
        control = table[f"plain__{model}"].to_numpy(float)
        control_error = control-observed
        scores.append(dict(
            model=key, rmse=float(np.sqrt(np.mean(e*e))),
            mae=float(np.mean(np.abs(e))),
            rmse_same_model_plain=float(np.sqrt(
                np.mean(control_error**2))),
            rmse_reference26=float(np.sqrt(np.mean(old**2))),
            delta_mse_vs_plain=float(np.mean(
                e*e-control_error*control_error
            )),
            delta_mse_vs_ref=float(np.mean(e*e-old*old)),
            rmse_historic_hard2=float(np.sqrt(np.mean(e[hard]**2))),
            rmse_other136=float(np.sqrt(np.mean(e[~hard]**2))),
            n_worsened_vs_ref_other136=int(np.sum(
                e[~hard]**2 > old[~hard]**2 + 1e-12
            )),
            n_improved_vs_ref=int(np.sum(e*e < old*old-1e-12)),
            max_absolute_delta_vs_ref=float(np.max(np.abs(pred-baseline))),
            real_train_rows=N-1,
            real_heldouts=len(observed),
            synthetic_rows_per_fit=(
                (N-1)*int(key.split("__")[2][1:])
                if key.startswith("balanced__") else 0
            ),
        ))
    return pd.DataFrame(scores).sort_values(
        ["rmse", "delta_mse_vs_plain"], kind="stable"
    )


def provenance(args, grid):
    return {
        "schema": 1,
        "evaluation": "138 real 137/1 LOOCV + per-fold synthetic/rowweights",
        "not_native_reference26_refit": True,
        "fold_local_y_dependent_hardness": True,
        "synthetic_points_do_not_add_independent_labels": True,
        "params": {name: getattr(args, name) for name in (
            "models","weight_modes","weight_factors","replicas","sigmas",
            "synth_masses","top_k","local_neighbors","ridge_alpha",
            "trees","leaf","seed")},
        "frozen_reference26_spec_sha": prior.digest(
            args.fresh_fold_dir.resolve()/"run_spec.json"),
        "bundle_manifest_sha": prior.digest(
            args.bundle.resolve()/"manifest.json"),
        "anchor_manifest_sha": prior.digest(args.anchor_file.resolve()),
        "this_source_sha": prior.digest(Path(__file__).resolve()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ("check", "smoke", "run", "summary"):
        mode.add_argument("--"+name, action="store_true")
    parser.add_argument("--bundle", type=Path, default=prior.historic.native.BUNDLE)
    parser.add_argument("--peer-repo", type=Path,
                        default=prior.historic.native.PEER_REPO)
    parser.add_argument("--anchor-file", type=Path,
                        default=prior.historic.ANCHOR_SOURCE)
    parser.add_argument("--config", type=Path,
                        default=prior.historic.native.CONFIG)
    parser.add_argument("--fresh-fold-dir", type=Path,
                        default=prior.historic.native.OUTPUT)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--models", default="ridge,extra_trees")
    parser.add_argument("--weight-modes", default="press,sparse")
    parser.add_argument("--weight-factors", default="2,5")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--replicas", default="2,10")
    parser.add_argument("--sigmas", default="0,0.1")
    parser.add_argument("--synth-masses", default="equal,full")
    parser.add_argument("--local-neighbors", type=int, default=20)
    parser.add_argument("--ridge-alpha", type=float, default=30.)
    parser.add_argument("--trees", type=int, default=64)
    parser.add_argument("--leaf", type=int, default=3)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    grid = parse_grid(args)
    folder = args.output.resolve()
    if args.summary:
        path = folder/"comparison_aggregate.csv"
        if not path.is_file():
            raise FileNotFoundError(f"No completed --run at {path}")
        print(pd.read_csv(path).sort_values("rmse").head(30).to_string(index=False))
        return
    frame, full, z, root, anchors, y, loo, features, ids = prior.loaded(args)
    ref = loo["Reference26_mean"].to_numpy(float)
    if args.check:
        print(f"[CHECK] 197 X-only rows; 138 y observed; 59 target X; "
              f"26 source features available={len(features)}.")
        print(f"[CHECK] {len(expected_columns(grid,args.top_k))} global "
              f"candidate fits per heldout, plus frozen original Reference26.")
        print(f"[CHECK] 137 true train + 1 true test per fold. "
              f"Balanced synthetic counts={[(N-1)*r for r in grid[3]}.")
        print("[CHECK] Difficult-training weights selected only inside each outer fold.")
        return
    dest = folder/"smoke" if args.smoke else folder
    dest.mkdir(parents=True, exist_ok=True)
    fp = dest/"run_spec.json"
    spec = provenance(args, grid)
    if fp.exists() and json.loads(fp.read_text(encoding="utf-8")) != spec:
        raise ValueError("Source/settings/input changed; use a new --output")
    if not fp.exists():
        if any(dest.glob("fold_*.csv")):
            raise ValueError("Existing folds lack the matching fingerprint")
        fp.write_text(json.dumps(spec,indent=2)+"\n",encoding="utf-8")
    requested = sorted(set([0,1,*anchors])) if args.smoke else list(range(N))
    roots = prior.local_roots(z, tuple(range(N)), args.local_neighbors)
    state = (
        z, y, ids, ref, roots,
        dict(ridge_alpha=args.ridge_alpha, trees=args.trees,
             leaf=args.leaf, seed=args.seed, top_k=args.top_k),
        grid,
    )
    expected = set([ID, "observed", "Reference26_mean",
                    *expected_columns(grid,args.top_k)])
    missing = []
    for held in requested:
        path = dest/f"fold_{held+1:03d}.csv"
        if path.is_file():
            cached = pd.read_csv(path, dtype={ID: str})
            if (len(cached)!=1 or set(cached.columns)!=expected
                or cached.iloc[0][ID]!=ids[held]
                or abs(float(cached.iloc[0]["observed"])-y[held])>1e-9
                or not np.isfinite(cached.drop(columns=ID).to_numpy(float)).all()):
                raise ValueError(f"Invalid cached fold: {path}")
        else:
            missing.append(held)
    print(f"[WEIGHT+BALANCED] {len(missing)} new real folds of "
          f"{len(requested)}; jobs={args.jobs}.",flush=True)
    if missing:
        with ProcessPoolExecutor(
            max_workers=min(args.jobs,len(missing)),
            initializer=init_worker, initargs=(state,)
        ) as executor:
            futures = {executor.submit(one_fold,i):i for i in missing}
            for j, future in enumerate(as_completed(futures),1):
                pos, record = future.result()
                pd.DataFrame([record]).to_csv(
                    dest/f"fold_{pos+1:03d}.csv", index=False,
                    float_format="%.16g",
                )
                if j%10==0 or j==len(missing):
                    print(f"[WEIGHT+BALANCED] {j}/{len(missing)} newly trained.",
                          flush=True)
    table = pd.concat([
        pd.read_csv(dest/f"fold_{pos+1:03d}.csv",dtype={ID:str})
        for pos in requested
    ],ignore_index=True)
    if args.smoke:
        print("[SMOKE] All small subset fits completed; NOT a full 138-fold score.")
        return
    anchor_ids = tuple(ids[i] for i in anchors)
    report = summarize(table,grid,args.top_k,anchor_ids)
    report.to_csv(dest/"comparison_aggregate.csv",index=False)
    table.to_csv(dest/"private_fold_predictions.csv",index=False)
    print(f"[FULL] {len(table)} genuine heldout labels; Reference26"
          " native frozen OOF used only as a comparison.")
    print(report.head(20).to_string(index=False))
    print("[CAUTION] Historical two hard IDs were selected post-hoc;"
          " this comparison is exploratory; no 59 target labels accessed.")
    print(f"[PRIVATE] {dest}")


if __name__=="__main__":
    main()
