#!/usr/bin/env python3
"""Three-class RANDOM FOREST router + two separately trained synthetic experts.

Class 0 -> unchanged Reference26 mean prediction.
Class 1 -> Ridge regressor trained on Gaussian-bridge group A (2 real anchors).
Class 2 -> Ridge regressor trained on a 5-segment bridge through group B
           (6 real anchors; fewer if one is held out).

Default augmentation: 3 paths * 7 points per segment -> 21 + 105 = 126
synthetic positions at full training size, NOT independent observed yields.
Uses existing authentic Reference26 mean 138-fold LOOCV predictions.

  python tools/run_three_regime_classifier.py --check
  python tools/run_three_regime_classifier.py --smoke
  python tools/run_three_regime_classifier.py --run
  python tools/run_three_regime_classifier.py --summary
  python tools/run_three_regime_classifier.py --predict59 --sigma 0.05 --min-prob 0.6 --blend 0.25
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge

import run_two_bridge_three_regime as bridges
import run_synthetic_bridge_three_regime as original

ROOT = original.ROOT
ID = original.ID
N = original.N
OUTPUT = ROOT / "reports/three_regime_classifier"
SEED = 20261019


def real_group_positions(train: np.ndarray, a: tuple, b: tuple):
    """Training-set group labels; never use held-out y or predicted error."""
    a_set, b_set = set(a), set(b)
    if a_set & b_set:
        raise ValueError("Rare groups overlap.")
    return np.array([
        1 if pos in a_set else 2 if pos in b_set else 0
        for pos in train
    ], dtype=int)


def build_classifier_training(z, train, groups, group_x, *, synthetic_weight=1.0):
    """Synthetic total sample weight is bounded by real anchor count.

    A group of 105 fake points is NOT treated as 105 independent labels.
    """
    train = np.asarray(train, int)
    groups = np.asarray(groups, int)
    if len(train) != len(groups) or not (0 < synthetic_weight <= 1):
        raise ValueError("Invalid training groups or synthetic weight.")
    x_parts, y_parts, w_parts = [], [], []
    present_classes = [0]
    for label in (1, 2):
        if label in group_x:
            present_classes.append(label)
    keep = np.isin(groups, present_classes)
    if not np.any((groups == 0) & keep):
        raise ValueError("No ordinary real training examples.")
    x_parts.append(z[train[keep]])
    y_parts.append(groups[keep])
    w_parts.append(np.ones(int(np.sum(keep)), float))
    for label in (1, 2):
        if label not in group_x:
            continue
        cloud = group_x[label]
        count = int(np.sum(groups == label))
        if len(cloud) == 0 or count < 2:
            raise ValueError("Specialist requires at least two real anchors.")
        x_parts.append(cloud)
        y_parts.append(np.full(len(cloud), label, int))
        w_parts.append(np.full(
            len(cloud), synthetic_weight*count/len(cloud), float
        ))
    x = np.vstack(x_parts)
    group_y = np.concatenate(y_parts)
    weight = np.concatenate(w_parts)
    # Class-balanced effective mass, using REAL sample counts rather
    # than inflated synthetic row counts.
    real_count = {j: max(int(np.sum(groups == j)), 1)
                  for j in np.unique(group_y)}
    for j in real_count:
        weight[group_y == j] /= real_count[j]
    if not np.isfinite(x).all() or not np.isfinite(weight).all():
        raise ValueError("Invalid classifier X / weights.")
    return x, group_y, weight


def fit_fold(
    z: np.ndarray, root: np.ndarray, y: np.ndarray,
    train: np.ndarray, query: np.ndarray, a: tuple, b: tuple,
    *, sigma: float, paths: int, steps: int, alpha: float,
    trees: int, seed: int, synthetic_weight: float,
):
    """Two independent regressors and a fold-trained 3-way classifier."""
    train, query = np.asarray(train, int), np.asarray(query, int)
    if (not len(train) or not len(query) or
            np.intersect1d(train, query).size):
        raise ValueError("Train/test overlap or empty train/test.")
    if (sigma < 0 or paths < 1 or steps < 3 or steps % 2 == 0
            or alpha <= 0 or trees < 10):
        raise ValueError("Invalid classifier/bridge configuration.")
    if not np.isfinite(y[train]).all():
        raise ValueError("Nonfinite training yields.")
    labels = real_group_positions(train, a, b)
    models = {}
    group_clouds = {}
    sample_counts = {1: 0, 2: 0}
    for cls, anchors in ((1, a), (2, b)):
        x_syn, y_syn, links = bridges.synthetic_group(
            z, root, y, train, anchors,
            sigma=sigma, paths=paths, steps=steps, seed=seed+100*cls,
        )
        sample_counts[cls] = len(x_syn)
        if links:
            models[cls] = Ridge(alpha=alpha).fit(x_syn, y_syn)
            group_clouds[cls] = x_syn
    cls_x, cls_y, weights = build_classifier_training(
        z, train, labels, group_clouds, synthetic_weight=synthetic_weight,
    )
    classifier = RandomForestClassifier(
        n_estimators=trees, max_depth=8, min_samples_leaf=3,
        max_features="sqrt", random_state=seed, n_jobs=1,
    )
    classifier.fit(cls_x, cls_y, sample_weight=weights)
    raw_proba = classifier.predict_proba(z[query])
    # Return three columns indexed by route 0, 1, 2.
    probability = np.zeros((len(query), 3), float)
    for j, label in enumerate(classifier.classes_):
        probability[:, int(label)] = raw_proba[:, j]
    if not np.isfinite(probability).all():
        raise ValueError("Nonfinite classifier probability.")
    expert_predictions = np.full((len(query), 3), np.nan)
    for label, model in models.items():
        expert_predictions[:, label] = model.predict(z[query])
    return {
        "probabilities": probability,
        "experts": expert_predictions,
        "samples_a": sample_counts[1],
        "samples_b": sample_counts[2],
        "real_train": len(train),
        "active_experts": sorted(models),
        "real_class_counts": {str(v): int(np.sum(labels == v))
                              for v in (0, 1, 2)},
    }


def route_predictions(
    baseline, fitted: dict, *, min_prob: float, blend: float,
    max_change: float, min_margin: float = 0.0,
):
    """Class 0 means *exact* untouched baseline. No geometric proxy routing."""
    if not (0 <= min_prob <= 1 and 0 <= blend <= 1 and
            max_change > 0 and 0 <= min_margin <= 1):
        raise ValueError("Bad classifier routing options.")
    baseline = np.asarray(baseline, float)
    proba = fitted["probabilities"]
    experts = fitted["experts"]
    if proba.shape != (len(baseline), 3) or experts.shape != proba.shape:
        raise ValueError("Classifier/experts/targets shapes differ.")
    best = np.argmax(proba, axis=1)
    probability = proba[np.arange(len(baseline)), best]
    sorted_proba = np.sort(proba, axis=1)
    margin = sorted_proba[:, -1]-sorted_proba[:, -2]
    chosen = (best != 0) & (probability >= min_prob) & (
        margin >= min_margin
    )
    expert = experts[np.arange(len(baseline)), best]
    chosen &= np.isfinite(expert)
    safe_expert = np.where(chosen, expert, baseline)
    adjustment = np.clip(
        blend*(safe_expert-baseline), -max_change, max_change
    )
    preds = baseline+adjustment
    routes = np.where(np.abs(adjustment) > 1e-12, best, 0).astype(int)
    return preds, routes, probability


def metrics(y, base, pred, route, groups):
    return bridges.scored(y, base, pred, route, groups)


def inputs(args):
    return bridges.verified_inputs(args)


def run(args, smoke=False):
    frame, full, z, root, y, loo, features, a, b = inputs(args)
    sigma_list = [float(item) for item in args.sigmas.split(",")]
    thresholds = [float(item) for item in args.min_probs.split(",")]
    blends = [float(item) for item in args.blends.split(",")]
    if (not sigma_list or not thresholds or not blends or
        any(not np.isfinite(s) or s < 0 for s in sigma_list) or
        any(not 0 <= p <= 1 for p in thresholds) or
        any(not 0 <= w <= 1 for w in blends) or
        not 0 <= args.synthetic_weight <= 1 or
        args.synthetic_weight == 0):
        raise ValueError("Invalid parameter grid.")
    folds = sorted(set(range(12)) | set(a) | set(b)) if smoke else list(range(N))
    observed = y[folds]
    baseline = loo["Reference26_mean"].to_numpy(float)[folds]
    real_classes = np.array([
        1 if idx in a else 2 if idx in b else 0 for idx in folds
    ])
    dest = args.output.resolve()
    spec = {
        "schema": 1,
        "method": "three_class_random_forest_two_synthetic_ridge_experts",
        "anchor_manifest_sha256": original.sha(args.anchor_file.resolve()),
        "reference26_run_spec_sha256": original.sha(
            args.fresh_fold_dir.resolve()/"run_spec.json"
        ),
        "bundle_sha256": original.sha(args.bundle.resolve()/"manifest.json"),
        "runner_sha256": original.sha(Path(__file__).resolve()),
        "posthoc_selection": True,
        "parameter_grid": {
            "sigmas": sigma_list, "min_probs": thresholds, "blends": blends,
            "paths": args.paths, "steps": args.steps,
            "ridge_alpha": args.alpha, "trees": args.trees,
            "synthetic_weight": args.synthetic_weight,
            "max_change": args.max_change, "min_margin": args.min_margin,
            "seed": args.seed,
        },
    }
    if not smoke:
        dest.mkdir(parents=True, exist_ok=True)
        path = dest/"run_spec.json"
        if path.is_file() and json.loads(path.read_text()) != spec:
            raise ValueError("Changed input/code/parameters; choose a fresh --output.")
        path.write_text(json.dumps(spec, indent=2)+"\n", encoding="utf-8")
    records, parcel_records = [], []
    for sigma in sigma_list:
        cache = []
        for i, held in enumerate(folds):
            train = np.delete(np.arange(N), held)
            fitted = fit_fold(
                z, root, y, train, np.asarray([held]), a, b,
                sigma=sigma, paths=args.paths, steps=args.steps,
                alpha=args.alpha, trees=args.trees,
                seed=args.seed+held, synthetic_weight=args.synthetic_weight,
            )
            cache.append(fitted)
        for threshold in thresholds:
            for blend in blends:
                pred = baseline.copy()
                routes = np.zeros(len(folds), int)
                confidence = np.zeros(len(folds))
                for j, fit in enumerate(cache):
                    newer, selected, prob = route_predictions(
                        baseline[j:j+1], fit, min_prob=threshold,
                        blend=blend, max_change=args.max_change,
                        min_margin=args.min_margin,
                    )
                    pred[j] = newer[0]
                    routes[j] = selected[0]
                    confidence[j] = prob[0]
                report = {
                    "sigma": sigma, "min_prob": threshold, "blend": blend,
                    "n_heldout": len(folds),
                    "n_true_training_labels": N-1,
                    "synthetic_a_full": args.paths*args.steps,
                    "synthetic_b_full": 5*args.paths*args.steps,
                    "trees": args.trees,
                    "max_change": args.max_change,
                    **metrics(observed, baseline, pred, routes, real_classes),
                }
                records.append(report)
                if not smoke:
                    parcel_records.extend({
                        ID: frame.iloc[held][ID],
                        "observed": float(observed[j]),
                        "reference26_mean": float(baseline[j]),
                        "three_regime": float(pred[j]),
                        "route": int(routes[j]),
                        "classifier_confidence": float(confidence[j]),
                        "sigma": sigma, "min_prob": threshold,
                        "blend": blend,
                    } for j, held in enumerate(folds))
        print(f"[CV] sigma={sigma:g}: {len(folds)} fresh training-only "
              f"3-class routers + two small augmented experts.", flush=True)
    ranking = pd.DataFrame(records).sort_values(
        ["rmse", "changed_ordinary130", "sigma"], kind="stable"
    ).reset_index(drop=True)
    if smoke:
        print(f"[SMOKE] {len(folds)} heldouts including all 8 anchors; "
              "not full CV.")
        print(ranking.head(12).to_string(index=False))
        return
    ranking.to_csv(dest/"comparison_aggregate.csv", index=False)
    pd.DataFrame(parcel_records).to_csv(
        dest/"parcel_predictions_private.csv", index=False
    )
    print(f"[FULL] {len(folds)} real heldouts; {len(records)} configurations. "
          f"Synthetic full bridge sizes: A={args.paths*args.steps}, "
          f"B={5*args.paths*args.steps}.")
    print(ranking.head(20).to_string(index=False))
    print("[CAUTION] The eight real regime labels were discovered using "
          "these same 138 LOOCV residuals. This CV is useful for "
          "mechanism screening but NOT unbiased model-selection validation.")
    print(f"[PRIVATE] Individual labels and routing predictions at {dest}.")


def predict59(args):
    _frame, full, z, root, y, loo, features, a, b = inputs(args)
    directory = args.fresh_fold_dir.resolve()/"final_artifacts"
    csv = directory/"predictions59_private.csv"
    manifest = directory/"final_manifest.json"
    if not (csv.is_file() and manifest.is_file()):
        raise FileNotFoundError("Missing verified original final Reference26 outputs.")
    spec = json.loads(manifest.read_text(encoding="utf-8"))
    if spec.get("n_train") != N or spec.get("n_unknown") != 59:
        raise ValueError("Original final model provenance counts disagree.")
    stored = pd.read_csv(csv, dtype={ID: str})
    ids = full.iloc[N:][ID].astype(str).to_numpy()
    if (len(stored) != 59 or stored[ID].duplicated().any() or
        not np.array_equal(ids, stored[ID].to_numpy())):
        raise ValueError("Original 59 target IDs differ.")
    base = stored["Reference26_mean_t_ha"].to_numpy(float)
    if not np.isfinite(base).all():
        raise ValueError("Missing/invalid original 59 forecasts.")
    fit = fit_fold(
        z, root, y, np.arange(N), np.arange(N,197), a, b,
        sigma=args.sigma, paths=args.paths, steps=args.steps,
        alpha=args.alpha, trees=args.trees,
        seed=args.seed, synthetic_weight=args.synthetic_weight,
    )
    predictions, routes, probs = route_predictions(
        base, fit, min_prob=args.min_prob, blend=args.blend,
        max_change=args.max_change, min_margin=args.min_margin,
    )
    out = args.output.resolve()
    dest = out/"predictions59_private.csv"
    if dest.exists():
        raise FileExistsError(f"Existing candidate file not overwritten: {dest}")
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        ID: ids, "Reference26_mean_t_ha": base,
        "ThreeRegime_t_ha": predictions, "route": routes,
        "classifier_confidence": probs,
        "correction_t_ha": predictions-base,
    }).to_csv(dest, index=False)
    print(f"[59] Routed: A={int((routes==1).sum())}, "
          f"B={int((routes==2).sum())}; unchanged={int((routes==0).sum())}. "
          f"Private file: {dest}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    for flag in ("check", "smoke", "run", "summary", "predict59"):
        mode.add_argument("--"+flag, action="store_true")
    p.add_argument("--anchor-file", type=Path, default=original.ANCHOR_SOURCE)
    p.add_argument("--bundle", type=Path, default=original.native.BUNDLE)
    p.add_argument("--peer-repo", type=Path, default=original.native.PEER_REPO)
    p.add_argument("--config", type=Path, default=original.native.CONFIG)
    p.add_argument("--fresh-fold-dir", type=Path, default=original.native.OUTPUT)
    p.add_argument("--output", type=Path, default=OUTPUT)
    p.add_argument("--paths", type=int, default=3)
    p.add_argument("--steps", type=int, default=7)
    p.add_argument("--alpha", type=float, default=5.)
    p.add_argument("--trees", type=int, default=80)
    p.add_argument("--synthetic-weight", type=float, default=1.)
    p.add_argument("--sigmas", default="0,0.05,0.15")
    p.add_argument("--min-probs", default="0.4,0.6,0.8")
    p.add_argument("--blends", default="0.25,0.5")
    p.add_argument("--max-change", type=float, default=0.35)
    p.add_argument("--min-margin", type=float, default=0.)
    p.add_argument("--sigma", type=float, default=0.05)
    p.add_argument("--min-prob", type=float, default=0.6)
    p.add_argument("--blend", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args()
    if args.summary:
        path = args.output.resolve()/"comparison_aggregate.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Run --run first: {path}")
        data = pd.read_csv(path)
        print(data.sort_values("rmse").to_string(index=False))
    elif args.check:
        frame, full, z, root, y, loo, features, a, b = inputs(args)
        print(f"[CHECK] Two REAL special regimes: A={len(a)} labels, "
              f"B={len(b)} labels, ordinary={N-len(a)-len(b} labels.")
        print(f"[CHECK] Default synthetic counts A={args.paths*args.steps}, "
              f"B={5*args.paths*args.steps} = "
              f"{6*args.paths*args.steps} positions total; "
              "80-tree RF classifier; two Ridge experts.")
        print("[CHECK] Genuine Reference26 138x137/1 OOF predictions reused.")
    elif args.predict59:
        predict59(args)
    else:
        run(args, smoke=args.smoke)


if __name__ == "__main__":
    main()
