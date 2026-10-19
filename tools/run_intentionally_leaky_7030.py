#!/usr/bin/env python3
"""INTENTIONALLY LEAKY 100x random 70/30 diagnostic; NOT a valid CV score.

Three independently distinguishable interventions:
  (i) no leak: real training -> real heldout tests;
  (ii) synthetic test expansion: train is clean; scores also include test-born
       Gaussian pseudo-observations whose responses are COPIED from test y;
  (iii) TARGET LEAKAGE: train incorporates test-born synthetic rows whose
        responses are COPIED from those same held-out test yields.

The latter two deliberately access heldout y during the experiment and must
NEVER be interpreted as generalization, LOOCV, hidden-59 validation, or a
proposed competition submission. Native Reference26 is NOT refitted; models
are global Ridge and ExtraTrees and compared to THEIR OWN clean controls.

Examples:
  python tools/run_intentionally_leaky_7030.py --check
  python tools/run_intentionally_leaky_7030.py --smoke
  python tools/run_intentionally_leaky_7030.py --run --jobs 4
  python tools/run_intentionally_leaky_7030.py --summary
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for _var in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_var] = "1"

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import run_global_synthetic_augmentation as base

ROOT = base.ROOT
ID = base.ID
N = base.N
SEED = 20261019
OUTPUT = ROOT / "reports/intentionally_leaky_70_30"
WORKER = None
SCENARIOS = (
    "clean_real_test",
    "clean_training_augmented_real_test",
    "clean_model_synthetic_expanded_test",
    "leaky_test_gaussian_training_real_test",
    "leaky_test_gaussian_training_expanded_test",
    "leaky_test_exact_copies_training_real_test",
)
MODELS = ("ridge", "extra_trees")


def split_indices(n: int, *, seed: int, test_fraction: float = 0.30):
    if not 0 < test_fraction < 1:
        raise ValueError("Test fraction must be inside (0,1).")
    tr, te = train_test_split(
        np.arange(n), test_size=test_fraction, shuffle=True,
        random_state=int(seed),
    )
    assert len(np.intersect1d(tr, te)) == 0
    assert len(np.union1d(tr, te)) == n
    return np.sort(tr), np.sort(te)


def test_origin_children(z, y, test, covariance_roots, *,
                         count: int, sigma: float, seed: int):
    """Test-origin Gaussian synthetic X with deliberately test-derived pseudo-y.

    Every test observation is represented as a source when count >= len(test).
    The fake parent labels are DELIBERATELY present, for leakage auditing.
    """
    test = np.asarray(test, int)
    if not len(test) or count < len(test) or sigma < 0:
        raise ValueError("Need >=1 child per heldout test parent and sigma >= 0.")
    rng = np.random.default_rng(seed)
    origin = np.tile(test, (count + len(test) - 1) // len(test))[:count]
    rng.shuffle(origin)
    noise = rng.normal(size=(count, z.shape[1]))
    x = np.empty((count, z.shape[1]), float)
    for source in np.unique(origin):
        where = origin == source
        x[where] = z[source] + sigma * (
            noise[where] @ covariance_roots[int(source)].T
        )
    x = np.clip(x, -9.0, 9.0)
    if not np.isfinite(x).all() or not np.isfinite(y[origin]).all():
        raise ValueError("Non-finite synthetic test samples.")
    return x, y[origin].copy(), origin


def fit(kind, x, y, *, train, synthetic, mass, seed, config):
    """Single regressor, with synthetic parents allowed from test ON PURPOSE."""
    xfit = x[train]
    yfit = y[train]
    weights = np.ones(len(train))
    if synthetic is not None:
        xs, ys = synthetic[:2]
        if not np.isfinite(xs).all() or not np.isfinite(ys).all():
            raise ValueError("Invalid synthetic X/y.")
        m = len(ys) if mass == "full" else int(mass)
        xfit = np.vstack((xfit, xs))
        yfit = np.r_[yfit, ys]
        weights = np.r_[weights, np.full(len(ys), m/len(ys))]
    reg = base.regressor(
        kind, alpha=config["alpha"], trees=config["trees"],
        leaf=config["leaf"], seed=seed,
    )
    reg.fit(xfit, yfit, sample_weight=weights)
    return reg


def metric(actual, forecast):
    actual, forecast = np.asarray(actual, float), np.asarray(forecast, float)
    if actual.shape != forecast.shape or not len(actual):
        raise ValueError("Mismatched/empty scoring vectors.")
    error = forecast - actual
    if not np.isfinite(error).all():
        raise ValueError("Nonfinite predictions.")
    return {
        "rmse_t_ha": float(np.sqrt(np.mean(error**2))),
        "mae_t_ha": float(np.mean(abs(error))),
        "bias_t_ha": float(np.mean(error)),
    }


def run_one(split_no: int, data):
    z, y, ids, roots, params = data
    tr, te = split_indices(N, seed=params["seed"] + split_no)
    count = params["synthetic_count"]
    sigma = params["sigma"]
    # Deliberate label leakage begins below. Never reuse in actual CV.
    # Training with synthetic rows derived from held-out real test parents
    # gives a model access to y[test] that ordinary validation forbids.
    leak_x, leak_y, leak_origins = test_origin_children(
        z, y, te, roots, count=count, sigma=sigma,
        seed=params["seed"] + 100000 + split_no,
    )
    exact_x, exact_y, exact_sources = test_origin_children(
        z, y, te, roots, count=count, sigma=0.,
        seed=params["seed"] + 200000 + split_no,
    )
    eval_x, eval_y, eval_origin = test_origin_children(
        z, y, te, roots, count=count, sigma=sigma,
        seed=params["seed"] + 300000 + split_no,
    )
    # CLEAN training-only hard cases (PRESS scoring uses y[train] exclusively).
    train_centers, _ = base.fold_hard_centers(
        z, y, tr, count=params["hard_centers"], alpha=params["alpha"]
    )
    clean_x, clean_y = base.gaussian_hard_samples(
        z, y, tr, train_centers, roots, count=count, sigma=sigma,
        seed=params["seed"] + 400000 + split_no,
    )
    values, predictions_real, predictions_fake = [], [], []
    for kind in params["models"]:
        plain = fit(kind, z, y, train=tr, synthetic=None,
                    mass="full", seed=params["seed"]+split_no,
                    config=params)
        clean_aug = fit(kind, z, y, train=tr,
                        synthetic=(clean_x, clean_y), mass=params["mass"],
                        seed=params["seed"]+split_no, config=params)
        contaminated = fit(kind, z, y, train=tr,
                           synthetic=(leak_x, leak_y), mass=params["mass"],
                           seed=params["seed"]+split_no, config=params)
        exact = fit(kind, z, y, train=tr,
                    synthetic=(exact_x, exact_y), mass=params["mass"],
                    seed=params["seed"]+split_no, config=params)
        real = {
            "clean_real_test": plain.predict(z[te]),
            "clean_training_augmented_real_test": clean_aug.predict(z[te]),
            "leaky_test_gaussian_training_real_test": contaminated.predict(z[te]),
            "leaky_test_exact_copies_training_real_test": exact.predict(z[te]),
        }
        fake = {
            "clean_model_synthetic_expanded_test": plain.predict(eval_x),
            "leaky_test_gaussian_training_expanded_test": contaminated.predict(eval_x),
        }
        for scenario in SCENARIOS:
            is_expanded = scenario in fake
            if is_expanded:
                forecast = np.r_[real[
                    "clean_real_test" if scenario.startswith("clean_")
                    else "leaky_test_gaussian_training_real_test"
                ], fake[scenario]]
                observed = np.r_[y[te], eval_y]
            else:
                forecast, observed = real[scenario], y[te]
            values.append({
                "repeat": split_no, "split_seed": params["seed"]+split_no,
                "model": kind, "scenario": scenario,
                "n_real_train": len(tr), "n_real_test": len(te),
                "n_synthetic_train": count if scenario.startswith(
                    ("clean_training_", "leaky_test_")
                ) else 0,
                "n_synthetic_test": count if is_expanded else 0,
                "n_evaluated": len(observed),
                "n_unique_real_test_parents": len(te),
                "accessed_heldout_y_for_training": scenario.startswith("leaky_test_"),
                "evaluated_synthetic_test_labels": is_expanded,
                **metric(observed, forecast),
            })
        for j, idx in enumerate(te):
            record = {
                "repeat": split_no, ID: ids[idx], "observed_real_test_y": y[idx],
                "model": kind, "split_seed": params["seed"] + split_no,
            }
            for name, vector in real.items():
                record[name] = float(vector[j])
            predictions_real.append(record)
        for j, origin in enumerate(eval_origin):
            predictions_fake.append({
                "repeat": split_no, "model": kind,
                "source_real_test_id": ids[origin],
                "pseudo_y_copied_from_test": float(eval_y[j]),
                "clean_model_prediction": float(fake["clean_model_synthetic_expanded_test"][j]),
                "leaky_model_prediction": float(fake["leaky_test_gaussian_training_expanded_test"][j]),
            })
    return (
        pd.DataFrame(values),
        pd.DataFrame(predictions_real),
        pd.DataFrame(predictions_fake),
    )


def init_worker(data):
    global WORKER
    WORKER = data


def process_repeat(i):
    if WORKER is None:
        raise RuntimeError("Worker payload not initialized.")
    return i, run_one(i, WORKER)


def summarize(folder: Path, *, expected_reps: int, models: tuple):
    score_paths = [folder / f"scores_{i:03d}.csv" for i in range(expected_reps)]
    if not all(p.is_file() for p in score_paths):
        raise FileNotFoundError("Some split scores are incomplete.")
    results = pd.concat([pd.read_csv(p) for p in score_paths], ignore_index=True)
    expected = expected_reps * len(models) * len(SCENARIOS)
    if (len(results) != expected
            or results.duplicated(["repeat", "model", "scenario"]).any()
            or set(results["scenario"]) != set(SCENARIOS)):
        raise ValueError("Incomplete or duplicated scenarios.")
    agg = results.groupby(["model", "scenario"], sort=True)["rmse_t_ha"].agg(
        mean="mean", median="median", sd="std",
        minimum="min", maximum="max",
    )
    agg = agg.reset_index()
    for q in (5, 25, 50, 75, 95):
        subset = results.groupby(["model", "scenario"])["rmse_t_ha"].quantile(q/100)
        agg[f"p{q:02d}"] = [subset.loc[(m, s)] for m, s in
                            zip(agg["model"], agg["scenario"])]
    agg["n_repetitions"] = expected_reps
    # Paired delta-RMSE against the clean model on IDENTICAL splits.
    clean = results.loc[results["scenario"] == "clean_real_test", [
        "repeat", "model", "rmse_t_ha"
    ]].rename(columns={"rmse_t_ha": "clean_rmse_t_ha"})
    paired = results.merge(clean, on=["repeat", "model"], validate="many_to_one")
    paired["delta_vs_same_split_clean_rmse"] = (
        paired["rmse_t_ha"] - paired["clean_rmse_t_ha"]
    )
    delta_mean = paired.groupby(["model", "scenario"])[
        "delta_vs_same_split_clean_rmse"
    ].mean()
    agg["mean_paired_delta_vs_clean"] = [
        delta_mean.loc[(m, s)] for m, s in
        zip(agg["model"], agg["scenario"])
    ]
    agg = agg.sort_values(["model", "mean"]).reset_index(drop=True)
    agg.to_csv(folder/"summary_rmse.csv", index=False)
    paired.to_csv(folder/"all_split_metrics.csv", index=False)
    agg.pivot(index="model", columns="scenario", values="mean").to_csv(
        folder/"mean_rmse_by_model.csv"
    )
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 5.5))
    selections = [
        "clean_real_test",
        "clean_training_augmented_real_test",
        "clean_model_synthetic_expanded_test",
        "leaky_test_gaussian_training_real_test",
        "leaky_test_gaussian_training_expanded_test",
        "leaky_test_exact_copies_training_real_test",
    ]
    ordered = [(kind, scene) for kind in models for scene in selections]
    values = [
        results.loc[(results["model"] == kind) &
                    (results["scenario"] == scene), "rmse_t_ha"].to_numpy()
        for kind, scene in ordered
    ]
    short_labels = [
        kind + chr(10) + scene.replace("leaky_test_", "LEAK ").replace(
            "clean_", "clean "
        ).replace("_", " ") for kind, scene in ordered
    ]
    try:
        ax.boxplot(values, tick_labels=short_labels, showfliers=False)
    except TypeError:  # Matplotlib before 3.9
        ax.boxplot(values, labels=short_labels, showfliers=False)
    ax.set_ylabel("Per-repetition RMSE (t/ha)")
    ax.set_title("Intentional test contamination: invalid apparent RMSE changes")
    ax.tick_params(axis="x", labelrotation=75, labelsize=7)
    fig.tight_layout()
    fig.savefig(folder/"rmse_distributions_leakage_comparison.png", dpi=160)
    plt.close(fig)
    print("[INTENTIONAL LEAKAGE — NOT VALIDATION] 100x 70/30 descriptive scores")
    print(agg.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    print("IMPORTANT: fake-test rows are dependent children of true test parents;")
    print("leaked-training scenarios explicitly COPY heldout y into training.")
    return agg


def main():
    p = argparse.ArgumentParser(description=__doc__)
    action = p.add_mutually_exclusive_group(required=True)
    for name in ("check", "smoke", "run", "summary"):
        action.add_argument("--"+name, action="store_true")
    p.add_argument("--bundle", type=Path, default=base.historic.native.BUNDLE)
    p.add_argument("--peer-repo", type=Path, default=base.historic.native.PEER_REPO)
    p.add_argument("--config", type=Path, default=base.historic.native.CONFIG)
    p.add_argument("--fresh-fold-dir", type=Path, default=base.historic.native.OUTPUT)
    p.add_argument("--anchor-file", type=Path, default=base.historic.ANCHOR_SOURCE)
    p.add_argument("--output", type=Path, default=OUTPUT)
    p.add_argument("--repetitions", type=int, default=100)
    p.add_argument("--synthetic-count", type=int, default=512)
    p.add_argument("--sigma", type=float, default=0.08)
    p.add_argument("--mass", choices=["32", "full"], default="full")
    p.add_argument("--hard-centers", type=int, default=8)
    p.add_argument("--trees", type=int, default=64)
    p.add_argument("--leaf", type=int, default=3)
    p.add_argument("--alpha", type=float, default=30.)
    p.add_argument("--neighbors", type=int, default=20)
    p.add_argument("--models", default="ridge,extra_trees")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--jobs", type=int, default=4)
    args = p.parse_args()
    models = tuple(x.strip() for x in args.models.split(","))
    if (args.repetitions < 1 or args.synthetic_count < 42
            or args.sigma < 0 or args.alpha <= 0 or args.trees < 4
            or args.leaf < 1 or args.hard_centers < 2
            or args.jobs < 1 or args.neighbors < 3
            or not models or set(models) - set(MODELS)):
        p.error("Invalid models/augmentation/splits.")
    folder = args.output.resolve()
    if args.summary:
        spec = json.loads((folder/"run_spec.json").read_text(encoding="utf-8"))
        summarize(folder, expected_reps=spec["repetitions"],
                  models=tuple(spec["models"]))
        return
    frame, full, z, root, anchors, y, loo, features, ids = base.loaded(args)
    if args.check:
        tr, te = split_indices(N, seed=args.seed)
        print(f"[CHECK] 197 observable X, 138 true labels, 59 hidden y. "
              f"First random split: {len(tr)} REAL train/{len(te)} REAL test.")
        print(f"[CHECK] {args.repetitions} random splits; {args.synthetic_count} "
              f"TEST-ORIGIN Gaussian children per repetition; {len(models)} models.")
        print("[WARNING] Test-derived synthetic targets deliberately copied from heldout y.")
        return
    count = min(3, args.repetitions) if args.smoke else args.repetitions
    folder = folder/"smoke" if args.smoke else folder
    params = {
        "seed": args.seed, "synthetic_count": args.synthetic_count,
        "sigma": args.sigma, "mass": args.mass,
        "hard_centers": args.hard_centers, "alpha": args.alpha,
        "trees": args.trees, "leaf": args.leaf, "models": list(models),
    }
    # Root covariance uses ONLY observable X and is shared across splits.
    roots = base.local_roots(z, tuple(range(N)), neighbors=args.neighbors)
    spec = {
        "protocol": "100 RANDOM 70/30 SPLITS; INTENTIONAL TEST ORIGIN LEAKAGE",
        "repetitions": count, "models": list(models), "settings": params,
        "base_bundle_sha256": base.digest(args.bundle.resolve()/"manifest.json"),
        "runner_sha256": base.digest(Path(__file__).resolve()),
        "source_augmentation_sha256": base.digest(
            ROOT/"tools/run_global_synthetic_augmentation.py"
        ),
        "synthetic_test_y_is_copied_from_heldout_y": True,
        "test_origin_children_used_for_model_training": True,
        "invalid_for_unbiased_generalization": True,
        "all_59_unknown_y_remain_unread": True,
    }
    folder.mkdir(parents=True, exist_ok=True)
    spec_path = folder/"run_spec.json"
    if spec_path.is_file():
        if json.loads(spec_path.read_text(encoding="utf-8")) != spec:
            raise ValueError("Different run/source settings; use a NEW --output.")
    else:
        if list(folder.glob("scores_*.csv")):
            raise ValueError("Scores exist without immutable manifest.")
        spec_path.write_text(json.dumps(spec, indent=2)+"\n", encoding="utf-8")
    expected_columns = {
        "repeat", "model", "scenario", "rmse_t_ha",
        "n_real_train", "n_real_test", "accessed_heldout_y_for_training"
    }
    missing = []
    for i in range(count):
        sc = folder/f"scores_{i:03d}.csv"
        real = folder/f"real_test_{i:03d}_private.csv"
        fake = folder/f"fake_test_{i:03d}_private.csv"
        if all(x.is_file() for x in (sc, real, fake)):
            saved = pd.read_csv(sc)
            if (len(saved) != len(models)*len(SCENARIOS)
                    or not expected_columns.issubset(saved.columns)
                    or saved["repeat"].nunique() != 1
                    or int(saved["repeat"].iloc[0]) != i):
                raise ValueError(f"Bad existing split output {sc}.")
        elif any(x.exists() for x in (sc, real, fake)):
            raise ValueError(f"Partial split {i}: remove partial local files manually.")
        else:
            missing.append(i)
    print(f"[SPLITS] {count} requested, {len(missing)} new, "
          f"jobs={args.jobs}, synthetic test children={args.synthetic_count}")
    payload = (z, y, ids, roots, params)
    if missing:
        with ProcessPoolExecutor(
            max_workers=min(len(missing), args.jobs),
            initializer=init_worker, initargs=(payload,),
        ) as pool:
            tasks = {pool.submit(process_repeat, i): i for i in missing}
            for done, future in enumerate(as_completed(tasks), 1):
                i, (scores, real, fake) = future.result()
                # Save the score last so partial outputs cannot be accepted.
                real.to_csv(folder/f"real_test_{i:03d}_private.csv", index=False)
                fake.to_csv(folder/f"fake_test_{i:03d}_private.csv", index=False)
                scores.to_csv(folder/f"scores_{i:03d}.csv", index=False)
                if done % 10 == 0 or done == len(missing):
                    print(f"[SPLITS] completed {done}/{len(missing)}", flush=True)
    if args.smoke:
        print("[SMOKE] Test-born artificial labels checked; not a 100-split result.")
    else:
        summarize(folder, expected_reps=count, models=models)


if __name__ == "__main__":
    main()
