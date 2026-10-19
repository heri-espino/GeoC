#!/usr/bin/env python3
"""Massive, zero-training convex ensemble search on five aligned 138-fold OOF models.

For 5 models at 1%-resolution, there are exactly C(104,4)=4,598,126
different nonnegative, sum-to-one weight combinations. Evaluate them using
the 5x5 error Gram matrix, never constructing each 138-row prediction vector.

Also solve the continuous constrained quadratic program (global optimum);
compare all nonempty predictor subsets, anchored regularization, and
descriptive paired robustness without claiming unbiased outer validation.

  python tools/run_loo_ensemble_massive.py --run
  python tools/run_loo_ensemble_massive.py --run --step 0.02
  python tools/run_loo_ensemble_massive.py --summary

Requires no base fitting and writes ONLY aggregated scores/weights, no rows
containing parcel IDs or private individual target/prediction data.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
from pathlib import Path

for _key in (
    "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_key] = "1"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

from run_loo_ensemble_screen import (  # noqa: E402
    ANCHOR,
    BASELINE_FILE,
    DBSCAN_DIR,
    ID,
    LOCAL_FILE,
    MEAN_DIR,
    OUTPUT,
    TARGET,
    join_models,
    load_baselines,
    load_extra,
    load_mean,
    read_dbscan,
    read_local,
)

DEFAULT_OUTPUT = OUTPUT / "massive"
DEFAULT_MODELS = 5
DEFAULT_STEP = 0.01
DEFAULT_CHUNK = 50000
DEFAULT_KEEP = 30
DEFAULT_BOOTSTRAP = 3000
DEFAULT_SEED = 20261019
PENALTIES = (0., 0.0001, 0.001, 0.01, 0.1)


def load_table(
    mean_dir: Path, baseline_file: Path,
    local_file: Path, dbscan_file: Path,
    extra: list[str],
) -> pd.DataFrame:
    original, compact = load_baselines(baseline_file)
    anchor = load_mean(mean_dir)
    others = [original, compact]
    try:
        others.append(read_local(local_file, anchor))
    except FileNotFoundError as err:
        print(f"[SKIP] Local04D unavailable: {err}", flush=True)
    try:
        others.append(read_dbscan(dbscan_file))
    except FileNotFoundError as err:
        print(f"[SKIP] DBSCAN unavailable: {err}", flush=True)
    for item in extra:
        _, model = load_extra(item)
        others.append(model)
    return join_models(anchor, others)


def gram_from_predictions(table: pd.DataFrame):
    names = [c for c in table if c not in (ID, TARGET)]
    if names[0] != ANCHOR:
        raise ValueError("Mean Reference26 must remain the first model.")
    observed = table[TARGET].to_numpy(dtype=np.float64)
    predictions = table[names].to_numpy(dtype=np.float64)
    errors = predictions - observed[:, None]
    gram = (errors.T @ errors) / len(observed)
    return names, observed, predictions, errors, gram


def count_simplex_weights(n_models: int, n_steps: int) -> int:
    if n_models < 1 or n_steps < 1:
        raise ValueError("Number of models and steps must be positive.")
    return math.comb(n_steps + n_models - 1, n_models - 1)


def discrete_steps(step: float) -> int:
    if not np.isfinite(step) or not (0 < step <= 1):
        raise ValueError("step must be finite and between 0 and 1.")
    steps = round(1 / step)
    if not np.isclose(steps * step, 1, rtol=0, atol=1e-10):
        raise ValueError("Use a step dividing 1 exactly (e.g. .01 or .02).")
    return steps


def simplex_batches(
    models: int, steps: int, chunk_size: int,
):
    """Complete stars-and-bars simplex enumeration, streamed in chunks."""
    if chunk_size < 1 or models < 1 or steps < 1:
        raise ValueError("Invalid simplex enumeration parameters.")
    values = np.empty((chunk_size, models), dtype=np.float64)
    filled = 0
    # Every set of bars has a unique integer composition of 'steps'.
    for cuts in itertools.combinations(range(steps + models - 1), models - 1):
        previous = -1
        for j, cut in enumerate(cuts):
            values[filled, j] = (cut - previous - 1) / steps
            previous = cut
        values[filled, -1] = (steps + models - 2 - previous) / steps
        filled += 1
        if filled == chunk_size:
            yield values.copy()
            filled = 0
    if filled:
        yield values[:filled].copy()


def gram_mse(weights: np.ndarray, gram: np.ndarray):
    """Exact pooled squared error of ANY mixture, no 138-row refit."""
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim == 1:
        return float(w @ gram @ w)
    return np.sum((w @ gram) * w, axis=1)


def exhaustive_grid(
    gram: np.ndarray, *, step: float, chunk: int, keep: int,
    max_evals: int,
) -> tuple[pd.DataFrame, int]:
    """Stream every discrete convex combination; retain only best 'keep'."""
    m = len(gram)
    steps = discrete_steps(step)
    n = count_simplex_weights(m, steps)
    if n > max_evals:
        raise ValueError(
            f"{n:,} combinations exceed --max-evals={max_evals:,}. "
            "Increase --max-evals explicitly or use a coarser --step; "
            "the constrained quadratic optimizer finds the continuous "
            "best weights without enumerating them."
        )
    saved_losses = np.empty(0, float)
    saved_weights = np.empty((0, m), float)
    examined = 0
    for batch in simplex_batches(m, steps, chunk):
        losses = gram_mse(batch, gram)
        how_many = min(keep, len(losses))
        selected = np.argpartition(losses, how_many-1)[:how_many]
        candidate_losses = np.concatenate([saved_losses, losses[selected]])
        candidate_weights = np.vstack([saved_weights, batch[selected]])
        how_many = min(keep, len(candidate_losses))
        chosen = np.argsort(candidate_losses, kind="stable")[:how_many]
        saved_losses = candidate_losses[chosen]
        saved_weights = candidate_weights[chosen]
        examined += len(batch)
    if examined != n:
        raise RuntimeError(f"Enumerated {examined} not {n} combinations.")
    ranking = pd.DataFrame(
        [{"rank": i+1, "mse": float(loss),
          "rmse": math.sqrt(max(0., float(loss))),
          "weights": json.dumps([float(x) for x in weights])}
         for i, (loss, weights) in enumerate(zip(
             saved_losses, saved_weights, strict=True
         ))]
    )
    return ranking, examined


def constrained_optimum(
    gram: np.ndarray, *, penalty: float = 0.,
    anchor_min: float = 0.,
    active: tuple[int, ...] | None = None,
):
    """Continuous simplex-optimal weights via convex quadratic programming.

    penalty * ||w - e_anchor||² encourages sticking with Reference26.
    Restricting to 'active' predictors permits exhaustive 2^M-1 subset search.
    """
    m = len(gram)
    if penalty < 0 or not (0 <= anchor_min <= 1):
        raise ValueError("Invalid regularization or minimum anchor weight.")
    included = tuple(range(m)) if active is None else tuple(active)
    if not included or len(set(included)) != len(included):
        raise ValueError("Use nonempty unique model indices.")
    if any(i < 0 or i >= m for i in included):
        raise ValueError("Invalid selected model indices.")
    if anchor_min > 0 and 0 not in included:
        raise ValueError("Anchor must be in an anchor-constrained subset.")
    local_gram = gram[np.ix_(included, included)]
    prior = np.zeros(len(included))
    if 0 in included:
        prior[included.index(0)] = 1.
    else:
        prior[:] = 1. / len(included)
    matrix = local_gram + penalty * np.eye(len(included))
    linear = penalty * prior
    bounds = [
        ((anchor_min if model == 0 else 0.), 1.) for model in included
    ]
    start = prior.copy()
    if 0 in included and start[included.index(0)] < anchor_min:
        start = start * (1.-anchor_min)
        start[included.index(0)] += anchor_min

    def loss(w):
        return float(w @ matrix @ w - 2. * linear @ w)

    def grad(w):
        return 2. * (matrix @ w-linear)

    result = minimize(
        loss, start, jac=grad, method="SLSQP",
        bounds=bounds,
        constraints=({
            "type": "eq",
            "fun": lambda w: np.sum(w)-1.,
            "jac": lambda w: np.ones(len(w)),
        },),
        options={"ftol": 1e-13, "maxiter": 1000},
    )
    if not result.success:
        raise RuntimeError(
            f"Constrained quadratic fit failed: {result.message}"
        )
    local_w = np.asarray(result.x, float)
    full_w = np.zeros(m, float)
    full_w[list(included)] = local_w
    if (not np.isclose(full_w.sum(), 1., atol=1e-7, rtol=0)
            or np.any(full_w < -1e-8)
            or full_w[0] < anchor_min-1e-7):
        raise RuntimeError("QP returned invalid convex weights.")
    return full_w


def model_score(
    weights: np.ndarray, names: list[str], gram: np.ndarray,
    baseline_mse: float, family: str, **meta,
):
    mse_value = gram_mse(weights, gram)
    return {
        "family": family,
        **meta,
        "rmse": math.sqrt(max(0., mse_value)),
        "delta_mse_vs_ref26mean": mse_value-baseline_mse,
        "nonzero_models": int(np.sum(weights > 1e-5)),
        "weights": json.dumps({
            name: round(float(w), 8)
            for name, w in zip(names, weights, strict=True)
            if w > 1e-7
        }, sort_keys=True),
    }


def optimizer_experiments(gram: np.ndarray, names: list[str]):
    m = len(names)
    base_loss = float(gram[0, 0])
    output = []
    cases = [
        (tuple(range(m)), 0., 0., "unrestricted_full_simplex"),
        (tuple(range(m)), 0., 0.50, "reference26_at_least_half"),
    ]
    for k in range(1, m+1):
        for active in itertools.combinations(range(m), k):
            for penalty in PENALTIES:
                if active == tuple(range(m)) and penalty == 0.:
                    # Already included unrestricted global optimum.
                    continue
                cases.append((active, penalty, 0., "subset_ridge_to_anchor"))
    for active, penalty, anchor_min, label in cases:
        weights = constrained_optimum(
            gram, penalty=penalty,
            anchor_min=anchor_min, active=active
        )
        output.append(model_score(
            weights, names, gram, base_loss, label,
            subset=",".join(names[i] for i in active),
            penalty=penalty, anchor_min=anchor_min,
        ))
    ranking = pd.DataFrame(output).sort_values(
        ["rmse", "nonzero_models", "penalty"], kind="stable"
    ).reset_index(drop=True)
    return ranking


def paired_bootstrap(
    errors: np.ndarray, weights: np.ndarray,
    *, reps: int, seed: int,
) -> dict:
    """Paired row bootstrap for a *fixed* selected blend, descriptively.

    Neither reoptimizes ensemble weights nor adjusts for the preceding
    extensive 138-label-guided search. Spatial dependence also matters.
    """
    if reps < 1:
        raise ValueError("bootstrap repetitions must be positive.")
    residual = errors @ weights
    original = errors[:, 0]
    delta = residual**2-original**2
    rng = np.random.default_rng(seed)
    n = len(delta)
    sample = rng.integers(0, n, (reps, n))
    values = delta[sample].mean(axis=1)
    ci = np.quantile(values, (0.025, 0.975))
    influence = np.sort(delta)
    return {
        "delta_mse": float(delta.mean()),
        "naive_paired_bootstrap_lo": float(ci[0]),
        "naive_paired_bootstrap_hi": float(ci[1]),
        "bootstrap_fraction_delta_negative": float(np.mean(values < 0)),
        "fraction_parcels_improved": float(np.mean(delta < -1e-12)),
        "delta_mse_after_removing_best_1": float(
            influence[1:].mean()
        ),
        "delta_mse_after_removing_best_5": float(
            influence[5:].mean()
        ),
        "delta_mse_after_removing_best_10": float(
            influence[10:].mean()
        ),
        "bootstrap_reps": reps,
        "bootstrap_seed": seed,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--summary", action="store_true")
    p.add_argument("--step", type=float, default=DEFAULT_STEP)
    p.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK)
    p.add_argument("--top", type=int, default=DEFAULT_KEEP)
    p.add_argument("--max-evals", type=int, default=5000000)
    p.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--mean-dir", type=Path, default=MEAN_DIR)
    p.add_argument("--baseline-file", type=Path, default=BASELINE_FILE)
    p.add_argument("--local-file", type=Path, default=LOCAL_FILE)
    p.add_argument("--dbscan-file", type=Path,
                   default=DBSCAN_DIR / "loo_predictions.csv")
    p.add_argument("--extra", action="append", default=[])
    args = p.parse_args()
    out = args.output.resolve()
    if args.summary:
        for file in ("grid_best.csv", "optimized_subsets.csv", "robustness.csv"):
            path = out / file
            if path.is_file():
                print(f"[{file}]\n"
                      + pd.read_csv(path).head(20).to_string(index=False),
                      flush=True)
        return 0
    if args.chunk_size < 1 or args.top < 1:
        p.error("--chunk-size and --top must be positive.")
    table = load_table(
        args.mean_dir.resolve(), args.baseline_file.resolve(),
        args.local_file.resolve(), args.dbscan_file.resolve(), args.extra,
    )
    names, y, predictions, errors, gram = gram_from_predictions(table)
    m = len(names)
    steps = discrete_steps(args.step)
    n = count_simplex_weights(m, steps)
    print(f"[MASSIVE] {m} models: {', '.join(names)}")
    print(f"[MASSIVE] {n:,} different convex combinations "
          f"at {100.*args.step:g}% resolution; n_holdouts={len(y)}.")
    grid, checked = exhaustive_grid(
        gram, step=args.step, chunk=args.chunk_size,
        keep=args.top, max_evals=args.max_evals,
    )
    best_grid = np.asarray(json.loads(grid.iloc[0]["weights"]), float)
    optimized = optimizer_experiments(gram, names)
    baseline_mse = float(gram[0, 0])
    continuous = constrained_optimum(gram)
    continuous_mse = gram_mse(continuous, gram)
    if continuous_mse > gram_mse(best_grid, gram) + 1e-9:
        raise RuntimeError(
            "Continuous simplex optimum is worse than exhaustive grid. "
            "Do not trust these results."
        )
    robustness = []
    # Include predeclared baseline+Compact14 (70/30) if available.
    if "Compact14" in names:
        declared = np.zeros(m)
        declared[0] = .7
        declared[names.index("Compact14")] = .3
        robustness.append({
            "candidate": "reference26mean_70_compact14_30",
            "weights": json.dumps(dict(zip(names, declared, strict=True))),
            **paired_bootstrap(
                errors, declared, reps=args.bootstrap, seed=args.seed,
            ),
        })
    for label, w in (
        ("exhaustive_grid_winner", best_grid),
        ("global_unregularized_qp_winner", continuous),
    ):
        robustness.append({
            "candidate": label,
            "weights": json.dumps(dict(zip(names, w, strict=True))),
            **paired_bootstrap(
                errors, w, reps=args.bootstrap, seed=args.seed,
            ),
        })
    out.mkdir(parents=True, exist_ok=True)
    grid["weights"] = grid["weights"].map(lambda row: json.dumps({
        name: round(float(w), 8)
        for name, w in zip(names, json.loads(row), strict=True)
        if w > 1e-7
    }, sort_keys=True))
    grid["delta_mse_vs_ref26mean"] = grid["mse"] - baseline_mse
    grid.to_csv(out / "grid_best.csv", index=False)
    optimized.to_csv(out / "optimized_subsets.csv", index=False)
    pd.DataFrame(robustness).to_csv(out / "robustness.csv", index=False)
    (out / "run_spec.json").write_text(json.dumps({
        "status": "EXPLORATORY 138-label descriptive screen, not nested stacking",
        "models": names,
        "grid_candidates": checked,
        "step": args.step,
        "no_base_model_training": True,
        "input_sources": {
            "mean_dir": str(args.mean_dir.resolve()),
            "baseline_file": str(args.baseline_file.resolve()),
            "local_file": str(args.local_file.resolve()),
            "dbscan_file": str(args.dbscan_file.resolve()),
            "extra": args.extra,
        },
        "warning": (
            "Weights optimized on the same OOF heldout target labels. "
            "Winner may be optimistically biased; resampling fixed selected "
            "weights does not eliminate selection bias or spatial dependence."
        ),
    }, indent=2) + "\n", encoding="utf-8")
    print("[MASSIVE] Top exact-grid combinations:\n"
          + grid.head(15).to_string(index=False), flush=True)
    print("[MASSIVE] Top constrained mathematical optimizations:\n"
          + optimized.head(15).to_string(index=False), flush=True)
    print(f"[MASSIVE] Reference26_mean RMSE={math.sqrt(baseline_mse):.6f}; "
          f"best exact grid RMSE={float(grid.iloc[0]['rmse']):.6f}; "
          f"continuous QP RMSE={math.sqrt(max(0.,continuous_mse)):.6f}.")
    print("[MASSIVE] Fixed-weight paired stability:\n"
          + pd.DataFrame(robustness).to_string(index=False), flush=True)
    print(
        "[CAUTION] Millions of weight combinations on 138 outcomes "
        "cannot establish unbiased generalization. This is not true "
        "meta-model LOOCV; do not promote a selected winner.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
