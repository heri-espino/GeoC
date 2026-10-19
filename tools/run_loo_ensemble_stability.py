#!/usr/bin/env python3
"""Test whether a small Reference26_mean + Compact14 blend is stable.

Uses only ALREADY SAVED true 138-fold predictions. No new base-model fits,
no model promotion, no parcel-level outputs. Results are descriptive since
the mixing weight and candidate family were chosen using these same labels.

    python tools/run_loo_ensemble_stability.py --run
    python tools/run_loo_ensemble_stability.py --summary

Optional:
    python tools/run_loo_ensemble_stability.py --run --candidate Local04D
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from run_loo_ensemble_screen import (
    ANCHOR,
    BASELINE_FILE,
    DBSCAN_DIR,
    LOCAL_FILE,
    MEAN_DIR,
    OUTPUT,
    ID,
    TARGET,
    join_models,
    load_baselines,
    load_mean,
    read_dbscan,
    read_local,
)


WEIGHTS = (0., 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.75, 1.)
DEFAULT_CANDIDATE = "Compact14"
DEFAULT_WEIGHT = 0.30
DEFAULT_BOOTSTRAP = 5000
DEFAULT_SEED = 20261019


def aligned_candidates(
    *, mean_dir: Path, baseline_file: Path,
    local_file: Path, dbscan_file: Path,
) -> pd.DataFrame:
    """Strictly align original y and every available candidate by parcel ID."""
    original, compact = load_baselines(baseline_file)
    anchor = load_mean(mean_dir)
    candidates = [original, compact]
    try:
        candidates.append(read_local(local_file, anchor))
    except FileNotFoundError:
        pass
    try:
        candidates.append(read_dbscan(dbscan_file))
    except FileNotFoundError:
        pass
    return join_models(anchor, candidates)


def optimal_convex_weight(y: np.ndarray, anchor: np.ndarray, other: np.ndarray):
    """In-sample least-squares weight only; NOT cross-validated."""
    residual = anchor-y
    shift = other-anchor
    denom = float(shift @ shift)
    if denom <= 0.:
        return 0.
    return float(np.clip(-float(residual @ shift) / denom, 0., 1.))


def summarize_weight_grid(
    y: np.ndarray, anchor: np.ndarray, other: np.ndarray,
) -> pd.DataFrame:
    base_error = (anchor-y)**2
    base_mse = float(np.mean(base_error))
    rows = []
    for w in WEIGHTS:
        blend = (1.-w)*anchor+w*other
        e = blend-y
        loss_improvement = base_error - e**2
        rows.append({
            "weight_candidate": w,
            "weight_reference26mean": round(1.-w, 3),
            "rmse": float(np.sqrt(np.mean(e**2))),
            "mae": float(np.mean(np.abs(e))),
            "bias": float(np.mean(e)),
            "delta_mse_vs_reference26mean": float(np.mean(e**2)-base_mse),
            "fraction_parcels_lower_squared_error": float(
                np.mean(loss_improvement > 1e-12)
            ),
        })
    return pd.DataFrame(rows)


def paired_diagnostics(
    y: np.ndarray, anchor: np.ndarray, other: np.ndarray,
    weight: float, *, bootstrap: int, seed: int,
) -> tuple[dict, pd.DataFrame]:
    if not (0 <= weight <= 1) or bootstrap <= 0:
        raise ValueError("Weight must be in [0,1], and bootstrap must be positive.")
    n = len(y)
    if n != 138 or len(anchor) != n or len(other) != n:
        raise ValueError("Require 138 common LOOCV heldout predictions.")
    base = anchor-y
    candidate = (1.-weight)*anchor + weight*other-y
    delta = candidate**2-base**2   # negative means the blend is better
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, n, size=(bootstrap, n))
    means = delta[draw].mean(axis=1)
    ci = np.quantile(means, (0.025, 0.975))
    # How many large wins make the apparent ensemble improvement?
    indices_by_benefit = np.argsort(delta, kind="stable")
    sensitivity = []
    for removed in (0, 1, 2, 5, 10):
        kept = np.ones(n, bool)
        if removed:
            kept[indices_by_benefit[:removed]] = False
        sensitivity.append({
            "largest_improvements_removed": removed,
            "remaining_parcels": int(kept.sum()),
            "rmse_reference26mean": float(
                np.sqrt(np.mean(base[kept]**2))
            ),
            "rmse_blend": float(
                np.sqrt(np.mean(candidate[kept]**2))
            ),
            "delta_mse_blend_minus_reference26mean": float(
                np.mean(delta[kept])
            ),
        })
    details = {
        "weight": weight,
        "n_heldouts": n,
        "rmse_reference26mean": float(math.sqrt(np.mean(base**2))),
        "rmse_blend": float(math.sqrt(np.mean(candidate**2))),
        "delta_mse_blend_minus_reference26mean": float(delta.mean()),
        "improved_parcels": int(np.sum(delta < -1e-12)),
        "worsened_parcels": int(np.sum(delta > 1e-12)),
        "unchanged_parcels": int(np.sum(abs(delta) <= 1e-12)),
        "mean_improvement_when_better": float(
            -delta[delta < -1e-12].mean()
        ) if np.any(delta < -1e-12) else 0.,
        "mean_deterioration_when_worse": float(
            delta[delta > 1e-12].mean()
        ) if np.any(delta > 1e-12) else 0.,
        "bootstrap_delta_mse_ci_lower": float(ci[0]),
        "bootstrap_delta_mse_ci_upper": float(ci[1]),
        "bootstrap_fraction_blend_better": float(np.mean(means < 0)),
        "bootstrap_samples": bootstrap,
        "bootstrap_seed": seed,
    }
    return details, pd.DataFrame(sensitivity)


def report(
    table: pd.DataFrame, candidate: str,
    weight: float, bootstrap: int, seed: int,
    output: Path,
):
    if candidate == ANCHOR or candidate not in table.columns:
        raise ValueError(
            f"Unknown candidate {candidate!r}. "
            f"Available: {sorted(set(table.columns)-{ID, TARGET})}"
        )
    y = table[TARGET].to_numpy(float)
    anchor = table[ANCHOR].to_numpy(float)
    other = table[candidate].to_numpy(float)
    grid = summarize_weight_grid(y, anchor, other)
    stats, influence = paired_diagnostics(
        y, anchor, other, weight, bootstrap=bootstrap, seed=seed
    )
    stats["candidate"] = candidate
    stats["exploratory_optimal_weight"] = optimal_convex_weight(
        y, anchor, other
    )
    output.mkdir(parents=True, exist_ok=True)
    label = candidate.lower()
    grid.to_csv(output / f"stability_{label}_weight_grid.csv", index=False)
    influence.to_csv(output / f"stability_{label}_influence.csv", index=False)
    pd.DataFrame([stats]).to_csv(
        output / f"stability_{label}_summary.csv", index=False
    )
    print(
        f"[STABILITY] {ANCHOR} vs {candidate}, fixed "
        f"weight={weight:g}, n=138.\n"
        f"  base RMSE={stats['rmse_reference26mean']:.6f}; "
        f"blend RMSE={stats['rmse_blend']:.6f}\n"
        f"  blend minus base ΔMSE={stats['delta_mse_blend_minus_reference26mean']:.7f}\n"
        f"  parcels improved={stats['improved_parcels']}, "
        f"worsened={stats['worsened_parcels']}, "
        f"unchanged={stats['unchanged_parcels']}\n"
        f"  naive paired-bootstrap 95% CI for ΔMSE="
        f"[{stats['bootstrap_delta_mse_ci_lower']:.7f}, "
        f"{stats['bootstrap_delta_mse_ci_upper']:.7f}]\n"
        f"  exploratory continuous optimum weight="
        f"{stats['exploratory_optimal_weight']:.3f}\n"
        f"[SENSITIVITY] Remove parcels with largest individual blend gains:\n"
        + influence.to_string(index=False) + "\n"
        "[WEIGHTS] RMSE along the convex blend:\n"
        + grid.to_string(index=False),
        flush=True,
    )
    print(
        "[LIMITATIONS] Weight and model were considered using these "
        "same 138 observed labels. Bootstrap assumes independent parcels, "
        "which spatial dependence may violate. These are diagnostic "
        "intervals, not independent confirmation of model selection.",
        flush=True,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    action = p.add_mutually_exclusive_group(required=True)
    action.add_argument("--run", action="store_true")
    action.add_argument("--summary", action="store_true")
    p.add_argument("--candidate", default=DEFAULT_CANDIDATE)
    p.add_argument("--weight", type=float, default=DEFAULT_WEIGHT)
    p.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--mean-dir", type=Path, default=MEAN_DIR)
    p.add_argument("--baseline-file", type=Path, default=BASELINE_FILE)
    p.add_argument("--local-file", type=Path, default=LOCAL_FILE)
    p.add_argument("--dbscan-file", type=Path,
                   default=DBSCAN_DIR / "loo_predictions.csv")
    p.add_argument("--output", type=Path, default=OUTPUT)
    args = p.parse_args()
    output = args.output.resolve()
    if not all(c.isalnum() or c == "_" for c in args.candidate):
        p.error("--candidate must be an existing safe model name.")
    label = args.candidate.lower()
    if args.summary:
        file = output / f"stability_{label}_summary.csv"
        if not file.is_file():
            raise FileNotFoundError(
                f"Saved stability report not found at {file}; run --run."
            )
        print(pd.read_csv(file).to_string(index=False))
        return 0
    table = aligned_candidates(
        mean_dir=args.mean_dir.resolve(),
        baseline_file=args.baseline_file.resolve(),
        local_file=args.local_file.resolve(),
        dbscan_file=args.dbscan_file.resolve(),
    )
    report(
        table, args.candidate, args.weight, args.bootstrap,
        args.seed, output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
