#!/usr/bin/env python3
"""Checkpoint 10: one interpretable nested PCA dimensionality × Ridge alpha CV.

Usage:
  python tools/run_checkpoint_10_pca.py --check
  python tools/run_checkpoint_10_pca.py --smoke
  python tools/run_checkpoint_10_pca.py --run
  python tools/run_checkpoint_10_pca.py --summary

Reuses the existing 16 X-only Checkpoint 08 confirmation memberships as
EXPLORATORY outer splits, with 97 training and 41 held-out labels each.
Inner threefold CV on 97 labels searches a joint 4-dimensional grid:
  PCA = 20 / 40 / 80 / 160; k = 12 / 20 / 24 / 36;
  Ridge alpha = 10 / 30 / 100 / 300; power p = 0.5 / 1 / 2.
  192 combinations per outer split; 16 splits; 3 inner folds.
The original PCA20/k24/alpha30/p1 stays frozen only as the paired control.
Does not overwrite any prior model or the 59 final predictions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from geocebada.evaluation.checkpoint08_local import ID, digest, atomic_json
from geocebada.evaluation.checkpoint10_pca import (
    ALPHA_GRID, DIMS, FROZEN, K_GRID, POWER_GRID, assemble_split,
    inner_scores, inputs, load_pca_data, predict, summarize,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "reports/checkpoint_10_pca"
MEMBERSHIPS = ROOT / "reports/checkpoint_08/confirmation_membership.csv"
MODELS = {*[f"PCA{n}_nested" for n in DIMS], "PCA20_frozen", "PCA_joint_nested"}


def _preflight(membership: Path) -> dict[str, Path]:
    paths = inputs(ROOT, membership)
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing required Checkpoint 10 input '{name}': {path}")
    frame = pd.read_csv(membership)
    if (set(frame["split_id"]) != {f"confirm_{n:02d}" for n in range(1, 17)}
            or len(frame) != 16 * 138):
        raise ValueError("Checkpoint 10 requires all 16 frozen Checkpoint 08 masks.")
    print("[10] Inputs and 16 Checkpoint 08 X-only outer masks verified.", flush=True)
    return paths


def _ensure_spec(output: Path, paths: dict[str, Path], *, smoke: bool) -> None:
    output.mkdir(parents=True, exist_ok=True)
    spec = {
        "schema": 1, "smoke": smoke, "components": list(DIMS),
        "neighbors_k": list((20, 24) if smoke else K_GRID),
        "alpha": list((30.0,) if smoke else ALPHA_GRID),
        "distance_power_p": list((1.0,) if smoke else POWER_GRID),
        "frozen_reference": FROZEN,
        "inner_folds": 2 if smoke else 3, "outer_splits": 1 if smoke else 16,
        "seed": 20261014,
        "sha256_inputs": {k: digest(v) for k, v in paths.items()},
        "sha256_code": {
            k: digest(ROOT / k) for k in (
                "src/geocebada/evaluation/checkpoint04a.py",
                "src/geocebada/evaluation/checkpoint04d1.py",
                "src/geocebada/evaluation/checkpoint10_pca.py",
                "tools/run_checkpoint_10_pca.py",
            )
        },
    }
    manifest = output / "run_spec.json"
    if manifest.exists():
        if json.loads(manifest.read_text(encoding="utf-8")) != spec:
            raise RuntimeError("Checkpoint 10 run fingerprint changed; use a fresh --output.")
    else:
        if list(output.glob("confirm_*")) or list(output.glob("*summary*")):
            raise RuntimeError("Existing study results have no matching run_spec.json.")
        atomic_json(manifest, spec)


def _check_checkpoint08_parity(data) -> None:
    path = ROOT / "reports/checkpoint_08/confirm_01__paired.csv"
    if not path.is_file():
        print("[10] No local checkpoint08 paired forecast found; PCA20 coordinate parity still required.", flush=True)
        return
    original = pd.read_csv(path)
    original = original.loc[original["model"].eq("Local04D_frozen")]
    train, test = data.folds["confirm_01"]
    if len(original) != len(test):
        raise ValueError("Checkpoint 08 reference has incomplete baseline coverage.")
    expected = original.set_index(ID).loc[test, "predicted"].to_numpy(float)
    actual = predict(data, train, test, components=20, **FROZEN)
    parity = float(np.max(np.abs(actual - expected)))
    if not np.isfinite(parity) or parity > 1e-7:
        raise RuntimeError(f"Frozen Checkpoint 08 baseline parity failed: {parity:.3g}")
    print(f"[10] Checkpoint 08 20-PC baseline prediction parity PASS ({parity:.2g}).", flush=True)


def _run(args: argparse.Namespace, paths: dict[str, Path], *, smoke: bool) -> None:
    output = args.output / "smoke" if smoke else args.output
    _ensure_spec(output, paths, smoke=smoke)
    data = load_pca_data(ROOT, MEMBERSHIPS, max_components=160)
    print(f"[10] Rebuilt 2,082-feature PCA; PC20 historical parity "
          f"{data.parity_max_diff:.2g}.", flush=True)
    print("[10] Cumulative PCA explained variance: " + ", ".join(
        f"{n}: {data.variance[n]:.2%}" for n in DIMS
    ), flush=True)
    _check_checkpoint08_parity(data)
    all_splits = sorted(data.folds)
    selected = all_splits[:1] if smoke else all_splits
    for i, split in enumerate(selected, 1):
        paired = output / f"{split}__paired.csv"
        inner_path = output / f"{split}__inner_grid.csv"
        meta_path = output / f"{split}__selection.json"
        if paired.is_file() and inner_path.is_file() and meta_path.is_file():
            found = pd.read_csv(paired)
            if len(found) != len(MODELS) * 41 or set(found["model"]) != MODELS:
                raise RuntimeError(f"Invalid cached paired results for {split}")
            print(f"[10 {i}/{len(selected)}] {split}: completed previously.", flush=True)
            continue
        train, _ = data.folds[split]
        grids = {
            n: inner_scores(
                data, train, components=n,
                seed=20261014 + int(split[-2:]), smoke=smoke,
            )
            for n in DIMS
        }
        predictions, selection = assemble_split(data, split, grids)
        if len(predictions) != len(MODELS) * 41 or set(predictions["model"]) != MODELS:
            raise RuntimeError("Incomplete paired output.")
        # Save after the complete split; resume never mixes half-written rows.
        inner_frame = pd.concat(grids.values(), ignore_index=True)
        temp_grid = inner_path.with_suffix(".csv.tmp")
        inner_frame.to_csv(temp_grid, index=False)
        temp_grid.replace(inner_path)
        temp_paired = paired.with_suffix(".csv.tmp")
        predictions.to_csv(temp_paired, index=False)
        temp_paired.replace(paired)
        atomic_json(meta_path, selection)
        print(f"[10 {i}/{len(selected)}] {split}: CV and outer forecasts completed.", flush=True)
    if smoke:
        print("[10 SMOKE] One split × 4 PCA ranks × 2 neighbors; not a final result.", flush=True)
    else:
        _summary(args.output)


def _summary(output: Path) -> None:
    paths = sorted(output.glob("confirm_*__paired.csv"))
    if len(paths) != 16:
        raise RuntimeError(f"Only {len(paths)}/16 paired splits present.")
    predictions = pd.concat((pd.read_csv(p) for p in paths), ignore_index=True)
    scores, intervals = summarize(predictions)
    scores.to_csv(output / "comparison_summary.csv", index=False)
    intervals.to_csv(output / "paired_cluster_bootstrap.csv", index=False)
    grid_files = sorted(output.glob("confirm_*__inner_grid.csv"))
    if len(grid_files) != 16:
        raise RuntimeError(f"Only {len(grid_files)}/16 inner-CV grids present.")
    inner = pd.concat((pd.read_csv(p).assign(split_id=p.stem.split("__")[0]) for p in grid_files), ignore_index=True)
    means = inner.groupby(
        ["components", "k", "alpha", "power"], as_index=False
    )["inner_rmse"].agg(["mean", "std"])
    means.to_csv(output / "inner_cv_grid_summary.csv", index=False)
    selections = [json.loads(p.read_text(encoding="utf-8"))
                  for p in sorted(output.glob("confirm_*__selection.json"))]
    if len(selections) != 16:
        raise RuntimeError("Incomplete split-selection reports.")
    chosen = pd.DataFrame({
        "split_id": s["split_id"], **s["joint_best"]
    } for s in selections)
    chosen.to_csv(output / "selected_hyperparameters.csv", index=False)
    print("[10] PAIRED outer-validation RMSE/R2, lower RMSE is better:\n"
          + scores.to_string(index=False), flush=True)
    print("[10] MSE difference against fixed PCA20, negative is better:\n"
          + intervals.to_string(index=False), flush=True)
    print("[10] Nested joint (PCA, k, alpha, p) selection counts:\n"
          + chosen["components"].value_counts().sort_index().to_string(), flush=True)
    print("[10] Results exploratory on previously used Checkpoint 08 splits."
          " No promotion, no changes to 59 forecasts.", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    for name in ("check", "smoke", "run", "summary"):
        action.add_argument("--" + name, action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT)
    args = parser.parse_args()
    paths = _preflight(MEMBERSHIPS)
    if args.check:
        data = load_pca_data(ROOT, MEMBERSHIPS, max_components=160)
        _check_checkpoint08_parity(data)
        print("[10 CHECK] Rebuilt C4 PCA160, PCA20 parity and source data PASS.", flush=True)
    elif args.summary:
        _summary(args.output)
    else:
        _run(args, paths, smoke=args.smoke)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
