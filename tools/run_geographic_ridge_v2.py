#!/usr/bin/env python3
"""Fair local geographic Ridge ablation: fixed PCA20, k24 and p1.

Run from repository root after restoring existing Checkpoint08 memberships:
  python tools/run_geographic_ridge_v2.py --check
  python tools/run_geographic_ridge_v2.py --smoke
  python tools/run_geographic_ridge_v2.py --run
  python tools/run_geographic_ridge_v2.py --summary

All outputs, including any label-bearing rows, remain in gitignored reports.
The existing canonical 59 predictions are never modified.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from geocebada.evaluation.checkpoint08_local import ID, atomic_json, digest
from geocebada.evaluation.checkpoint10_pca import (
    FROZEN,
    inputs,
    load_pca_data,
)
from geocebada.evaluation.geographic_ridge_v2 import (
    ALL_ARMS,
    ALPHAS,
    FROZEN_NAME,
    HUBER_EPSILON,
    HUBER_UPDATES,
    N_FOLDS,
    N_REPEATS,
    NORMALIZED,
    POWER,
    RAW,
    SEED,
    TUNED_ARMS,
    K,
    _evaluate_split,
    _predict,
    _select_alphas,
    _summarize,
)

ROOT = Path(__file__).resolve().parents[1]
MEMBERSHIP = ROOT / "reports/checkpoint_08/confirmation_membership.csv"
OUTPUT = ROOT / "reports/geographic_ridge_v2"
SOURCE_FILES = (
    "src/geocebada/evaluation/geographic_ridge_v2.py",
    "src/geocebada/evaluation/checkpoint04d1.py",
    "src/geocebada/evaluation/checkpoint04a.py",
    "src/geocebada/evaluation/checkpoint08_local.py",
    "src/geocebada/evaluation/checkpoint10_pca.py",
    "tools/run_geographic_ridge_v2.py",
)


def _write_csv(path: Path, table: pd.DataFrame) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    table.to_csv(temp, index=False)
    temp.replace(path)


def _input_files() -> dict[str, Path]:
    paths = inputs(ROOT, MEMBERSHIP)
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name}: {path}")
    return paths


def _fingerprint(folder: Path, paths: dict[str, Path]) -> None:
    meta = {
        "schema": 1,
        "experiment": "geographic_pca20_local_ridge_v2_fair",
        "official_sha256": {name: digest(path) for name, path in paths.items()},
        "implementation_sha256": {
            path: digest(ROOT / path) for path in SOURCE_FILES
        },
        "arms": list(ALL_ARMS),
        "alpha_grid": list(ALPHAS),
        "frozen_parameters": dict(FROZEN),
        "k": K, "power": POWER,
        "folds": N_FOLDS, "repeats": N_REPEATS,
        "seed": SEED,
        "huber_epsilon": HUBER_EPSILON, "huber_updates": HUBER_UPDATES,
        "outer_folds": 16, "outer_train": 97, "outer_holdout": 41,
        "transductive_X_only": True,
        "hidden_target_y_used": False,
    }
    folder.mkdir(parents=True, exist_ok=True)
    spec = folder / "run_spec.json"
    if spec.is_file():
        if json.loads(spec.read_text(encoding="utf-8")) != meta:
            raise RuntimeError(
                "Input or implementation changed. Preserve prior results and "
                "choose a NEW --output directory; do not mix runs."
            )
    elif list(folder.iterdir()):
        raise RuntimeError("Existing result files without run_spec.json.")
    else:
        atomic_json(spec, meta)


def _load():
    data = load_pca_data(ROOT, MEMBERSHIP, max_components=160)
    if len(data.folds) != 16:
        raise RuntimeError("Expected 16 frozen confirmation splits.")
    if not np.isfinite(data.x[20]).all():
        raise ValueError("Nonfinite frozen PCA20 coordinates.")
    reference = ROOT / "reports/checkpoint_08/confirm_01__paired.csv"
    if reference.is_file():
        saved = pd.read_csv(reference)
        saved = saved.loc[saved.model.eq("Local04D_frozen")]
        train, held = data.folds["confirm_01"]
        expected = saved.set_index(ID).loc[held, "predicted"].to_numpy(float)
        recomputed = _predict(data, train, held, FROZEN_NAME, FROZEN["alpha"])
        gap = float(np.max(np.abs(expected - recomputed)))
        if not np.isfinite(gap) or gap > 1e-7:
            raise RuntimeError(f"Original predictions parity FAIL: {gap:.6g}")
        print(f"[RIDGE-V2] Original 08 prediction parity PASS, max Δ={gap:.3g}")
    return data


def _smoke(data) -> None:
    train, held = data.folds["confirm_01"]
    subset = held[:3]
    reference = {
        arm: _predict(data, train, subset, arm, 30.0)
        for arm in ALL_ARMS
    }
    old = data.y.loc[held].copy()
    try:
        data.y.loc[held] = 1000000.0
        for arm in ALL_ARMS:
            np.testing.assert_allclose(
                reference[arm], _predict(data, train, subset, arm, 30.0),
                rtol=0.0, atol=1e-10,
            )
        # The same inner folds, and changing outer held-out yields must not
        # affect their CV scores or chosen alpha.
        a, picked_a = _select_alphas(
            data, train, "confirm_01", arms=(RAW, NORMALIZED), alphas=(3., 30.),
        )
    finally:
        data.y.loc[held] = old
    b, picked_b = _select_alphas(
        data, train, "confirm_01", arms=(RAW, NORMALIZED), alphas=(3., 30.),
    )
    pd.testing.assert_frame_equal(a, b)
    if picked_a != picked_b:
        raise RuntimeError("Outer held-out y changed an inner CV decision.")
    print(
        "[RIDGE-V2 SMOKE] Frozen original + five challengers, equal inner folds, "
        "outer-label isolation PASS; no official predictions touched.",
        flush=True,
    )


def _run(folder: Path, data) -> None:
    for i, split in enumerate(sorted(data.folds), 1):
        prefix = folder / split
        outputs = {
            "predictions": Path(f"{prefix}__paired.csv"),
            "cv": Path(f"{prefix}__inner_cv.csv"),
            "selection": Path(f"{prefix}__selection.json"),
            "diagnostics": Path(f"{prefix}__neighbors.csv"),
        }
        exists = {name: path.is_file() for name, path in outputs.items()}
        if all(exists.values()):
            prior = pd.read_csv(outputs["predictions"])
            if (len(prior) != 41 * len(ALL_ARMS)
                    or set(prior["model"]) != set(ALL_ARMS)
                    or prior.groupby(ID)["model"].nunique().ne(len(ALL_ARMS)).any()):
                raise RuntimeError(f"Incomplete cached outputs for {split}.")
            cv = pd.read_csv(outputs["cv"])
            if (len(cv) != len(TUNED_ARMS) * len(ALPHAS)
                    or cv["inner_folds"].ne(N_FOLDS * N_REPEATS).any()
                    or cv["n_inner_predictions"].ne(97 * N_REPEATS).any()):
                raise RuntimeError(f"Incomplete inner CV cache for {split}.")
            if len(pd.read_csv(outputs["diagnostics"])) != 41:
                raise RuntimeError(f"Incomplete neighbor diagnostics for {split}.")
            print(f"[RIDGE-V2 {i}/16] {split}: verified cached outputs.")
            continue
        if any(exists.values()):
            raise RuntimeError(
                f"Incomplete files for {split}; inspect before resuming. "
                "Do not mix partial results."
            )
        prediction, cv, choices, neighbors = _evaluate_split(data, split)
        if (len(prediction) != 41 * len(ALL_ARMS)
                or len(cv) != len(TUNED_ARMS) * len(ALPHAS)):
            raise RuntimeError(f"Unexpected score coverage for {split}.")
        _write_csv(outputs["predictions"], prediction)
        _write_csv(outputs["cv"], cv)
        _write_csv(outputs["diagnostics"], neighbors)
        atomic_json(outputs["selection"], choices)
        print(f"[RIDGE-V2 {i}/16] {split}: completed 41 paired predictions.",
              flush=True)
    _summary(folder)


def _summary(folder: Path) -> None:
    paths = sorted(folder.glob("confirm_*__paired.csv"))
    if len(paths) != 16:
        raise ValueError(f"Expected 16 completed splits, found {len(paths)}.")
    frame = pd.concat((pd.read_csv(path) for path in paths), ignore_index=True)
    summary, deltas, errors = _summarize(frame)
    _write_csv(folder / "comparison_summary.csv", summary)
    _write_csv(folder / "paired_cluster_bootstrap.csv", deltas)
    _write_csv(folder / "per_parcel_error_diagnostics.csv", errors)
    print("[RIDGE-V2] Paired pooled RMSE (t/ha):\n"
          + summary.to_string(index=False), flush=True)
    print("[RIDGE-V2] ΔMSE vs frozen original (negative favors contender):\n"
          + deltas.to_string(index=False), flush=True)
    print(
        "[RIDGE-V2] Same 16 historically reused split masks. Exploratory only. "
        "No untouched confirmation set, no new labels, no 59-value promotion.",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    command = parser.add_mutually_exclusive_group(required=True)
    for mode in ("check", "smoke", "run", "summary"):
        command.add_argument(f"--{mode}", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    paths = _input_files()
    if args.check:
        data = _load()
        print(
            f"[RIDGE-V2 CHECK] {len(data.ids)} X rows; {len(data.y)} known y; "
            "16 complete 97/41 masks, frozen original PCA20 PASS.",
            flush=True,
        )
        return 0
    _fingerprint(args.output, paths)
    if args.summary:
        _summary(args.output)
        return 0
    data = _load()
    if args.smoke:
        _smoke(data)
    else:
        _run(args.output, data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
