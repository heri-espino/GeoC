#!/usr/bin/env python3
"""Checkpoint 12: inexpensive local representations and ElasticNet evaluation.

  python tools/run_checkpoint_12_representations.py --check
  python tools/run_checkpoint_12_representations.py --smoke
  python tools/run_checkpoint_12_representations.py --run
  python tools/run_checkpoint_12_representations.py --summary

Reuses the 16 previously examined Checkpoint08 97/41 outer splits.
This is exploratory and NEVER changes 59 canonical forecasts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from geocebada.evaluation.checkpoint08_local import ID, atomic_json, digest
from geocebada.evaluation.checkpoint10_pca import FROZEN, inputs, predict
from geocebada.evaluation.checkpoint12_representations import (
    BASELINE,
    FAMILIES,
    SEED,
    configurations,
    inner_grid,
    paired_split,
    prepare,
    summarize,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports/checkpoint_12_representations"
MEMBERSHIPS = ROOT / "reports/checkpoint_08/confirmation_membership.csv"


def _inputs() -> dict[str, Path]:
    paths = inputs(ROOT, MEMBERSHIPS)
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint12 missing input {name}: {path}")
    membership = pd.read_csv(MEMBERSHIPS)
    if (len(membership) != 16 * 138
            or set(membership["split_id"]) != {
                f"confirm_{j:02d}" for j in range(1, 17)
            }):
        raise ValueError("Expected all 16 frozen Checkpoint08 memberships.")
    print("[12] Required C4 sources and 16 outer splits PASS.", flush=True)
    return paths


def _spec(folder: Path, paths: dict[str, Path], smoke: bool) -> None:
    specification = {
        "schema": 1, "seed": SEED, "smoke": smoke,
        "families": list(FAMILIES), "control": BASELINE,
        "neighbor_k": 24, "distance_power": 1.0,
        "inner_folds": 2 if smoke else 3,
        "outer_splits": 1 if smoke else 16,
        "candidate_grid": {
            family: configurations(family, smoke=smoke) for family in FAMILIES
        },
        "source_sha256": {name: digest(path) for name, path in paths.items()},
        "code_sha256": {
            path: digest(ROOT / path) for path in (
                "src/geocebada/evaluation/checkpoint04a.py",
                "src/geocebada/evaluation/checkpoint04d1.py",
                "src/geocebada/evaluation/checkpoint08_local.py",
                "src/geocebada/evaluation/checkpoint10_pca.py",
                "src/geocebada/evaluation/checkpoint12_representations.py",
                "tools/run_checkpoint_12_representations.py",
            )
        },
    }
    folder.mkdir(parents=True, exist_ok=True)
    frozen = folder / "run_spec.json"
    if frozen.exists():
        if json.loads(frozen.read_text(encoding="utf-8")) != specification:
            raise ValueError(
                "Run fingerprint changed: choose a new --output for this experiment."
            )
    else:
        if any(folder.glob("confirm_*")):
            raise RuntimeError("Untracked previous results without fingerprint.")
        atomic_json(frozen, specification)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    temp = path.with_suffix(".csv.tmp")
    frame.to_csv(temp, index=False)
    temp.replace(path)


def _assert_baseline(data) -> None:
    local = ROOT / "reports/checkpoint_08/confirm_01__paired.csv"
    if not local.exists():
        print("[12] Historical PCA20 coordinate parity checked; no 08 forecast CSV.", flush=True)
        return
    previous = pd.read_csv(local)
    previous = previous.loc[previous["model"].eq("Local04D_frozen")]
    train, test = data.baseline.folds["confirm_01"]
    if set(previous[ID].astype(str)) != set(test):
        raise ValueError("Frozen checkpoint08 forecast IDs mismatch.")
    expected = previous.set_index(ID).loc[test, "predicted"].to_numpy(float)
    current = predict(data.baseline, train, test, components=20, **FROZEN)
    difference = float(np.max(np.abs(current - expected)))
    if not np.isfinite(difference) or difference > 1e-7:
        raise RuntimeError(f"Local04D parity check failed ({difference:.3g}).")
    print(f"[12] Original Local04D forecast parity PASS ({difference:.2g}).", flush=True)


def _run(args, paths, *, smoke: bool) -> None:
    folder = args.output / "smoke" if smoke else args.output
    _spec(folder, paths, smoke=smoke)
    data = prepare(ROOT, MEMBERSHIPS)
    _assert_baseline(data)
    if data.ica_warnings:
        print("[12] WARNING: " + " ".join(data.ica_warnings), flush=True)
    split_ids = sorted(data.baseline.folds)
    if smoke:
        split_ids = split_ids[:1]
    for i, split_id in enumerate(split_ids, 1):
        paired_path = folder / f"{split_id}__paired.csv"
        selected_path = folder / f"{split_id}__selection.json"
        if paired_path.exists() and selected_path.exists():
            old = pd.read_csv(paired_path)
            if len(old) != 41 * (len(FAMILIES) + 1):
                raise RuntimeError("Incomplete cached paired predictions.")
            if set(old["model"]) != {BASELINE, *FAMILIES}:
                raise RuntimeError("Cached models mismatch.")
            print(f"[12 {i}/{len(split_ids)}] {split_id} cached.", flush=True)
            continue
        train, _ = data.baseline.folds[split_id]
        grids = {}
        for family in FAMILIES:
            cache = folder / f"{split_id}__{family}__inner.csv"
            if cache.exists():
                frame = pd.read_csv(cache)
                if (len(frame) != len(configurations(family, smoke=smoke))
                        or frame["inner_rmse"].isna().any()):
                    raise ValueError(f"Corrupt grid {cache}.")
            else:
                print(
                    f"[12 {i}/{len(split_ids)}] {split_id} {family}...",
                    flush=True,
                )
                frame = inner_grid(
                    data, train, family,
                    seed=SEED + int(split_id[-2:]), smoke=smoke,
                )
                _write_csv(cache, frame)
            grids[family] = frame
        predicted, selected = paired_split(data, split_id, grids)
        if len(predicted) != 41 * (len(FAMILIES) + 1):
            raise RuntimeError("Incomplete paired model predictions.")
        _write_csv(paired_path, predicted)
        atomic_json(selected_path, selected)
        print(f"[12 {i}/{len(split_ids)}] {split_id} finished.", flush=True)
    if smoke:
        print("[12 SMOKE] Synthetic-sized model grid on one real split; not a final result.")
    else:
        _summary(args.output)


def _summary(folder: Path) -> None:
    files = sorted(folder.glob("confirm_*__paired.csv"))
    if len(files) != 16:
        raise ValueError(f"Only {len(files)}/16 paired split files found.")
    rows = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    scores, uncertainty = summarize(rows)
    _write_csv(folder / "comparison_summary.csv", scores)
    _write_csv(folder / "paired_cluster_bootstrap.csv", uncertainty)
    selection_files = sorted(folder.glob("confirm_*__selection.json"))
    if len(selection_files) != 16:
        raise RuntimeError("Missing per-split hyperparameter selections.")
    selection = pd.DataFrame([
        {"split_id": record["split_id"], "model": family, **parameters}
        for path in selection_files
        for record in [json.loads(path.read_text(encoding="utf-8"))]
        for family, parameters in record["selected"].items()
    ])
    _write_csv(folder / "selected_hyperparameters.csv", selection)
    print("[12] PAIRED RMSE/R², smaller RMSE is better:\n"
          + scores.to_string(index=False), flush=True)
    print("[12] Cluster-bootstrap delta-MSE vs frozen Local04D:\n"
          + uncertainty.to_string(index=False), flush=True)
    print("[12] Exploratory model-family comparison. No prediction promotion.", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    for name in ("check", "smoke", "run", "summary"):
        actions.add_argument(f"--{name}", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    paths = _inputs()
    if args.check:
        data = prepare(ROOT, MEMBERSHIPS)
        _assert_baseline(data)
        print("[12 CHECK] Target-free representations and baseline verified.", flush=True)
    elif args.summary:
        _spec(args.output, paths, smoke=False)
        _summary(args.output)
    else:
        _run(args, paths, smoke=args.smoke)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
