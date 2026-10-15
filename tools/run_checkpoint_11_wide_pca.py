#!/usr/bin/env python3
"""Checkpoint 11: larger Local04D grid with two-stage, repeated inner CV.

Commands from repo root:
  python tools/run_checkpoint_11_wide_pca.py --check
  python tools/run_checkpoint_11_wide_pca.py --smoke
  python tools/run_checkpoint_11_wide_pca.py --run
  python tools/run_checkpoint_11_wide_pca.py --summary

Full search: PCA 10/20/30/40/60/80/120/160; k 8/12/16/20/24/32/48;
alpha 1/3/10/30/100/300/1000; distance exponent p 0/.5/1/1.5/2.
Stage 1: 1x 3-fold on 1,960 candidates per outer split.
Stage 2: 2x 5-fold on four finalists per PCA + incumbent parameters.
The already inspected 16 Checkpoint 08 outer splits are EXPLORATORY.
Never promotes any model or changes the 59 final parcel predictions.
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
    predict,
)
from geocebada.evaluation.checkpoint11_wide_pca import (
    ALPHAS,
    DIMS,
    NEIGHBORS,
    POWERS,
    TOP_PER_DIM,
    choose_nested,
    enlarge_c4,
    make_grid,
    models,
    score_grid,
    shortlist,
    summarize,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports/checkpoint_11_wide_pca"
MEMBERSHIPS = ROOT / "reports/checkpoint_08/confirmation_membership.csv"
SEED = 20261015


def _required_inputs() -> dict[str, Path]:
    paths = inputs(ROOT, MEMBERSHIPS)
    for label, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing required input {label}: {path}")
    mask = pd.read_csv(MEMBERSHIPS)
    if (len(mask) != 16 * 138
            or set(mask["split_id"]) != {f"confirm_{i:02d}" for i in range(1, 17)}):
        raise ValueError("16 frozen Checkpoint 08 confirmation memberships required.")
    print("[11] All 16 existing outer masks and C4 inputs found.", flush=True)
    return paths


def _load_data(dimensions: tuple[int, ...]):
    # Canonical 2082-variable PCA is built once (with X of all 197 parcels).
    data = load_pca_data(ROOT, MEMBERSHIPS, max_components=160)
    if data.parity_max_diff > 1e-6:
        raise RuntimeError("C4 PCA20 historical coordinate parity failed.")
    data = enlarge_c4(data, dimensions)
    prior = ROOT / "reports/checkpoint_08/confirm_01__paired.csv"
    if prior.exists():
        control = pd.read_csv(prior)
        control = control.loc[control["model"].eq("Local04D_frozen")].copy()
        fit, held_out = data.folds["confirm_01"]
        if set(control[ID].astype(str)) != set(held_out):
            raise ValueError("Checkpoint 08 baseline is misaligned.")
        reference = control.set_index(ID).loc[held_out, "predicted"].to_numpy(float)
        forecast = predict(data, fit, held_out, components=20, **FROZEN)
        difference = float(np.max(np.abs(reference - forecast)))
        if not np.isfinite(difference) or difference > 1e-7:
            raise ValueError(
                f"Checkpoint 08 frozen baseline forecast parity FAILED: {difference}"
            )
        print(f"[11] Frozen Local04D prediction parity PASS ({difference:.2g}).", flush=True)
    print(f"[11] PCA20 coordinate parity PASS ({data.parity_max_diff:.2g}).", flush=True)
    return data


def _spec(output: Path, paths: dict[str, Path], *, smoke: bool) -> None:
    dimensions = (20, 40) if smoke else DIMS
    manifest = {
        "schema": 1, "experiment": "checkpoint11_stable_wide_grid",
        "smoke": smoke, "seed": SEED, "dimensions": list(dimensions),
        "neighbors": [20, 24] if smoke else list(NEIGHBORS),
        "alphas": [10.0, 30.0] if smoke else list(ALPHAS),
        "powers": [0.5, 1.0] if smoke else list(POWERS),
        "top_per_dimension": 2 if smoke else TOP_PER_DIM,
        "coarse_cv": {"folds": 2 if smoke else 3, "repeats": 1},
        "finalist_cv": {"folds": 3 if smoke else 5, "repeats": 1 if smoke else 2},
        "outer_splits": 1 if smoke else 16, "frozen": FROZEN,
        "source_sha256": {label: digest(path) for label, path in paths.items()},
        "code_sha256": {
            p: digest(ROOT / p) for p in (
                "src/geocebada/evaluation/checkpoint04a.py",
                "src/geocebada/evaluation/checkpoint04b.py",
                "src/geocebada/evaluation/checkpoint04d1.py",
                "src/geocebada/evaluation/checkpoint08_local.py",
                "src/geocebada/evaluation/checkpoint10_pca.py",
                "src/geocebada/evaluation/checkpoint11_wide_pca.py",
                "tools/run_checkpoint_11_wide_pca.py",
            )
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    path = output / "run_spec.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("Checkpoint 11 fingerprints differ. Use a fresh --output.")
    else:
        if list(output.glob("confirm_*")):
            raise ValueError("Unidentified previous results in output directory.")
        atomic_json(path, manifest)


def _save_frame(path: Path, frame: pd.DataFrame) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False)
    temp.replace(path)


def _check_coarse(frame: pd.DataFrame, n: int, smoke: bool) -> None:
    expected = make_grid(n, smoke=smoke)
    columns = ["components", "k", "alpha", "power"]
    if len(frame) != len(expected):
        raise ValueError(f"Incomplete PCA{n} coarse grid; expected {len(expected)} rows.")
    got = {tuple(x) for x in frame[columns].itertuples(index=False, name=None)}
    want = {tuple(x[c] for c in columns) for x in expected}
    if got != want or frame["inner_rmse"].isna().any():
        raise ValueError(f"Corrupt cached PCA{n} coarse grid.")


def _run(args, paths: dict[str, Path], *, smoke: bool) -> None:
    dimensions = (20, 40) if smoke else DIMS
    selected = args.output / "smoke" if smoke else args.output
    _spec(selected, paths, smoke=smoke)
    data = _load_data(dimensions)
    split_names = sorted(data.folds)[:1] if smoke else sorted(data.folds)
    n_coarse = sum(len(make_grid(n, smoke=smoke)) for n in dimensions)
    print(
        f"[11] {len(split_names)} outer splits × {n_coarse} coarse candidates; "
        f"finalists reranked with {'3-fold' if smoke else '2×5-fold'} CV.",
        flush=True,
    )
    for pos, split in enumerate(split_names, 1):
        rows_path = selected / f"{split}__paired.csv"
        rank_path = selected / f"{split}__rerank.csv"
        report_path = selected / f"{split}__selection.json"
        if rows_path.exists() and rank_path.exists() and report_path.exists():
            previous = pd.read_csv(rows_path)
            if (len(previous) != 41 * len(models(dimensions))
                    or set(previous["model"]) != models(dimensions)):
                raise ValueError(f"Invalid completed split {split}.")
            print(f"[11 {pos}/{len(split_names)}] {split}: previously completed.", flush=True)
            continue
        train, _ = data.folds[split]
        coarse = {}
        for n in dimensions:
            path = selected / f"{split}__coarse_pca{n}.csv"
            if path.exists():
                frame = pd.read_csv(path)
                _check_coarse(frame, n, smoke)
            else:
                print(f"[11 {pos}/{len(split_names)}] {split} PCA{n} coarse...", flush=True)
                frame = score_grid(
                    data, train, make_grid(n, smoke=smoke),
                    n_folds=2 if smoke else 3, n_repeats=1,
                    seed=SEED + int(split[-2:]),
                )
                _check_coarse(frame, n, smoke)
                _save_frame(path, frame)
            coarse[n] = frame
        shortlisted = shortlist(coarse, top_per_dim=2 if smoke else TOP_PER_DIM)
        if rank_path.exists():
            rerank = pd.read_csv(rank_path)
            required = ("components", "k", "alpha", "power")
            available = {
                tuple(x) for x in rerank[list(required)].itertuples(index=False, name=None)
            }
            expected = {tuple(config[k] for k in required) for config in shortlisted}
            if (available != expected or len(rerank) != len(shortlisted)
                    or rerank["inner_rmse"].isna().any()):
                raise ValueError(f"Corrupt cached second-stage CV for {split}.")
        else:
            print(
                f"[11 {pos}/{len(split_names)}] {split} repeated-CV rerank "
                f"({len(shortlisted)} candidates)...",
                flush=True,
            )
            rerank = score_grid(
                data, train, shortlisted, n_folds=3 if smoke else 5,
                n_repeats=1 if smoke else 2,
                seed=SEED + 1000 + int(split[-2:]) * 10,
            )
            _save_frame(rank_path, rerank)
        if rows_path.exists() != report_path.exists():
            # An interruption between atomic CSV and JSON writes is safe:
            # archive the orphan and deterministically rebuild both outputs.
            orphan = rows_path if rows_path.exists() else report_path
            archived = orphan.with_suffix(orphan.suffix + ".incomplete")
            if archived.exists():
                raise ValueError(f"Repeated partial output for {split}: inspect manually.")
            orphan.replace(archived)
            print(f"[11] Archived interrupted {orphan.name}; rebuilding.", flush=True)
        paired, selection = choose_nested(
            data, split, rerank, dimensions=dimensions
        )
        _save_frame(rows_path, paired)
        atomic_json(report_path, selection)
        best = selection["joint_best"]
        print(
            f"[11 {pos}/{len(split_names)}] {split}: "
            f"joint PCA{best['components']} / k{best['k']} / "
            f"alpha{best['alpha']:g} / p{best['power']:g}; saved.",
            flush=True,
        )
    if smoke:
        print("[11 SMOKE] One 41-target outer split passed. Not a final model score.", flush=True)
    else:
        _summary(args.output)


def _summary(output: Path) -> None:
    files = sorted(output.glob("confirm_*__paired.csv"))
    if len(files) != 16:
        raise ValueError(f"Only {len(files)} of 16 complete outer prediction files.")
    predictions = pd.concat((pd.read_csv(p) for p in files), ignore_index=True)
    summary, ci = summarize(predictions)
    _save_frame(output / "comparison_summary.csv", summary)
    _save_frame(output / "paired_cluster_bootstrap.csv", ci)
    selections = [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted(output.glob("confirm_*__selection.json"))
    ]
    if len(selections) != 16:
        raise ValueError("Only some hyperparameter selections are available.")
    best = pd.DataFrame([
        {"split_id": s["split_id"], **s["joint_best"]} for s in selections
    ])
    _save_frame(output / "selected_hyperparameters.csv", best)
    if len(best) != 16 or best["split_id"].nunique() != 16:
        raise ValueError("Duplicated or missing split selections.")
    print("[11] Nested outer paired results (RMSE in t/ha):\n"
          + summary.to_string(index=False), flush=True)
    print("[11] Paired delta-MSE against frozen PCA20:\n"
          + ci.to_string(index=False), flush=True)
    print("[11] Selected joint PCA dimensions:\n"
          + best["components"].value_counts().sort_index().to_string(),
          flush=True)
    print(
        "[11] Exploratory on repeatedly examined Checkpoint 08 labels. "
        "No promotion, no modifications to the fixed 59 forecasts.",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    for name in ("check", "smoke", "run", "summary"):
        actions.add_argument(f"--{name}", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    paths = _required_inputs()
    if args.check:
        _load_data(DIMS)
        print("[11 CHECK] C4 160-PC representation and frozen baseline verified.", flush=True)
    elif args.summary:
        # Never summarize incompatible cached results after changing inputs/code.
        _spec(args.output, paths, smoke=False)
        _summary(args.output)
    else:
        _run(args, paths, smoke=args.smoke)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
