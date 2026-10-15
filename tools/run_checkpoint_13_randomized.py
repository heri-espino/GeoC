#!/usr/bin/env python3
"""Checkpoint 13: 10,000 seeded random local-model configurations.

Run:
  python tools/run_checkpoint_13_randomized.py --check
  python tools/run_checkpoint_13_randomized.py --smoke
  python tools/run_checkpoint_13_randomized.py --run --trials 10000
  python tools/run_checkpoint_13_randomized.py --summary --trials 10000

Exactly 10,000 candidates are screened INSIDE EACH of 16 frozen 97/41
outer splits; 12 inner training examples are held out for the first pass.
A balanced top 12 from each of five families are reranked by repeated 4-fold
CV (2 repeats) using only the outer 97 yields. Only then are 41 external
pseudo-targets evaluated. Old outer bank is exploratory, not independent.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from geocebada.evaluation.checkpoint08_local import ID, atomic_json, digest
from geocebada.evaluation.checkpoint10_pca import FROZEN, inputs, predict
from geocebada.evaluation.checkpoint12_representations import prepare
from geocebada.evaluation.checkpoint13_randomized import (
    FAMILIES,
    SCREEN_TARGETS,
    SEED,
    SHORTLIST_PER_FAMILY,
    choose_and_evaluate,
    extend_pca,
    model_names,
    rerank_finalists,
    sample_candidates,
    screen_split,
    select_finalists,
    summarize,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports/checkpoint_13_randomized"
MEMBERSHIPS = ROOT / "reports/checkpoint_08/confirmation_membership.csv"


def _inputs() -> dict[str, Path]:
    files = inputs(ROOT, MEMBERSHIPS)
    for name, path in files.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing Checkpoint13 source {name}: {path}")
    masks = pd.read_csv(MEMBERSHIPS)
    if (len(masks) != 16 * 138
            or set(masks["split_id"]) != {
                f"confirm_{n:02d}" for n in range(1, 17)
            }):
        raise ValueError("Checkpoint13 requires all 16 frozen Checkpoint08 masks.")
    return files


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False)
    temp.replace(path)


def _spec(folder: Path, files: dict[str, Path], args, *, smoke: bool) -> None:
    trial_count = 60 if smoke else args.trials
    finalist_count = 2 if smoke else SHORTLIST_PER_FAMILY
    manifest = {
        "schema": 1, "seed": SEED, "smoke": smoke,
        "n_random_configs": trial_count,
        "families": list(FAMILIES),
        "screen_targets": 6 if smoke else SCREEN_TARGETS,
        "screen_chunk_size": 12 if smoke else args.chunk_size,
        "finalists_per_family": finalist_count,
        "final_cv": {"folds": 2 if smoke else 4, "repeats": 1 if smoke else 2},
        "n_outer_splits": 1 if smoke else 16,
        "baseline": {"components": 20, **FROZEN},
        "source_sha256": {name: digest(path) for name, path in files.items()},
        "code_sha256": {
            p: digest(ROOT / p) for p in (
                "src/geocebada/evaluation/checkpoint04a.py",
                "src/geocebada/evaluation/checkpoint04d1.py",
                "src/geocebada/evaluation/checkpoint08_local.py",
                "src/geocebada/evaluation/checkpoint10_pca.py",
                "src/geocebada/evaluation/checkpoint12_representations.py",
                "src/geocebada/evaluation/checkpoint13_randomized.py",
                "tools/run_checkpoint_13_randomized.py",
            )
        },
    }
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "run_spec.json"
    if path.is_file():
        if json.loads(path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("Checkpoint13 source/config changed; choose a new --output.")
    else:
        if any(folder.glob("confirm_*")) or (folder / "candidates.csv").exists():
            raise ValueError("Unidentified previous results: use an empty --output.")
        atomic_json(path, manifest)


def _candidates(folder: Path, count: int) -> pd.DataFrame:
    expected = sample_candidates(count=count)
    path = folder / "candidates.csv"
    if path.is_file():
        previous = pd.read_csv(path)
        if (len(previous) != count
                or previous["candidate_id"].tolist() != expected["candidate_id"].tolist()):
            raise ValueError("Cached random candidate list doesn't match this run.")
        for field in ("components", "k", "alpha", "power", "l1_ratio"):
            if not np.allclose(
                previous[field].astype(float),
                expected[field].astype(float),
                equal_nan=True, rtol=1e-10, atol=1e-12,
            ):
                raise ValueError(f"Candidate hyperparameters changed in {field}.")
        return previous
    _write_csv(path, expected)
    return expected


def _data():
    data = extend_pca(prepare(ROOT, MEMBERSHIPS, include_rp=False))
    old = ROOT / "reports/checkpoint_08/confirm_01__paired.csv"
    if old.is_file():
        reference = pd.read_csv(old)
        reference = reference.loc[reference["model"].eq("Local04D_frozen")]
        fit, test = data.baseline.folds["confirm_01"]
        if set(reference[ID].astype(str)) != set(test):
            raise ValueError("Incorrect stored checkpoint08 baseline rows.")
        expected = reference.set_index(ID).loc[test, "predicted"].to_numpy(float)
        actual = predict(data.baseline, fit, test, components=20, **FROZEN)
        difference = float(np.max(np.abs(actual - expected)))
        if not np.isfinite(difference) or difference > 1e-7:
            raise RuntimeError(f"Frozen Local04D numerical parity FAILED: {difference}")
        print(f"[13] Frozen Checkpoint08 Local04D parity PASS ({difference:.2g}).",
              flush=True)
    print(
        "[13] Historical C4 PCA20 parity PASS, "
        f"max abs diff {data.baseline.parity_max_diff:.3g}.",
        flush=True,
    )
    if data.ica_warnings:
        print("[13 WARNING] " + "; ".join(data.ica_warnings), flush=True)
    return data


def _screen(
    folder: Path,
    split: str,
    train: list[str],
    data,
    candidates: pd.DataFrame,
    *,
    smoke: bool,
    chunk_size: int,
) -> pd.DataFrame:
    target_count = 6 if smoke else SCREEN_TARGETS
    stage1_cache = {}
    pieces = []
    for start in range(0, len(candidates), chunk_size):
        block = candidates.iloc[start:start + chunk_size].copy()
        path = folder / f"{split}__screen__{start:05d}.csv"
        if path.exists():
            part = pd.read_csv(path)
            if (part["candidate_id"].tolist() != block["candidate_id"].tolist()
                    or not np.isfinite(part["screen_rmse"]).all()):
                raise ValueError(f"Corrupt random-search cache: {path}")
        else:
            part = screen_split(
                data, train, block, seed=SEED + int(split[-2:]),
                target_count=target_count, cache=stage1_cache,
            )
            _write_csv(path, part)
            print(
                f"[13] {split} screening {start + len(block)}/{len(candidates)}...",
                flush=True,
            )
        pieces.append(part)
    complete = pd.concat(pieces, ignore_index=True)
    if (len(complete) != len(candidates)
            or not complete["candidate_id"].is_unique
            or set(complete["candidate_id"]) != set(candidates["candidate_id"])):
        raise ValueError(f"Incomplete screening on {split}.")
    return complete


def _run(args, files, *, smoke: bool) -> None:
    folder = args.output / "smoke" if smoke else args.output
    _spec(folder, files, args, smoke=smoke)
    count = 60 if smoke else args.trials
    candidates = _candidates(folder, count)
    data = _data()
    split_names = sorted(data.baseline.folds)
    if smoke:
        split_names = split_names[:1]
    print(
        f"[13] {len(split_names)} outer splits, {count} random configs "
        f"per split; {len(FAMILIES)} families; freeze PCA20 baseline.",
        flush=True,
    )
    for i, split in enumerate(split_names, 1):
        out = folder / f"{split}__paired.csv"
        selection_path = folder / f"{split}__selection.json"
        reranked_path = folder / f"{split}__reranked.csv"
        if out.exists() and selection_path.exists():
            old = pd.read_csv(out)
            if (len(old) != 41 * len(model_names())
                    or set(old["model"]) != model_names()
                    or old.groupby(ID)["model"].nunique().ne(len(model_names())).any()):
                raise RuntimeError(f"Corrupt completed paired results for {split}.")
            print(f"[13 {i}/{len(split_names)}] {split}: already complete.", flush=True)
            continue
        train, _ = data.baseline.folds[split]
        screened = _screen(
            folder, split, train, data, candidates,
            smoke=smoke, chunk_size=12 if smoke else args.chunk_size,
        )
        finalists = select_finalists(
            candidates, screened,
            per_family=2 if smoke else SHORTLIST_PER_FAMILY,
        )
        if reranked_path.is_file():
            reranked = pd.read_csv(reranked_path)
            if (len(reranked) != len(finalists)
                    or set(reranked["candidate_id"]) != set(finalists["candidate_id"])
                    or not np.isfinite(reranked["inner_rmse"]).all()):
                raise ValueError(f"Invalid finalist cache: {reranked_path}")
        else:
            print(
                f"[13 {i}/{len(split_names)}] {split} reranking "
                f"{len(finalists)} finalists with repeated inner CV...",
                flush=True,
            )
            reranked = rerank_finalists(
                data, train, finalists,
                seed=SEED + 1000 + int(split[-2:]) * 4,
                repeats=1 if smoke else 2,
                folds=2 if smoke else 4,
            )
            _write_csv(reranked_path, reranked)
        if out.exists() != selection_path.exists():
            raise RuntimeError(
                f"Partial terminal results for {split}; inspect or archive first."
            )
        predicted, selected = choose_and_evaluate(
            data, split, candidates, reranked
        )
        if len(predicted) != len(model_names()) * 41:
            raise ValueError("Incomplete external paired predictions.")
        _write_csv(out, predicted)
        atomic_json(selection_path, selected)
        joint = selected["winners"]["Random_joint_nested"]
        print(
            f"[13 {i}/{len(split_names)}] {split}: joint {joint['family']} "
            f"{joint['components']} PCs / k={joint['k']} / "
            f"alpha={joint['alpha']:.3g} / p={joint['power']:.2f}.",
            flush=True,
        )
    if smoke:
        print("[13 SMOKE] One split, 60 candidate configs. No final scores.", flush=True)
    else:
        _summary(args.output)


def _summary(folder: Path) -> None:
    pairs = sorted(folder.glob("confirm_*__paired.csv"))
    if len(pairs) != 16:
        raise ValueError(f"Only {len(pairs)}/16 complete outer-prediction files.")
    predictions = pd.concat((pd.read_csv(p) for p in pairs), ignore_index=True)
    scores, confidence = summarize(predictions)
    selections = sorted(folder.glob("confirm_*__selection.json"))
    if len(selections) != 16:
        raise ValueError("Incomplete finalist choices for outer folds.")
    records = []
    for path in selections:
        meta = json.loads(path.read_text(encoding="utf-8"))
        for name, chosen in meta["winners"].items():
            records.append({"split_id": meta["split_id"], "model": name, **chosen})
    chosen = pd.DataFrame(records)
    _write_csv(folder / "comparison_summary.csv", scores)
    _write_csv(folder / "paired_cluster_bootstrap.csv", confidence)
    _write_csv(folder / "selected_hyperparameters.csv", chosen)
    print("[13] Matched outer RMSE/R², lower RMSE better:\n"
          + scores.to_string(index=False), flush=True)
    print("[13] Paired parcel bootstrap MSE differences:\n"
          + confidence.to_string(index=False), flush=True)
    joint = chosen.loc[chosen["model"].eq("Random_joint_nested")]
    print("[13] Joint selector families:\n"
          + joint["family"].value_counts().to_string(), flush=True)
    print(
        "[13] Exploratory reuse of the Checkpoint08 138 reported yields. "
        "No promotion or modification of the final 59 forecasts.",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    for name in ("check", "smoke", "run", "summary"):
        actions.add_argument(f"--{name}", action="store_true")
    parser.add_argument("--trials", type=int, default=10000)
    parser.add_argument("--chunk-size", type=int, default=250)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.trials < len(FAMILIES):
        parser.error(f"--trials must be at least {len(FAMILIES)}.")
    if args.chunk_size < 1:
        parser.error("--chunk-size must be positive.")
    files = _inputs()
    if args.check:
        data = _data()
        candidates = sample_candidates(count=args.trials)
        print(
            f"[13 CHECK] {len(candidates)} seeded configurations from "
            f"{len(FAMILIES)} families; 16 outer masks; "
            f"{len(data.baseline.ids)} C4 parcel rows PASS.",
            flush=True,
        )
    elif args.summary:
        _spec(args.output, files, args, smoke=False)
        _summary(args.output)
    else:
        _run(args, files, smoke=args.smoke)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
