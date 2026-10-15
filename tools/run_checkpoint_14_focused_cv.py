#!/usr/bin/env python3
"""Checkpoint14: focused 1,000-config Local04D search with *full* inner CV.

Commands:
  python tools/run_checkpoint_14_focused_cv.py --check
  python tools/run_checkpoint_14_focused_cv.py --smoke
  python tools/run_checkpoint_14_focused_cv.py --run
  python tools/run_checkpoint_14_focused_cv.py --summary

Runs on exactly the frozen 16 x 97/41 Checkpoint08 outer masks (exploratory).
Every candidate is first scored on all 97 outer-training IDs by 3-fold CV.
Best by family PLUS historical Local04D are reranked on 2x4fold CV. The
41 external pseudo-targets are held until the evaluation, and 59 hidden
targets are NEVER used. Writes only to separately ignored output.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from geocebada.evaluation.checkpoint08_local import ID, atomic_json, digest
from geocebada.evaluation.checkpoint10_pca import FROZEN, inputs, predict
from geocebada.evaluation.checkpoint14_focused_cv import (
    ANCHOR_NAME,
    BASELINE,
    COUNT,
    FAMILIES,
    JOINT,
    SEED,
    evaluate_outer,
    focused_candidates,
    model_names,
    prepare_focused,
    score_candidates,
    shortlist_candidates,
    summarize,
)

ROOT = Path(__file__).resolve().parents[1]
MEMBERSHIPS = ROOT / "reports/checkpoint_08/confirmation_membership.csv"
OUTPUT = ROOT / "reports/checkpoint_14_focused_1k"


def input_checks() -> dict[str, Path]:
    files = inputs(ROOT, MEMBERSHIPS)
    for label, path in files.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing Checkpoint14 input {label}: {path}")
    mask = pd.read_csv(MEMBERSHIPS)
    if (len(mask) != 16 * 138
            or set(mask["split_id"]) != {
                f"confirm_{i:02d}" for i in range(1, 17)
            }):
        raise RuntimeError("Checkpoint14 needs the frozen 16 Checkpoint08 masks.")
    return files


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False)
    temp.replace(path)


def manifest(folder: Path, paths: dict[str, Path], *, smoke: bool) -> None:
    spec = {
        "schema": 1,
        "experiment": "checkpoint14_focused_fullcv",
        "seed": SEED, "smoke": smoke,
        "model_families": list(FAMILIES),
        "n_candidates": COUNT,
        "initial_inner_cv": {"folds": 2 if smoke else 3, "repeats": 1},
        "rerank_inner_cv": {
            "folds": 2 if smoke else 4, "repeats": 1 if smoke else 2
        },
        "n_outer_splits": 1 if smoke else 16,
        "incumbent": {"candidate_id": ANCHOR_NAME, "pca": 20, **FROZEN},
        "source_sha256": {key: digest(path) for key, path in paths.items()},
        "code_sha256": {
            path: digest(ROOT / path) for path in (
                "src/geocebada/evaluation/checkpoint04a.py",
                "src/geocebada/evaluation/checkpoint04d1.py",
                "src/geocebada/evaluation/checkpoint08_local.py",
                "src/geocebada/evaluation/checkpoint10_pca.py",
                "src/geocebada/evaluation/checkpoint12_representations.py",
                "src/geocebada/evaluation/checkpoint13_randomized.py",
                "src/geocebada/evaluation/checkpoint14_focused_cv.py",
                "tools/run_checkpoint_14_focused_cv.py",
            )
        },
    }
    folder.mkdir(parents=True, exist_ok=True)
    file = folder / "run_spec.json"
    if file.exists():
        if json.loads(file.read_text(encoding="utf-8")) != spec:
            raise ValueError(
                "Checkpoint14 inputs/code/protocol differ from saved run_spec. "
                "Use a new --output; don't delete or mix earlier results."
            )
    else:
        if list(folder.glob("confirm_*")) or (folder / "candidates.csv").exists():
            raise RuntimeError("Previous results without run_spec in output directory.")
        atomic_json(file, spec)


def candidate_manifest(folder: Path, *, smoke: bool) -> pd.DataFrame:
    full = focused_candidates()
    if smoke:
        # Keep the anchor + 2 novel proposals from each of four families.
        frames = []
        for family in FAMILIES:
            group = full.loc[full["family"].eq(family)]
            frames.append(group.head(3))
        full = pd.concat(frames, ignore_index=True)
    path = folder / "candidates.csv"
    if path.exists():
        cached = pd.read_csv(path)
        if (cached["candidate_id"].tolist() != full["candidate_id"].tolist()
                or not np.allclose(cached["alpha"], full["alpha"], atol=1e-12)
                or not np.allclose(cached["power"], full["power"], atol=1e-12)):
            raise ValueError("Candidate generation no longer matches cached trials.")
    else:
        write_csv(path, full)
    return full


def load_data():
    data = prepare_focused(ROOT, MEMBERSHIPS)
    print(
        f"[14] C4 197 rows, 2082 features, "
        f"original PCA20 parity max diff "
        f"{data.baseline.parity_max_diff:.3g}.",
        flush=True,
    )
    old = ROOT / "reports/checkpoint_08/confirm_01__paired.csv"
    if old.exists():
        reference = pd.read_csv(old)
        reference = reference.loc[reference["model"].eq("Local04D_frozen")]
        train, held = data.baseline.folds["confirm_01"]
        if set(reference[ID].astype(str)) != set(held):
            raise RuntimeError("Frozen old Local04D prediction IDs differ.")
        original = reference.set_index(ID).loc[held, "predicted"].to_numpy(float)
        fresh = predict(data.baseline, train, held, components=20, **FROZEN)
        delta = float(np.max(np.abs(fresh - original)))
        if not np.isfinite(delta) or delta > 1e-7:
            raise RuntimeError(f"Original Local04D numerical parity FAIL: {delta:.3g}")
        print(f"[14] Original Local04D forecast parity PASS ({delta:.3g}).",
              flush=True)
    return data


def validate_scores(
    scores: pd.DataFrame, candidates: pd.DataFrame, *, inner_predictions: int
) -> None:
    if (len(scores) != len(candidates)
            or scores["candidate_id"].tolist() != candidates["candidate_id"].tolist()
            or scores["family"].tolist() != candidates["family"].tolist()
            or scores["inner_predictions"].ne(inner_predictions).any()
            or not np.isfinite(scores["inner_rmse"]).all()):
        raise RuntimeError("Cached CV batch incomplete, shifted or corrupt.")


def initial_cv(
    folder: Path, split: str, data, train: list[str],
    candidates: pd.DataFrame, *, smoke: bool, chunk_size: int,
) -> pd.DataFrame:
    pieces = []
    cache = {}
    for start in range(0, len(candidates), chunk_size):
        subset = candidates.iloc[start:start + chunk_size]
        path = folder / f"{split}__cv3__{start:04d}.csv"
        if path.exists():
            scores = pd.read_csv(path)
        else:
            scores = score_candidates(
                data, train, subset, seed=SEED + int(split[-2:]),
                folds=2 if smoke else 3, repeats=1, cache=cache,
            )
            validate_scores(scores, subset, inner_predictions=len(train))
            write_csv(path, scores)
            print(
                f"[14] {split}: all-fold inner-CV "
                f"{start + len(subset)}/{len(candidates)} candidates",
                flush=True,
            )
        validate_scores(scores, subset, inner_predictions=len(train))
        pieces.append(scores)
    if not pieces:
        raise RuntimeError("No inner CV scores.")
    result = pd.concat(pieces, ignore_index=True)
    validate_scores(result, candidates, inner_predictions=len(train))
    return result


def _run(args, paths, *, smoke: bool) -> None:
    folder = args.output / "smoke" if smoke else args.output
    manifest(folder, paths, smoke=smoke)
    candidates = candidate_manifest(folder, smoke=smoke)
    data = load_data()
    ids = sorted(data.baseline.folds)
    if smoke:
        ids = ids[:1]
    print(
        f"[14] {len(ids)} outer splits × {len(candidates)} candidates, "
        "every candidate scored on all 97 training targets in inner CV. "
        "Fixed Local04D included in both stages.",
        flush=True,
    )
    for i, split in enumerate(ids, 1):
        prediction_path = folder / f"{split}__paired.csv"
        selection_path = folder / f"{split}__selection.json"
        rerank_path = folder / f"{split}__rerank.csv"
        if prediction_path.exists() and selection_path.exists():
            old = pd.read_csv(prediction_path)
            if (len(old) != 41 * len(model_names())
                    or set(old["model"]) != model_names()
                    or old.groupby(ID)["model"].nunique().ne(len(model_names())).any()):
                raise RuntimeError(f"Invalid cached outer paired predictions: {split}.")
            print(f"[14 {i}/{len(ids)}] {split} previously completed.", flush=True)
            continue
        train, _ = data.baseline.folds[split]
        first = initial_cv(
            folder, split, data, train, candidates,
            smoke=smoke, chunk_size=4 if smoke else args.chunk_size,
        )
        finalists = shortlist_candidates(candidates, first)
        if ANCHOR_NAME not in set(finalists["candidate_id"]):
            raise ValueError("Original Local04D was not promoted to repeated inner CV.")
        if rerank_path.exists():
            second = pd.read_csv(rerank_path)
            if (len(second) != len(finalists)
                    or second["candidate_id"].tolist() != finalists["candidate_id"].tolist()
                    or not np.isfinite(second["inner_rmse"]).all()
                    or second["inner_predictions"].ne(
                        len(train) * (1 if smoke else 2)
                    ).any()):
                raise RuntimeError(f"Rerank cache inconsistent for {split}.")
        else:
            second = score_candidates(
                data, train, finalists,
                seed=SEED + 1000 + int(split[-2:]) * 4,
                folds=2 if smoke else 4,
                repeats=1 if smoke else 2,
            )
            write_csv(rerank_path, second)
        if prediction_path.exists() != selection_path.exists():
            raise RuntimeError(
                f"Half-written outer results for {split}; inspect the files first."
            )
        paired, chosen = evaluate_outer(data, split, candidates, second)
        incumbent = first.loc[first["candidate_id"].eq(ANCHOR_NAME)].iloc[0]
        chosen["incumbent_initial_cv_rmse"] = float(incumbent["inner_rmse"])
        chosen["incumbent_initial_cv_rank_all"] = 1 + int(
            first["inner_rmse"].lt(float(incumbent["inner_rmse"])).sum()
        )
        chosen["n_full_initial_cv_candidates"] = len(first)
        chosen["n_repeated_cv_finalists"] = len(second)
        write_csv(prediction_path, paired)
        atomic_json(selection_path, chosen)
        selected = chosen["chosen"][JOINT]
        print(
            f"[14 {i}/{len(ids)}] {split}: selected {selected['family']} "
            f"PC{selected['components']} k{selected['k']} "
            f"alpha{selected['alpha']:.3g} p{selected['power']:.2f}; "
            f"original Local04D rank {chosen['incumbent_repeated_cv_rank_all']}/"
            f"{len(second)}.",
            flush=True,
        )
    if smoke:
        print("[14 SMOKE] Fold-local search and fixed anchor PASS. No full result.",
              flush=True)
    else:
        summary(args.output)


def summary(folder: Path) -> None:
    saved = sorted(folder.glob("confirm_*__paired.csv"))
    if len(saved) != 16:
        raise ValueError(f"Only {len(saved)}/16 paired outer splits complete.")
    forecast = pd.concat((pd.read_csv(p) for p in saved), ignore_index=True)
    result, intervals = summarize(forecast)
    selections = sorted(folder.glob("confirm_*__selection.json"))
    if len(selections) != 16:
        raise ValueError("Saved hyperparameter selection files are incomplete.")
    details = pd.DataFrame([
        {
            "split_id": record["split_id"],
            "incumbent_initial_cv_rmse": record["incumbent_initial_cv_rmse"],
            "incumbent_repeated_cv_rmse": record["incumbent_repeated_cv_rmse"],
            "incumbent_initial_cv_rank_all": record["incumbent_initial_cv_rank_all"],
            "incumbent_repeated_cv_rank_all":
                record["incumbent_repeated_cv_rank_all"],
            "incumbent_selected_by_joint": record["incumbent_selected_by_joint"],
            "joint_family": record["chosen"][JOINT]["family"],
            "joint_candidate": record["chosen"][JOINT]["candidate_id"],
            "joint_components": record["chosen"][JOINT]["components"],
            "joint_k": record["chosen"][JOINT]["k"],
            "joint_alpha": record["chosen"][JOINT]["alpha"],
            "joint_power": record["chosen"][JOINT]["power"],
        }
        for p in selections
        for record in [json.loads(p.read_text(encoding="utf-8"))]
    ])
    write_csv(folder / "comparison_summary.csv", result)
    write_csv(folder / "paired_cluster_bootstrap.csv", intervals)
    write_csv(folder / "selected_hyperparameters.csv", details)
    print("[14] Paired RMSE/R² by outer model procedure:\n"
          + result.to_string(index=False), flush=True)
    print("[14] MSE difference to frozen PCA20 (negative better):\n"
          + intervals.to_string(index=False), flush=True)
    print(
        "[14] Original Local04D selected as joint finalist in "
        f"{int(details['incumbent_selected_by_joint'].sum())}/16 outer splits.",
        flush=True,
    )
    print(
        "[14] Exploratory outer bank previously reused in 08/10/13; "
        "never changes final 59 forecasts.",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    for name in ("check", "smoke", "run", "summary"):
        actions.add_argument(f"--{name}", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--chunk-size", type=int, default=50)
    args = parser.parse_args()
    if args.chunk_size < 1:
        parser.error("chunk size must be at least 1.")
    paths = input_checks()
    if args.check:
        data = load_data()
        candidates = focused_candidates()
        print(
            f"[14 CHECK] C4, masks, historical Local04D and "
            f"{len(candidates)} fixed candidates PASS. "
            f"Models: {', '.join(FAMILIES)}.",
            flush=True,
        )
    elif args.summary:
        manifest(args.output, paths, smoke=False)
        summary(args.output)
    else:
        _run(args, paths, smoke=args.smoke)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
