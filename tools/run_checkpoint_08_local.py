#!/usr/bin/env python3
"""Final symmetric nested comparison: old Local04D vs HLS/SMAP Local07.

Usage, from repository root:
  python tools/run_checkpoint_08_local.py --prepare
  python tools/run_checkpoint_08_local.py --smoke
  python tools/run_checkpoint_08_local.py --run --trials 96
  python tools/run_checkpoint_08_local.py --summary --trials 96

No GPU is needed: both local models use sklearn Ridge/PCA. Heavy validation
outputs remain local. This tool cannot edit the canonical 59 predictions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from geocebada.evaluation.checkpoint08_local import (
    FAMILIES,
    ID,
    atomic_json,
    digest,
    fixed_baseline,
    input_paths,
    label_blind_memberships,
    load_inputs,
    summarize,
    tune_and_predict,
    validate_memberships,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "reports/checkpoint_08"


def ensure_spec(folder: Path, spec: dict) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "run_spec.json"
    if path.is_file():
        if json.loads(path.read_text(encoding="utf-8")) != spec:
            raise ValueError("Checkpoint 08 fingerprint differs. Use the original "
                             "options and code, or a separate --output directory.")
    else:
        if list(folder.glob("*.sqlite")) or list(folder.glob("*__paired.csv")):
            raise RuntimeError("Existing studies lack run_spec.json; refuse unsafe resume.")
        atomic_json(path, spec)


def freeze_memberships(args: argparse.Namespace) -> Path:
    target = args.output / "confirmation_membership.csv"
    metadata = pd.read_csv(
        input_paths(ROOT)["base"],
        usecols=[ID, "CONJUNTO", "meta_estado", "meta_municipio"],
    )
    historical = pd.read_csv(input_paths(ROOT)["original_splits"])
    result = label_blind_memberships(metadata, historical, seed=args.seed)
    validate_memberships(metadata, result)
    if target.exists():
        saved = pd.read_csv(target)
        pd.testing.assert_frame_equal(result, saved, check_dtype=False, atol=1e-12)
        print("[08 PREPARE] Reused identical frozen membership; no overwrite.", flush=True)
    else:
        args.output.mkdir(parents=True, exist_ok=True)
        result.to_csv(target, index=False)
        print(f"[08 PREPARE] Frozen 16 label-blind confirmation splits in {target}", flush=True)
    return target


def check(args: argparse.Namespace) -> Path:
    for name, path in input_paths(ROOT).items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name}: {path}")
    member_path = args.output / "confirmation_membership.csv"
    if not member_path.is_file():
        raise FileNotFoundError(f"Prepare the new outer bank first: "
                                f"python tools/run_checkpoint_08_local.py --prepare")
    metadata = pd.read_csv(input_paths(ROOT)["base"], usecols=[ID, "CONJUNTO"])
    validate_memberships(metadata, pd.read_csv(member_path))
    print("[08] Inputs and 16 fixed confirmation memberships PASS.", flush=True)
    return member_path


def run(args: argparse.Namespace, *, smoke: bool = False) -> None:
    if args.trials < 2 or args.inner_folds < 2:
        raise ValueError("Need at least two Optuna trials and two inner folds.")
    member_path = check(args)
    # A one-split smoke has its own checkpoints and cannot contaminate the
    # official comparison. Keep its smaller budget in a different directory.
    folder = args.output / "smoke" if smoke else args.output
    trials = 2 if smoke else args.trials
    watched = {**input_paths(ROOT), "confirmation_membership": member_path,
               "module": ROOT / "src/geocebada/evaluation/checkpoint08_local.py",
               "runner": ROOT / "tools/run_checkpoint_08_local.py"}
    specification = {
        "schema": 1, "models": [*FAMILIES, "Local04D_frozen"],
        "trials": trials, "inner_folds": 2 if smoke else args.inner_folds,
        "seed": args.seed, "smoke": smoke,
        "sha256": {name: digest(path) for name, path in watched.items()},
    }
    ensure_spec(folder, specification)
    membership = pd.read_csv(member_path)
    data = load_inputs(ROOT, membership)
    names = sorted(data.folds)
    if smoke:
        names = names[:1]

    for j, split in enumerate(names, start=1):
        paired = folder / f"{split}__paired.csv"
        if paired.exists():
            previous = pd.read_csv(paired)
            if len(previous) != 123 or set(previous["model"]) != {*FAMILIES, "Local04D_frozen"}:
                raise ValueError(f"Corrupted completed paired split {paired}")
            print(f"[08 {j}/{len(names)}] {split} previously complete; skipping.", flush=True)
            continue
        parts = [fixed_baseline(data, split)]
        for family in FAMILIES:
            parts.append(tune_and_predict(
                data, split, family,
                trials=trials, seed=args.seed + int(split[-2:]),
                n_folds=2 if smoke else args.inner_folds, folder=folder,
            ))
        rows = pd.concat(parts, ignore_index=True)
        if len(rows) != 3 * 41:
            raise ValueError(f"Incomplete paired prediction for {split}")
        temp = paired.with_suffix(".csv.tmp")
        rows.to_csv(temp, index=False)
        temp.replace(paired)
        print(f"[08 {j}/{len(names)}] {split} finished and checkpointed.", flush=True)

    if smoke:
        print("[08 SMOKE] One paired split completed, not a final RMSE.", flush=True)
    else:
        summarize_files(folder)


def summarize_files(folder: Path) -> None:
    files = sorted(folder.glob("confirm_*__paired.csv"))
    if len(files) != 16:
        raise RuntimeError(f"Only {len(files)}/16 paired confirmation splits complete.")
    predictions = pd.concat([pd.read_csv(x) for x in files], ignore_index=True)
    summary, intervals = summarize(predictions)
    summary.to_csv(folder / "comparison_summary.csv", index=False)
    intervals.to_csv(folder / "paired_cluster_bootstrap.csv", index=False)
    print("[08] FULL nested paired RMSE, smaller is better:\n"
          + summary.to_string(index=False), flush=True)
    print("[08] Paired delta-MSE vs frozen Local04D (negative is better):\n"
          + intervals.to_string(index=False), flush=True)
    print("[08] No automatic promotion or modification of the fixed 59 forecasts.", flush=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    action = p.add_mutually_exclusive_group(required=True)
    for name in ("prepare", "check", "smoke", "run", "summary"):
        action.add_argument("--" + name, action="store_true")
    p.add_argument("--output", type=Path, default=DEFAULT)
    p.add_argument("--trials", type=int, default=96)
    p.add_argument("--inner-folds", type=int, default=3)
    p.add_argument("--seed", type=int, default=20261012)
    args = p.parse_args()
    if args.prepare:
        freeze_memberships(args)
    elif args.check:
        check(args)
    elif args.summary:
        check(args)
        summarize_files(args.output)
    else:
        run(args, smoke=args.smoke)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
