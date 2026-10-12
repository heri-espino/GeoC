#!/usr/bin/env python3
"""Checkpoint 09: test robust Local04D under possible noisy yield labels.

From the repository root, in the geocebada environment:
  python tools/run_checkpoint_09_robust.py --preflight
  python tools/run_checkpoint_09_robust.py --prepare
  python tools/run_checkpoint_09_robust.py --smoke
  python tools/run_checkpoint_09_robust.py --run
  python tools/run_checkpoint_09_robust.py --summary

Only when development has passed its predeclared LOSO stability gate:
  python tools/run_checkpoint_09_robust.py --confirm

Runs on CPU with ordinary scikit-learn; no 07B downloads or GPU necessary.
Does not modify the published 59 predictions or introduce missing target y.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from geocebada.evaluation.checkpoint08_local import (
    ID,
    atomic_json,
    digest,
    label_blind_memberships,
    validate_memberships,
    verify_original_control,
)
from geocebada.evaluation.checkpoint09_robust import (
    METHODS,
    build_review_queue,
    evaluate_bank,
    leave_one_split_out_selection,
    load_legacy_inputs,
    paired_cluster_interval,
    summarize_bank,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/checkpoint_09"
HIST = ROOT / "reports/checkpoint_04b/pseudo_competition_splits.csv"
SEED = 20261013
MIN_GAIN = 0.003
MAE_TOLERANCE = 0.002
TAIL_TOLERANCE = 0.005
EXPECTED_BASELINE = 0.49574142871073495


def _necessary_files() -> dict[str, Path]:
    return {
        "base_X": ROOT / "data/processed/features_v1/parcel_features_all.csv",
        "official_y": ROOT / "data/source/tabular/ID_area_rendimiento_70_30_Reto_AgroCebada.csv",
        "C4_embedding": ROOT / "reports/checkpoint_04a/transductive_embeddings.csv",
        "historical_memberships": HIST,
        "historical_control": ROOT / "reports/checkpoint_06/parcel_oof_residuals.csv",
        "implementation": ROOT / "src/geocebada/evaluation/checkpoint09_robust.py",
        "runner": ROOT / "tools/run_checkpoint_09_robust.py",
        "config": ROOT / "configs/checkpoint09_robust.yaml",
    }


def _historical() -> pd.DataFrame:
    all_memberships = pd.read_csv(HIST)
    primary = all_memberships.loc[
        all_memberships["split_id"].astype(str).str.startswith("target_matched_")
        & all_memberships["family"].eq("target_matched")
    ].copy()
    return primary


def _check():
    required = _necessary_files()
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Checkpoint 09 missing inputs:\n" + "\n".join(missing))
    data = load_legacy_inputs(ROOT)
    historical = _historical()
    validate_memberships(data.meta, historical)
    diff = verify_original_control(ROOT, data)
    if diff > 1e-7:
        raise ValueError(f"Frozen Local04D parity mismatch: {diff}")
    print("[09] 197/138/59, 16 frozen historical masks and numerical baseline parity PASS")
    print(f"[09] C4 numerical parity max absolute difference: {diff:.3g} t/ha")
    return data, historical


def _prepare(data, history, output: Path) -> Path:
    destination = output / "confirmation_membership.csv"
    if (output / "development_predictions.csv").exists() and not destination.exists():
        raise RuntimeError("Refuse to choose confirmation masks AFTER looking at development.")
    proposed = label_blind_memberships(
        data.meta, history, seed=SEED, repeats=16, n_heldout=41,
    )
    validate_memberships(data.meta, proposed)
    output.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        pd.testing.assert_frame_equal(
            pd.read_csv(destination), proposed, check_dtype=False, atol=1e-12,
        )
        print("[09 PREPARE] Frozen label-blind confirmation masks verified; unchanged.")
    else:
        temp = destination.with_suffix(".csv.tmp")
        proposed.to_csv(temp, index=False)
        temp.replace(destination)
        print(f"[09 PREPARE] New confirmation masks frozen, seed={SEED}.")
    return destination


def _validate_spec(folder: Path, member: Path) -> None:
    watched = _necessary_files() | {"confirmation_membership": member}
    contract = {
        "schema": 1,
        "method_names": list(METHODS),
        "seed": SEED,
        "min_rmse_gain": MIN_GAIN,
        "mae_tolerance": MAE_TOLERANCE,
        "tail_tolerance": TAIL_TOLERANCE,
        "source_sha256": {key: digest(path) for key, path in watched.items()},
    }
    path = folder / "run_spec.json"
    if path.is_file():
        if json.loads(path.read_text(encoding="utf-8")) != contract:
            raise RuntimeError("09 run_spec differs: use original code/data or NEW output folder.")
    else:
        if any(folder.glob("*predictions.csv")) or (folder / "decision.json").exists():
            raise RuntimeError("Existing checkpoint outputs lack run_spec.json.")
        atomic_json(path, contract)


def _save_csv(frame: pd.DataFrame, path: Path) -> None:
    temp = path.with_suffix(".csv.tmp")
    frame.to_csv(temp, index=False)
    temp.replace(path)


def _report(data: pd.DataFrame, baseline: float, *, stage: str) -> None:
    summary = summarize_bank(data)
    print(f"[09 {stage}] Paired RMSE/MAE against REPORTED yields:")
    print(summary.to_string(index=False), flush=True)
    if len(data["split_id"].unique()) == 16 and "Local04D_frozen" in set(data["model"]):
        actual = float(
            summary.loc[summary["model"].eq("Local04D_frozen"), "rmse_pooled"].iloc[0]
        )
        if stage == "DEVELOPMENT" and abs(actual - baseline) > 1e-7:
            raise RuntimeError(
                f"Historical Local04D failed full 656-row parity: {actual} vs {baseline}"
            )


def _run_development(data, history, output: Path) -> None:
    member = output / "confirmation_membership.csv"
    if not member.is_file():
        raise RuntimeError("Freeze confirmation masks first with --prepare.")
    _validate_spec(output, member)
    prediction_path = output / "development_predictions.csv"
    if prediction_path.exists():
        print("[09] Development already frozen; refusing to recompute or overwrite.")
        _report(pd.read_csv(prediction_path), EXPECTED_BASELINE, stage="DEVELOPMENT")
        return
    predictions = evaluate_bank(data, history)
    _report(predictions, EXPECTED_BASELINE, stage="DEVELOPMENT")
    summary = summarize_bank(predictions)
    selection, selected, decision = leave_one_split_out_selection(
        predictions,
        min_gain=MIN_GAIN,
        mae_tolerance=MAE_TOLERANCE,
        tail_tolerance=TAIL_TOLERANCE,
    )
    _save_csv(predictions, prediction_path)
    _save_csv(summary, output / "development_summary.csv")
    _save_csv(selection, output / "loso_selections.csv")
    _save_csv(selected, output / "loso_selected_predictions.csv")
    _save_csv(build_review_queue(data), output / "human_label_review_candidates.csv")
    atomic_json(output / "decision.json", {
        **decision,
        "stage": "development",
        "baseline_model": "Local04D_frozen",
        "results_compare_to": "reported_yield_not_verified_true_yield",
        "hidden_fira_y_used": False,
        "official_59_predictions_changed": False,
        "human_review_only": True,
        "can_confirm": decision["LOSO_STABILITY_GATE"] == "PASS",
        "confirmation_membership_digest": digest(member),
    })
    print(f"[09] LOSO_STABILITY_GATE = {decision['LOSO_STABILITY_GATE']}")
    print(f"[09] Label review queue (NOT errors): {output / 'human_label_review_candidates.csv'}")
    if decision["LOSO_STABILITY_GATE"] == "PASS":
        print("[09] One frozen challenger may be tested on --confirm.")
    else:
        print("[09] No confirmation run: Local04D retains priority.")


def _confirm(data, output: Path) -> None:
    member = output / "confirmation_membership.csv"
    decision_path = output / "decision.json"
    if not decision_path.is_file() or not member.is_file():
        raise RuntimeError("First --prepare and --run on development.")
    _validate_spec(output, member)
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if decision["LOSO_STABILITY_GATE"] != "PASS":
        raise RuntimeError("09 LOSO stability gate failed. Do not spend confirmation.")
    if decision["confirmation_membership_digest"] != digest(member):
        raise RuntimeError("Frozen confirmation memberships changed.")
    challenger = decision["candidate_to_freeze_for_new_mask_confirmation"]
    if challenger not in METHODS[1:]:
        raise ValueError("No frozen eligible challenger.")
    outfile = output / "confirmation_predictions.csv"
    if outfile.is_file():
        predictions = pd.read_csv(outfile)
        if set(predictions["model"]) != {"Local04D_frozen", challenger}:
            raise ValueError("Existing confirmation artifact belongs to another challenger.")
    else:
        predictions = evaluate_bank(
            data, pd.read_csv(member),
            selected_methods=("Local04D_frozen", challenger),
        )
        _save_csv(predictions, outfile)
    _report(predictions, EXPECTED_BASELINE, stage="CONFIRMATION")
    summary = summarize_bank(predictions)
    _save_csv(summary, output / "confirmation_summary.csv")
    bootstrap = paired_cluster_interval(predictions, challenger, seed=SEED)
    atomic_json(output / "confirmation_bootstrap.json", bootstrap)
    baseline = summary.loc[summary["model"].eq("Local04D_frozen")].iloc[0]
    rival = summary.loc[summary["model"].eq(challenger)].iloc[0]
    supports = (
        float(bootstrap["ci95_high"]) < 0
        and float(rival["mae_pooled"]) <= float(baseline["mae_pooled"]) + MAE_TOLERANCE
        and float(rival["tail_rmse"]) <= float(baseline["tail_rmse"]) + TAIL_TOLERANCE
    )
    atomic_json(output / "confirmation_decision.json", {
        "status": "SUPPORTS_HUMAN_REVIEW" if supports else "NO_PROMOTION",
        "challenger": challenger,
        "paired_bootstrap": bootstrap,
        "caution": "Different masks reuse the same 138 y; measurement error not verified.",
        "hidden_y_used": False,
        "modified_canonical_predictions": False,
        "automatic_promotion": False,
    })
    print(f"[09] CONFIRMATION = {'SUPPORTS_HUMAN_REVIEW' if supports else 'NO_PROMOTION'}")
    print("[09] No automatic modification of 59 predictions.")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    for flag in ("preflight", "prepare", "smoke", "run", "confirm", "summary"):
        mode.add_argument("--" + flag, action="store_true")
    p.add_argument("--output", type=Path, default=OUT)
    args = p.parse_args()
    output = args.output.resolve()

    if args.summary:
        for name in ("decision.json", "confirmation_decision.json"):
            path = output / name
            if path.is_file():
                print(path.read_text(encoding="utf-8"))
        for name in ("development_summary.csv", "confirmation_summary.csv"):
            path = output / name
            if path.is_file():
                print(f"\n{name}\n" + pd.read_csv(path).to_string(index=False))
        return 0

    data, history = _check()
    if args.preflight:
        return 0
    if args.prepare:
        _prepare(data, history, output)
    elif args.smoke:
        primary = next(iter(sorted(history["split_id"].unique())))
        subset = history.loc[history["split_id"].eq(primary)]
        smoke = evaluate_bank(data, subset, expected_splits=1)
        output_smoke = output / "smoke"
        output_smoke.mkdir(parents=True, exist_ok=True)
        _save_csv(smoke, output_smoke / "smoke_predictions.csv")
        print("[09 SMOKE] 1 split × 41 pseudo-targets × 5 models: PASS.")
        print("[09 SMOKE] No selection, no confirmation, no promotion.")
    elif args.run:
        _run_development(data, history, output)
    elif args.confirm:
        _confirm(data, output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
