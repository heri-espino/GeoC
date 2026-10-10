#!/usr/bin/env python3
"""Single unattended Checkpoint 07 run: CatBoost -> Local07 -> Prithvi.

Only the three requested research branches; no XGBoost/LightGBM/model zoo.
Prepared HLS/SMAP/07B features and predownloaded Prithvi weights are inputs.
This orchestrator DOES NOT call Git, download archives, or access hidden y.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from geocebada.evaluation.checkpoint07c import aggregate_07c_results

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "reports/checkpoint_07_three"
STAGES = ("catboost", "local07", "prithvi")


def _utc() -> str:
    return datetime.now(UTC).isoformat()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stages(args: argparse.Namespace, *, smoke: bool = False) -> dict[str, list[str]]:
    out = args.output.resolve()
    common = [sys.executable, "-u"]
    cat = common + [
        "tools/run_checkpoint_07c.py",
        "--device", "gpu" if not smoke else "cpu",
        "--families", "catboost", "--trials", str(args.catboost_trials),
        "--threads", str(args.threads),
        "--output", str(out / "catboost"),
    ]
    loc = common + [
        "tools/run_checkpoint_07_local.py",
        "--trials", str(args.local_trials),
        "--output", str(out / "local07"),
    ]
    pre = common + [
        "tools/run_checkpoint_07_prithvi.py",
        "--model", args.prithvi_model,
        "--output", str(out / "prithvi"),
    ]
    if smoke:
        for command in (cat, loc, pre):
            command.append("--smoke")
    return {"catboost": cat, "local07": loc, "prithvi": pre}


def _check(args: argparse.Namespace) -> None:
    if args.catboost_trials < 1 or args.local_trials < 1 or args.threads < 1:
        raise ValueError("Positive CatBoost/Local07 budgets and threads required.")
    for name, cmd in _stages(args).items():
        print(f"[CHECK] {name} dependencies and inputs", flush=True)
        check = subprocess.run(
            [*cmd, "--preflight"], cwd=ROOT, check=False,
        )
        if check.returncode != 0:
            raise RuntimeError(
                f"Preflight failed for {name}, return code {check.returncode}. "
                "Correct it before starting the unattended run."
            )
    print("[07 THREE] Preflight PASS: CatBoost, Local07, Prithvi.", flush=True)


def _run_spec(args: argparse.Namespace) -> dict:
    data = {
        "features": ROOT / "reports/checkpoint_07/highdim/expanded_parcel_features.csv",
        "membership": ROOT / "reports/checkpoint_04b/pseudo_competition_splits.csv",
        "baseline": ROOT / "reports/checkpoint_06/parcel_oof_residuals.csv",
        "targets": ROOT / "data/source/tabular/ID_area_rendimiento_70_30_Reto_AgroCebada.csv",
        "chip_manifest": ROOT / "data/processed/checkpoint_07/prithvi_chip_manifest.json",
    }
    script_paths = [
        "tools/run_checkpoint_07_three.py",
        "tools/run_checkpoint_07c.py",
        "tools/run_checkpoint_07_local.py",
        "tools/run_checkpoint_07_prithvi.py",
        "src/geocebada/evaluation/checkpoint07c.py",
        "src/geocebada/evaluation/checkpoint07_local.py",
        "src/geocebada/evaluation/checkpoint07_prithvi.py",
        "src/geocebada/evaluation/checkpoint07_prithvi_head.py",
        "src/geocebada/evaluation/checkpoint07_pca.py",
    ]
    return {
        "schema": 1, "models": list(STAGES),
        "catboost_trials": args.catboost_trials,
        "local_trials": args.local_trials, "threads": args.threads,
        "prithvi_model": args.prithvi_model,
        "python_executable": str(Path(sys.executable).resolve()),
        "input_sha256": {key: _sha(value) for key, value in data.items()},
        "code_sha256": {name: _sha(ROOT / name) for name in script_paths},
    }


def _atomic_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2)+"\n", encoding="utf-8")
    tmp.replace(path)


def _ensure_spec(folder: Path, spec: dict) -> None:
    path = folder / "run_spec.json"
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous != spec:
            raise ValueError(
                "Prior run specification differs from current files/options. "
                "For a new experiment choose a NEW --output directory. "
                "Do not merge incompatible completed Optuna studies."
            )
    else:
        if any(folder.glob("*/target_matched_*__*.sqlite")):
            raise RuntimeError(
                "Existing studies lack a run_spec.json; refuse unsafe resume."
            )
        _atomic_json(path, spec)


def _summary(folder: Path, *, require_complete: bool) -> pd.DataFrame:
    inputs = {
        "catboost": folder / "catboost",
        "local07": folder / "local07",
        "prithvi": folder / "prithvi" / "300M" / "heads",
    }
    if (folder / "prithvi" / "600M" / "heads").is_dir():
        inputs["prithvi"] = folder / "prithvi" / "600M" / "heads"
    scores = []
    for source, path in inputs.items():
        try:
            _, summary = aggregate_07c_results(path)
        except FileNotFoundError:
            if require_complete:
                raise
            continue
        selected = summary.loc[
            (summary["model"] != "Local04D")
            | summary["family"].eq("catboost")
        ].copy()
        selected["branch"] = source
        scores.append(selected)
    if not scores:
        raise FileNotFoundError("No completed 07C three-branch results to summarize.")
    result = pd.concat(scores, ignore_index=True)
    if require_complete:
        incomplete = result.loc[result["n_splits_completed"].ne(16)]
        if len(incomplete):
            raise RuntimeError(
                f"Incomplete 16-split three-branch coverage: "
                f"{incomplete[['branch', 'model', 'n_splits_completed']].to_dict('records')}"
            )
    result = result.sort_values(
        ["n_splits_completed", "rmse_mean_split"],
        ascending=[False, True],
    )
    result.to_csv(folder / "three_model_summary.csv", index=False)
    print(
        "[07 THREE] Matched target-held-out RMSE:\n"
        + result.head(25).to_string(index=False),
        flush=True,
    )
    return result


def run(args: argparse.Namespace) -> int:
    _check(args)  # All three availability checks run BEFORE any expensive stage.
    folder = args.output.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    _ensure_spec(folder, _run_spec(args))
    status_path = folder / "status.json"
    state = (
        json.loads(status_path.read_text(encoding="utf-8"))
        if status_path.is_file() else {"stages": {}, "created_at_utc": _utc()}
    )
    failed = []
    for stage, command in _stages(args).items():
        if state["stages"].get(stage, {}).get("status") == "completed":
            print(f"[07 THREE] {stage}: completed previously; skip.", flush=True)
            continue
        stage_path = folder / stage
        stage_path.mkdir(parents=True, exist_ok=True)
        log_path = stage_path / "stage.log"
        state["stages"][stage] = {
            "status": "running", "started_at_utc": _utc(),
            "command": command[1:], "log": str(log_path.relative_to(folder)),
        }
        _atomic_json(status_path, state)
        print(f"[07 THREE] Starting {stage}, log={log_path}", flush=True)
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n===== {stage} resumed/started {_utc()} =====\n")
            log.flush()
            try:
                outcome = subprocess.run(
                    command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                    check=False,
                ).returncode
            except Exception as exc:
                outcome = 1
                log.write(f"\nStage invocation failed: {exc!r}\n")
        state["stages"][stage]["ended_at_utc"] = _utc()
        state["stages"][stage]["exit_code"] = outcome
        state["stages"][stage]["status"] = (
            "completed" if outcome == 0 else "failed"
        )
        _atomic_json(status_path, state)
        print(
            f"[07 THREE] {stage}: {'PASS' if outcome == 0 else 'FAILED'} "
            f"exit={outcome}; {log_path}",
            flush=True,
        )
        if outcome:
            failed.append(stage)
            # Continue independent branches, making errors visible at end.
    if failed:
        print(
            f"[07 THREE] INCOMPLETE: failed={failed}. Fix error(s) from "
            "per-stage logs; rerun the SAME command to resume.",
            flush=True,
        )
        return 1
    _summary(folder, require_complete=True)
    print("[07 THREE] COMPLETED all three branches; no model promoted.", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check", action="store_true")
    action.add_argument("--run", action="store_true")
    action.add_argument("--smoke", action="store_true")
    action.add_argument("--status", action="store_true")
    args0 = ROOT / "reports/checkpoint_07_three"
    parser.add_argument("--output", type=Path, default=args0)
    parser.add_argument("--catboost-trials", type=int, default=48)
    parser.add_argument("--local-trials", type=int, default=48)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--prithvi-model", choices=("300", "600"), default="300")
    args = parser.parse_args()
    if args.status:
        file = args.output / "status.json"
        print(file.read_text(encoding="utf-8") if file.exists() else
              "No unattended run has started yet.")
        if args.output.is_dir():
            try:
                _summary(args.output, require_complete=False)
            except FileNotFoundError:
                pass
        return 0
    if args.check:
        _check(args)
        return 0
    if args.smoke:
        _check(args)
        for stage, command in _stages(args, smoke=True).items():
            print(f"[07 THREE SMOKE] {stage}", flush=True)
            if subprocess.run(command, cwd=ROOT).returncode != 0:
                raise RuntimeError(f"Real {stage} smoke failed; do not launch full.")
        return 0
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
