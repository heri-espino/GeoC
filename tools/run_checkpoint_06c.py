#!/usr/bin/env python
"""Run Checkpoint 06C low-capacity tail calibration."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from geocebada.evaluation.checkpoint06 import normalize_honest_prediction_table
from geocebada.evaluation.checkpoint06c import (
    build_calibration_predictions,
    leave_one_split_out_calibration_selection,
    summarize_calibration_predictions,
)
from geocebada.paths import find_project_root

ROOT = Path(__file__).resolve().parents[1]


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping in {path}.")
    return payload


def _git_commit(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _preflight(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    inputs = config["inputs"]
    paths = {key: root / str(value) for key, value in inputs.items()}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing 06C inputs: {missing}")

    decision = json.loads(paths["checkpoint06_report"].read_text(encoding="utf-8"))
    status = str(decision["decision"]["TAIL_HYPOTHESIS"])
    if status != str(config["stage"]["required_tail_hypothesis"]):
        raise RuntimeError(
            f"06C requires TAIL_HYPOTHESIS=SUPPORTED, observed {status}."
        )

    base_oof = pd.read_csv(paths["base_oof_predictions"])
    long_test = pd.read_csv(paths["honest_pseudo_predictions"])
    pseudo = normalize_honest_prediction_table(
        long_test,
        required_methods=["Local04D", "CatBoost_C1"],
    )
    train = base_oof.loc[base_oof["family"].eq("target_matched")]
    test = pseudo.loc[pseudo["family"].eq("target_matched")]
    if train.groupby("split_id")["ID_POLIGONO"].nunique().ne(97).any():
        raise ValueError("06C expected 97 honest pseudo-train OOF rows per target-matched split.")
    if test.groupby("split_id")["ID_POLIGONO"].nunique().ne(41).any():
        raise ValueError("06C expected 41 honest pseudo-target rows per target-matched split.")
    if train["split_id"].nunique() != 16 or test["split_id"].nunique() != 16:
        raise ValueError("06C expected exactly 16 target-matched development splits.")

    return {
        "TAIL_HYPOTHESIS": status,
        "target_matched_splits": 16,
        "pseudo_train_oof_per_split": 97,
        "pseudo_targets_per_split": 41,
        "calibration_fit_uses_cross_fitted_predictions": True,
        "hidden_fira_y_used_or_scored": False,
    }


def _markdown(
    *,
    generated_at: str,
    git_commit: str | None,
    summary: pd.DataFrame,
    loso: dict[str, Any],
) -> str:
    best = summary.iloc[0].to_dict()
    lines = [
        "# Checkpoint 06C — low-capacity tail calibration",
        "",
        f"Generated: {generated_at}",
        f"Git commit: {git_commit}",
        "",
        "## Contract",
        "",
        "- prerequisite 06A/06B: SUPPORTED",
        "- calibration fit rows are cross-fitted pseudo-train predictions",
        "- score rows are untouched pseudo-target predictions",
        "- hidden 59 FIRA y used/scored: no",
        "- clustering and specialists: not run",
        "",
        "## Development ranking",
        "",
        summary.head(12).to_string(index=False),
        "",
        "## Best development candidate",
        "",
        json.dumps(best, indent=2, default=float),
        "",
        "## LOSO selection",
        "",
        json.dumps(loso, indent=2, sort_keys=True),
        "",
    ]
    passed = (
        loso["rmse_mean_improvement"] > 0
        and loso["pooled_rmse_improvement"] > 0
        and loso["tail_rmse_improvement"] > 0
    )
    if passed:
        lines.extend(
            [
                "## Decision",
                "",
                "**06C_LOSO_GATE = PASS**",
                "",
                "Calibration is promising enough to freeze one challenger for fresh confirmation.",
                "Do not start clustering unless fresh confirmation fails or "
                "the calibration mechanism is rejected.",
            ]
        )
    else:
        lines.extend(
            [
                "## Decision",
                "",
                "**06C_LOSO_GATE = FAIL**",
                "",
                "Simple calibration did not survive split-excluded selection.",
                "Proceed to 06D X-only regime discovery rather than adding "
                "more calibration flexibility.",
            ]
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/checkpoint06c.yaml"),
    )
    parser.add_argument("--preflight", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    if not (root / "pyproject.toml").is_file():
        root = find_project_root(root)
    config_path = args.config if args.config.is_absolute() else root / args.config
    config = _load_yaml(config_path)

    preflight = _preflight(root, config)
    print("Checkpoint 06C preflight: PASS")
    for key, value in preflight.items():
        print(f"{key}: {value}")
    if args.preflight:
        return 0

    base_oof = pd.read_csv(root / str(config["inputs"]["base_oof_predictions"]))
    pseudo_long = pd.read_csv(root / str(config["inputs"]["honest_pseudo_predictions"]))
    pseudo = normalize_honest_prediction_table(
        pseudo_long,
        required_methods=["Local04D", "CatBoost_C1"],
    )

    predictions = build_calibration_predictions(
        base_oof,
        pseudo,
        config=config,
    )
    split_metrics, summary = summarize_calibration_predictions(predictions)
    selections, loso_predictions, loso_summary = (
        leave_one_split_out_calibration_selection(
            predictions,
            split_metrics,
            center_tolerance_absolute=float(
                config["selection"]["center_tolerance_absolute"]
            ),
            center_tolerance_fraction=float(
                config["selection"]["center_tolerance_fraction"]
            ),
        )
    )

    output = root / str(config["outputs"]["directory"])
    output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output / str(config["outputs"]["predictions"]), index=False)
    split_metrics.to_csv(output / str(config["outputs"]["split_metrics"]), index=False)
    summary.to_csv(output / str(config["outputs"]["summary"]), index=False)
    selections.to_csv(output / str(config["outputs"]["loso_selection"]), index=False)
    loso_predictions.to_csv(output / str(config["outputs"]["loso_predictions"]), index=False)
    (output / str(config["outputs"]["loso_summary"])).write_text(
        json.dumps(loso_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    generated_at = datetime.now(UTC).isoformat()
    git_commit = _git_commit(root)
    report = {
        "schema_version": 1,
        "checkpoint": "06C",
        "generated_at": generated_at,
        "git_commit": git_commit,
        "hidden_fira_y_used_or_scored": False,
        "preflight": preflight,
        "best_development": summary.iloc[0].to_dict(),
        "loso": loso_summary,
        "loso_gate_passed": bool(
            loso_summary["rmse_mean_improvement"] > 0
            and loso_summary["pooled_rmse_improvement"] > 0
            and loso_summary["tail_rmse_improvement"] > 0
        ),
    }
    (output / str(config["outputs"]["report_json"])).write_text(
        json.dumps(report, indent=2, sort_keys=True, default=float) + "\n",
        encoding="utf-8",
    )
    (output / str(config["outputs"]["report_markdown"])).write_text(
        _markdown(
            generated_at=generated_at,
            git_commit=git_commit,
            summary=summary,
            loso=loso_summary,
        ),
        encoding="utf-8",
    )

    print("Checkpoint 06C calibration: PASS")
    print(f"Best development: {summary.iloc[0]['method']}")
    print(
        "06C_LOSO_GATE = "
        + (
            "PASS"
            if report["loso_gate_passed"]
            else "FAIL"
        )
    )
    print(f"Inspect: {output.relative_to(root) / str(config['outputs']['report_markdown'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
