#!/usr/bin/env python
"""Run Checkpoint 06E supervised center/tail mixture."""

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
from geocebada.evaluation.checkpoint06e import (
    build_supervised_mixture_predictions,
    leave_one_split_out_mixture_selection,
    summarize_mixture_predictions,
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
    prereqs = {
        "tail": root / str(config["stage"]["prerequisite_tail"]),
        "calibration": root / str(config["stage"]["prerequisite_calibration"]),
        "regimes": root / str(config["stage"]["prerequisite_regimes"]),
    }
    for key, path in prereqs.items():
        if not path.is_file():
            raise FileNotFoundError(f"06E missing prerequisite {key}: {path}")

    tail = json.loads(prereqs["tail"].read_text(encoding="utf-8"))
    calibration = json.loads(prereqs["calibration"].read_text(encoding="utf-8"))
    regimes = json.loads(prereqs["regimes"].read_text(encoding="utf-8"))
    if tail["decision"]["TAIL_HYPOTHESIS"] != "SUPPORTED":
        raise RuntimeError("06E requires 06A/06B TAIL_HYPOTHESIS=SUPPORTED.")
    if bool(calibration["loso_gate_passed"]):
        raise RuntimeError("06E should not supersede a 06C calibration that passed LOSO.")
    if regimes["decision"]["REGIME_HYPOTHESIS"] != "NOT_SUPPORTED":
        raise RuntimeError(
            "06E supervised gates are entered only after unsupervised 06D regimes fail."
        )

    inputs = {
        key: root / str(value)
        for key, value in config["inputs"].items()
    }
    missing = [str(path) for path in inputs.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"06E missing inputs: {missing}")

    base_oof = pd.read_csv(inputs["base_oof_predictions"])
    pseudo_long = pd.read_csv(inputs["honest_pseudo_predictions"])
    pseudo = normalize_honest_prediction_table(
        pseudo_long,
        required_methods=["Local04D", "CatBoost_C1"],
    )
    agronomic = pd.read_csv(inputs["agronomic_table"])

    train = base_oof.loc[base_oof["family"].eq("target_matched")]
    test = pseudo.loc[pseudo["family"].eq("target_matched")]
    if train["split_id"].nunique() != 16 or test["split_id"].nunique() != 16:
        raise ValueError("06E expects 16 target-matched development splits.")
    if train.groupby("split_id")["ID_POLIGONO"].nunique().ne(97).any():
        raise ValueError("06E expects 97 pseudo-train OOF rows per split.")
    if test.groupby("split_id")["ID_POLIGONO"].nunique().ne(41).any():
        raise ValueError("06E expects 41 pseudo-target rows per split.")
    if len(agronomic) != 197 or agronomic["ID_POLIGONO"].duplicated().any():
        raise ValueError("06E expects the 197-row one-to-one agronomic table.")

    return {
        "tail_hypothesis": "SUPPORTED",
        "06c_loso_gate_passed": False,
        "06d_regime_hypothesis": "NOT_SUPPORTED",
        "target_matched_splits": 16,
        "pseudo_train_oof_per_split": 97,
        "pseudo_targets_per_split": 41,
        "hidden_fira_y_used_or_scored": False,
        "low_gate_family": config["gates"]["low_family"],
        "high_gate_family": config["gates"]["high_family"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/checkpoint06e.yaml"),
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
    print("Checkpoint 06E preflight: PASS")
    for key, value in preflight.items():
        print(f"{key}: {value}")
    if args.preflight:
        return 0

    inputs = config["inputs"]
    base_oof = pd.read_csv(root / str(inputs["base_oof_predictions"]))
    pseudo_long = pd.read_csv(root / str(inputs["honest_pseudo_predictions"]))
    pseudo = normalize_honest_prediction_table(
        pseudo_long,
        required_methods=["Local04D", "CatBoost_C1"],
    )
    agronomic = pd.read_csv(root / str(inputs["agronomic_table"]))
    manifest = json.loads(
        (root / str(inputs["agronomic_manifest"])).read_text(encoding="utf-8")
    )

    predictions, gates = build_supervised_mixture_predictions(
        base_oof,
        pseudo,
        agronomic,
        manifest,
        config=config,
    )
    split_metrics, summary = summarize_mixture_predictions(predictions)
    selections, loso_predictions, loso = leave_one_split_out_mixture_selection(
        predictions,
        split_metrics,
        min_center_improvement=float(config["selection"]["min_center_improvement"]),
        tail_tolerance_absolute=float(
            config["selection"]["tail_tolerance_absolute"]
        ),
        tail_tolerance_fraction=float(
            config["selection"]["tail_tolerance_fraction"]
        ),
    )

    out = root / str(config["outputs"]["directory"])
    out.mkdir(parents=True, exist_ok=True)
    gates.to_csv(out / str(config["outputs"]["gate_weights"]), index=False)
    predictions.to_csv(out / str(config["outputs"]["predictions"]), index=False)
    split_metrics.to_csv(out / str(config["outputs"]["split_metrics"]), index=False)
    summary.to_csv(out / str(config["outputs"]["summary"]), index=False)
    selections.to_csv(out / str(config["outputs"]["loso_selection"]), index=False)
    loso_predictions.to_csv(
        out / str(config["outputs"]["loso_predictions"]), index=False
    )
    (out / str(config["outputs"]["loso_summary"])).write_text(
        json.dumps(loso, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    gate_passed = bool(
        loso["rmse_mean_improvement"] > 0
        and loso["pooled_rmse_improvement"] > 0
        and loso["center_rmse_improvement"] > 0
        and loso["tail_rmse_change"]
        <= max(
            float(config["selection"]["tail_tolerance_absolute"]),
            float(config["selection"]["tail_tolerance_fraction"])
            * loso["incumbent_tail_rmse"],
        )
    )

    generated_at = datetime.now(UTC).isoformat()
    git_commit = _git_commit(root)
    report = {
        "schema_version": 1,
        "checkpoint": "06E",
        "generated_at": generated_at,
        "git_commit": git_commit,
        "hidden_fira_y_used_or_scored": False,
        "preflight": preflight,
        "best_development": summary.iloc[0].to_dict(),
        "loso": loso,
        "loso_gate_passed": gate_passed,
    }
    (out / str(config["outputs"]["report_json"])).write_text(
        json.dumps(report, indent=2, sort_keys=True, default=float) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Checkpoint 06E — supervised center/tail mixture",
        "",
        f"Generated: {generated_at}",
        f"Git commit: {git_commit}",
        "",
        "## Architecture",
        "",
        "Local04D + w_center*center_residual_expert(X)"
        " + w_low*low_offset + w_high*high_offset",
        "",
        f"low gate family: {config['gates']['low_family']}",
        f"high gate family: {config['gates']['high_family']}",
        "",
        "## Development ranking",
        "",
        summary.head(20).to_string(index=False),
        "",
        "## LOSO",
        "",
        json.dumps(loso, indent=2, sort_keys=True),
        "",
        "## Decision",
        "",
        f"**06E_LOSO_GATE = {'PASS' if gate_passed else 'FAIL'}**",
        "",
    ]
    if gate_passed:
        lines.extend(
            [
                "Freeze exactly one supervised mixture challenger and run fresh confirmation.",
                "Do not tune 06E further on the development splits.",
            ]
        )
    else:
        lines.extend(
            [
                "The center/tail supervised mixture did not survive split-excluded selection.",
                "Retain Local04D and close the planned Checkpoint 06 modeling path unless a new, predeclared hypothesis is introduced.",
            ]
        )
    (out / str(config["outputs"]["report_markdown"])).write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print("Checkpoint 06E supervised mixture: PASS")
    print(f"Best development: {summary.iloc[0]['method']}")
    print(f"06E_LOSO_GATE = {'PASS' if gate_passed else 'FAIL'}")
    print(f"Inspect: {out.relative_to(root) / str(config['outputs']['report_markdown'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
