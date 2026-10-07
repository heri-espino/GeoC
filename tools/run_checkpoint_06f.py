#!/usr/bin/env python
"""Run Checkpoint 06F selective center expert with abstention."""

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
from geocebada.evaluation.checkpoint06e import summarize_mixture_predictions
from geocebada.evaluation.checkpoint06f import (
    _build_selective_center_predictions,
    _leave_one_split_out_selective_selection,
    _summarize_selective_routing,
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
    prerequisite = root / str(config["stage"]["prerequisite_06e"])
    if not prerequisite.is_file():
        raise FileNotFoundError(f"06F missing 06E prerequisite: {prerequisite}")

    report_06e = json.loads(prerequisite.read_text(encoding="utf-8"))
    if bool(report_06e["loso_gate_passed"]):
        raise RuntimeError("06F is only defined after 06E fails its LOSO gate.")

    if str(config["center"]["representation"]) != "all_agronomic_pca":
        raise ValueError("06F freezes the G1-PCA center representation from 06E.")
    if float(config["center"]["ridge_alpha"]) != 1000.0:
        raise ValueError("06F freezes Ridge alpha=1000 from the 06E center signal.")
    if float(config["center"]["pca_variance"]) != 0.80:
        raise ValueError("06F freezes PCA retained variance at 0.80.")

    cutoffs = [float(value) for value in config["gate"]["tail_score_cutoffs"]]
    if not cutoffs or cutoffs != sorted(set(cutoffs)):
        raise ValueError("06F tail-score cutoffs must be unique and sorted.")
    if any(not 0.0 < value < 1.0 for value in cutoffs):
        raise ValueError("06F tail-score cutoffs must lie strictly between 0 and 1.")

    inputs = {
        key: root / str(value)
        for key, value in config["inputs"].items()
    }
    missing = [str(path) for path in inputs.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"06F missing inputs: {missing}")

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
        raise ValueError("06F expects 16 target-matched development splits.")
    if train.groupby("split_id")["ID_POLIGONO"].nunique().ne(97).any():
        raise ValueError("06F expects 97 pseudo-train OOF rows per split.")
    if test.groupby("split_id")["ID_POLIGONO"].nunique().ne(41).any():
        raise ValueError("06F expects 41 pseudo-target rows per split.")
    if len(agronomic) != 197 or agronomic["ID_POLIGONO"].duplicated().any():
        raise ValueError("06F expects the 197-row one-to-one agronomic table.")

    return {
        "06e_loso_gate_passed": False,
        "target_matched_splits": 16,
        "pseudo_train_oof_per_split": 97,
        "pseudo_targets_per_split": 41,
        "hidden_fira_y_used_or_scored": False,
        "center_representation": config["center"]["representation"],
        "center_ridge_alpha": float(config["center"]["ridge_alpha"]),
        "center_pca_variance": float(config["center"]["pca_variance"]),
        "low_gate_family": config["gate"]["low_family"],
        "high_gate_family": config["gate"]["high_family"],
        "tail_score_cutoffs": cutoffs,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/checkpoint06f.yaml"),
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
    print("Checkpoint 06F preflight: PASS")
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

    predictions, gates = _build_selective_center_predictions(
        base_oof,
        pseudo,
        agronomic,
        manifest,
        config=config,
    )
    split_metrics, summary = summarize_mixture_predictions(predictions)
    routing = _summarize_selective_routing(predictions)
    selections, loso_predictions, loso = _leave_one_split_out_selective_selection(
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
    gates.to_csv(out / str(config["outputs"]["gate_scores"]), index=False)
    predictions.to_csv(out / str(config["outputs"]["predictions"]), index=False)
    split_metrics.to_csv(out / str(config["outputs"]["split_metrics"]), index=False)
    summary.to_csv(out / str(config["outputs"]["summary"]), index=False)
    routing.to_csv(out / str(config["outputs"]["routing_summary"]), index=False)
    selections.to_csv(out / str(config["outputs"]["loso_selection"]), index=False)
    loso_predictions.to_csv(
        out / str(config["outputs"]["loso_predictions"]),
        index=False,
    )
    (out / str(config["outputs"]["loso_summary"])).write_text(
        json.dumps(loso, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    tail_tolerance = max(
        float(config["selection"]["tail_tolerance_absolute"]),
        float(config["selection"]["tail_tolerance_fraction"])
        * float(loso["incumbent_tail_rmse"]),
    )
    gate_passed = bool(
        loso["rmse_mean_improvement"] > 0
        and loso["pooled_rmse_improvement"] > 0
        and loso["center_rmse_improvement"]
        >= float(config["selection"]["min_center_improvement"])
        and loso["tail_rmse_change"] <= tail_tolerance
    )

    generated_at = datetime.now(UTC).isoformat()
    git_commit = _git_commit(root)
    report = {
        "schema_version": 1,
        "checkpoint": "06F",
        "generated_at": generated_at,
        "git_commit": git_commit,
        "hidden_fira_y_used_or_scored": False,
        "post_hoc_followup_to_06e": True,
        "preflight": preflight,
        "best_development": summary.iloc[0].to_dict(),
        "routing": routing.to_dict(orient="records"),
        "loso": loso,
        "loso_gate_passed": gate_passed,
        "fresh_confirmation_required_before_promotion": True,
    }
    (out / str(config["outputs"]["report_json"])).write_text(
        json.dumps(report, indent=2, sort_keys=True, default=float) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Checkpoint 06F — selective center expert / abstention gate",
        "",
        f"Generated: {generated_at}",
        f"Git commit: {git_commit}",
        "",
        "## Architecture",
        "",
        "If both supervised tail scores are below a frozen cutoff:",
        "    prediction = Local04D + frozen G1-PCA Ridge center residual correction",
        "Else:",
        "    prediction = Local04D",
        "",
        "No low-tail or high-tail correction is applied.",
        "",
        "## Frozen center expert",
        "",
        f"representation: {config['center']['representation']}",
        f"PCA retained variance: {config['center']['pca_variance']}",
        f"Ridge alpha: {config['center']['ridge_alpha']}",
        "",
        "## Development ranking",
        "",
        summary.to_string(index=False),
        "",
        "## Routing diagnostics",
        "",
        routing.to_string(index=False),
        "",
        "## LOSO",
        "",
        json.dumps(loso, indent=2, sort_keys=True),
        "",
        "## Decision",
        "",
        f"**06F_LOSO_GATE = {'PASS' if gate_passed else 'FAIL'}**",
        "",
    ]
    if gate_passed:
        lines.extend(
            [
                "This is a post-hoc follow-up to 06E. A LOSO PASS is screening evidence only.",
                "Freeze exactly one selected cutoff and run fresh confirmation against Local04D.",
                "Do not tune thresholds further on these 16 development splits.",
                "Do not change the 59 canonical predictions before fresh confirmation passes.",
            ]
        )
    else:
        lines.extend(
            [
                "Selective hard routing did not survive the predeclared LOSO gate.",
                "Retain Local04D and close center/tail correction work on "
                "these development splits.",
            ]
        )

    (out / str(config["outputs"]["report_markdown"])).write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print("Checkpoint 06F selective center expert: PASS")
    print(f"Best development: {summary.iloc[0]['method']}")
    print(f"06F_LOSO_GATE = {'PASS' if gate_passed else 'FAIL'}")
    print(
        "Inspect: "
        f"{out.relative_to(root) / str(config['outputs']['report_markdown'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
