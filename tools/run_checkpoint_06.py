#!/usr/bin/env python
"""Run Checkpoint 06A/06B tail-aware diagnostics."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from geocebada.evaluation.checkpoint06 import (
    checkpoint06_preflight,
    run_checkpoint06_diagnostics,
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


def _write_figures(result: dict[str, Any], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)

    parcel = result["parcel_summary"]
    for method in ["Local04D", "CatBoost_C1"]:
        subset = parcel.loc[parcel["method"].eq(method)].copy()
        if subset.empty:
            continue
        subset = subset.sort_values(
            "mean_squared_error", ascending=False
        ).reset_index(drop=True)
        cumulative = (
            subset["mean_squared_error"].cumsum()
            / subset["mean_squared_error"].sum()
        )
        x = (
            pd.Series(range(1, len(subset) + 1), dtype=float) / len(subset)
        ).to_numpy()
        fig, ax = plt.subplots(figsize=(7.0, 4.5))
        ax.plot(x, cumulative.to_numpy())
        ax.plot([0, 1], [0, 1], linestyle="--")
        ax.set_xlabel("Fraction of parcels, ordered by squared error")
        ax.set_ylabel("Cumulative share of parcel SSE")
        ax.set_title(f"{method}: concentration of honest squared error")
        fig.tight_layout()
        fig.savefig(
            directory / f"sse_concentration_{method}.png",
            dpi=180,
        )
        plt.close(fig)

    residuals = result["residuals"]
    for method in ["Local04D", "CatBoost_C1"]:
        subset = residuals.loc[residuals["method"].eq(method)].copy()
        if subset.empty:
            continue
        fig, ax = plt.subplots(figsize=(7.0, 4.5))
        ax.scatter(
            subset["observed"],
            subset["residual"],
            alpha=0.45,
            s=18,
        )
        ax.axhline(0.0, linewidth=1)
        ax.set_xlabel("Observed yield (t/ha)")
        ax.set_ylabel("Residual = observed - predicted")
        ax.set_title(f"{method}: honest residual vs observed yield")
        fig.tight_layout()
        fig.savefig(
            directory / f"residual_vs_y_{method}.png",
            dpi=180,
        )
        plt.close(fig)

    pivot = (
        parcel.loc[parcel["method"].isin(["Local04D", "CatBoost_C1"])]
        .pivot(
            index="ID_POLIGONO",
            columns="method",
            values="mean_residual",
        )
        .dropna()
    )
    if not pivot.empty:
        fig, ax = plt.subplots(figsize=(5.5, 5.0))
        ax.scatter(
            pivot["Local04D"],
            pivot["CatBoost_C1"],
            alpha=0.65,
            s=24,
        )
        ax.axhline(0.0, linewidth=1)
        ax.axvline(0.0, linewidth=1)
        ax.set_xlabel("Local04D mean honest residual")
        ax.set_ylabel("CatBoost mean honest residual")
        ax.set_title("Do Local04D and CatBoost miss the same parcels?")
        fig.tight_layout()
        fig.savefig(
            directory / "local_vs_catboost_residuals.png",
            dpi=180,
        )
        plt.close(fig)

    family = result["tail_classifier_summary"]
    family = family.loc[family["task"].eq("tail_vs_center")].copy()
    if not family.empty:
        family = family.sort_values("mean_split_roc_auc", ascending=True)
        fig, ax = plt.subplots(
            figsize=(8.0, max(4.0, 0.35 * len(family)))
        )
        ax.barh(family["family"], family["mean_split_roc_auc"])
        ax.axvline(0.5, linestyle="--", linewidth=1)
        ax.set_xlabel("Mean split ROC-AUC")
        ax.set_title("Tail-vs-center predictability by agronomic family")
        fig.tight_layout()
        fig.savefig(
            directory / "tail_classifier_family_auc.png",
            dpi=180,
        )
        plt.close(fig)


def _report_markdown(
    result: dict[str, Any],
    *,
    generated_at: str,
    git_commit: str | None,
) -> str:
    decision = result["decision"]
    concentration = result["concentration"]
    bias = result["tail_bias"]
    overlap = result["overlap"]
    classifier = result["tail_classifier_summary"]
    risk = result["error_risk_summary"]

    local_conc = concentration.loc[
        concentration["method"].eq("Local04D")
        & concentration["selection"].isin(
            ["top_5_parcels", "top_10pct"]
        )
    ]
    local_bias = bias.loc[
        bias["method"].eq("Local04D")
        & bias["tail_regime"].isin(["low", "high"])
    ]
    cat_bias = bias.loc[
        bias["method"].eq("CatBoost_C1")
        & bias["tail_regime"].isin(["low", "high"])
    ]
    tail_classifier = (
        classifier.loc[classifier["task"].eq("tail_vs_center")]
        .sort_values("mean_split_roc_auc", ascending=False)
        .head(10)
    )
    local_risk = (
        risk.loc[risk["method"].eq("Local04D")]
        .sort_values("risk_spearman", ascending=False)
        .head(10)
    )

    lines = [
        "# Checkpoint 06A/06B — Tail-aware diagnostic report",
        "",
        f"Generated: {generated_at}",
        f"Git commit: {git_commit}",
        "",
        "## Contract",
        "",
        "- active track: competition only",
        "- hidden 59 FIRA y used or scored: no",
        "- base predictions: frozen honest target-matched Checkpoint 05 predictions",
        "- clustering/specialists/calibration: not run in this stage",
        "",
        "## Diagnostic decision",
        "",
        f"**TAIL_HYPOTHESIS = {decision['TAIL_HYPOTHESIS']}**",
        "",
        "This is a continuation gate, not promotion of a competition model.",
        "",
        "Decision payload:",
        json.dumps(decision, indent=2, sort_keys=True),
        "",
        "## Local04D SSE concentration",
        "",
        local_conc.to_string(index=False)
        if not local_conc.empty
        else "No rows.",
        "",
        "## Local04D tail bias",
        "",
        local_bias.to_string(index=False)
        if not local_bias.empty
        else "No rows.",
        "",
        "## CatBoost tail bias",
        "",
        cat_bias.to_string(index=False)
        if not cat_bias.empty
        else "No rows.",
        "",
        "## Local04D vs CatBoost overlap",
        "",
        overlap.to_string(index=False)
        if not overlap.empty
        else "No rows.",
        "",
        "## Best agronomic-family tail classifiers",
        "",
        tail_classifier.to_string(index=False)
        if not tail_classifier.empty
        else "No rows.",
        "",
        "## Best Local04D error-risk family models",
        "",
        local_risk.to_string(index=False)
        if not local_risk.empty
        else "No rows.",
        "",
        "## Next step",
        "",
    ]
    status = str(decision["TAIL_HYPOTHESIS"])
    if status == "SUPPORTED":
        lines.extend(
            [
                "The diagnostic gate supports continuing to Checkpoint 06C.",
                "Implement low-capacity calibration first; do not jump directly to clustering.",
            ]
        )
    elif status == "AMBIGUOUS":
        lines.extend(
            [
                "The premise is only partially supported.",
                "Inspect feature/tail diagnostics before implementing 06C/06D.",
            ]
        )
    else:
        lines.extend(
            [
                "The diagnostic gate does not support the proposed tail-regime mechanism.",
                "Close Checkpoint 06 unless a specific diagnostic flaw is identified.",
            ]
        )
    lines.extend(
        [
            "",
            "Local04D and reports/checkpoint_05/final_predictions.csv remain canonical.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/checkpoint06.yaml"),
    )
    parser.add_argument("--preflight", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    if not (root / "pyproject.toml").is_file():
        root = find_project_root(root)

    config_path = (
        args.config
        if args.config.is_absolute()
        else root / args.config
    )
    config = _load_yaml(config_path)

    preflight = checkpoint06_preflight(root, config)
    print("Checkpoint 06A/06B preflight: PASS")
    for key, value in preflight.items():
        print(f"{key}: {value}")
    if args.preflight:
        return 0

    result = run_checkpoint06_diagnostics(root, config)
    output_cfg = config["outputs"]
    output_dir = root / str(output_cfg["directory"])
    output_dir.mkdir(parents=True, exist_ok=True)

    table_map = {
        "parcel_oof_residuals": "residuals",
        "parcel_residual_summary": "parcel_summary",
        "tail_definition_summary": "tail_thresholds",
        "descriptive_tail_quantiles": "descriptive_quantiles",
        "tail_sse_concentration": "concentration",
        "tail_bias_summary": "tail_bias",
        "expert_tail_overlap": "overlap",
        "feature_associations_by_split": "feature_associations_by_split",
        "feature_tail_associations": "feature_associations",
        "tail_classifier_oof": "tail_classifier_oof",
        "tail_classifier_summary": "tail_classifier_summary",
        "feature_family_tail_importance": "feature_family_tail_importance",
        "error_risk_oof": "error_risk_oof",
        "error_risk_summary": "error_risk_summary",
        "feature_family_error_risk": "feature_family_error_risk",
    }
    for output_key, result_key in table_map.items():
        result[result_key].to_csv(
            output_dir / str(output_cfg[output_key]),
            index=False,
        )

    _write_figures(
        result,
        output_dir / str(output_cfg["figures_directory"]),
    )

    generated_at = datetime.now(UTC).isoformat()
    git_commit = _git_commit(root)
    report = {
        "schema_version": 1,
        "generated_at": generated_at,
        "git_commit": git_commit,
        "checkpoint": "06A/06B",
        "hidden_fira_y_used_or_scored": False,
        "preflight": result["preflight"],
        "decision": result["decision"],
        "row_counts": {
            output_key: int(len(result[result_key]))
            for output_key, result_key in table_map.items()
        },
    }
    (output_dir / str(output_cfg["report_json"])).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / str(output_cfg["report_markdown"])).write_text(
        _report_markdown(
            result,
            generated_at=generated_at,
            git_commit=git_commit,
        ),
        encoding="utf-8",
    )

    print("Checkpoint 06A/06B diagnostics: PASS")
    print(
        f"TAIL_HYPOTHESIS = "
        f"{result['decision']['TAIL_HYPOTHESIS']}"
    )
    print(
        "Inspect: "
        f"{output_dir.relative_to(root) / str(output_cfg['report_markdown'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
