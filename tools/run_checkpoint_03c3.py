#!/usr/bin/env python
"""Run Checkpoint 03C.3 finalist equal-weight ensemble evaluation."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from geocebada.evaluation.checkpoint03c3 import (
    build_finalist_ensemble_predictions,
    summarize_finalist_ensemble_results,
)
from geocebada.paths import find_project_root

ROOT = Path(__file__).resolve().parents[1]


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping in {path}.")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join(["---"] * len(columns)) + "|",
    ]
    for row in frame.loc[:, columns].itertuples(index=False, name=None):
        values: list[str] = []
        for value in row:
            if isinstance(value, float):
                values.append("" if pd.isna(value) else f"{value:.4f}")
            else:
                values.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def _render_report(
    *,
    report: dict[str, Any],
    protocol_summary: pd.DataFrame,
    robustness: pd.DataFrame,
    residual_correlation: pd.DataFrame,
) -> str:
    lines = [
        "# Checkpoint 03C.3 — Finalist equal-weight ensembles",
        "",
        f"Generated: {report['generated_at']}",
        "",
        "## Scope",
        "",
        "- source: Checkpoint 03C.2 out-of-fold predictions",
        "- active track: **competition only**",
        "- hidden prediction targets used: **no**",
        "- final 59 predictions generated: **no**",
        "- continuous ensemble-weight tuning: **no**",
        f"- preserved finalists: **{', '.join(report['finalists'])}**",
        "",
        "The checkpoint compares only three reviewed individual finalists and four",
        "pre-specified equal-weight combinations. It does not refit models or optimize",
        "weights against these OOF residuals.",
        "",
        "## Cross-protocol robustness",
        "",
    ]
    lines.extend(
        _markdown_table(
            robustness,
            [
                "candidate",
                "candidate_type",
                "state_oof_rmse",
                "grouped_oof_rmse",
                "mean_oof_rmse",
                "worst_protocol_rmse",
                "protocol_gap",
            ],
        )
    )

    for protocol in report["protocols"]:
        lines.extend(["", f"## {protocol}", ""])
        subset = protocol_summary.loc[
            protocol_summary["protocol"].eq(protocol)
        ].sort_values(["oof_rmse", "candidate"])
        lines.extend(
            _markdown_table(
                subset,
                [
                    "candidate",
                    "candidate_type",
                    "oof_rmse",
                    "oof_mae",
                    "oof_r2",
                    "fold_rmse_mean",
                    "fold_rmse_std",
                ],
            )
        )

    lines.extend(
        [
            "",
            "## Residual correlation among individual finalists",
            "",
        ]
    )
    lines.extend(
        _markdown_table(
            residual_correlation,
            [
                "protocol",
                "candidate_a",
                "candidate_b",
                "residual_correlation",
            ],
        )
    )

    lines.extend(
        [
            "",
            "## Decision",
            "",
            "- E13_PLS_CatBoost and E123_equal_top3 are retained.",
            "- Their municipality-grouped RMSE values differ by less than 0.001.",
            "- The small difference is not treated as evidence for a unique winner.",
            "- ExtraTrees is not carried into this finalist ensemble checkpoint because",
            "  its municipality-grouped robustness was materially weaker in 03C.2.",
            "- The next phase should freeze the full-data fitting rule for the retained",
            "  finalist(s), then generate the 59 challenge predictions once.",
            "",
            "These are development estimates from reused frozen OOF folds, not an",
            "independent final-test estimate.",
            "",
        ]
    )
    return "\n".join(lines)


def run_checkpoint_03c3(*, root: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Run the deterministic 03C.3 ensemble review and write report artifacts."""

    input_path = root / str(config["inputs"]["oof_predictions"])
    output_dir = root / str(config["outputs"]["directory"])
    output_dir.mkdir(parents=True, exist_ok=True)

    oof = pd.read_csv(input_path)
    predictions = build_finalist_ensemble_predictions(oof, config)
    fold_metrics, protocol_summary, robustness, residual_correlation = (
        summarize_finalist_ensemble_results(predictions, config)
    )

    outputs = config["outputs"]
    predictions.to_csv(
        output_dir / str(outputs["candidate_oof_predictions"]),
        index=False,
    )
    fold_metrics.to_csv(
        output_dir / str(outputs["fold_metrics"]),
        index=False,
    )
    protocol_summary.to_csv(
        output_dir / str(outputs["protocol_summary"]),
        index=False,
    )
    robustness.to_csv(
        output_dir / str(outputs["robustness_summary"]),
        index=False,
    )
    residual_correlation.to_csv(
        output_dir / str(outputs["residual_correlation"]),
        index=False,
    )

    report = {
        "schema_version": int(config["schema_version"]),
        "generated_at": datetime.now(UTC).isoformat(),
        "stage": str(config["stage"]["name"]),
        "active_track": str(config["scope"]["active_track"]),
        "source_oof_sha256": _sha256(input_path),
        "training_rows": int(config["validation"]["expected_training_rows"]),
        "protocols": [
            str(config["validation"]["state_protocol"]),
            str(config["validation"]["grouped_protocol"]),
        ],
        "base_candidates": list(map(str, config["base_candidates"].keys())),
        "ensembles": list(map(str, config["ensembles"].keys())),
        "finalists": list(map(str, config["decision"]["finalists"])),
        "continuous_weight_tuning": False,
        "hidden_prediction_targets_used": False,
        "final_predictions_generated": False,
    }

    (output_dir / str(outputs["run_report_json"])).write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / str(outputs["run_report_markdown"])).write_text(
        _render_report(
            report=report,
            protocol_summary=protocol_summary,
            robustness=robustness,
            residual_correlation=residual_correlation,
        ),
        encoding="utf-8",
    )
    return report


def main() -> int:
    root = ROOT
    if not (root / "pyproject.toml").is_file():
        root = find_project_root(root)
    config = _load_yaml(root / "configs" / "checkpoint03c3.yaml")
    report = run_checkpoint_03c3(root=root, config=config)
    print("Checkpoint 03C.3 finalist ensemble review: PASS")
    print(f"Training rows: {report['training_rows']}")
    print(f"Candidates: {len(report['base_candidates']) + len(report['ensembles'])}")
    print(f"Finalists retained: {', '.join(report['finalists'])}")
    print("Inspect: reports/checkpoint_03c3/checkpoint_03c3_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
