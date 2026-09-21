#!/usr/bin/env python
"""Freeze the final 59-value reconstruction for Checkpoint 04F."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from geocebada.evaluation import (
    build_final_prediction_diagnostics,
    build_final_prediction_table,
    build_local_graph_blends,
    leave_one_split_out_blend_selection,
    regression_metrics,
    summarize_blend_predictions,
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


def _summarize_loso(predictions: pd.DataFrame) -> dict[str, float | int]:
    split_rmses: list[float] = []
    for _, group in predictions.groupby("split_id", sort=False):
        split_rmses.append(
            regression_metrics(group["observed"], group["predicted"])["rmse"]
        )
    pooled = regression_metrics(predictions["observed"], predictions["predicted"])
    return {
        "n_splits": int(predictions["split_id"].nunique()),
        "n_predictions": int(len(predictions)),
        "rmse_mean": float(np.mean(split_rmses)),
        "rmse_median": float(np.median(split_rmses)),
        "rmse_worst": float(np.max(split_rmses)),
        "pooled_rmse": pooled["rmse"],
        "pooled_mae": pooled["mae"],
        "pooled_r2": pooled["r2"],
    }


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join(["---"] * len(columns)) + "|",
    ]
    for row in frame.loc[:, columns].itertuples(index=False, name=None):
        rendered: list[str] = []
        for value in row:
            if isinstance(value, float):
                rendered.append("" if pd.isna(value) else f"{value:.6f}")
            else:
                rendered.append(str(value))
        lines.append("| " + " | ".join(rendered) + " |")
    return lines


def _render_report(
    *,
    report: dict[str, Any],
    blend_summary: pd.DataFrame,
    loso_selection: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> str:
    lines = [
        "# Checkpoint 04F — Final transductive reconstruction",
        "",
        f"Generated: {report['generated_at']}",
        f"Git commit: {report.get('git_commit') or 'unknown'}",
        "",
        "## Frozen final rule",
        "",
        f"- selected method: **{report['selected_method']}**",
        f"- final rows: **{report['prediction_rows']}**",
        "- hidden 59 yields used or scored: **no**",
        (
            f"- frozen target-matched mean RMSE: "
            f"**{report['selected_target_matched_rmse_mean']:.6f}**"
        ),
        "",
        (
            "The final rule is intentionally simpler than the full development suite. "
            "SIAP and CatBoost did not add reproducible target-matched improvement, "
            "and support-tier routing was not promoted after constrained checks."
        ),
        "",
        "## Local/Graph sensitivity",
        "",
    ]
    lines.extend(
        _markdown_table(
            blend_summary,
            [
                "method",
                "graph_weight",
                "rmse_mean",
                "pooled_rmse",
                "pooled_mae",
                "rmse_worst",
            ],
        )
    )
    lines.extend(
        [
            "",
            (
                "The Local/Graph blend table is a development sensitivity diagnostic, "
                "not an automatic final selector."
            ),
            "",
            "## Constrained LOSO blend-weight selection",
            "",
            (
                f"Mean holdout RMSE: **{report['loso_blend']['rmse_mean']:.6f}**; "
                f"pooled RMSE: **{report['loso_blend']['pooled_rmse']:.6f}**."
            ),
            "",
            "| selected method | holdouts |",
            "|---|---:|",
        ]
    )
    for method, count in (
        loso_selection["selected_method"].value_counts().sort_index().items()
    ):
        lines.append(f"| {method} | {int(count)} |")

    disagreement = diagnostics["local_graph_abs_disagreement"].astype(float)
    lines.extend(
        [
            "",
            "## Actual-target disagreement diagnostics",
            "",
            (
                f"- mean |Local-Graph| disagreement: **{disagreement.mean():.4f} t/ha**"
            ),
            (
                f"- median |Local-Graph| disagreement: "
                f"**{disagreement.median():.4f} t/ha**"
            ),
            (
                f"- maximum |Local-Graph| disagreement: "
                f"**{disagreement.max():.4f} t/ha**"
            ),
            "",
            (
                "These diagnostics do not modify any parcel prediction. They are retained "
                "for uncertainty interpretation and final reporting."
            ),
            "",
            "## Output",
            "",
            (
                "final_predictions.csv contains exactly two columns: "
                "ID_POLIGONO, RENDIMIENTO_T_HA."
            ),
            "",
            (
                "final_prediction_diagnostics.csv contains the frozen Local04D value, "
                "Graph04D reference, X-support information and disagreement diagnostics."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def run_checkpoint_04f(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Freeze and write the final 59-value prediction table."""

    inputs = config["inputs"]
    outputs = config["outputs"]
    identity = config["identity"]
    final_rule = config["final_rule"]
    validation = config["validation"]

    print("[04F 1/6] Loading frozen candidate and validation artifacts...")
    parcels = pd.read_csv(root / str(inputs["parcel_table"]))
    pseudo = pd.read_csv(root / str(inputs["pseudo_predictions"]))
    actual_04d = pd.read_csv(root / str(inputs["actual_candidates_04d"]))
    actual_04e = pd.read_csv(root / str(inputs["actual_candidates_04e"]))
    protocol_04c = pd.read_csv(root / str(inputs["protocol_summary_04c"]))

    id_column = str(identity["id_column"])
    target_column = str(identity["target_column"])
    split_column = str(identity["split_column"])
    train_value = str(identity["training_value"])
    prediction_value = str(identity["prediction_value"])

    counts = (
        len(parcels),
        int(parcels[split_column].eq(train_value).sum()),
        int(parcels[split_column].eq(prediction_value).sum()),
    )
    expected = (
        int(validation["expected_total_rows"]),
        int(validation["expected_training_rows"]),
        int(validation["expected_prediction_rows"]),
    )
    if counts != expected:
        raise ValueError(f"Unexpected parcel counts: {counts}; expected {expected}.")

    hidden = pd.to_numeric(
        parcels.loc[parcels[split_column].eq(prediction_value), target_column],
        errors="coerce",
    )
    if hidden.notna().any():
        raise ValueError("Prediction rows unexpectedly contain hidden target values.")

    print("[04F 2/6] Verifying frozen Local04D provenance across 04D and 04E...")
    selected_method = str(final_rule["selected_method"])
    source_method = str(final_rule["source_method_04d"])
    secondary_method = str(final_rule["secondary_method"])
    secondary_source = str(final_rule["secondary_source_method_04d"])

    selected_04d = actual_04d.loc[
        actual_04d["method"].eq(source_method),
        [id_column, "predicted"],
    ].copy()
    selected_04e = actual_04e.loc[
        actual_04e["method"].eq(selected_method),
        [id_column, "predicted"],
    ].copy()
    selected_04d[id_column] = selected_04d[id_column].astype(str)
    selected_04e[id_column] = selected_04e[id_column].astype(str)
    merged_selected = selected_04d.merge(
        selected_04e,
        on=id_column,
        how="inner",
        validate="one_to_one",
        suffixes=("_04d", "_04e"),
    )
    if len(merged_selected) != int(validation["expected_prediction_rows"]):
        raise ValueError("04D/04E final Local04D candidates do not cover all 59 targets.")
    max_selected_diff = float(
        np.max(
            np.abs(
                merged_selected["predicted_04d"].to_numpy(dtype=float)
                - merged_selected["predicted_04e"].to_numpy(dtype=float)
            )
        )
    )
    if max_selected_diff > float(validation["actual_prediction_tolerance"]):
        raise ValueError(
            "Frozen Local04D actual-target predictions disagree across 04D and 04E."
        )

    graph_04d = actual_04d.loc[
        actual_04d["method"].eq(secondary_source),
        [id_column, "predicted"],
    ].copy()
    graph_04e = actual_04e.loc[
        actual_04e["method"].eq(secondary_method),
        [id_column, "predicted"],
    ].copy()
    graph_04d[id_column] = graph_04d[id_column].astype(str)
    graph_04e[id_column] = graph_04e[id_column].astype(str)
    merged_graph = graph_04d.merge(
        graph_04e,
        on=id_column,
        how="inner",
        validate="one_to_one",
        suffixes=("_04d", "_04e"),
    )
    if len(merged_graph) != int(validation["expected_prediction_rows"]):
        raise ValueError("04D/04E final Graph04D candidates do not cover all 59 targets.")
    max_graph_diff = float(
        np.max(
            np.abs(
                merged_graph["predicted_04d"].to_numpy(dtype=float)
                - merged_graph["predicted_04e"].to_numpy(dtype=float)
            )
        )
    )
    if max_graph_diff > float(validation["actual_prediction_tolerance"]):
        raise ValueError(
            "Frozen Graph04D actual-target predictions disagree across 04D and 04E."
        )

    print("[04F 3/6] Reconstructing constrained Local/Graph blend diagnostics...")
    family = str(config["blend_diagnostic"]["family"])
    graph_weights = [
        float(value) for value in config["blend_diagnostic"]["graph_weights"]
    ]
    blends = build_local_graph_blends(
        pseudo,
        family=family,
        graph_weights=graph_weights,
        local_method=selected_method,
        graph_method=secondary_method,
    )
    blend_summary = summarize_blend_predictions(blends)
    primary_splits = int(blends["split_id"].nunique())
    if primary_splits != int(validation["expected_primary_splits"]):
        raise ValueError(
            f"Expected {validation['expected_primary_splits']} primary splits, "
            f"found {primary_splits}."
        )

    print("[04F 4/6] Running constrained leave-one-split-out blend check...")
    loso_selection, loso_predictions = leave_one_split_out_blend_selection(blends)
    loso_summary = _summarize_loso(loso_predictions)

    frozen_metric = protocol_04c.loc[
        protocol_04c["family"].eq(family)
        & protocol_04c["method"].eq(selected_method),
        "rmse_mean",
    ]
    if len(frozen_metric) != 1:
        raise ValueError("Could not resolve one frozen Local04D target-matched RMSE.")
    selected_rmse = float(frozen_metric.iloc[0])
    expected_rmse = float(final_rule["expected_target_matched_rmse_mean"])
    if abs(selected_rmse - expected_rmse) > float(
        validation["metric_tolerance"]
    ):
        raise ValueError(
            f"Frozen Local04D RMSE changed: {selected_rmse} vs {expected_rmse}."
        )

    print("[04F 5/6] Freezing exactly 59 final predictions and diagnostics...")
    final_predictions = build_final_prediction_table(
        actual_04e,
        selected_method=selected_method,
        target_column=target_column,
        expected_rows=int(validation["expected_prediction_rows"]),
    )
    target_ids = set(
        parcels.loc[
            parcels[split_column].eq(prediction_value),
            id_column,
        ].astype(str)
    )
    if set(final_predictions[id_column]) != target_ids:
        raise ValueError("Final prediction IDs do not match the official 59 target IDs.")

    diagnostics = build_final_prediction_diagnostics(
        actual_04e,
        selected_method=selected_method,
        secondary_method=secondary_method,
        diagnostic_graph_weight=float(
            config["blend_diagnostic"]["diagnostic_weight"]
        ),
    )

    print("[04F 6/6] Writing final reconstruction artifacts...")
    output_dir = root / str(outputs["directory"])
    output_dir.mkdir(parents=True, exist_ok=True)
    blend_summary.to_csv(
        output_dir / str(outputs["blend_validation_summary"]),
        index=False,
    )
    loso_selection.to_csv(
        output_dir / str(outputs["loso_blend_selection"]),
        index=False,
    )
    loso_predictions.to_csv(
        output_dir / str(outputs["loso_blend_predictions"]),
        index=False,
    )
    final_predictions.to_csv(
        output_dir / str(outputs["final_predictions"]),
        index=False,
    )
    diagnostics.to_csv(
        output_dir / str(outputs["final_prediction_diagnostics"]),
        index=False,
    )

    disagreement = diagnostics["local_graph_abs_disagreement"].astype(float)
    report = {
        "schema_version": int(config["schema_version"]),
        "generated_at": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(root),
        "stage": str(config["stage"]["name"]),
        "selected_method": selected_method,
        "selected_source_method_04d": source_method,
        "secondary_method": secondary_method,
        "prediction_rows": int(len(final_predictions)),
        "hidden_prediction_targets_used": False,
        "actual_hidden_y_scored": False,
        "selected_target_matched_rmse_mean": selected_rmse,
        "local04d_cross_checkpoint_max_abs_diff": max_selected_diff,
        "graph04d_cross_checkpoint_max_abs_diff": max_graph_diff,
        "blend_candidate_count": int(blend_summary["method"].nunique()),
        "loso_blend": loso_summary,
        "actual_disagreement": {
            "mean_abs_local_graph": float(disagreement.mean()),
            "median_abs_local_graph": float(disagreement.median()),
            "max_abs_local_graph": float(disagreement.max()),
        },
        "final_columns": list(final_predictions.columns),
    }
    (output_dir / str(outputs["report_json"])).write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / str(outputs["report_markdown"])).write_text(
        _render_report(
            report=report,
            blend_summary=blend_summary,
            loso_selection=loso_selection,
            diagnostics=diagnostics,
        ),
        encoding="utf-8",
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/checkpoint04f.yaml"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    if not (root / "pyproject.toml").is_file():
        root = find_project_root(root)

    config_path = args.config if args.config.is_absolute() else root / args.config
    report = run_checkpoint_04f(root, _load_yaml(config_path))

    print("Checkpoint 04F final transductive reconstruction: PASS")
    print(f"Selected method: {report['selected_method']}")
    print(f"Final predictions: {report['prediction_rows']}")
    print(
        "Target-matched RMSE: "
        f"{report['selected_target_matched_rmse_mean']:.6f}"
    )
    print(
        "LOSO blend-check RMSE: "
        f"{report['loso_blend']['rmse_mean']:.6f}"
    )
    print("Inspect: reports/checkpoint_04f/checkpoint_04f_report.md")
    print("Final table: reports/checkpoint_04f/final_predictions.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
