#!/usr/bin/env python
"""Run Checkpoint 03C.2 competition-only nested model-family evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from geocebada.evaluation import (
    build_competition_representation_specs,
    evaluate_nested_model_families,
    summarize_nested_results,
)
from geocebada.features import join_agronomic_features, join_empirical_features
from geocebada.paths import find_project_root

ROOT = Path(__file__).resolve().parents[1]


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping in {path}.")
    return payload


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}.")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _parse_csv_list(value: str | None) -> list[str] | None:
    if value is None:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or None


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
    representation_manifest: list[dict[str, Any]],
) -> str:
    representation_rows = pd.DataFrame(
        [
            {
                "representation": item["name"],
                "n_features": item["n_features"],
                "n_primitives": item["n_primitives"],
                "discovery": item["discovery"],
            }
            for item in representation_manifest
        ]
    )

    lines = [
        "# Checkpoint 03C.2 — Competition-only nested model benchmark",
        "",
        f"Generated: {report['generated_at']}",
        f"Git commit: {report.get('git_commit') or 'unknown'}",
        "",
        "## Scope",
        "",
        "- active feature track: **competition only**",
        "- clean track evaluated: **no**",
        f"- training parcels: **{report['training_rows']}**",
        "- prediction parcels scored: **0**",
        f"- outer protocols: **{', '.join(report['protocols'])}**",
        f"- inner folds: **{report['inner_folds']}**",
        f"- representations: **{', '.join(report['representations'])}**",
        f"- model families: **{', '.join(report['models'])}**",
        f"- CatBoost execution: **{report['catboost_task_type_final']}**",
        "",
        (
            "Hyperparameters are selected only from each outer training split using inner CV. "
            "The frozen Checkpoint 02 folds remain the outer evaluation."
        ),
        "",
        "CHIRPS remains excluded pending QC.",
        "",
        "## Representation sizes",
        "",
    ]
    lines.extend(
        _markdown_table(
            representation_rows,
            ["representation", "n_features", "n_primitives", "discovery"],
        )
    )

    lines.extend(
        [
            "",
            "## Cross-protocol robustness ranking",
            "",
            (
                "The table is sorted by worst-protocol OOF RMSE, then mean OOF RMSE. "
                "This is development evidence, not a fresh independent test after 03C.1."
            ),
            "",
        ]
    )
    top = robustness.head(30).copy()
    lines.extend(
        _markdown_table(
            top,
            [
                "representation",
                "model",
                "state_oof_rmse",
                "grouped_oof_rmse",
                "mean_oof_rmse",
                "worst_protocol_rmse",
                "protocol_gap",
            ],
        )
    )

    for protocol in report["protocols"]:
        lines.extend([f"", f"## {protocol}", ""])
        subset = protocol_summary.loc[
            protocol_summary["protocol"].eq(protocol)
        ].sort_values(["oof_rmse", "representation", "model"])
        lines.extend(
            _markdown_table(
                subset,
                [
                    "representation",
                    "model",
                    "n_features",
                    "n_discovered",
                    "oof_rmse",
                    "oof_mae",
                    "oof_r2",
                    "fold_rmse_mean",
                    "fold_rmse_std",
                    "inner_best_rmse_mean",
                    "elapsed_seconds",
                ],
            )
        )

    lines.extend(
        [
            "",
            "## Interpretation rules",
            "",
            "- Do not compare or revive the clean track in 03C.2.",
            "- Do not use the 59 hidden targets for selection.",
            "- B/C discovery remains fitted inside inner/outer training folds only.",
            "- PCA and PLS are fitted inside the nested pipeline.",
            "- Prefer models that remain competitive under municipality-grouped CV.",
            "- Do not treat the minimum development score as an unbiased final error estimate.",
            "- Final challenge prediction is a later frozen-model step.",
            "",
        ]
    )
    return "\n".join(lines)


def run_checkpoint_03c2(
    *,
    root: Path,
    config: dict[str, Any],
    selected_models: list[str] | None = None,
    selected_representations: list[str] | None = None,
    catboost_task_type: str | None = None,
) -> dict[str, Any]:
    """Run Checkpoint 03C.2 and write reproducible report artifacts."""

    inputs = config["inputs"]
    outputs = config["outputs"]
    identity = config["identity"]

    if str(config["scope"]["active_track"]) != "competition":
        raise ValueError("03C.2 configuration must be competition-only.")
    if bool(config["scope"].get("clean_track_enabled", False)):
        raise ValueError("Clean track must be disabled in 03C.2.")

    base_dir = root / str(inputs["base_directory"])
    agronomic_dir = root / str(inputs["agronomic_directory"])
    empirical_dir = root / str(inputs["empirical_directory"])

    base_table_path = base_dir / str(inputs["base_table"])
    agronomic_table_path = agronomic_dir / str(inputs["agronomic_table"])
    empirical_table_path = empirical_dir / str(inputs["empirical_table"])
    base_manifest_path = base_dir / str(inputs["base_manifest"])
    agronomic_manifest_path = agronomic_dir / str(inputs["agronomic_manifest"])
    empirical_manifest_path = empirical_dir / str(inputs["empirical_manifest"])
    empirical_config_path = root / str(inputs["empirical_config"])
    folds_path = root / str(inputs["frozen_folds"])

    base = pd.read_csv(base_table_path)
    agronomic = pd.read_csv(agronomic_table_path)
    empirical = pd.read_csv(empirical_table_path)
    folds = pd.read_csv(folds_path)

    joined = join_agronomic_features(base, agronomic)
    joined = join_empirical_features(joined, empirical)

    manifests = {
        "base": _load_json(base_manifest_path),
        "agronomic": _load_json(agronomic_manifest_path),
        "empirical": _load_json(empirical_manifest_path),
    }
    empirical_config = _load_yaml(empirical_config_path)

    representations = build_competition_representation_specs(
        joined=joined,
        manifests=manifests,
        config=config,
        empirical_config=empirical_config,
    )

    chosen_representations = (
        selected_representations
        if selected_representations
        else list(representations)
    )
    chosen_models = (
        selected_models
        if selected_models
        else list(config["models"])
    )

    output_dir = root / str(outputs["directory"])
    output_dir.mkdir(parents=True, exist_ok=True)

    representation_manifest = [
        representations[name].as_dict()
        for name in chosen_representations
    ]
    (output_dir / str(outputs["representation_manifest"])).write_text(
        json.dumps(
            {
                "schema_version": int(config["schema_version"]),
                "generated_at": datetime.now(UTC).isoformat(),
                "active_track": "competition",
                "representations": representation_manifest,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    outer, oof, inner, discovery, actual_catboost_task = (
        evaluate_nested_model_families(
            joined=joined,
            folds=folds,
            representations=representations,
            config=config,
            empirical_config=empirical_config,
            selected_models=chosen_models,
            selected_representations=chosen_representations,
            catboost_task_type_override=catboost_task_type,
        )
    )
    protocol_summary, robustness = summarize_nested_results(outer, oof)

    outer.to_csv(
        output_dir / str(outputs["outer_fold_metrics"]),
        index=False,
    )
    oof.to_csv(
        output_dir / str(outputs["oof_predictions"]),
        index=False,
    )
    inner.to_csv(
        output_dir / str(outputs["inner_search_results"]),
        index=False,
    )
    discovery.to_csv(
        output_dir / str(outputs["discovery_by_outer_fold"]),
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

    train_mask = joined[str(identity["split_column"])].eq(
        str(identity["train_value"])
    )
    validation = config["validation"]
    report = {
        "schema_version": int(config["schema_version"]),
        "generated_at": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(root),
        "stage": str(config["stage"]["name"]),
        "active_track": "competition",
        "clean_track_evaluated": False,
        "training_rows": int(train_mask.sum()),
        "total_rows": int(len(joined)),
        "prediction_rows_scored": 0,
        "hidden_prediction_targets_used": False,
        "protocols": list(map(str, validation["outer_protocols"])),
        "inner_folds": int(validation["inner_folds"]),
        "representations": chosen_representations,
        "models": chosen_models,
        "outer_fits": int(len(outer)),
        "oof_prediction_rows": int(len(oof)),
        "inner_candidate_rows": int(len(inner)),
        "discovery_rows": int(len(discovery)),
        "catboost_task_type_final": actual_catboost_task,
        "chirps_excluded": (
            "external_chirps_daily"
            in set(map(str, config["feature_policy"]["exclude_sources"]))
        ),
        "final_model_selected": False,
        "final_predictions_generated": False,
        "source_sha256": {
            "base_table": _sha256(base_table_path),
            "agronomic_table": _sha256(agronomic_table_path),
            "empirical_table": _sha256(empirical_table_path),
            "base_manifest": _sha256(base_manifest_path),
            "agronomic_manifest": _sha256(agronomic_manifest_path),
            "empirical_manifest": _sha256(empirical_manifest_path),
            "frozen_folds": _sha256(folds_path),
        },
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
            representation_manifest=representation_manifest,
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
        default=Path("configs/checkpoint03c2.yaml"),
    )
    parser.add_argument(
        "--models",
        help="Comma-separated subset, e.g. Ridge,ExtraTrees,CatBoost",
    )
    parser.add_argument(
        "--representations",
        help="Comma-separated subset, e.g. C0_base,C1_agronomic",
    )
    parser.add_argument(
        "--catboost-task-type",
        choices=("GPU", "CPU"),
        help="Override CatBoost task type. Config defaults to GPU.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    if not (root / "pyproject.toml").is_file():
        root = find_project_root(root)

    config_path = args.config if args.config.is_absolute() else root / args.config
    report = run_checkpoint_03c2(
        root=root,
        config=_load_yaml(config_path),
        selected_models=_parse_csv_list(args.models),
        selected_representations=_parse_csv_list(args.representations),
        catboost_task_type=args.catboost_task_type,
    )

    print("Checkpoint 03C.2 competition-only benchmark: PASS")
    print(f"Training rows: {report['training_rows']}")
    print(f"Representations: {len(report['representations'])}")
    print(f"Models: {len(report['models'])}")
    print(f"Outer fits: {report['outer_fits']}")
    print(f"OOF prediction rows: {report['oof_prediction_rows']}")
    print(f"Inner candidate rows: {report['inner_candidate_rows']}")
    print(f"CatBoost task type: {report['catboost_task_type_final']}")
    print("Inspect: reports/checkpoint_03c2/checkpoint_03c2_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
