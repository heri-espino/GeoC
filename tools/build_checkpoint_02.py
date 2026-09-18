#!/usr/bin/env python
"""Build Checkpoint 02 diagnostics, frozen CV folds and lightweight baselines."""

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
    build_feature_catalog,
    build_fixed_fold_assignments,
    evaluate_fixed_folds,
    initial_regressors,
    resolve_ablation_features,
    summarize_feature_inventory,
    summarize_fixed_fold_scores,
    train_prediction_diagnostics,
    validate_fixed_fold_assignments,
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


def _validate_expected(
    frame: pd.DataFrame,
    manifest: dict[str, Any],
    config: dict[str, Any],
) -> None:
    expected = config["expected"]
    identity = config["identity"]
    split_column = identity["split_column"]
    target_column = identity["target_column"]
    train_value = identity["train_value"]
    prediction_value = identity["prediction_value"]

    if len(frame) != int(expected["rows"]):
        raise ValueError(f"Expected {expected['rows']} rows, found {len(frame)}.")

    train = frame[split_column].eq(train_value)
    prediction = frame[split_column].eq(prediction_value)
    if int(train.sum()) != int(expected["train_rows"]):
        raise ValueError("Unexpected training row count.")
    if int(prediction.sum()) != int(expected["prediction_rows"]):
        raise ValueError("Unexpected prediction row count.")
    if frame.loc[train, target_column].isna().any():
        raise ValueError("Training targets contain missing values.")
    if frame.loc[prediction, target_column].notna().any():
        raise ValueError("Prediction rows unexpectedly expose hidden targets.")

    records = manifest["features"]
    total = len(records)
    clean = sum(str(record.get("mode")) == "clean" for record in records)
    competition = sum(str(record.get("mode")) == "competition" for record in records)
    if total != int(expected["total_features"]):
        raise ValueError(f"Expected {expected['total_features']} features, found {total}.")
    if clean != int(expected["clean_features"]):
        raise ValueError(f"Expected {expected['clean_features']} clean features, found {clean}.")
    if competition != int(expected["competition_only_features"]):
        raise ValueError(
            "Unexpected competition-only feature count: "
            f"expected {expected['competition_only_features']}, found {competition}."
        )


def _resolve_ablations(
    catalog: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[dict[str, list[str]], dict[str, dict[str, Any]]]:
    resolved: dict[str, list[str]] = {}
    manifest: dict[str, dict[str, Any]] = {}
    for name, spec in config["ablations"].items():
        features = resolve_ablation_features(catalog, spec)
        resolved[str(name)] = features
        manifest[str(name)] = {
            "description": str(spec.get("description", "")),
            "n_features": len(features),
            "features": features,
        }
    return resolved, manifest


def _markdown_table(
    frame: pd.DataFrame,
    columns: list[str],
    *,
    limit: int | None = None,
) -> list[str]:
    subset = frame[columns]
    if limit is not None:
        subset = subset.head(limit)
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join(["---"] * len(columns)) + "|",
    ]
    for _, row in subset.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append("" if pd.isna(value) else f"{value:.4f}")
            else:
                values.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def _render_report(
    report: dict[str, Any],
    inventory: pd.DataFrame,
    diagnostics: pd.DataFrame,
    ablation_manifest: dict[str, dict[str, Any]],
    baseline_summary: pd.DataFrame | None,
) -> str:
    flagged = diagnostics.loc[diagnostics["flag_any_shift"]].copy()
    lines = [
        "# Checkpoint 02 run report",
        "",
        f"Generated: {report['generated_at']}",
        f"Git commit: {report.get('git_commit') or 'unknown'}",
        f"Feature-table SHA256: {report['feature_table_sha256']}",
        "",
        "## Frozen feature table",
        "",
        f"- rows: **{report['rows']}**",
        f"- training parcels: **{report['train_rows']}**",
        f"- prediction parcels: **{report['prediction_rows']}**",
        f"- clean features: **{report['clean_features']}**",
        f"- competition-only features: **{report['competition_only_features']}**",
        f"- total features: **{report['total_features']}**",
        "",
        "No hidden prediction target is used anywhere in this checkpoint.",
        "",
        "## Feature inventory",
        "",
    ]
    lines.extend(
        _markdown_table(
            inventory,
            [
                "family",
                "mode",
                "source",
                "feature_count",
                "numeric_features",
                "non_numeric_features",
            ],
        )
    )

    lines.extend(
        [
            "",
            "## Train vs prediction diagnostics",
            "",
            (
                f"Numeric features screened: **{report['numeric_features_screened']}**; "
                f"features with at least one shift/support flag: "
                f"**{report['shift_flagged_features']}**."
            ),
            "",
            (
                "Flags are descriptive diagnostics only. They are not used to select "
                "features or tune models in the clean track."
            ),
            "",
        ]
    )
    if not flagged.empty:
        lines.extend(
            _markdown_table(
                flagged,
                [
                    "feature",
                    "family",
                    "mode",
                    "abs_smd",
                    "p_adjusted",
                    "missing_gap_abs",
                    "prediction_outside_train_range_fraction",
                ],
                limit=20,
            )
        )
    else:
        lines.append("No feature crossed the configured diagnostic thresholds.")

    lines.extend(
        [
            "",
            "## Frozen validation",
            "",
            (
                f"Primary protocol: {report['primary_protocol']}; robustness protocol: "
                f"{report['robustness_protocol']}; folds: **{report['n_splits']}**."
            ),
            "",
            (
                "The primary folds preserve state composition. The robustness folds keep "
                "municipalities together to expose spatial/generalization sensitivity."
            ),
            "",
            "## Ablations",
            "",
            "| ablation | n_features | rationale |",
            "|---|---:|---|",
        ]
    )
    for name, payload in ablation_manifest.items():
        description = str(payload["description"]).replace("|", "\\|")
        lines.append(f"| {name} | {payload['n_features']} | {description} |")

    lines.extend(
        [
            "",
            "## Initial models",
            "",
            (
                "Only untuned baselines are run here: DummyMean, fixed-alpha Ridge and "
                "regularized ExtraTrees. Median imputation is fitted inside each fold. "
                "PCA, supervised feature selection, Optuna, CatBoost and final hyperparameter "
                "search are intentionally deferred."
            ),
            "",
        ]
    )

    if baseline_summary is None:
        lines.append("Baseline evaluation was skipped for this run.")
    else:
        lines.extend(
            _markdown_table(
                baseline_summary,
                [
                    "protocol",
                    "ablation",
                    "model",
                    "n_features",
                    "rmse_mean",
                    "rmse_std",
                    "mae_mean",
                    "r2_mean",
                ],
            )
        )

    lines.extend(
        [
            "",
            "## Interpretation policy",
            "",
            "- Do not select a final model from this report alone.",
            "- Compare ablation deltas under both frozen protocols.",
            "- Treat large disagreement between protocols as evidence of spatial sensitivity.",
            "- Keep clean and competition tracks separate in all subsequent reporting.",
            "- Do not use prediction-set X diagnostics as a hidden-label substitute.",
            "",
        ]
    )
    return "\n".join(lines)


def build_checkpoint(
    *,
    root: Path,
    config: dict[str, Any],
    force_folds: bool,
    skip_models: bool,
) -> dict[str, Any]:
    inputs = config["inputs"]
    outputs = config["outputs"]
    identity = config["identity"]

    feature_directory = root / inputs["feature_directory"]
    table_path = feature_directory / inputs["feature_table"]
    manifest_path = feature_directory / inputs["feature_manifest"]
    build_report_path = feature_directory / inputs["build_report"]
    output_directory = root / outputs["directory"]
    output_directory.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(table_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    build_report = json.loads(build_report_path.read_text(encoding="utf-8"))
    _validate_expected(frame, manifest, config)

    catalog = build_feature_catalog(manifest, frame)
    inventory = summarize_feature_inventory(catalog)
    catalog.to_csv(output_directory / outputs["feature_catalog"], index=False)
    inventory.to_csv(output_directory / outputs["feature_inventory"], index=False)

    numeric_features = catalog.loc[catalog["numeric"], "feature"].tolist()
    shift_cfg = config["shift_diagnostics"]
    diagnostics = train_prediction_diagnostics(
        frame,
        features=numeric_features,
        split_column=identity["split_column"],
        train_value=identity["train_value"],
        prediction_value=identity["prediction_value"],
        smd_threshold=float(shift_cfg["smd_threshold"]),
        adjusted_p_threshold=float(shift_cfg["adjusted_p_threshold"]),
        missing_gap_threshold=float(shift_cfg["missing_gap_threshold"]),
        outside_support_threshold=float(shift_cfg["outside_support_threshold"]),
    )
    diagnostics = diagnostics.merge(
        catalog[["feature", "family", "mode", "source"]],
        on="feature",
        how="left",
        validate="1:1",
    )
    diagnostics.to_csv(
        output_directory / outputs["train_prediction_diagnostics"],
        index=False,
    )

    validation = config["validation"]
    folds_path = output_directory / outputs["cv_folds"]
    reused_folds = folds_path.exists() and not force_folds
    if reused_folds:
        folds = pd.read_csv(folds_path)
        validate_fixed_fold_assignments(
            frame,
            folds,
            id_column=identity["id_column"],
            split_column=identity["split_column"],
            train_value=identity["train_value"],
        )
    else:
        folds = build_fixed_fold_assignments(
            frame,
            n_splits=int(validation["n_splits"]),
            random_state=int(validation["random_state"]),
            id_column=identity["id_column"],
            split_column=identity["split_column"],
            train_value=identity["train_value"],
            state_column=identity["state_column"],
            municipality_column=identity["municipality_column"],
        )
        folds.to_csv(folds_path, index=False)

    ablations, ablation_manifest = _resolve_ablations(catalog, config)
    (output_directory / outputs["ablation_features"]).write_text(
        json.dumps(ablation_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    baseline_summary: pd.DataFrame | None = None
    if not skip_models:
        model_cfg = config["baseline_models"]
        models = initial_regressors(
            random_state=int(validation["random_state"]),
            ridge_alpha=float(model_cfg["ridge_alpha"]),
            extra_trees_estimators=int(model_cfg["extra_trees_estimators"]),
            extra_trees_min_samples_leaf=int(
                model_cfg["extra_trees_min_samples_leaf"]
            ),
        )
        score_frames = []
        for fold_column in [
            str(validation["primary_protocol"]),
            str(validation["robustness_protocol"]),
        ]:
            score_frames.append(
                evaluate_fixed_folds(
                    frame,
                    folds,
                    ablations,
                    fold_column=fold_column,
                    regressors=models,
                    id_column=identity["id_column"],
                    target_column=identity["target_column"],
                    split_column=identity["split_column"],
                    train_value=identity["train_value"],
                )
            )
        baseline_scores = pd.concat(score_frames, ignore_index=True)
        baseline_summary = summarize_fixed_fold_scores(baseline_scores)
        baseline_scores.to_csv(
            output_directory / outputs["baseline_fold_scores"],
            index=False,
        )
        baseline_summary.to_csv(
            output_directory / outputs["baseline_summary"],
            index=False,
        )

    train_mask = frame[identity["split_column"]].eq(identity["train_value"])
    prediction_mask = frame[identity["split_column"]].eq(
        identity["prediction_value"]
    )
    report = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(root),
        "feature_table_sha256": _sha256(table_path),
        "feature_build_generated_at": build_report.get("generated_at"),
        "rows": int(len(frame)),
        "train_rows": int(train_mask.sum()),
        "prediction_rows": int(prediction_mask.sum()),
        "clean_features": int(
            sum(record.get("mode") == "clean" for record in manifest["features"])
        ),
        "competition_only_features": int(
            sum(
                record.get("mode") == "competition"
                for record in manifest["features"]
            )
        ),
        "total_features": int(len(manifest["features"])),
        "numeric_features_screened": int(len(numeric_features)),
        "shift_flagged_features": int(diagnostics["flag_any_shift"].sum()),
        "n_splits": int(validation["n_splits"]),
        "primary_protocol": str(validation["primary_protocol"]),
        "robustness_protocol": str(validation["robustness_protocol"]),
        "folds_reused": bool(reused_folds),
        "models_skipped": bool(skip_models),
        "ablations": {
            name: int(len(features)) for name, features in ablations.items()
        },
    }
    (output_directory / outputs["run_report_json"]).write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown = _render_report(
        report,
        inventory,
        diagnostics,
        ablation_manifest,
        baseline_summary,
    )
    (output_directory / outputs["run_report_markdown"]).write_text(
        markdown,
        encoding="utf-8",
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/checkpoint02.yaml"),
    )
    parser.add_argument(
        "--force-folds",
        action="store_true",
        help="Regenerate frozen folds intentionally instead of reusing cv_folds.csv.",
    )
    parser.add_argument(
        "--skip-models",
        action="store_true",
        help="Build inventory/shift/folds/ablations without fitting baseline models.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    if not (root / "pyproject.toml").is_file():
        root = find_project_root(root)
    config_path = args.config
    if not config_path.is_absolute():
        config_path = root / config_path

    report = build_checkpoint(
        root=root,
        config=_load_yaml(config_path),
        force_folds=bool(args.force_folds),
        skip_models=bool(args.skip_models),
    )
    print("Checkpoint 02 pipeline: PASS")
    print(
        f"Rows: {report['rows']} | train={report['train_rows']} | "
        f"prediction={report['prediction_rows']}"
    )
    print(
        f"Features: clean={report['clean_features']} | "
        f"competition-only={report['competition_only_features']} | "
        f"total={report['total_features']}"
    )
    print(
        f"Shift flags: {report['shift_flagged_features']} / "
        f"{report['numeric_features_screened']} numeric features"
    )
    print("Output: reports/checkpoint_02")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
