#!/usr/bin/env python
"""Run Checkpoint 03C.1: frozen-fold representation benchmark.

This stage compares feature representations using the same fixed Ridge10 and ExtraTrees
baselines. It does not tune a final model and does not generate predictions for the 59 hidden
targets.
"""

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
    build_representation_specs,
    evaluate_representation_benchmark,
    summarize_representation_benchmark,
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
    report: dict[str, Any],
    summary: pd.DataFrame,
    representation_manifest: list[dict[str, Any]],
) -> str:
    counts = pd.DataFrame(
        [
            {
                "track": item["track"],
                "representation": item["name"],
                "n_features": item["n_features"],
                "discovery": item["discovery"],
                "n_primitives": item["n_primitives"],
            }
            for item in representation_manifest
        ]
    ).sort_values(["track", "representation"])

    lines = [
        "# Checkpoint 03C.1 — Representation benchmark",
        "",
        f"Generated: {report['generated_at']}",
        f"Git commit: {report.get('git_commit') or 'unknown'}",
        "",
        "## Contract",
        "",
        f"- training parcels: **{report['training_rows']}**",
        "- prediction parcels scored: **0**",
        f"- tracks: **{', '.join(report['tracks'])}**",
        f"- representations per track: **{report['representations_per_track']}**",
        f"- fixed model families: **{', '.join(report['models'])}**",
        f"- frozen protocols: **{', '.join(report['protocols'])}**",
        "",
        (
            "This stage isolates representation effects. Ridge10 and ExtraTrees use the same "
            "fixed settings across representations; there is no broad hyperparameter tuning."
        ),
        "",
        "CHIRPS is explicitly excluded pending QC.",
        "",
        "## Representation sizes",
        "",
    ]
    lines.extend(
        _markdown_table(
            counts,
            ["track", "representation", "n_features", "discovery", "n_primitives"],
        )
    )

    lines.extend(
        [
            "",
            "## OOF results",
            "",
            (
                "OOF metrics pool all 138 held-out predictions and therefore weight parcels "
                "rather than folds. Mean-fold metrics are also retained because the grouped "
                "folds are intentionally unequal in size."
            ),
            "",
            (
                "A negative delta_oof_rmse_vs_B0 means the representation improved over "
                "Feature Table v1 for the same track/model/protocol."
            ),
            "",
        ]
    )

    for protocol in report["protocols"]:
        for track in report["tracks"]:
            lines.extend([f"### {protocol} — {track}", ""])
            subset = summary.loc[
                summary["protocol"].eq(protocol) & summary["track"].eq(track)
            ].copy()
            subset = subset.sort_values(["model", "oof_rmse", "representation"])
            lines.extend(
                _markdown_table(
                    subset,
                    [
                        "model",
                        "representation",
                        "n_features",
                        "n_discovered",
                        "oof_rmse",
                        "oof_mae",
                        "oof_r2",
                        "fold_rmse_mean",
                        "fold_rmse_std",
                        "delta_oof_rmse_vs_B0",
                        "delta_oof_rmse_vs_B5",
                    ],
                )
            )
            lines.append("")

    lines.extend(
        [
            "## Interpretation rules",
            "",
            "- This is not final model selection.",
            "- Compare the same model across representations before comparing model families.",
            "- Treat municipality-grouped results as a first-class robustness signal.",
            "- B7 discovers expressions inside each outer training fold only.",
            "- Do not promote a discovered expression because its training association is large.",
            "- Use these results to choose a small representation set for 03C.2.",
            "- 03C.2 may then tune CatBoost/ElasticNet/kernels/PCA/PLS inside folds.",
            "",
        ]
    )
    return "\n".join(lines)


def run_checkpoint_03c(
    *,
    root: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Run the frozen-fold 03C.1 representation benchmark."""

    inputs = config["inputs"]
    outputs = config["outputs"]
    identity = config["identity"]

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

    representations = build_representation_specs(
        joined=joined,
        manifests=manifests,
        config=config,
        empirical_config=empirical_config,
    )

    output_dir = root / str(outputs["directory"])
    output_dir.mkdir(parents=True, exist_ok=True)

    representation_manifest = [
        spec.as_dict()
        for _, spec in sorted(
            representations.items(),
            key=lambda item: (item[0][0], item[0][1]),
        )
    ]
    (output_dir / str(outputs["representation_manifest"])).write_text(
        json.dumps(
            {
                "schema_version": int(config["schema_version"]),
                "generated_at": datetime.now(UTC).isoformat(),
                "representations": representation_manifest,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    validation = config["validation"]
    fold_scores, oof, discovery = evaluate_representation_benchmark(
        joined=joined,
        folds=folds,
        representations=representations,
        model_configs=config["models"],
        empirical_config=empirical_config,
        protocols=tuple(map(str, validation["protocols"])),
        random_state=int(validation["random_state"]),
        id_column=str(identity["id_column"]),
        target_column=str(identity["target_column"]),
        split_column=str(identity["split_column"]),
        train_value=str(identity["train_value"]),
    )
    summary = summarize_representation_benchmark(fold_scores, oof)

    fold_scores.to_csv(output_dir / str(outputs["fold_metrics"]), index=False)
    summary.to_csv(output_dir / str(outputs["summary_metrics"]), index=False)
    oof.to_csv(output_dir / str(outputs["oof_predictions"]), index=False)
    discovery.to_csv(output_dir / str(outputs["discovery_by_fold"]), index=False)

    train_mask = joined[str(identity["split_column"])].eq(
        str(identity["train_value"])
    )
    report = {
        "schema_version": int(config["schema_version"]),
        "generated_at": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(root),
        "stage": str(config["stage"]["name"]),
        "training_rows": int(train_mask.sum()),
        "total_rows": int(len(joined)),
        "tracks": list(map(str, config["tracks"])),
        "protocols": list(map(str, validation["protocols"])),
        "models": list(map(str, config["models"].keys())),
        "representations_per_track": int(len(config["representations"])),
        "experiment_combinations": int(
            len(config["tracks"])
            * len(config["representations"])
            * len(config["models"])
            * len(validation["protocols"])
        ),
        "outer_fits": int(len(fold_scores)),
        "oof_prediction_rows": int(len(oof)),
        "discovery_rows": int(len(discovery)),
        "hidden_prediction_targets_used": False,
        "prediction_rows_scored": 0,
        "final_model_selected": False,
        "chirps_excluded": (
            "external_chirps_daily"
            in set(map(str, config["feature_policy"]["exclude_sources"]))
        ),
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
        _render_report(report, summary, representation_manifest),
        encoding="utf-8",
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/checkpoint03c.yaml"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    if not (root / "pyproject.toml").is_file():
        root = find_project_root(root)

    config_path = args.config if args.config.is_absolute() else root / args.config
    report = run_checkpoint_03c(root=root, config=_load_yaml(config_path))

    print("Checkpoint 03C.1 representation benchmark: PASS")
    print(f"Training rows: {report['training_rows']}")
    print(f"Experiment combinations: {report['experiment_combinations']}")
    print(f"Outer fits: {report['outer_fits']}")
    print(f"OOF prediction rows: {report['oof_prediction_rows']}")
    print(f"Discovery rows: {report['discovery_rows']}")
    print("Inspect: reports/checkpoint_03c/checkpoint_03c_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
