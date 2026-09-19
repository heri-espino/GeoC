#!/usr/bin/env python
"""Run the Checkpoint 03B fold-local expression-stability audit.

This is a discovery audit, not a model benchmark. Each miner is fitted only on the training
portion of one frozen Checkpoint 02 fold. Validation targets are never read for ranking or
reporting. Aggregate stability describes recurrence under training-set perturbations; it is
not an unbiased estimate of predictive performance.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

from geocebada.features import (
    FoldLocalExpressionMiner,
    join_agronomic_features,
    join_empirical_features,
)
from geocebada.paths import find_project_root

ROOT = Path(__file__).resolve().parents[1]
PROTOCOLS = ("fold_state_stratified", "fold_municipality_grouped")


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping in {path}.")
    return payload


def _signed_spearman(values: np.ndarray, target: np.ndarray) -> float:
    finite = np.isfinite(values) & np.isfinite(target)
    if finite.sum() < 3:
        return float("nan")
    x = values[finite]
    y = target[finite]
    if np.std(x) <= 0 or np.std(y) <= 0:
        return float("nan")
    statistic = spearmanr(x, y).statistic
    return float(statistic) if np.isfinite(statistic) else float("nan")


def _load_training_table(
    root: Path,
    config: dict[str, Any],
) -> pd.DataFrame:
    inputs = config["inputs"]
    identity = config["identity"]
    id_column = str(identity["id_column"])
    target_column = str(identity["target_column"])
    split_column = str(identity["split_column"])

    feature_dir = root / str(inputs["feature_directory"])
    agronomic_dir = root / str(inputs["agronomic_directory"])
    empirical_dir = root / str(config["outputs"]["directory"])

    base = pd.read_csv(feature_dir / str(inputs["feature_table"]))
    agronomic = pd.read_csv(agronomic_dir / str(inputs["agronomic_table"]))
    empirical = pd.read_csv(empirical_dir / str(config["outputs"]["feature_csv"]))

    joined = join_agronomic_features(base, agronomic)
    joined = join_empirical_features(joined, empirical)

    training = joined.loc[
        joined[split_column].eq("ENTRENAMIENTO")
        & pd.to_numeric(joined[target_column], errors="coerce").notna()
    ].copy()
    training[target_column] = pd.to_numeric(
        training[target_column],
        errors="raise",
    ).astype(float)

    folds = pd.read_csv(root / "reports" / "checkpoint_02" / "cv_folds.csv")
    required = [id_column, *PROTOCOLS]
    missing = [column for column in required if column not in folds.columns]
    if missing:
        raise KeyError(f"Frozen fold file missing columns: {missing}")
    if folds[id_column].duplicated().any():
        raise ValueError("Frozen fold file contains duplicate parcel IDs.")

    training[id_column] = training[id_column].astype(str)
    fold_columns = folds[required].copy()
    fold_columns[id_column] = fold_columns[id_column].astype(str)
    training_ids = set(training[id_column])
    fold_ids = set(fold_columns[id_column])
    if training_ids != fold_ids:
        raise ValueError(
            "Frozen fold IDs do not match current training IDs; "
            f"missing={sorted(training_ids-fold_ids)[:5]}, "
            f"extra={sorted(fold_ids-training_ids)[:5]}"
        )

    return training.merge(
        fold_columns,
        on=id_column,
        how="left",
        validate="one_to_one",
    )


def run_expression_discovery(
    *,
    root: Path,
    config: dict[str, Any],
    output_directory: Path,
) -> dict[str, Any]:
    """Run training-fold-only expression discovery on both frozen protocols."""

    output_directory.mkdir(parents=True, exist_ok=True)
    target_column = str(config["identity"]["target_column"])
    discovery = config["supervised_discovery"]

    training = _load_training_table(root, config)
    primitive_columns = tuple(str(x) for x in discovery["primitive_columns"])
    missing = [
        column for column in primitive_columns if column not in training.columns
    ]
    if missing:
        raise KeyError(f"Discovery primitive(s) missing from joined table: {missing}")

    selections: list[dict[str, Any]] = []
    fold_sizes: list[dict[str, Any]] = []

    for protocol in PROTOCOLS:
        fold_values = sorted(training[protocol].dropna().astype(int).unique())
        if fold_values != [1, 2, 3, 4, 5]:
            raise ValueError(
                f"{protocol} must contain frozen folds 1..5; got {fold_values}"
            )

        for fold in fold_values:
            fit_mask = training[protocol].astype(int).ne(fold)
            holdout_mask = ~fit_mask
            fit_frame = training.loc[fit_mask].copy()
            y_fit = fit_frame[target_column].to_numpy(dtype=float)

            miner = FoldLocalExpressionMiner(
                primitive_columns,
                top_primitives=int(discovery["top_primitives"]),
                top_expressions=int(discovery["top_expressions"]),
                operations=tuple(str(x) for x in discovery["operations"]),
                epsilon=float(discovery["epsilon"]),
            )
            miner.fit(fit_frame, y_fit)
            selected = miner.discovery_report()
            transformed_fit = miner.transform(fit_frame)

            fold_sizes.append(
                {
                    "protocol": protocol,
                    "fold": int(fold),
                    "fit_rows": int(fit_mask.sum()),
                    "held_out_rows_not_scored": int(holdout_mask.sum()),
                    "selected_primitives": "|".join(miner.selected_primitives_),
                }
            )

            for row in selected.itertuples(index=False):
                values = transformed_fit[row.feature].to_numpy(dtype=float)
                signed = _signed_spearman(values, y_fit)
                selections.append(
                    {
                        "protocol": protocol,
                        "fold": int(fold),
                        "rank": int(row.rank),
                        "operation": str(row.operation),
                        "left": str(row.left),
                        "right": str(row.right),
                        "formula": str(row.formula),
                        "abs_spearman_train": float(row.abs_spearman_train),
                        "signed_spearman_train": signed,
                        "expression_key": (
                            f"{row.operation}::{row.left}::{row.right}"
                        ),
                    }
                )

    selected_frame = pd.DataFrame(selections)
    fold_frame = pd.DataFrame(fold_sizes)

    stability_rows: list[dict[str, Any]] = []
    for (protocol, expression_key), group in selected_frame.groupby(
        ["protocol", "expression_key"],
        sort=True,
    ):
        signs = np.sign(group["signed_spearman_train"].to_numpy(dtype=float))
        finite_signs = signs[np.isfinite(signs) & (signs != 0)]
        direction_consistency = (
            float(abs(np.mean(finite_signs)))
            if len(finite_signs)
            else float("nan")
        )
        first = group.iloc[0]
        stability_rows.append(
            {
                "protocol": protocol,
                "expression_key": expression_key,
                "operation": first["operation"],
                "left": first["left"],
                "right": first["right"],
                "formula": first["formula"],
                "selected_folds": int(group["fold"].nunique()),
                "selection_rate": float(group["fold"].nunique() / 5.0),
                "mean_rank": float(group["rank"].mean()),
                "mean_abs_spearman_train": float(
                    group["abs_spearman_train"].mean()
                ),
                "mean_signed_spearman_train": float(
                    group["signed_spearman_train"].mean()
                ),
                "direction_consistency": direction_consistency,
            }
        )

    stability = pd.DataFrame(stability_rows).sort_values(
        [
            "protocol",
            "selected_folds",
            "mean_rank",
            "mean_abs_spearman_train",
        ],
        ascending=[True, False, True, False],
        kind="stable",
    )

    selected_frame.to_csv(
        output_directory / "expression_selections_by_fold.csv",
        index=False,
    )
    stability.to_csv(
        output_directory / "expression_stability.csv",
        index=False,
    )
    fold_frame.to_csv(
        output_directory / "discovery_fold_audit.csv",
        index=False,
    )

    pivot = stability.pivot_table(
        index="expression_key",
        columns="protocol",
        values="selected_folds",
        aggfunc="max",
        fill_value=0,
    )
    stable_both: list[str] = []
    if set(PROTOCOLS) <= set(pivot.columns):
        stable_both = sorted(
            pivot.index[
                (pivot[PROTOCOLS[0]] >= 3)
                & (pivot[PROTOCOLS[1]] >= 3)
            ].tolist()
        )

    report = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "training_rows": int(len(training)),
        "protocols": list(PROTOCOLS),
        "folds_per_protocol": 5,
        "primitive_count": len(primitive_columns),
        "top_primitives_per_fit": int(discovery["top_primitives"]),
        "top_expressions_per_fit": int(discovery["top_expressions"]),
        "selection_rows": int(len(selected_frame)),
        "unique_expressions": int(selected_frame["expression_key"].nunique()),
        "expressions_selected_in_3plus_folds_both_protocols": stable_both,
        "validation_targets_used_for_discovery_or_reporting": False,
        "interpretation": (
            "Training-fold recurrence audit only. Do not treat selection frequency "
            "or training association as unbiased predictive performance."
        ),
    }
    (output_directory / "discovery_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Checkpoint 03B — Expression Discovery Stability",
        "",
        "This report is a training-fold recurrence audit, not a model benchmark.",
        "Validation targets are never used to select, rank or report expressions.",
        "",
        f"- training parcels: {report['training_rows']}",
        f"- primitives configured: {report['primitive_count']}",
        f"- top primitives per fit: {report['top_primitives_per_fit']}",
        f"- top expressions per fit: {report['top_expressions_per_fit']}",
        f"- unique selected expressions: {report['unique_expressions']}",
        "",
        "An expression recurring across folds is only a candidate for later testing. "
        "Training Spearman values are not held-out performance estimates.",
        "",
    ]
    for protocol in PROTOCOLS:
        lines.extend([f"## {protocol}", ""])
        subset = stability.loc[stability["protocol"].eq(protocol)].head(15)
        lines.append(
            "| Selected folds | Mean rank | Mean abs(rho) train | "
            "Direction consistency | Formula |"
        )
        lines.append("|---:|---:|---:|---:|---|")
        for row in subset.itertuples(index=False):
            lines.append(
                f"| {row.selected_folds}/5 | {row.mean_rank:.2f} | "
                f"{row.mean_abs_spearman_train:.3f} | "
                f"{row.direction_consistency:.2f} | {row.formula} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Cross-protocol recurrence",
            "",
            "Expressions selected in at least 3/5 fits under both protocols:",
            "",
        ]
    )
    if stable_both:
        lines.extend([f"- {key}" for key in stable_both])
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "Any predictive evaluation belongs in Checkpoint 03C, where the miner "
            "is refit inside each outer training fold.",
            "",
        ]
    )
    (output_directory / "expression_discovery_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/empirical_features_v1.yaml"),
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("reports/checkpoint_03b"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    if not (root / "pyproject.toml").is_file():
        root = find_project_root(root)

    config_path = args.config if args.config.is_absolute() else root / args.config
    output_directory = (
        args.output_directory
        if args.output_directory.is_absolute()
        else root / args.output_directory
    )
    report = run_expression_discovery(
        root=root,
        config=_load_yaml(config_path),
        output_directory=output_directory,
    )
    print("Checkpoint 03B expression discovery: PASS")
    print(f"Training rows: {report['training_rows']}")
    print(f"Unique selected expressions: {report['unique_expressions']}")
    print(
        "Stable 3+/5 in both protocols: "
        f"{len(report['expressions_selected_in_3plus_folds_both_protocols'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
