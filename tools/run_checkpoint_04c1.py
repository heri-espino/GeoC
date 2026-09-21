#!/usr/bin/env python
"""Run focused Checkpoint 04C.1 global CatBoost anchor evaluation."""

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
    blend_anchor_predictions,
    leave_one_split_out_method_selection,
    predict_catboost_seed_ensemble,
    regression_metrics,
    residual_correlation_table,
    resolve_agronomic_competition_features,
    summarize_nested_predictions,
    summarize_predictions_by_method,
)
from geocebada.features import join_agronomic_features
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


def _weight_label(weight: float) -> str:
    return f"{float(weight):.2f}".replace(".", "p")


def _split_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (family, split_id, method), group in predictions.groupby(
        ["family", "split_id", "method"]
    ):
        rows.append(
            {
                "family": family,
                "split_id": split_id,
                "method": method,
                **regression_metrics(group["observed"], group["predicted"]),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["family", "split_id", "rmse", "method"]
    ).reset_index(drop=True)


def _append_fixed_blends(
    predictions: pd.DataFrame,
    *,
    anchor_method: str,
    baseline_map: dict[str, str],
    weights: list[float],
) -> pd.DataFrame:
    additions: list[pd.DataFrame] = []

    for (family, split_id), group in predictions.groupby(["family", "split_id"]):
        anchor = group.loc[group["method"].eq(anchor_method), :]
        if anchor.empty:
            raise ValueError(f"Missing {anchor_method} for {family}/{split_id}.")
        anchor = anchor.set_index("ID_POLIGONO")

        for label, baseline_method in baseline_map.items():
            baseline = group.loc[group["method"].eq(baseline_method), :]
            if baseline.empty:
                raise ValueError(f"Missing {baseline_method} for {family}/{split_id}.")
            baseline = baseline.set_index("ID_POLIGONO")
            ids = baseline.index.intersection(anchor.index)
            if len(ids) != len(baseline) or len(ids) != len(anchor):
                raise ValueError(
                    f"Baseline/anchor ID mismatch for {family}/{split_id}/{label}."
                )

            for weight in weights:
                predicted = blend_anchor_predictions(
                    baseline.loc[ids, "predicted"].to_numpy(dtype=float),
                    anchor.loc[ids, "predicted"].to_numpy(dtype=float),
                    anchor_weight=float(weight),
                )
                additions.append(
                    pd.DataFrame(
                        {
                            "family": str(family),
                            "split_id": str(split_id),
                            "method": (
                                f"Blend_{label}_CatBoost_w{_weight_label(weight)}"
                            ),
                            "ID_POLIGONO": ids.astype(str),
                            "observed": baseline.loc[ids, "observed"].to_numpy(
                                dtype=float
                            ),
                            "predicted": predicted,
                        }
                    )
                )

    if not additions:
        return predictions.copy()
    return pd.concat([predictions, *additions], ignore_index=True)


def _actual_candidates(
    *,
    joined: pd.DataFrame,
    features: tuple[str, ...],
    config: dict[str, Any],
    actual_baselines: pd.DataFrame,
    task_type: str,
) -> pd.DataFrame:
    identity = config["identity"]
    split_column = str(identity["split_column"])
    target_column = str(identity["target_column"])
    train_value = str(identity["train_value"])
    prediction_value = str(identity["prediction_value"])

    train_positions = np.flatnonzero(
        joined[split_column].eq(train_value).to_numpy()
    )
    target_positions = np.flatnonzero(
        joined[split_column].eq(prediction_value).to_numpy()
    )
    y = pd.to_numeric(joined[target_column], errors="coerce").to_numpy(dtype=float)
    x = joined.loc[:, list(features)]

    catboost_cfg = config["catboost"]
    seeds = list(map(int, catboost_cfg["seeds"]))
    devices = str(catboost_cfg.get("devices", "0"))
    iterations = int(catboost_cfg["iterations"])

    blocks: list[pd.DataFrame] = []
    anchor_predictions: dict[str, np.ndarray] = {}
    for method, raw_params in catboost_cfg["candidates"].items():
        params = {
            "iterations": iterations,
            "depth": int(raw_params["depth"]),
            "learning_rate": float(raw_params["learning_rate"]),
            "l2_leaf_reg": float(raw_params["l2_leaf_reg"]),
        }
        predicted = predict_catboost_seed_ensemble(
            x,
            y,
            train_positions,
            target_positions,
            params=params,
            seeds=seeds,
            task_type=task_type,
            devices=devices,
        )
        anchor_predictions[str(method)] = predicted
        blocks.append(
            pd.DataFrame(
                {
                    "ID_POLIGONO": joined.iloc[target_positions]["ID_POLIGONO"]
                    .astype(str)
                    .to_numpy(),
                    "method": str(method),
                    "predicted": predicted,
                }
            )
        )

    anchor_method = str(config["blends"]["anchor_method"])
    stable = anchor_predictions[anchor_method]
    weights = [float(value) for value in config["blends"]["anchor_weights"]]
    baseline_map = {
        str(key): str(value)
        for key, value in config["blends"]["baselines"].items()
    }
    target_ids = joined.iloc[target_positions]["ID_POLIGONO"].astype(str).to_numpy()

    for label, baseline_method in baseline_map.items():
        baseline = actual_baselines.loc[
            actual_baselines["method"].eq(baseline_method),
            ["ID_POLIGONO", "predicted"],
        ].copy()
        baseline["ID_POLIGONO"] = baseline["ID_POLIGONO"].astype(str)
        baseline = baseline.set_index("ID_POLIGONO").reindex(target_ids)
        if baseline["predicted"].isna().any():
            raise ValueError(f"Actual baseline {baseline_method} is incomplete.")

        for weight in weights:
            blocks.append(
                pd.DataFrame(
                    {
                        "ID_POLIGONO": target_ids,
                        "method": f"Blend_{label}_CatBoost_w{_weight_label(weight)}",
                        "predicted": blend_anchor_predictions(
                            baseline["predicted"].to_numpy(dtype=float),
                            stable,
                            anchor_weight=weight,
                        ),
                    }
                )
            )

    return pd.concat(blocks, ignore_index=True)


def _render_report(
    *,
    report: dict[str, Any],
    primary: pd.DataFrame,
    loso_selection: pd.DataFrame,
    residual_correlation: pd.DataFrame,
) -> str:
    ranking = primary.sort_values(["rmse_mean", "method"]).copy()
    lines = [
        "# Checkpoint 04C.1 — Focused global CatBoost anchor",
        "",
        f"Generated: {report['generated_at']}",
        f"Git commit: {report.get('git_commit') or 'unknown'}",
        "",
        "## Scope",
        "",
        "- active track: **competition only**",
        "- hidden 59 yields used: **no**",
        f"- agronomic features: **{report['n_features']}**",
        f"- pseudo-split families: **{', '.join(report['families'])}**",
        f"- CatBoost task type: **{report['catboost_task_type']}**",
        f"- CatBoost seeds per candidate: **{report['n_seeds']}**",
        f"- CatBoost candidates: **{report['n_anchor_candidates']}**",
        "",
        (
            "This checkpoint is intentionally narrow. The three CatBoost "
            "hyperparameter candidates come from the frozen 03C.2 grid; it is "
            "not a new broad search."
        ),
        "",
        "## Primary target-matched ranking",
        "",
        "| method | rmse_mean | rmse_median | rmse_worst | pooled_rmse | pooled_mae |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in ranking.itertuples(index=False):
        lines.append(
            f"| {row.method} | {row.rmse_mean:.4f} | {row.rmse_median:.4f} | "
            f"{row.rmse_worst:.4f} | {row.pooled_rmse:.4f} | {row.pooled_mae:.4f} |"
        )

    counts = loso_selection["selected_method"].value_counts()
    lines.extend(
        [
            "",
            "## Controlled LOSO anchor gate",
            "",
            (
                "The gate can choose only the frozen Local04D baseline, the stable "
                "CatBoost anchor, or the three low-weight Local+CatBoost blends."
            ),
            "",
        ]
    )
    for method, count in counts.items():
        lines.append(f"- {method}: **{int(count)}/{len(loso_selection)}** holdouts")

    lines.extend(
        [
            "",
            "## Residual diversity on target-matched pseudo-targets",
            "",
            "| method A | method B | n | residual correlation |",
            "|---|---|---:|---:|",
        ]
    )
    for row in residual_correlation.itertuples(index=False):
        lines.append(
            f"| {row.method_a} | {row.method_b} | {row.n_pairs} | "
            f"{row.residual_correlation:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Decision rule for 04F",
            "",
            (
                "CatBoost should enter 04F only if the frozen target-matched evidence "
                "shows either a reproducible low-weight blend improvement or useful "
                "residual diversity without a material robustness penalty. A weaker "
                "standalone CatBoost score is not sufficient by itself."
            ),
            "",
            (
                "Actual 59 predictions in actual_anchor_candidates.csv are candidates "
                "only. No hidden target is loaded or scored."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def run_checkpoint_04c1(
    root: Path,
    config: dict[str, Any],
    *,
    task_type_override: str | None = None,
) -> dict[str, Any]:
    """Run the focused CatBoost anchor experiment."""

    inputs = config["inputs"]
    outputs = config["outputs"]
    identity = config["identity"]
    validation = config["validation"]

    print("[04C.1 1/7] Loading competition features and frozen split artifacts...")
    base = pd.read_csv(root / str(inputs["base_table"]))
    agronomic = pd.read_csv(root / str(inputs["agronomic_table"]))
    joined = join_agronomic_features(base, agronomic)
    manifest = _load_json(root / str(inputs["agronomic_manifest"]))
    membership = pd.read_csv(root / str(inputs["pseudo_membership"]))
    baseline_predictions = pd.read_csv(
        root / str(inputs["frozen_baseline_predictions"])
    )
    actual_baselines = pd.read_csv(root / str(inputs["frozen_actual_baselines"]))

    id_column = str(identity["id_column"])
    target_column = str(identity["target_column"])
    split_column = str(identity["split_column"])
    train_value = str(identity["train_value"])
    prediction_value = str(identity["prediction_value"])

    expected = (
        int(validation["expected_total_rows"]),
        int(validation["expected_training_rows"]),
        int(validation["expected_prediction_rows"]),
    )
    actual_counts = (
        len(joined),
        int(joined[split_column].eq(train_value).sum()),
        int(joined[split_column].eq(prediction_value).sum()),
    )
    if actual_counts != expected:
        raise ValueError(f"Unexpected parcel counts: {actual_counts}, expected {expected}.")

    hidden = pd.to_numeric(
        joined.loc[joined[split_column].eq(prediction_value), target_column],
        errors="coerce",
    )
    if hidden.notna().any():
        raise ValueError("Prediction rows unexpectedly contain hidden target values.")

    joined[id_column] = joined[id_column].astype(str)
    if joined[id_column].duplicated().any():
        raise ValueError("Joined parcel table contains duplicate IDs.")
    features = resolve_agronomic_competition_features(joined, manifest)
    x = joined.loc[:, list(features)]
    y = pd.to_numeric(joined[target_column], errors="coerce").to_numpy(dtype=float)
    position_by_id = {
        parcel_id: position
        for position, parcel_id in enumerate(joined[id_column].astype(str))
    }

    baseline_methods = list(map(str, validation["baseline_methods"]))
    baseline_predictions = baseline_predictions.loc[
        baseline_predictions["method"].isin(baseline_methods),
        ["family", "split_id", "method", id_column, "observed", "predicted"],
    ].copy()
    baseline_predictions[id_column] = baseline_predictions[id_column].astype(str)

    print("[04C.1 2/7] Evaluating focused CatBoost anchors on frozen pseudo-splits...")
    catboost_cfg = config["catboost"]
    task_type = (
        str(task_type_override).upper()
        if task_type_override
        else str(catboost_cfg["task_type"]).upper()
    )
    devices = str(catboost_cfg.get("devices", "0"))
    seeds = list(map(int, catboost_cfg["seeds"]))
    iterations = int(catboost_cfg["iterations"])
    progress = bool(config.get("runtime", {}).get("progress", True))

    anchor_rows: list[pd.DataFrame] = []
    grouped_membership = list(membership.groupby(["family", "split_id"], sort=False))
    for split_number, ((family, split_id), split_frame) in enumerate(
        grouped_membership,
        start=1,
    ):
        train_ids = split_frame.loc[
            split_frame["role"].eq("pseudo_train"),
            id_column,
        ].astype(str)
        target_ids = split_frame.loc[
            split_frame["role"].eq("pseudo_target"),
            id_column,
        ].astype(str)

        observed_positions = [position_by_id[value] for value in train_ids]
        query_positions = [position_by_id[value] for value in target_ids]
        observed = y[np.asarray(query_positions, dtype=int)]
        if not np.isfinite(observed).all():
            raise ValueError(f"Pseudo-target y missing for {split_id}.")

        for method, raw_params in catboost_cfg["candidates"].items():
            params = {
                "iterations": iterations,
                "depth": int(raw_params["depth"]),
                "learning_rate": float(raw_params["learning_rate"]),
                "l2_leaf_reg": float(raw_params["l2_leaf_reg"]),
            }
            try:
                predicted = predict_catboost_seed_ensemble(
                    x,
                    y,
                    observed_positions,
                    query_positions,
                    params=params,
                    seeds=seeds,
                    task_type=task_type,
                    devices=devices,
                )
            except Exception as exc:
                if task_type == "GPU":
                    raise RuntimeError(
                        "CatBoost GPU execution failed. Automatic CPU fallback is "
                        "disabled. Fix CUDA/CatBoost or rerun explicitly with "
                        "--catboost-task-type CPU --confirm-cpu y."
                    ) from exc
                raise

            anchor_rows.append(
                pd.DataFrame(
                    {
                        "family": str(family),
                        "split_id": str(split_id),
                        "method": str(method),
                        id_column: target_ids.to_numpy(),
                        "observed": observed,
                        "predicted": predicted,
                    }
                )
            )

        if progress:
            print(
                f"  split {split_number}/{len(grouped_membership)} "
                f"{family}/{split_id}"
            )

    anchors = pd.concat(anchor_rows, ignore_index=True)
    predictions = pd.concat([baseline_predictions, anchors], ignore_index=True)

    print("[04C.1 3/7] Building fixed low-weight baseline + CatBoost blends...")
    predictions = _append_fixed_blends(
        predictions,
        anchor_method=str(config["blends"]["anchor_method"]),
        baseline_map={
            str(key): str(value)
            for key, value in config["blends"]["baselines"].items()
        },
        weights=[float(value) for value in config["blends"]["anchor_weights"]],
    )

    print("[04C.1 4/7] Summarizing primary and stress protocols...")
    split_metrics = _split_metrics(predictions)
    protocol_summary = summarize_predictions_by_method(predictions)
    primary_family = str(validation["primary_family"])
    primary = protocol_summary.loc[
        protocol_summary["family"].eq(primary_family)
    ].copy()

    print("[04C.1 5/7] Running controlled leave-one-split-out anchor gate...")
    gate_methods = list(map(str, validation["anchor_gate_methods"]))
    loso_selection, loso_predictions = leave_one_split_out_method_selection(
        predictions,
        family=primary_family,
        candidate_methods=gate_methods,
    )
    loso_summary = summarize_nested_predictions(
        loso_predictions,
        label="04C.1 controlled anchor gate",
    )

    stable_method = str(config["blends"]["anchor_method"])
    correlations = residual_correlation_table(
        predictions,
        family=primary_family,
        methods=[
            "Baseline_Local04D",
            "Baseline_Graph04D",
            stable_method,
        ],
    )

    print("[04C.1 6/7] Fitting full-label CatBoost candidates for the actual 59...")
    actual_candidates = _actual_candidates(
        joined=joined,
        features=features,
        config=config,
        actual_baselines=actual_baselines,
        task_type=task_type,
    )

    print("[04C.1 7/7] Writing frozen artifacts and report...")
    output_dir = root / str(outputs["directory"])
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / str(outputs["pseudo_predictions"]), index=False)
    split_metrics.to_csv(output_dir / str(outputs["split_metrics"]), index=False)
    protocol_summary.to_csv(
        output_dir / str(outputs["protocol_summary"]),
        index=False,
    )
    loso_selection.to_csv(
        output_dir / str(outputs["loso_anchor_gate_selection"]),
        index=False,
    )
    loso_predictions.to_csv(
        output_dir / str(outputs["loso_anchor_gate_predictions"]),
        index=False,
    )
    correlations.to_csv(
        output_dir / str(outputs["residual_correlation"]),
        index=False,
    )
    actual_candidates.to_csv(
        output_dir / str(outputs["actual_anchor_candidates"]),
        index=False,
    )

    primary_split_count = int(
        membership.loc[membership["family"].eq(primary_family), "split_id"].nunique()
    )
    if primary_split_count != int(validation["expected_primary_splits"]):
        raise ValueError(
            f"Expected {validation['expected_primary_splits']} primary splits, "
            f"found {primary_split_count}."
        )

    report = {
        "schema_version": int(config["schema_version"]),
        "generated_at": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(root),
        "stage": str(config["stage"]["name"]),
        "active_track": "competition",
        "total_rows": int(len(joined)),
        "training_rows": int(joined[split_column].eq(train_value).sum()),
        "prediction_rows": int(joined[split_column].eq(prediction_value).sum()),
        "hidden_prediction_targets_used": False,
        "n_features": int(len(features)),
        "families": sorted(membership["family"].astype(str).unique().tolist()),
        "primary_family": primary_family,
        "primary_splits": primary_split_count,
        "n_anchor_candidates": int(len(catboost_cfg["candidates"])),
        "n_seeds": int(len(seeds)),
        "catboost_task_type": task_type,
        "automatic_cpu_fallback": False,
        "loso_anchor_gate": loso_summary,
        "actual_candidate_rows": int(len(actual_candidates)),
        "actual_hidden_y_scored": False,
    }
    (output_dir / str(outputs["report_json"])).write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / str(outputs["report_markdown"])).write_text(
        _render_report(
            report=report,
            primary=primary,
            loso_selection=loso_selection,
            residual_correlation=correlations,
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
        default=Path("configs/checkpoint04c1.yaml"),
    )
    parser.add_argument(
        "--catboost-task-type",
        choices=("GPU", "CPU"),
        help="Override the configured task type. GPU is the default.",
    )
    parser.add_argument(
        "--confirm-cpu",
        default="n",
        help="Must be exactly 'y' when explicitly requesting CPU execution.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    if not (root / "pyproject.toml").is_file():
        root = find_project_root(root)

    config_path = args.config if args.config.is_absolute() else root / args.config
    config = _load_yaml(config_path)
    requested_task = (
        str(args.catboost_task_type).upper()
        if args.catboost_task_type
        else str(config["catboost"]["task_type"]).upper()
    )
    if requested_task == "CPU" and str(args.confirm_cpu).casefold() != "y":
        raise SystemExit(
            "CPU execution was requested but not confirmed. Rerun with "
            "--catboost-task-type CPU --confirm-cpu y."
        )

    report = run_checkpoint_04c1(
        root,
        config,
        task_type_override=args.catboost_task_type,
    )
    print("Checkpoint 04C.1 focused CatBoost anchor: PASS")
    print(f"Training rows: {report['training_rows']}")
    print(f"Prediction rows kept hidden: {report['prediction_rows']}")
    print(f"Agronomic features: {report['n_features']}")
    print(f"CatBoost task type: {report['catboost_task_type']}")
    print(f"Primary pseudo-splits: {report['primary_splits']}")
    print("Inspect: reports/checkpoint_04c1/checkpoint_04c1_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
