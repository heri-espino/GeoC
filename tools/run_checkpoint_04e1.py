#!/usr/bin/env python
"""Run Checkpoint 04E.1: exact SIAP municipal-prior validation."""

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
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

from geocebada.data.external import load_siap_barley_detail
from geocebada.evaluation.checkpoint04a import (
    ID_COLUMN,
    PREDICTION_VALUE,
    SPLIT_COLUMN,
    TARGET_COLUMN,
    TRAIN_VALUE,
    haversine_distance_matrix,
)
from geocebada.evaluation.checkpoint04b import (
    build_knn_graph,
    mixed_distance_matrix,
    normalized_distance_matrix,
    predict_graph_laplacian,
    regression_metrics,
)
from geocebada.evaluation.checkpoint04d1 import (
    leave_one_split_out_method_selection,
    predict_weighted_local_ridge,
    split_positions_from_membership,
    summarize_nested_predictions,
    summarize_predictions_by_method,
)
from geocebada.evaluation.checkpoint04e import (
    attach_siap_panel_to_parcels,
    blend_predictions,
    build_siap_external_panel,
    complete_external_prior,
    predict_affine_external_prior,
    predict_graph_external_residual,
    predict_local_external_residual,
)
from geocebada.paths import find_project_root
from geocebada.visualization.checkpoint04e import (
    plot_external_ranking,
    plot_siap_proxy_vs_yield,
    plot_siap_scope_coverage,
)

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


def _validate_contract(frame: pd.DataFrame) -> None:
    required = {
        ID_COLUMN,
        SPLIT_COLUMN,
        TARGET_COLUMN,
        "admin_cvegeo",
        "base_centroid_lat",
        "base_centroid_lon",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"04E.1 missing required parcel columns: {sorted(missing)}")
    if len(frame) != 197 or not frame[ID_COLUMN].is_unique:
        raise ValueError("04E.1 requires exactly 197 unique parcels.")
    train = frame[SPLIT_COLUMN].eq(TRAIN_VALUE)
    target = frame[SPLIT_COLUMN].eq(PREDICTION_VALUE)
    if int(train.sum()) != 138 or int(target.sum()) != 59:
        raise ValueError("04E.1 requires 138 labeled + 59 fixed targets.")
    if frame.loc[train, TARGET_COLUMN].isna().any():
        raise ValueError("All 138 training yields must be observed.")
    if frame.loc[target, TARGET_COLUMN].notna().any():
        raise ValueError("Hidden FIRA yields must remain missing.")


def _embedding_matrix(
    embeddings: pd.DataFrame,
    frame: pd.DataFrame,
    representation: str,
) -> np.ndarray:
    block = embeddings.loc[embeddings["representation"].eq(representation)].copy()
    pc_columns = [column for column in block.columns if column.startswith("pc")]
    if not pc_columns:
        raise ValueError(f"No PCA columns found for {representation}.")
    block = block.set_index(ID_COLUMN).reindex(frame[ID_COLUMN].astype(str))
    if block[pc_columns].isna().any().any():
        raise ValueError(f"Embedding {representation} does not cover all parcels.")
    return block[pc_columns].to_numpy(float)


def _support_lookup(pseudo_predictions: pd.DataFrame) -> pd.DataFrame:
    return (
        pseudo_predictions[[
            "split_id",
            ID_COLUMN,
            "x_support_score",
            "x_support_tier",
        ]]
        .drop_duplicates()
        .reset_index(drop=True)
    )


def _method_predictions(
    *,
    config: dict[str, Any],
    prior: np.ndarray,
    x_c4: np.ndarray,
    geo_distance: np.ndarray,
    graph: np.ndarray,
    y: np.ndarray,
    observed: tuple[int, ...] | list[int] | np.ndarray,
    query: tuple[int, ...] | list[int] | np.ndarray,
) -> dict[str, np.ndarray]:
    observed_idx = np.asarray(list(map(int, observed)), dtype=int)
    query_idx = np.asarray(list(map(int, query)), dtype=int)
    prior_filled, _ = complete_external_prior(
        prior,
        reference_positions=observed_idx,
    )

    local_cfg = config["baselines"]["local"]
    graph_cfg = config["baselines"]["graph"]
    baseline_local = predict_weighted_local_ridge(
        x_c4,
        geo_distance,
        y,
        observed_idx,
        query_idx,
        k=int(local_cfg["k"]),
        alpha=float(local_cfg["alpha"]),
        distance_power=float(local_cfg["distance_power"]),
    )
    baseline_graph = predict_graph_laplacian(
        graph,
        y,
        observed_idx,
        query_idx,
        regularization=float(graph_cfg["regularization"]),
    )
    external_local = predict_local_external_residual(
        prior_filled,
        x_c4,
        geo_distance,
        y,
        observed_idx,
        query_idx,
        k=int(local_cfg["k"]),
        alpha=float(local_cfg["alpha"]),
        distance_power=float(local_cfg["distance_power"]),
    )
    external_graph = predict_graph_external_residual(
        prior_filled,
        graph,
        y,
        observed_idx,
        query_idx,
        regularization=float(graph_cfg["regularization"]),
    )

    result: dict[str, np.ndarray] = {
        "Baseline_Local04D": baseline_local,
        "Baseline_Graph04D": baseline_graph,
        "SIAP_Direct": prior_filled[query_idx],
        "SIAP_LocalResidual": external_local,
        "SIAP_GraphResidual": external_graph,
    }
    for alpha in config["external_models"]["affine_alphas"]:
        label = f"{float(alpha):g}".replace(".", "p")
        result[f"SIAP_Affine_a{label}"] = predict_affine_external_prior(
            prior_filled,
            y,
            observed_idx,
            query_idx,
            alpha=float(alpha),
        )
    for weight in config["external_models"]["blend_weights"]:
        label = f"{float(weight):g}".replace(".", "p")
        result[f"Blend_Local_SIAPLocal_w{label}"] = blend_predictions(
            baseline_local,
            external_local,
            right_weight=float(weight),
        )
        result[f"Blend_Graph_SIAPGraph_w{label}"] = blend_predictions(
            baseline_graph,
            external_graph,
            right_weight=float(weight),
        )
    return result


def _split_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in predictions.groupby(["family", "split_id", "method"]):
        family, split_id, method = keys
        rows.append(
            {
                "family": str(family),
                "split_id": str(split_id),
                "method": str(method),
                **regression_metrics(group["observed"], group["predicted"]),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["family", "split_id", "rmse", "method"]
    ).reset_index(drop=True)


def _md_table(frame: pd.DataFrame, columns: list[str], max_rows: int | None = None) -> list[str]:
    block = frame.loc[:, columns]
    if max_rows is not None:
        block = block.head(int(max_rows))
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join(["---"] * len(columns)) + "|",
    ]
    for row in block.itertuples(index=False, name=None):
        values = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                values.append("" if pd.isna(value) else f"{float(value):.4f}")
            else:
                values.append(str(value).replace("|", "/"))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def _render_report(
    report: dict[str, Any],
    scope_audit: pd.DataFrame,
    summary: pd.DataFrame,
    loso_summary: dict[str, Any],
) -> str:
    primary = summary.loc[summary["family"].eq(report["primary_family"])].sort_values(
        ["rmse_mean", "method"]
    )
    lines = [
        "# Checkpoint 04E.1 — SIAP external localization",
        "",
        f"Generated: {report['generated_at']}",
        f"Git commit: {report.get('git_commit') or 'unknown'}",
        "",
        "## Information boundary",
        "",
        "The 2025 SIAP municipal closure is public external competition-mode evidence.",
        "No hidden parcel yield is loaded or scored. Pseudo-target y is revealed only after",
        "predictions are frozen for each committed 04B split.",
        "",
        "## SIAP scope audit",
        "",
    ]
    lines.extend(
        _md_table(
            scope_audit,
            ["scope", "cycle", "modality", "rows_all_years", "municipalities_2025", "rows_2025"],
        )
    )
    lines.extend(
        [
            "",
            f"Exact parcel coverage: **{report['exact_2025_parcel_coverage']}/197**.",
            (
                "Selected-prior parcel coverage before median fallback: "
                f"**{report['prior_parcel_coverage']}/197**."
            ),
            (
                "Actual-target selected-prior coverage: "
                f"**{report['actual_target_prior_coverage']}/59**."
            ),
            "",
            "## Primary target-matched ranking",
            "",
        ]
    )
    lines.extend(
        _md_table(
            primary,
            ["method", "rmse_mean", "rmse_median", "rmse_std", "rmse_worst", "pooled_mae"],
            max_rows=15,
        )
    )
    lines.extend(
        [
            "",
            "## Leave-one-split-out method selection",
            "",
            f"- mean RMSE: **{float(loso_summary['rmse_mean']):.4f}**",
            f"- median RMSE: **{float(loso_summary['rmse_median']):.4f}**",
            f"- worst split RMSE: **{float(loso_summary['rmse_worst']):.4f}**",
            f"- pooled RMSE: **{float(loso_summary['pooled_rmse']):.4f}**",
            "",
            "## Full-label proxy diagnostics",
            "",
            (
                "- SIAP selected prior direct RMSE on the 138 observed parcels: "
                f"**{report['full_train_prior_rmse']:.4f}**"
            ),
            f"- Spearman(prior, observed yield): **{report['full_train_prior_spearman']:.4f}**",
            (
                "- legacy aggregated SIAP 2025 direct RMSE: "
                f"**{report['legacy_siap_direct_rmse']:.4f}**"
            ),
            "",
            "## Interpretation boundary",
            "",
            "- Exact SIAP means Cebada grano + Primavera-Verano + Temporal + CVEGEO.",
            (
                "- Broader SIAP scopes are explicit fallbacks, never silently mixed "
                "into the exact audit."
            ),
            (
                "- Actual 59 predictions written here are candidates for 04F, "
                "not the final submission."
            ),
            (
                "- 04E.1 decides whether the external municipal prior adds validated "
                "information beyond 04D.1."
            ),
            "",
            "## Figures",
            "",
            "![Scope coverage](figures/siap_scope_coverage.png)",
            "",
            "![Primary ranking](figures/external_method_ranking.png)",
            "",
            "![SIAP vs observed](figures/siap_proxy_vs_yield.png)",
            "",
        ]
    )
    return "\n".join(lines)


def run_checkpoint_04e1(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Run exact-SIAP audit and pseudo-competition validation."""

    inputs = config["inputs"]
    outputs = config["outputs"]
    output_dir = root / str(outputs["directory"])
    figures_dir = output_dir / str(outputs["figures_directory"])
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(
        "[04E.1 1/8] Loading parcels, frozen validation artifacts and detailed SIAP...",
        flush=True,
    )
    base_dir = root / str(inputs["base_directory"])
    frame = pd.read_csv(base_dir / str(inputs["base_table"]))
    _validate_contract(frame)
    contract = _load_yaml(root / str(inputs["data_contract"]))
    siap_cfg = config["siap"]
    years = list(
        range(
            int(siap_cfg["history_start_year"]),
            int(siap_cfg["competition_year"]) + 1,
        )
    )
    detail = load_siap_barley_detail(contract, root=root, years=years)
    a04_dir = root / str(inputs["checkpoint04a_directory"])
    b04_dir = root / str(inputs["checkpoint04b_directory"])
    embeddings = pd.read_csv(a04_dir / str(inputs["checkpoint04a_embeddings"]))
    membership = pd.read_csv(b04_dir / str(inputs["checkpoint04b_membership"]))
    pseudo_support = pd.read_csv(
        b04_dir / str(inputs["checkpoint04b_pseudo_predictions"]),
        usecols=["split_id", ID_COLUMN, "x_support_score", "x_support_tier"],
    )
    actual_support = pd.read_csv(b04_dir / str(inputs["checkpoint04b_support"]))

    print(
        "[04E.1 2/8] Auditing Cebada grano + Primavera-Verano + Temporal + municipality...",
        flush=True,
    )
    panel, scope_audit = build_siap_external_panel(
        detail,
        crop=str(siap_cfg["crop"]),
        cycle=str(siap_cfg["cycle"]),
        modality=str(siap_cfg["modality"]),
        competition_year=int(siap_cfg["competition_year"]),
        historical_years=range(
            int(siap_cfg["history_start_year"]),
            int(siap_cfg["history_end_year"]) + 1,
        ),
    )
    parcel_siap = attach_siap_panel_to_parcels(
        frame,
        panel,
        cvegeo_column=str(config["identity"]["cvegeo_column"]),
    )
    parcel_siap = parcel_siap.merge(
        frame[[ID_COLUMN, SPLIT_COLUMN, TARGET_COLUMN, "meta_estado", "meta_municipio"]],
        on=ID_COLUMN,
        how="left",
        validate="1:1",
    )
    scope_audit.to_csv(output_dir / str(outputs["siap_scope_audit"]), index=False)
    panel.to_csv(output_dir / str(outputs["siap_municipal_panel"]), index=False)
    parcel_siap.to_csv(output_dir / str(outputs["siap_parcel_panel"]), index=False)

    print("[04E.1 3/8] Rebuilding frozen 04D local/graph geometry...", flush=True)
    c1 = _embedding_matrix(embeddings, frame, str(config["representations"]["c1_name"]))
    c4 = _embedding_matrix(embeddings, frame, str(config["representations"]["c4_name"]))
    x_c4 = StandardScaler().fit_transform(c4)
    geographic = haversine_distance_matrix(
        pd.to_numeric(frame["base_centroid_lat"], errors="raise"),
        pd.to_numeric(frame["base_centroid_lon"], errors="raise"),
    )
    agronomic = pairwise_distances(c1, metric="euclidean")
    train_positions = np.flatnonzero(frame[SPLIT_COLUMN].eq(TRAIN_VALUE).to_numpy())
    target_positions = np.flatnonzero(frame[SPLIT_COLUMN].eq(PREDICTION_VALUE).to_numpy())
    geo_norm, _ = normalized_distance_matrix(geographic, reference_positions=train_positions)
    agro_norm, _ = normalized_distance_matrix(agronomic, reference_positions=train_positions)
    graph_cfg = config["baselines"]["graph"]
    graph_distance = mixed_distance_matrix(
        [
            (geo_norm, float(graph_cfg["geographic_weight"])),
            (agro_norm, float(graph_cfg["agronomic_weight"])),
        ]
    )
    graph = build_knn_graph(graph_distance, k=int(graph_cfg["k"]))

    print("[04E.1 4/8] Resolving exact committed 04B pseudo-competition splits...", flush=True)
    splits = split_positions_from_membership(
        frame,
        membership,
        train_value=TRAIN_VALUE,
        split_column=SPLIT_COLUMN,
    )
    support_lookup = _support_lookup(pseudo_support).set_index(["split_id", ID_COLUMN])
    y = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce").to_numpy(float)
    prior = pd.to_numeric(parcel_siap["siap_prior_yield"], errors="coerce").to_numpy(float)
    source = parcel_siap["siap_prior_source"].astype("string").to_numpy()

    print(
        f"[04E.1 5/8] Evaluating external-prior methods on {len(splits)} frozen splits...",
        flush=True,
    )
    prediction_rows: list[dict[str, Any]] = []
    for split_index, split in enumerate(splits, start=1):
        predictions = _method_predictions(
            config=config,
            prior=prior,
            x_c4=x_c4,
            geo_distance=geo_norm,
            graph=graph,
            y=y,
            observed=split.observed_positions,
            query=split.query_positions,
        )
        _, filled_mask = complete_external_prior(
            prior,
            reference_positions=split.observed_positions,
        )
        for method, values in predictions.items():
            for local_index, position in enumerate(split.query_positions):
                parcel_id = str(frame.iloc[position][ID_COLUMN])
                support_row = support_lookup.loc[(split.split_id, parcel_id)]
                if isinstance(support_row, pd.DataFrame):
                    support_row = support_row.iloc[0]
                prediction_rows.append(
                    {
                        "split_id": split.split_id,
                        "family": split.family,
                        "repeat": split.repeat,
                        "method": method,
                        ID_COLUMN: parcel_id,
                        "observed": float(y[position]),
                        "predicted": float(values[local_index]),
                        "error": float(y[position] - values[local_index]),
                        "x_support_score": float(support_row["x_support_score"]),
                        "x_support_tier": str(support_row["x_support_tier"]),
                        "siap_prior_source": "visible_median_fallback"
                        if bool(filled_mask[position])
                        else str(source[position]),
                    }
                )
        print(
            f"  [{split_index:02d}/{len(splits):02d}] {split.split_id}",
            flush=True,
        )

    predictions = pd.DataFrame(prediction_rows)
    split_metrics = _split_metrics(predictions)
    summary = summarize_predictions_by_method(predictions)
    predictions.to_csv(output_dir / str(outputs["predictions"]), index=False)
    split_metrics.to_csv(output_dir / str(outputs["split_metrics"]), index=False)
    summary.to_csv(output_dir / str(outputs["protocol_summary"]), index=False)

    print("[04E.1 6/8] Running leave-one-target-matched-split-out method selection...", flush=True)
    primary_family = str(config["validation"]["primary_family"])
    loso_selection, loso_predictions = leave_one_split_out_method_selection(
        predictions,
        family=primary_family,
    )
    loso_summary = summarize_nested_predictions(
        loso_predictions,
        label="LOSO_04E_method_selection",
    )
    loso_selection.to_csv(output_dir / str(outputs["loso_selection"]), index=False)
    loso_predictions.to_csv(output_dir / str(outputs["loso_predictions"]), index=False)

    print(
        "[04E.1 7/8] Fitting candidate external-prior models for the actual 59 targets...",
        flush=True,
    )
    actual_predictions = _method_predictions(
        config=config,
        prior=prior,
        x_c4=x_c4,
        geo_distance=geo_norm,
        graph=graph,
        y=y,
        observed=train_positions,
        query=target_positions,
    )
    actual_prior_filled, actual_filled = complete_external_prior(
        prior,
        reference_positions=train_positions,
    )
    actual_support_lookup = actual_support.set_index(ID_COLUMN)
    actual_rows: list[dict[str, Any]] = []
    for method, values in actual_predictions.items():
        for local_index, position in enumerate(target_positions):
            parcel_id = str(frame.iloc[position][ID_COLUMN])
            support_row = actual_support_lookup.loc[parcel_id]
            actual_rows.append(
                {
                    ID_COLUMN: parcel_id,
                    "method": method,
                    "predicted": float(values[local_index]),
                    "x_support_score": float(support_row["x_support_score"]),
                    "x_support_tier": str(support_row["x_support_tier"]),
                    "siap_prior_yield": float(actual_prior_filled[position]),
                    "siap_prior_source": "visible_median_fallback"
                    if bool(actual_filled[position])
                    else str(source[position]),
                }
            )
    actual_candidates = pd.DataFrame(actual_rows)
    actual_candidates.to_csv(output_dir / str(outputs["actual_candidates"]), index=False)

    print("[04E.1 8/8] Writing report and figures...", flush=True)
    exact_column = f"siap_exact_{int(siap_cfg['competition_year'])}_yield"
    exact_coverage = int(pd.to_numeric(parcel_siap[exact_column], errors="coerce").notna().sum())
    prior_coverage = int(np.isfinite(prior).sum())
    actual_prior_coverage = int(np.isfinite(prior[target_positions]).sum())
    filled_full, _ = complete_external_prior(prior, reference_positions=train_positions)
    direct_metrics = regression_metrics(y[train_positions], filled_full[train_positions])
    spearman = float(
        pd.Series(filled_full[train_positions]).corr(
            pd.Series(y[train_positions]),
            method="spearman",
        )
    )
    if "siap_2025__yield" in frame.columns:
        legacy = pd.to_numeric(frame["siap_2025__yield"], errors="coerce").to_numpy(float)
        legacy_filled, _ = complete_external_prior(legacy, reference_positions=train_positions)
        legacy_rmse = regression_metrics(y[train_positions], legacy_filled[train_positions])["rmse"]
    else:
        legacy_rmse = float("nan")

    primary_summary = summary.loc[summary["family"].eq(primary_family)].sort_values(
        ["rmse_mean", "method"]
    )
    best = primary_summary.iloc[0]
    report = {
        "schema_version": int(config["schema_version"]),
        "generated_at": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(root),
        "stage": str(config["stage"]["name"]),
        "hidden_prediction_targets_used": False,
        "public_2025_external_outcome_used": True,
        "siap_crop": str(siap_cfg["crop"]),
        "siap_cycle": str(siap_cfg["cycle"]),
        "siap_modality": str(siap_cfg["modality"]),
        "primary_family": primary_family,
        "total_splits": int(len(splits)),
        "methods_evaluated": int(predictions["method"].nunique()),
        "best_primary_method": str(best["method"]),
        "best_primary_rmse_mean": float(best["rmse_mean"]),
        "best_primary_rmse_worst": float(best["rmse_worst"]),
        "loso_rmse_mean": float(loso_summary["rmse_mean"]),
        "loso_pooled_rmse": float(loso_summary["pooled_rmse"]),
        "exact_2025_parcel_coverage": exact_coverage,
        "prior_parcel_coverage": prior_coverage,
        "actual_target_prior_coverage": actual_prior_coverage,
        "prior_source_counts": {
            str(key): int(value)
            for key, value in parcel_siap["siap_prior_source"].value_counts(dropna=False).items()
        },
        "full_train_prior_rmse": float(direct_metrics["rmse"]),
        "full_train_prior_mae": float(direct_metrics["mae"]),
        "full_train_prior_spearman": spearman,
        "legacy_siap_direct_rmse": float(legacy_rmse),
    }
    (output_dir / str(outputs["report_json"])).write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / str(outputs["report_markdown"])).write_text(
        _render_report(report, scope_audit, summary, loso_summary),
        encoding="utf-8",
    )
    plot_siap_scope_coverage(
        scope_audit,
        figures_dir / "siap_scope_coverage.png",
        dpi=int(config["figures"]["dpi"]),
    )
    plot_external_ranking(
        summary,
        figures_dir / "external_method_ranking.png",
        family=primary_family,
        top_n=int(config["figures"]["ranking_top_n"]),
        dpi=int(config["figures"]["dpi"]),
    )
    plot_siap_proxy_vs_yield(
        parcel_siap,
        figures_dir / "siap_proxy_vs_yield.png",
        dpi=int(config["figures"]["dpi"]),
    )

    print("Checkpoint 04E.1 SIAP localization: PASS")
    print(f"SIAP exact parcel coverage: {exact_coverage}/197")
    print(f"Best target-matched method: {best['method']}")
    print(f"Mean RMSE: {float(best['rmse_mean']):.4f}")
    print(f"LOSO selection RMSE: {float(loso_summary['rmse_mean']):.4f}")
    print(f"Inspect: {outputs['directory']}/{outputs['report_markdown']}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/checkpoint04e1.yaml",
        help="Checkpoint configuration path relative to repository root.",
    )
    args = parser.parse_args()
    root = find_project_root(ROOT)
    config = _load_yaml(root / args.config)
    run_checkpoint_04e1(root, config)


if __name__ == "__main__":
    main()
