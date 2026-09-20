#!/usr/bin/env python
"""Run Checkpoint 04A target topology, similarity and support diagnostics."""

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

from geocebada.evaluation.checkpoint04a import (
    ID_COLUMN,
    PREDICTION_VALUE,
    SPLIT_COLUMN,
    TARGET_COLUMN,
    TRAIN_VALUE,
    adversarial_train_target_validation,
    build_checkpoint03_ensemble_residuals,
    build_monthly_temporal_arrays,
    build_target_support_profile,
    build_train_pair_validation,
    build_transductive_pca_embedding,
    feature_neighbor_tables,
    feature_shift_table,
    geographic_neighbor_tables,
    knn_weight_matrix,
    morans_i,
    resolve_numeric_representation_features,
    summarize_similarity_yield_relationships,
    temporal_neighbor_tables,
    temporal_pair_metrics,
)
from geocebada.features import join_agronomic_features, join_empirical_features
from geocebada.paths import find_project_root
from geocebada.visualization.checkpoint04a import (
    plot_adversarial_roc,
    plot_moran_scatter,
    plot_nearest_distance_distribution,
    plot_pca_embedding,
    plot_primary_temporal_neighbor_panels,
    plot_similarity_vs_yield_difference,
    plot_support_ranking,
    plot_temporal_similarity_distribution,
    plot_top_feature_shift,
    plot_train_target_map,
)

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


def _validate_contract(frame: pd.DataFrame) -> None:
    required = {
        ID_COLUMN,
        SPLIT_COLUMN,
        TARGET_COLUMN,
        "meta_estado",
        "meta_municipio",
        "base_centroid_lat",
        "base_centroid_lon",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"04A missing columns: {sorted(missing)}")
    if len(frame) != 197 or not frame[ID_COLUMN].is_unique:
        raise ValueError("04A requires exactly 197 unique parcels.")
    train = frame[SPLIT_COLUMN].eq(TRAIN_VALUE)
    target = frame[SPLIT_COLUMN].eq(PREDICTION_VALUE)
    if int(train.sum()) != 138 or int(target.sum()) != 59:
        raise ValueError("04A requires 138 labeled and 59 target parcels.")
    if frame.loc[train, TARGET_COLUMN].isna().any():
        raise ValueError("Training yield is missing.")
    if frame.loc[target, TARGET_COLUMN].notna().any():
        raise ValueError("Hidden FIRA targets must remain missing.")


def _md_table(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join(["---"] * len(columns)) + "|",
    ]
    for row in frame.loc[:, columns].itertuples(index=False, name=None):
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
    support: pd.DataFrame,
    similarity: pd.DataFrame,
    shift: pd.DataFrame,
    spatial: dict[str, Any],
    embedding_manifest: list[dict[str, Any]],
) -> str:
    lines = [
        "# Checkpoint 04A — Target topology, similarity and support",
        "",
        f"Generated: {report['generated_at']}",
        f"Git commit: {report.get('git_commit') or 'unknown'}",
        "",
        "## Objective",
        "",
        "Characterize the exact 59 fixed target parcels relative to the 138 labeled parcels.",
        "Target covariates are used transductively. The 59 hidden FIRA yields are never accessed.",
        "",
        "The support score is diagnostic. It is not a predicted yield or a calibrated probability.",
        "",
        "## Data",
        "",
        f"- 197 parcels = {report['training_rows']} labeled + {report['prediction_rows']} targets",
        f"- temporal series found: {len(report['temporal_series_found'])}",
        f"- labeled-labeled pairs evaluated: {report['train_pair_rows']}",
        f"- target-labeled temporal pairs evaluated: {report['target_train_temporal_pairs']}",
        "",
        "## Transductive PCA representations",
        "",
    ]
    emb = pd.DataFrame(embedding_manifest)
    lines.extend(
        _md_table(
            emb,
            [
                "representation",
                "n_features_used",
                "n_components",
                "explained_variance_ratio_sum",
            ],
        )
    )

    lines.extend(
        [
            "",
            "## Covariate shift",
            "",
            f"Cross-validated adversarial AUC: **{report['adversarial_auc']:.4f}**.",
            "AUC near 0.5 means weak separation; larger AUC means the target X distribution "
            "is more distinguishable.",
            "",
            "Largest standardized train-target mean shifts:",
            "",
        ]
    )
    lines.extend(
        _md_table(
            shift.head(20),
            [
                "feature",
                "standardized_mean_difference",
                "train_missing_fraction",
                "target_missing_fraction",
            ],
        )
    )

    lines.extend(
        [
            "",
            "## Does similarity transfer yield?",
            "",
            "Spearman associations below use only pairs among the 138 labeled parcels.",
            "",
        ]
    )
    if not similarity.empty:
        lines.extend(
            _md_table(
                similarity,
                [
                    "metric",
                    "n_pairs",
                    "spearman_rho_with_abs_yield_difference",
                    "p_value",
                    "expected_direction_if_useful",
                ],
            )
        )

    spatial_rows = [
        {"signal": name, **item}
        for name, item in spatial.items()
        if isinstance(item, dict) and "morans_i" in item
    ]
    lines.extend(["", "## Spatial autocorrelation", ""])
    if spatial_rows:
        lines.extend(
            _md_table(
                pd.DataFrame(spatial_rows),
                [
                    "signal",
                    "morans_i",
                    "expected_i",
                    "permutation_p_value_two_sided",
                    "permutations",
                ],
            )
        )

    tiers = (
        support["support_tier"]
        .value_counts()
        .sort_index()
        .rename_axis("support_tier")
        .reset_index(name="n_targets")
    )
    lines.extend(["", "## Support taxonomy", ""])
    lines.extend(_md_table(tiers, ["support_tier", "n_targets"]))
    lines.extend(["", "### All 59 target parcels", ""])
    lines.extend(
        _md_table(
            support,
            [
                ID_COLUMN,
                "meta_estado",
                "meta_municipio",
                "support_score",
                "support_tier",
                "nearest_geo_train_id",
                "nearest_geo_km",
                "nearest_feature_train_id",
                "nearest_feature_distance",
                "nearest_temporal_train_id",
                "nearest_temporal_similarity",
                "feature_neighbor_y_mean",
                "feature_neighbor_y_sd",
                "adversarial_target_probability",
            ],
        )
    )

    lines.extend(
        [
            "",
            "## Figures",
            "",
            "![Map](figures/map_train_target.png)",
            "",
            "![Geographic support](figures/geographic_nn_distribution.png)",
            "",
            f"![PCA](figures/pca_{report['primary_representation']}.png)",
            "",
            "![Feature support](figures/feature_nn_distribution.png)",
            "",
            "![Adversarial ROC](figures/adversarial_roc.png)",
            "",
            "![Feature shift](figures/adversarial_top_shift_features.png)",
            "",
            "![Temporal support](figures/temporal_similarity_distribution.png)",
            "",
            "![Geo versus yield difference](figures/similarity_vs_yield_geo.png)",
            "",
            "![Feature versus yield difference](figures/similarity_vs_yield_feature.png)",
            "",
            "![Temporal versus yield difference](figures/similarity_vs_yield_temporal.png)",
            "",
            "![Yield Moran scatter](figures/moran_yield.png)",
            "",
            "![E123 residual Moran scatter](figures/moran_E123_grouped_residual.png)",
            "",
            "![Support ranking](figures/target_support_ranking.png)",
            "",
            "Individual target-versus-best-temporal-neighbor panels are in "
            "figures/target_temporal_panels.",
            "",
            "## Boundary",
            "",
            "- 04A does not predict the 59 yields.",
            "- Neighbor yields are observed labels from the 138 training parcels only.",
            "- Checkpoint 04B must convert these diagnostics into pseudo-competition RMSE.",
            "",
        ]
    )
    return "\n".join(lines)


def run_checkpoint_04a(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Run 04A and write all diagnostic tables, figures and report."""

    inputs = config["inputs"]
    outputs = config["outputs"]
    identity = config["identity"]
    output_dir = root / str(outputs["directory"])
    figures_dir = output_dir / str(outputs["figures_directory"])
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    base_dir = root / str(inputs["base_directory"])
    agro_dir = root / str(inputs["agronomic_directory"])
    empirical_dir = root / str(inputs["empirical_directory"])

    print("[04A 1/8] Loading canonical parcel tables...")
    base = pd.read_csv(base_dir / str(inputs["base_table"]))
    agronomic = pd.read_csv(agro_dir / str(inputs["agronomic_table"]))
    empirical = pd.read_csv(empirical_dir / str(inputs["empirical_table"]))
    joined = join_agronomic_features(base, agronomic)
    joined = join_empirical_features(joined, empirical)
    _validate_contract(joined)

    manifests = {
        "base": _load_json(base_dir / str(inputs["base_manifest"])),
        "agronomic": _load_json(agro_dir / str(inputs["agronomic_manifest"])),
        "empirical": _load_json(empirical_dir / str(inputs["empirical_manifest"])),
    }

    print("[04A 2/8] Building X-only transductive PCA spaces over all 197 parcels...")
    policy = config["feature_policy"]
    representation_features: dict[str, tuple[str, ...]] = {}
    embeddings = {}
    embedding_manifest = []
    for name, spec in config["representations"].items():
        features = resolve_numeric_representation_features(
            joined=joined,
            manifests=manifests,
            layers=spec["layers"],
            allowed_modes=policy["allowed_modes"],
            exclude_sources=policy.get("exclude_sources", []),
            exclude_columns=policy.get("exclude_columns", []),
        )
        representation_features[name] = features
        embedding = build_transductive_pca_embedding(
            joined,
            name=name,
            features=features,
            n_components=int(config["embedding"]["pca_components"]),
        )
        embeddings[name] = embedding
        embedding_manifest.append(
            {
                "representation": name,
                "n_features_used": len(embedding.features_used),
                "n_components": embedding.n_components,
                "explained_variance_ratio_sum": embedding.explained_variance_ratio_sum,
            }
        )

    all_embeddings = []
    for name, embedding in embeddings.items():
        block = embedding.frame.copy()
        block.insert(0, "representation", name)
        all_embeddings.append(block)
    pd.concat(all_embeddings, ignore_index=True).to_csv(
        output_dir / str(outputs["embeddings"]), index=False
    )
    (output_dir / str(outputs["embedding_manifest"])).write_text(
        json.dumps(embedding_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("[04A 3/8] Geographic and multivariate nearest neighbors...")
    geo_target, geo_train, geo_matrix = geographic_neighbor_tables(
        joined,
        latitude_column=identity["latitude_column"],
        longitude_column=identity["longitude_column"],
        k=int(config["neighbors"]["k"]),
    )
    geo_target.insert(0, "neighbor_scope", "target_to_train")
    geo_train.insert(0, "neighbor_scope", "train_leave_one_out")
    pd.concat([geo_target, geo_train], ignore_index=True).to_csv(
        output_dir / str(outputs["geographic_neighbors"]), index=False
    )

    feature_target_parts = []
    feature_train_parts = []
    feature_matrices = {}
    for name, embedding in embeddings.items():
        target_neighbors, train_neighbors, matrix = feature_neighbor_tables(
            joined, embedding, k=int(config["neighbors"]["k"])
        )
        target_neighbors.insert(0, "neighbor_scope", "target_to_train")
        train_neighbors.insert(0, "neighbor_scope", "train_leave_one_out")
        feature_target_parts.append(target_neighbors)
        feature_train_parts.append(train_neighbors)
        feature_matrices[name] = matrix
    feature_target = pd.concat(feature_target_parts, ignore_index=True)
    feature_train = pd.concat(feature_train_parts, ignore_index=True)
    pd.concat([feature_target, feature_train], ignore_index=True).to_csv(
        output_dir / str(outputs["feature_neighbors"]), index=False
    )

    print("[04A 4/8] Monthly temporal similarity, lags and DTW...")
    temporal_cfg = config["temporal"]
    temporal_arrays = build_monthly_temporal_arrays(
        base, series=temporal_cfg["series"], months=temporal_cfg["months"]
    )
    temporal_pairs = temporal_pair_metrics(
        base,
        series_arrays=temporal_arrays,
        max_lag=int(temporal_cfg["max_lag_months"]),
        min_points=int(temporal_cfg["min_points"]),
    )
    temporal_pairs.to_csv(output_dir / str(outputs["temporal_pairs"]), index=False)
    temporal_target, temporal_train = temporal_neighbor_tables(
        temporal_pairs, k=int(config["neighbors"]["k"])
    )
    temporal_target.insert(0, "neighbor_scope", "target_to_train")
    temporal_train.insert(0, "neighbor_scope", "train_leave_one_out")
    pd.concat([temporal_target, temporal_train], ignore_index=True).to_csv(
        output_dir / str(outputs["temporal_neighbors"]), index=False
    )

    print("[04A 5/8] Testing whether closeness predicts smaller yield differences...")
    pair_validation = build_train_pair_validation(
        joined,
        geographic_distances=geo_matrix,
        feature_distances=feature_matrices,
        temporal_pairs=temporal_pairs,
    )
    pair_validation.to_csv(
        output_dir / str(outputs["train_pair_validation"]), index=False
    )
    similarity_summary = summarize_similarity_yield_relationships(pair_validation)
    similarity_summary.to_csv(
        output_dir / str(outputs["similarity_summary"]), index=False
    )

    print("[04A 6/8] Adversarial train-vs-target shift...")
    adversarial_rep = str(config["adversarial"]["representation"])
    adversarial_scores, adversarial_roc, adversarial_summary = (
        adversarial_train_target_validation(
            joined,
            embeddings[adversarial_rep],
            random_state=int(config["adversarial"]["random_state"]),
            folds=int(config["adversarial"]["folds"]),
        )
    )
    adversarial_scores.to_csv(
        output_dir / str(outputs["adversarial_scores"]), index=False
    )
    adversarial_roc.to_csv(output_dir / str(outputs["adversarial_roc"]), index=False)
    shift = feature_shift_table(
        joined, features=representation_features[adversarial_rep]
    )
    shift.to_csv(output_dir / str(outputs["adversarial_feature_shift"]), index=False)

    print("[04A 7/8] Spatial autocorrelation of yield and Checkpoint 03 residuals...")
    train_mask = joined[SPLIT_COLUMN].eq(TRAIN_VALUE).to_numpy()
    train_pos = np.flatnonzero(train_mask)
    weights = knn_weight_matrix(
        geo_matrix[np.ix_(train_pos, train_pos)],
        k=int(config["spatial"]["k_neighbors"]),
    )
    y = pd.to_numeric(joined.loc[train_mask, TARGET_COLUMN], errors="coerce").to_numpy()
    y_summary, y_centered, y_lag = morans_i(
        y,
        weights,
        permutations=int(config["spatial"]["permutations"]),
        random_state=int(config["spatial"]["random_state"]),
    )
    spatial_summary: dict[str, Any] = {"yield": y_summary}
    moran_payload = {"yield": (y_centered, y_lag, y_summary)}

    oof_path = root / str(inputs["checkpoint03_oof"])
    if oof_path.exists():
        oof = pd.read_csv(oof_path)
        train_ids = joined.loc[train_mask, ID_COLUMN].astype(str).tolist()
        for protocol in config["spatial"]["baseline_protocols"]:
            try:
                residuals = build_checkpoint03_ensemble_residuals(
                    oof, protocol=str(protocol)
                ).set_index(ID_COLUMN)
            except ValueError:
                continue
            for ensemble in ["E13", "E123"]:
                values = residuals.reindex(train_ids)[f"{ensemble}_residual"].to_numpy(float)
                if not np.isfinite(values).all():
                    continue
                summary, centered, lag = morans_i(
                    values,
                    weights,
                    permutations=int(config["spatial"]["permutations"]),
                    random_state=int(config["spatial"]["random_state"]),
                )
                key = f"{ensemble}_{protocol}_residual"
                spatial_summary[key] = summary
                moran_payload[key] = (centered, lag, summary)

    (output_dir / str(outputs["spatial_autocorrelation"])).write_text(
        json.dumps(spatial_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("[04A 8/8] Target profiles, figures and report...")
    primary_rep = str(config["embedding"]["primary_representation"])
    support = build_target_support_profile(
        joined,
        geographic_target_neighbors=geo_target,
        geographic_train_neighbors=geo_train,
        feature_target_neighbors=feature_target,
        feature_train_neighbors=feature_train,
        temporal_target_neighbors=temporal_target,
        temporal_train_neighbors=temporal_train,
        adversarial_scores=adversarial_scores,
        primary_representation=primary_rep,
        weights=config["support"]["weights"],
        high_threshold=float(config["support"]["high_threshold"]),
        low_threshold=float(config["support"]["low_threshold"]),
    )
    support.to_csv(output_dir / str(outputs["target_support_profile"]), index=False)

    dpi = int(config["figures"]["dpi"])
    plot_train_target_map(
        joined,
        figures_dir / "map_train_target.png",
        latitude_column=identity["latitude_column"],
        longitude_column=identity["longitude_column"],
        dpi=dpi,
    )
    plot_nearest_distance_distribution(
        geo_target,
        geo_train,
        figures_dir / "geographic_nn_distribution.png",
        xlabel="Nearest labeled parcel distance (km)",
        title="Geographic support: target versus labeled leave-one-out",
        dpi=dpi,
    )
    plot_pca_embedding(
        embeddings[primary_rep].frame,
        figures_dir / f"pca_{primary_rep}.png",
        title=f"{primary_rep}: transductive PCA over all 197 X rows",
        dpi=dpi,
    )
    plot_nearest_distance_distribution(
        feature_target.loc[feature_target["representation"].eq(primary_rep)],
        feature_train.loc[feature_train["representation"].eq(primary_rep)],
        figures_dir / "feature_nn_distribution.png",
        xlabel="Nearest labeled PCA distance",
        title=f"Multivariate support in {primary_rep}",
        dpi=dpi,
    )
    plot_adversarial_roc(
        adversarial_roc,
        float(adversarial_summary["auc"]),
        figures_dir / "adversarial_roc.png",
        dpi=dpi,
    )
    plot_top_feature_shift(
        shift,
        figures_dir / "adversarial_top_shift_features.png",
        top_n=int(config["adversarial"]["top_shift_features"]),
        dpi=dpi,
    )
    plot_temporal_similarity_distribution(
        temporal_target,
        temporal_train,
        figures_dir / "temporal_similarity_distribution.png",
        dpi=dpi,
    )
    plot_similarity_vs_yield_difference(
        pair_validation,
        figures_dir / "similarity_vs_yield_geo.png",
        metric="geo_distance_km",
        dpi=dpi,
    )
    plot_similarity_vs_yield_difference(
        pair_validation,
        figures_dir / "similarity_vs_yield_feature.png",
        metric=f"{primary_rep}__distance",
        dpi=dpi,
    )
    plot_similarity_vs_yield_difference(
        pair_validation,
        figures_dir / "similarity_vs_yield_temporal.png",
        metric="temporal_best_lag_corr_mean",
        dpi=dpi,
    )
    plot_moran_scatter(
        y_centered,
        y_lag,
        figures_dir / "moran_yield.png",
        title="Observed yield spatial autocorrelation",
        morans_i=float(y_summary["morans_i"]),
        p_value=float(y_summary["permutation_p_value_two_sided"]),
        dpi=dpi,
    )
    grouped_key = "E123_fold_municipality_grouped_residual"
    if grouped_key in moran_payload:
        centered, lag, summary = moran_payload[grouped_key]
        plot_moran_scatter(
            centered,
            lag,
            figures_dir / "moran_E123_grouped_residual.png",
            title="E123 municipality-grouped OOF residual autocorrelation",
            morans_i=float(summary["morans_i"]),
            p_value=float(summary["permutation_p_value_two_sided"]),
            dpi=dpi,
        )
    plot_support_ranking(
        support, figures_dir / "target_support_ranking.png", dpi=dpi
    )

    if bool(config["figures"].get("target_temporal_panels", True)):
        panel = temporal_cfg["panel_series"]
        plot_primary_temporal_neighbor_panels(
            base=base,
            temporal_neighbors=temporal_target,
            series_name=str(panel["name"]),
            prefix=str(panel["prefix"]),
            months=list(map(int, temporal_cfg["months"])),
            output_dir=figures_dir / "target_temporal_panels",
            dpi=max(120, dpi - 20),
        )

    report = {
        "schema_version": int(config["schema_version"]),
        "generated_at": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(root),
        "stage": str(config["stage"]["name"]),
        "active_track": "competition",
        "total_rows": int(len(joined)),
        "training_rows": int(train_mask.sum()),
        "prediction_rows": int((~train_mask).sum()),
        "hidden_prediction_targets_used": False,
        "target_covariates_used_transductively": True,
        "primary_representation": primary_rep,
        "representations": list(embeddings),
        "temporal_series_found": list(temporal_arrays),
        "train_pair_rows": int(len(pair_validation)),
        "target_train_temporal_pairs": int(
            temporal_pairs["pair_type"].eq("target_train").sum()
        ),
        "adversarial_auc": float(adversarial_summary["auc"]),
        "support_tier_counts": {
            str(key): int(value)
            for key, value in support["support_tier"].value_counts().items()
        },
    }
    (output_dir / str(outputs["run_report_json"])).write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    markdown = _render_report(
        report,
        support,
        similarity_summary,
        shift,
        spatial_summary,
        embedding_manifest,
    )
    (output_dir / str(outputs["run_report_markdown"])).write_text(
        markdown + "\n", encoding="utf-8"
    )

    print("Checkpoint 04A target topology: PASS")
    print(f"Parcels: 197 = {int(train_mask.sum())} labeled + {int((~train_mask).sum())} targets")
    print(f"Adversarial AUC: {float(adversarial_summary['auc']):.4f}")
    print(f"Temporal series found: {len(temporal_arrays)}")
    print(f"Train-pair comparisons: {len(pair_validation):,}")
    print(f"Inspect: {output_dir / str(outputs['run_report_markdown'])}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Checkpoint 04A transductive target diagnostics."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/checkpoint04a.yaml"),
    )
    parser.add_argument("--root", type=Path, default=None)
    args = parser.parse_args()

    root = (
        args.root.resolve()
        if args.root is not None
        else find_project_root(start=ROOT)
    )
    config_path = args.config if args.config.is_absolute() else root / args.config
    run_checkpoint_04a(root, _load_yaml(config_path))


if __name__ == "__main__":
    main()
