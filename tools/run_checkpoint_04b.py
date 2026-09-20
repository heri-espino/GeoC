#!/usr/bin/env python
"""Run Checkpoint 04B transductive pseudo-competition validation."""

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
from sklearn.impute import SimpleImputer
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

from geocebada.evaluation.checkpoint04a import (
    ID_COLUMN,
    PREDICTION_VALUE,
    SPLIT_COLUMN,
    TARGET_COLUMN,
    TRAIN_VALUE,
    haversine_distance_matrix,
    resolve_numeric_representation_features,
)
from geocebada.evaluation.checkpoint04b import (
    PseudoSplit,
    build_knn_graph,
    build_legacy_stress_splits,
    build_static_x_profile,
    generate_state_random_splits,
    generate_target_matched_splits,
    mixed_distance_matrix,
    normalized_distance_matrix,
    predict_global_mean,
    predict_global_pls,
    predict_global_ridge,
    predict_graph_laplacian,
    predict_knn,
    predict_local_ridge,
    predict_municipality_shrinkage,
    predict_state_mean,
    regression_metrics,
    split_x_support,
    standardized_profile_coordinates,
    summarize_method_results,
    summarize_support_tier_results,
    temporal_distance_matrix,
    temporal_similarity_matrix,
    transductive_standardize,
)
from geocebada.features import join_agronomic_features, join_empirical_features
from geocebada.paths import find_project_root
from geocebada.visualization.checkpoint04b import (
    plot_actual_method_routing,
    plot_actual_vs_pseudo_support,
    plot_primary_method_ranking,
    plot_split_match_quality,
    plot_support_tier_method_rmse,
    plot_target_support_routing_scatter,
    plot_top_method_rmse_boxplot,
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
        raise KeyError(f"04B missing required columns: {sorted(missing)}")
    if len(frame) != 197 or not frame[ID_COLUMN].is_unique:
        raise ValueError("04B requires exactly 197 unique parcels.")
    train = frame[SPLIT_COLUMN].eq(TRAIN_VALUE)
    target = frame[SPLIT_COLUMN].eq(PREDICTION_VALUE)
    if int(train.sum()) != 138 or int(target.sum()) != 59:
        raise ValueError("04B requires 138 labeled and 59 target parcels.")
    if frame.loc[train, TARGET_COLUMN].isna().any():
        raise ValueError("All 138 training yields must be observed.")
    if frame.loc[target, TARGET_COLUMN].notna().any():
        raise ValueError("The 59 hidden FIRA yields must remain missing.")


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
        raise ValueError(f"Embedding {representation} does not cover all 197 parcels.")
    return block.loc[:, pc_columns].to_numpy(float)


def _prepare_c0_transductive_matrix(
    joined: pd.DataFrame,
    *,
    manifests: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> tuple[np.ndarray, tuple[str, ...]]:
    policy = config["feature_policy"]
    features = resolve_numeric_representation_features(
        joined=joined,
        manifests=manifests,
        layers=config["representations"]["c0_layers"],
        allowed_modes=policy["allowed_modes"],
        exclude_sources=policy.get("exclude_sources", []),
        exclude_columns=policy.get("exclude_columns", []),
    )
    numeric = joined.loc[:, list(features)].apply(pd.to_numeric, errors="coerce")
    usable = [
        column
        for column in numeric.columns
        if numeric[column].notna().any() and numeric[column].nunique(dropna=True) > 1
    ]
    imputer = SimpleImputer(strategy="median", keep_empty_features=False)
    imputed = imputer.fit_transform(numeric.loc[:, usable])
    scaled = StandardScaler().fit_transform(imputed)
    return scaled, tuple(usable)


def _fmt_number(value: float) -> str:
    numeric = float(value)
    if numeric.is_integer():
        return str(int(numeric))
    return f"{numeric:g}".replace(".", "p")


def _split_membership(
    frame: pd.DataFrame,
    splits: list[PseudoSplit],
) -> pd.DataFrame:
    train_positions = set(
        np.flatnonzero(frame[SPLIT_COLUMN].eq(TRAIN_VALUE).to_numpy()).tolist()
    )
    rows: list[dict[str, Any]] = []
    for split in splits:
        pseudo = set(split.pseudo_target_positions)
        for position in sorted(train_positions):
            rows.append(
                {
                    "split_id": split.split_id,
                    "family": split.family,
                    "repeat": split.repeat,
                    ID_COLUMN: str(frame.iloc[position][ID_COLUMN]),
                    "meta_estado": str(frame.iloc[position]["meta_estado"]),
                    "meta_municipio": str(frame.iloc[position]["meta_municipio"]),
                    "role": "pseudo_target" if position in pseudo else "pseudo_train",
                }
            )
    return pd.DataFrame(rows)


def _split_quality_frame(splits: list[PseudoSplit]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "split_id": split.split_id,
                "family": split.family,
                "repeat": split.repeat,
                "n_pseudo_targets": len(split.pseudo_target_positions),
                "match_objective": split.match_objective,
                "municipality_tv": split.municipality_tv,
                "profile_mean_distance": split.profile_mean_distance,
            }
            for split in splits
        ]
    )


def _robustness_summary(protocol_summary: pd.DataFrame) -> pd.DataFrame:
    pivot = protocol_summary.pivot(
        index="method",
        columns="family",
        values="rmse_mean",
    ).reset_index()
    metric_columns = [column for column in pivot.columns if column != "method"]
    pivot["mean_family_rmse"] = pivot[metric_columns].mean(axis=1)
    pivot["worst_family_rmse"] = pivot[metric_columns].max(axis=1)
    return pivot.sort_values(
        ["worst_family_rmse", "mean_family_rmse", "method"]
    ).reset_index(drop=True)


def _routing_tables(
    support_tier_summary: pd.DataFrame,
    actual_support: pd.DataFrame,
    *,
    primary_family: str,
    tiers: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    route_rows: list[dict[str, Any]] = []
    for tier in tiers:
        block = support_tier_summary.loc[
            support_tier_summary["family"].eq(primary_family)
            & support_tier_summary["x_support_tier"].eq(tier)
        ].sort_values(["rmse", "mae", "method"])
        if block.empty:
            continue
        best = block.iloc[0]
        route_rows.append(
            {
                "x_support_tier": tier,
                "recommended_method": str(best["method"]),
                "pseudo_rmse": float(best["rmse"]),
                "pseudo_mae": float(best["mae"]),
                "n_predictions": int(best["n_predictions"]),
            }
        )
    tier_routing = pd.DataFrame(route_rows)
    method_by_tier = dict(
        zip(
            tier_routing["x_support_tier"],
            tier_routing["recommended_method"],
            strict=True,
        )
    )

    actual = actual_support.copy()
    actual["recommended_method"] = actual["x_support_tier"].map(method_by_tier)
    return tier_routing, actual


def _md_table(frame: pd.DataFrame, columns: list[str], *, max_rows: int | None = None) -> list[str]:
    subset = frame.loc[:, columns]
    if max_rows is not None:
        subset = subset.head(int(max_rows))
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join(["---"] * len(columns)) + "|",
    ]
    for row in subset.itertuples(index=False, name=None):
        values: list[str] = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                values.append("" if pd.isna(value) else f"{float(value):.4f}")
            else:
                values.append(str(value).replace("|", "/"))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def _render_report(
    report: dict[str, Any],
    split_quality: pd.DataFrame,
    protocol_summary: pd.DataFrame,
    robustness: pd.DataFrame,
    tier_routing: pd.DataFrame,
    actual_routing: pd.DataFrame,
) -> str:
    primary = str(report["primary_family"])
    primary_summary = (
        protocol_summary.loc[protocol_summary["family"].eq(primary)]
        .sort_values("rmse_mean")
        .reset_index(drop=True)
    )
    matched_quality = split_quality.loc[split_quality["family"].eq("target_matched")]
    random_quality = split_quality.loc[split_quality["family"].eq("state_random")]

    lines = [
        "# Checkpoint 04B — Transductive pseudo-competition validation",
        "",
        f"Generated: {report['generated_at']}",
        f"Git commit: {report.get('git_commit') or 'unknown'}",
        "",
        "## Validation contract",
        "",
        "Each pseudo-competition hides y for selected labeled parcels but leaves their X visible.",
        "All X-only PCA, distances, support calculations and graph topology may use the full 197-X universe.",
        "Every y-aware estimator sees only the pseudo-training labels.",
        "",
        f"Primary split family: **{primary}**",
        f"Pseudo-targets per matched/random split: **{report['n_pseudo_targets']}**",
        f"Target-matched repeats: **{report['target_matched_repeats']}**",
        f"State-random repeats: **{report['state_random_repeats']}**",
        f"Legacy stress splits: **{report['legacy_stress_splits']}**",
        "",
        "## Does target matching improve the pseudo-test geometry?",
        "",
    ]
    if not matched_quality.empty:
        lines.append(
            f"- target-matched mean municipality TV: "
            f"**{matched_quality['municipality_tv'].mean():.4f}**"
        )
        lines.append(
            f"- target-matched mean profile distance: "
            f"**{matched_quality['profile_mean_distance'].mean():.4f}**"
        )
    if not random_quality.empty:
        lines.append(
            f"- state-random mean municipality TV: "
            f"**{random_quality['municipality_tv'].mean():.4f}**"
        )
        lines.append(
            f"- state-random mean profile distance: "
            f"**{random_quality['profile_mean_distance'].mean():.4f}**"
        )

    lines.extend(
        [
            "",
            "## Primary target-matched ranking",
            "",
        ]
    )
    lines.extend(
        _md_table(
            primary_summary.head(15),
            [
                "method",
                "n_splits",
                "rmse_mean",
                "rmse_median",
                "rmse_std",
                "rmse_worst",
                "mae_mean",
                "r2_mean",
            ],
        )
    )

    lines.extend(["", "## Robustness across split families", ""])
    robustness_columns = [
        column
        for column in [
            "method",
            "target_matched",
            "state_random",
            "fold_state_stratified",
            "fold_municipality_grouped",
            "mean_family_rmse",
            "worst_family_rmse",
        ]
        if column in robustness.columns
    ]
    lines.extend(_md_table(robustness.head(15), robustness_columns))

    lines.extend(
        [
            "",
            "## X-only support-tier routing",
            "",
            (
                "Routing is learned only from pseudo-target performance. The actual 59 hidden "
                "y values are never inspected."
            ),
            "",
        ]
    )
    lines.extend(
        _md_table(
            tier_routing,
            [
                "x_support_tier",
                "recommended_method",
                "pseudo_rmse",
                "pseudo_mae",
                "n_predictions",
            ],
        )
    )

    counts = (
        actual_routing["recommended_method"]
        .value_counts(dropna=False)
        .rename_axis("recommended_method")
        .reset_index(name="n_actual_targets")
    )
    lines.extend(["", "### Routing proposal for the actual 59 targets", ""])
    lines.extend(_md_table(counts, ["recommended_method", "n_actual_targets"]))

    lines.extend(
        [
            "",
            "## Figures",
            "",
            "![Primary method ranking](figures/primary_method_ranking.png)",
            "",
            "![Top method dispersion](figures/top_method_rmse_boxplot.png)",
            "",
            "![Split matching quality](figures/split_match_quality.png)",
            "",
            "![Support-tier RMSE](figures/support_tier_method_rmse.png)",
            "",
            "![Actual vs pseudo support](figures/actual_vs_pseudo_support.png)",
            "",
            "![Actual routing counts](figures/actual_method_routing.png)",
            "",
            "![Actual target routing](figures/actual_target_support_routing.png)",
            "",
            "## Interpretation boundary",
            "",
            "- 04B ranks methods with observed pseudo-target y only after predictions are frozen.",
            "- Actual-target rows receive support/routing metadata only; no hidden yield is scored.",
            "- Target-matched RMSE is the primary fixed-target simulation; legacy state/municipality folds remain stress tests.",
            "- 04C may spend more compute only on globally competitive model families.",
            "- 04D should refine local/graph mixtures only when 04B demonstrates pseudo-test gain.",
            "",
        ]
    )
    return "\n".join(lines)


def run_checkpoint_04b(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Run repeated transductive pseudo-competition validation."""

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
    a04_dir = root / str(inputs["checkpoint04a_directory"])

    print("[04B 1/9] Loading canonical parcel tables and frozen 04A X-only artifacts...")
    base = pd.read_csv(base_dir / str(inputs["base_table"]))
    agronomic = pd.read_csv(agro_dir / str(inputs["agronomic_table"]))
    empirical = pd.read_csv(empirical_dir / str(inputs["empirical_table"]))
    joined = join_agronomic_features(base, agronomic)
    joined = join_empirical_features(joined, empirical)
    _validate_contract(joined)

    embeddings = pd.read_csv(a04_dir / str(inputs["checkpoint04a_embeddings"]))
    adversarial_scores = pd.read_csv(
        a04_dir / str(inputs["checkpoint04a_adversarial_scores"])
    )
    temporal_pairs = pd.read_csv(a04_dir / str(inputs["checkpoint04a_temporal_pairs"]))
    frozen_folds = pd.read_csv(root / str(inputs["frozen_folds"]))

    representation_cfg = config["representations"]
    c0_embedding = _embedding_matrix(embeddings, joined, representation_cfg["c0_name"])
    c1_embedding = _embedding_matrix(embeddings, joined, representation_cfg["c1_name"])

    print("[04B 2/9] Building transductive distance/support geometry...")
    latitude = pd.to_numeric(joined[identity["latitude_column"]], errors="raise")
    longitude = pd.to_numeric(joined[identity["longitude_column"]], errors="raise")
    geographic = haversine_distance_matrix(latitude, longitude)
    agronomic = pairwise_distances(c1_embedding, metric="euclidean")
    temporal_similarity = temporal_similarity_matrix(joined, temporal_pairs)

    train_positions = np.flatnonzero(joined[SPLIT_COLUMN].eq(TRAIN_VALUE).to_numpy())
    actual_target_positions = np.flatnonzero(
        joined[SPLIT_COLUMN].eq(PREDICTION_VALUE).to_numpy()
    )

    static_profile = build_static_x_profile(
        joined,
        geographic_distances=geographic,
        agronomic_distances=agronomic,
        temporal_similarities=temporal_similarity,
        adversarial_scores=adversarial_scores,
    )
    profile_coordinates = standardized_profile_coordinates(
        static_profile,
        columns=config["profile_matching"]["columns"],
    )

    geo_norm, geo_scale = normalized_distance_matrix(
        geographic, reference_positions=train_positions
    )
    agro_norm, agro_scale = normalized_distance_matrix(
        agronomic, reference_positions=train_positions
    )
    temporal_distance = temporal_distance_matrix(temporal_similarity)
    temporal_norm, temporal_scale = normalized_distance_matrix(
        temporal_distance, reference_positions=train_positions
    )

    mixed_distances: dict[str, np.ndarray] = {}
    for name, weights in config["distance"]["mixes"].items():
        mixed_distances[str(name)] = mixed_distance_matrix(
            [
                (geo_norm, float(weights.get("geographic", 0.0))),
                (agro_norm, float(weights.get("agronomic", 0.0))),
                (temporal_norm, float(weights.get("temporal", 0.0))),
            ]
        )

    print("[04B 3/9] Generating target-matched, random and frozen stress pseudo-splits...")
    split_cfg = config["split_design"]
    target_cfg = split_cfg["target_matched"]
    matched = generate_target_matched_splits(
        joined,
        profile_coordinates=profile_coordinates,
        n_pseudo_targets=int(split_cfg["n_pseudo_targets"]),
        repeats=int(target_cfg["repeats"]),
        candidates_per_repeat=int(target_cfg["candidates_per_repeat"]),
        temperature=float(target_cfg["temperature"]),
        municipality_smoothing=float(target_cfg["municipality_smoothing"]),
        diversity_penalty=float(target_cfg["diversity_penalty"]),
        random_state=int(split_cfg["random_state"]),
    )
    random_splits = generate_state_random_splits(
        joined,
        profile_coordinates=profile_coordinates,
        n_pseudo_targets=int(split_cfg["n_pseudo_targets"]),
        repeats=int(split_cfg["state_random"]["repeats"]),
        random_state=int(split_cfg["random_state"])
        + int(split_cfg["state_random"]["random_state_offset"]),
    )
    legacy = build_legacy_stress_splits(
        joined,
        frozen_folds,
        protocols=split_cfg["legacy_stress"]["protocols"],
        profile_coordinates=profile_coordinates,
    )
    splits = [*matched, *random_splits, *legacy]

    membership = _split_membership(joined, splits)
    membership.to_csv(
        output_dir / str(outputs["pseudo_split_membership"]),
        index=False,
    )
    split_quality = _split_quality_frame(splits)
    split_quality.to_csv(output_dir / str(outputs["split_quality"]), index=False)

    print("[04B 4/9] Computing actual-target X-only support without hidden y...")
    actual_support = split_x_support(
        joined,
        query_positions=actual_target_positions,
        observed_positions=train_positions,
        geographic_distances=geographic,
        agronomic_distances=agronomic,
        temporal_similarities=temporal_similarity,
        adversarial_scores=adversarial_scores,
        high_threshold=float(config["support"]["high_threshold"]),
        low_threshold=float(config["support"]["low_threshold"]),
    )
    actual_support.insert(
        2,
        "meta_estado",
        joined.iloc[actual_support["position"]]["meta_estado"].astype(str).to_numpy(),
    )
    actual_support.insert(
        3,
        "meta_municipio",
        joined.iloc[actual_support["position"]]["meta_municipio"].astype(str).to_numpy(),
    )
    actual_support.to_csv(
        output_dir / str(outputs["actual_target_x_support"]),
        index=False,
    )

    print("[04B 5/9] Preparing global and local model representations...")
    manifests = {
        "base": _load_json(base_dir / str(inputs["base_manifest"])),
        "agronomic": _load_json(agro_dir / str(inputs["agronomic_manifest"])),
        "empirical": _load_json(empirical_dir / str(inputs["empirical_manifest"])),
    }
    c0_x, c0_features = _prepare_c0_transductive_matrix(
        joined,
        manifests=manifests,
        config=config,
    )
    c1_x = transductive_standardize(c1_embedding)
    _ = c0_embedding

    graph_cfg = config["models"]["graph"]
    graph_distance = mixed_distances[str(graph_cfg["distance_name"])]
    graphs: dict[int, np.ndarray] = {}
    for variant in graph_cfg["variants"]:
        k = int(variant["k"])
        if k not in graphs:
            graphs[k] = build_knn_graph(graph_distance, k=k)

    y = pd.to_numeric(joined[TARGET_COLUMN], errors="coerce").to_numpy(float)
    train_set = set(map(int, train_positions))
    prediction_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []

    print(f"[04B 6/9] Evaluating {len(splits)} pseudo-competition splits...")
    for split_index, split in enumerate(splits, start=1):
        query = list(map(int, split.pseudo_target_positions))
        observed = sorted(train_set.difference(query))
        support = split_x_support(
            joined,
            query_positions=query,
            observed_positions=observed,
            geographic_distances=geographic,
            agronomic_distances=agronomic,
            temporal_similarities=temporal_similarity,
            adversarial_scores=adversarial_scores,
            high_threshold=float(config["support"]["high_threshold"]),
            low_threshold=float(config["support"]["low_threshold"]),
        ).set_index("position")

        predictions: dict[str, np.ndarray] = {}
        predictions["GlobalMean"] = predict_global_mean(y, observed, query)
        predictions["StateMean"] = predict_state_mean(joined, y, observed, query)

        for prior in config["models"]["municipality_shrinkage"]["prior_strengths"]:
            name = f"MunicipalityShrink_{_fmt_number(float(prior))}"
            predictions[name] = predict_municipality_shrinkage(
                joined,
                y,
                observed,
                query,
                prior_strength=float(prior),
            )

        for k in config["models"]["geo_knn"]["k_values"]:
            name = f"GeoKNN_{int(k)}"
            predictions[name] = predict_knn(
                geo_norm,
                y,
                observed,
                query,
                k=int(k),
                power=float(config["models"]["geo_knn"]["power"]),
            )

        for k in config["models"]["agronomic_knn"]["k_values"]:
            name = f"AgroKNN_{int(k)}"
            predictions[name] = predict_knn(
                agro_norm,
                y,
                observed,
                query,
                k=int(k),
                power=float(config["models"]["agronomic_knn"]["power"]),
            )

        mixed_cfg = config["models"]["mixed_knn"]
        for distance_name in mixed_cfg["distance_names"]:
            for k in mixed_cfg["k_values"]:
                name = f"{distance_name}_KNN{int(k)}"
                predictions[name] = predict_knn(
                    mixed_distances[str(distance_name)],
                    y,
                    observed,
                    query,
                    k=int(k),
                    power=float(mixed_cfg["power"]),
                )

        for alpha in config["models"]["ridge_c1"]["alphas"]:
            name = f"GlobalRidge_C1_{_fmt_number(float(alpha))}"
            predictions[name] = predict_global_ridge(
                c1_x,
                y,
                observed,
                query,
                alpha=float(alpha),
            )

        for components in config["models"]["pls_c0"]["n_components"]:
            name = f"GlobalPLS_C0_{int(components)}"
            predictions[name] = predict_global_pls(
                c0_x,
                y,
                observed,
                query,
                n_components=int(components),
            )

        local_cfg = config["models"]["local_ridge"]
        local_distance = mixed_distances[str(local_cfg["distance_name"])]
        for variant in local_cfg["variants"]:
            k = int(variant["k"])
            alpha = float(variant["alpha"])
            name = f"LocalRidge_k{k}_a{_fmt_number(alpha)}"
            predictions[name] = predict_local_ridge(
                c1_x,
                local_distance,
                y,
                observed,
                query,
                k=k,
                alpha=alpha,
            )

        for variant in graph_cfg["variants"]:
            k = int(variant["k"])
            regularization = float(variant["regularization"])
            name = f"Graph_k{k}_lam{_fmt_number(regularization)}"
            predictions[name] = predict_graph_laplacian(
                graphs[k],
                y,
                observed,
                query,
                regularization=regularization,
            )

        for blend in config["models"]["blends"]:
            members = list(map(str, blend["members"]))
            weights = np.asarray(blend["weights"], dtype=float)
            if len(members) != len(weights):
                raise ValueError(f"Blend {blend['name']} has incompatible members/weights.")
            missing_members = [member for member in members if member not in predictions]
            if missing_members:
                raise KeyError(
                    f"Blend {blend['name']} references missing methods: {missing_members}"
                )
            weights = weights / float(np.sum(weights))
            stacked = np.vstack([predictions[member] for member in members])
            predictions[str(blend["name"])] = np.average(
                stacked,
                axis=0,
                weights=weights,
            )

        observed_y = y[query]
        for method, predicted in predictions.items():
            metrics = regression_metrics(observed_y, predicted)
            metric_rows.append(
                {
                    "split_id": split.split_id,
                    "family": split.family,
                    "repeat": split.repeat,
                    "method": method,
                    "n_train": len(observed),
                    "n_pseudo_target": len(query),
                    "match_objective": split.match_objective,
                    "municipality_tv": split.municipality_tv,
                    "profile_mean_distance": split.profile_mean_distance,
                    **metrics,
                }
            )
            for local_index, position in enumerate(query):
                support_row = support.loc[position]
                prediction_rows.append(
                    {
                        "split_id": split.split_id,
                        "family": split.family,
                        "repeat": split.repeat,
                        "method": method,
                        ID_COLUMN: str(joined.iloc[position][ID_COLUMN]),
                        "meta_estado": str(joined.iloc[position]["meta_estado"]),
                        "meta_municipio": str(joined.iloc[position]["meta_municipio"]),
                        "observed": float(y[position]),
                        "predicted": float(predicted[local_index]),
                        "error": float(y[position] - predicted[local_index]),
                        "x_support_score": float(support_row["x_support_score"]),
                        "x_support_tier": str(support_row["x_support_tier"]),
                    }
                )

        print(
            f"  [{split_index:02d}/{len(splits):02d}] {split.split_id}: "
            f"{len(observed)} train / {len(query)} pseudo-target"
        )

    predictions_frame = pd.DataFrame(prediction_rows)
    split_metrics = pd.DataFrame(metric_rows)
    predictions_frame.to_csv(
        output_dir / str(outputs["pseudo_predictions"]),
        index=False,
    )
    split_metrics.to_csv(output_dir / str(outputs["split_metrics"]), index=False)

    print("[04B 7/9] Aggregating method rankings and support-tier behavior...")
    protocol_summary = summarize_method_results(split_metrics)
    protocol_summary.to_csv(
        output_dir / str(outputs["protocol_summary"]),
        index=False,
    )
    tier_summary = summarize_support_tier_results(predictions_frame)
    tier_summary.to_csv(
        output_dir / str(outputs["support_tier_summary"]),
        index=False,
    )
    robustness = _robustness_summary(protocol_summary)
    robustness.to_csv(
        output_dir / str(outputs["robustness_summary"]),
        index=False,
    )

    tier_routing, actual_routing = _routing_tables(
        tier_summary,
        actual_support,
        primary_family=str(config["routing"]["primary_family"]),
        tiers=list(map(str, config["routing"]["support_tiers"])),
    )
    tier_routing.to_csv(
        output_dir / str(outputs["support_tier_method_routing"]),
        index=False,
    )
    actual_routing.to_csv(
        output_dir / str(outputs["actual_target_method_routing"]),
        index=False,
    )

    print("[04B 8/9] Writing diagnostic figures...")
    dpi = int(config["figures"]["dpi"])
    top_n = int(config["figures"]["top_methods"])
    plot_primary_method_ranking(
        protocol_summary,
        figures_dir / "primary_method_ranking.png",
        family=str(config["routing"]["primary_family"]),
        top_n=top_n,
        dpi=dpi,
    )
    plot_top_method_rmse_boxplot(
        split_metrics,
        protocol_summary,
        figures_dir / "top_method_rmse_boxplot.png",
        family=str(config["routing"]["primary_family"]),
        top_n=min(8, top_n),
        dpi=dpi,
    )
    plot_split_match_quality(
        split_quality,
        figures_dir / "split_match_quality.png",
        dpi=dpi,
    )
    plot_support_tier_method_rmse(
        tier_summary,
        protocol_summary,
        figures_dir / "support_tier_method_rmse.png",
        family=str(config["routing"]["primary_family"]),
        top_n=min(8, top_n),
        dpi=dpi,
    )
    plot_actual_vs_pseudo_support(
        actual_support,
        predictions_frame,
        figures_dir / "actual_vs_pseudo_support.png",
        dpi=dpi,
    )
    plot_actual_method_routing(
        actual_routing,
        figures_dir / "actual_method_routing.png",
        dpi=dpi,
    )
    plot_target_support_routing_scatter(
        actual_routing,
        figures_dir / "actual_target_support_routing.png",
        dpi=dpi,
    )

    print("[04B 9/9] Writing canonical report and machine-readable run summary...")
    primary_family = str(config["routing"]["primary_family"])
    primary_summary = protocol_summary.loc[
        protocol_summary["family"].eq(primary_family)
    ].sort_values("rmse_mean")
    best_primary = primary_summary.iloc[0]

    report = {
        "schema_version": int(config["schema_version"]),
        "generated_at": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(root),
        "stage": str(config["stage"]["name"]),
        "active_track": "competition",
        "hidden_prediction_targets_used": False,
        "target_covariates_used_transductively": True,
        "total_rows": int(len(joined)),
        "training_rows": int(len(train_positions)),
        "prediction_rows": int(len(actual_target_positions)),
        "n_pseudo_targets": int(split_cfg["n_pseudo_targets"]),
        "target_matched_repeats": int(target_cfg["repeats"]),
        "state_random_repeats": int(split_cfg["state_random"]["repeats"]),
        "legacy_stress_splits": int(len(legacy)),
        "total_splits": int(len(splits)),
        "methods_evaluated": int(split_metrics["method"].nunique()),
        "primary_family": primary_family,
        "best_primary_method": str(best_primary["method"]),
        "best_primary_rmse_mean": float(best_primary["rmse_mean"]),
        "best_primary_rmse_median": float(best_primary["rmse_median"]),
        "best_primary_rmse_worst": float(best_primary["rmse_worst"]),
        "distance_scales": {
            "geographic_nearest_median_km": geo_scale,
            "agronomic_nearest_median": agro_scale,
            "temporal_distance_nearest_median": temporal_scale,
        },
        "c0_features_used": int(len(c0_features)),
        "actual_target_x_support_counts": {
            str(key): int(value)
            for key, value in actual_support["x_support_tier"].value_counts().items()
        },
    }
    (output_dir / str(outputs["run_report_json"])).write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown = _render_report(
        report,
        split_quality,
        protocol_summary,
        robustness,
        tier_routing,
        actual_routing,
    )
    (output_dir / str(outputs["run_report_markdown"])).write_text(
        markdown + "\n",
        encoding="utf-8",
    )

    print("Checkpoint 04B transductive pseudo-competition: PASS")
    print(f"Splits: {len(splits)}")
    print(f"Methods: {split_metrics['method'].nunique()}")
    print(
        f"Best target-matched method: {report['best_primary_method']} "
        f"RMSE={report['best_primary_rmse_mean']:.4f}"
    )
    print(f"Inspect: {output_dir / str(outputs['run_report_markdown'])}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Checkpoint 04B transductive pseudo-competition validation."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/checkpoint04b.yaml"),
    )
    parser.add_argument("--root", type=Path, default=None)
    args = parser.parse_args()

    root = (
        args.root.resolve()
        if args.root is not None
        else find_project_root(start=ROOT)
    )
    config_path = args.config if args.config.is_absolute() else root / args.config
    run_checkpoint_04b(root, _load_yaml(config_path))


if __name__ == "__main__":
    main()
