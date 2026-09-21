#!/usr/bin/env python
"""Run Checkpoint 04D.1 local/graph refinement."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
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
    build_knn_graph,
    mixed_distance_matrix,
    normalized_distance_matrix,
    predict_graph_laplacian,
    predict_knn,
    regression_metrics,
)
from geocebada.evaluation.checkpoint04d1 import (
    SplitPositions,
    fit_pls_anchor_with_cross_fitted_residuals,
    leave_one_split_out_method_selection,
    leave_one_split_out_tier_routing,
    predict_residual_graph_correction,
    predict_weighted_local_ridge,
    split_positions_from_membership,
    summarize_nested_predictions,
    summarize_predictions_by_method,
)
from geocebada.paths import find_project_root
from geocebada.visualization.checkpoint04d1 import (
    plot_actual_candidate_spread,
    plot_graph_k_profiles,
    plot_local_k_profile,
    plot_loso_selection_frequency,
    plot_nested_validation_comparison,
    plot_refinement_ranking,
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
        "base_centroid_lat",
        "base_centroid_lon",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"04D.1 missing required columns: {sorted(missing)}")
    if len(frame) != 197 or not frame[ID_COLUMN].is_unique:
        raise ValueError("04D.1 requires exactly 197 unique parcels.")
    train = frame[SPLIT_COLUMN].eq(TRAIN_VALUE)
    target = frame[SPLIT_COLUMN].eq(PREDICTION_VALUE)
    if int(train.sum()) != 138 or int(target.sum()) != 59:
        raise ValueError("04D.1 requires 138 labeled and 59 fixed target parcels.")
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
    frame: pd.DataFrame,
    *,
    manifest: dict[str, Any],
    config: dict[str, Any],
) -> tuple[np.ndarray, tuple[str, ...]]:
    policy = config["feature_policy"]
    features = resolve_numeric_representation_features(
        joined=frame,
        manifests={"base": manifest},
        layers=config["representations"]["c0_layers"],
        allowed_modes=policy["allowed_modes"],
        exclude_sources=policy.get("exclude_sources", []),
        exclude_columns=policy.get("exclude_columns", []),
    )
    numeric = frame.loc[:, list(features)].apply(pd.to_numeric, errors="coerce")
    usable = [
        column
        for column in numeric.columns
        if numeric[column].notna().any() and numeric[column].nunique(dropna=True) > 1
    ]
    imputed = SimpleImputer(
        strategy="median",
        keep_empty_features=False,
    ).fit_transform(numeric.loc[:, usable])
    return StandardScaler().fit_transform(imputed), tuple(usable)


def _fmt(value: float | int) -> str:
    numeric = float(value)
    if numeric.is_integer():
        return str(int(numeric))
    return f"{numeric:g}".replace(".", "p")


def _build_method_manifest(config: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = [
        {
            "method": "Baseline_GeoKNN10",
            "method_family": "baseline",
            "kind": "geo_knn",
            "representation": "",
            "distance_name": str(config["baselines"]["geo_knn"]["distance_name"]),
            "k": int(config["baselines"]["geo_knn"]["k"]),
            "alpha": np.nan,
            "distance_power": float(config["baselines"]["geo_knn"]["power"]),
            "regularization": np.nan,
        },
        {
            "method": "Anchor_PLS4",
            "method_family": "baseline",
            "kind": "pls_anchor",
            "representation": "C0_base",
            "distance_name": "",
            "k": np.nan,
            "alpha": np.nan,
            "distance_power": np.nan,
            "regularization": np.nan,
        },
    ]

    for distance_name in config["distance_topologies"]:
        for k in config["graph_search"]["k_values"]:
            for regularization in config["graph_search"]["regularization_values"]:
                for mode in config["graph_search"]["modes"]:
                    residual = str(mode) == "residual_pls4"
                    family = "graph_residual" if residual else "graph_direct"
                    prefix = "GraphResidualPLS4" if residual else "GraphDirect"
                    rows.append(
                        {
                            "method": (
                                f"{prefix}_{distance_name}_k{int(k)}_"
                                f"lam{_fmt(float(regularization))}"
                            ),
                            "method_family": family,
                            "kind": family,
                            "representation": "C0_base" if residual else "",
                            "distance_name": str(distance_name),
                            "k": int(k),
                            "alpha": np.nan,
                            "distance_power": np.nan,
                            "regularization": float(regularization),
                        }
                    )

    local_cfg = config["local_ridge_search"]
    for representation in local_cfg["representations"]:
        for distance_name in local_cfg["distance_names"]:
            for k in local_cfg["k_values"]:
                for alpha in local_cfg["alpha_values"]:
                    for power in local_cfg["distance_powers"]:
                        rows.append(
                            {
                                "method": (
                                    f"LocalRidge_{representation}_{distance_name}_"
                                    f"k{int(k)}_a{_fmt(float(alpha))}_p{_fmt(float(power))}"
                                ),
                                "method_family": "local",
                                "kind": "local",
                                "representation": str(representation),
                                "distance_name": str(distance_name),
                                "k": int(k),
                                "alpha": float(alpha),
                                "distance_power": float(power),
                                "regularization": np.nan,
                            }
                        )
    manifest = pd.DataFrame(rows)
    if manifest["method"].duplicated().any():
        raise ValueError("04D.1 method names must be unique.")
    return manifest


def _build_distance_matrices(
    config: dict[str, Any],
    geographic: np.ndarray,
    agronomic: np.ndarray,
    train_positions: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    geo_norm, geo_scale = normalized_distance_matrix(
        geographic,
        reference_positions=train_positions,
    )
    agro_norm, agro_scale = normalized_distance_matrix(
        agronomic,
        reference_positions=train_positions,
    )
    matrices: dict[str, np.ndarray] = {}
    for name, weights in config["distance_topologies"].items():
        matrices[str(name)] = mixed_distance_matrix(
            [
                (geo_norm, float(weights["geographic"])),
                (agro_norm, float(weights["agronomic"])),
            ]
        )
    return matrices, {
        "geographic_nearest_median_km": geo_scale,
        "agronomic_nearest_median": agro_scale,
    }


def _support_lookup(pseudo_predictions: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "split_id",
        ID_COLUMN,
        "x_support_score",
        "x_support_tier",
    ]
    return pseudo_predictions.loc[:, columns].drop_duplicates().reset_index(drop=True)


def _support_for_split(
    lookup: pd.DataFrame,
    split: SplitPositions,
    frame: pd.DataFrame,
) -> pd.DataFrame:
    ids = frame.iloc[list(split.query_positions)][ID_COLUMN].astype(str)
    block = (
        lookup.loc[
            lookup["split_id"].eq(split.split_id)
            & lookup[ID_COLUMN].astype(str).isin(ids),
            [
                ID_COLUMN,
                "x_support_score",
                "x_support_tier",
            ],
        ]
        .drop_duplicates(ID_COLUMN)
        .set_index(ID_COLUMN)
    )
    missing = set(ids).difference(block.index.astype(str))
    if missing:
        raise KeyError(f"04B support lookup missing split {split.split_id}: {sorted(missing)}")
    return block


def _graph_cache(
    manifest: pd.DataFrame,
    distances: dict[str, np.ndarray],
) -> dict[tuple[str, int], np.ndarray]:
    pairs = (
        manifest.loc[
            manifest["method_family"].isin(["graph_direct", "graph_residual"]),
            ["distance_name", "k"],
        ]
        .drop_duplicates()
        .itertuples(index=False)
    )
    result: dict[tuple[str, int], np.ndarray] = {}
    for distance_name, k in pairs:
        key = (str(distance_name), int(k))
        result[key] = build_knn_graph(
            distances[str(distance_name)],
            k=int(k),
        )
    return result


def _predict_one(
    spec: pd.Series,
    *,
    split: SplitPositions,
    y: np.ndarray,
    x_by_representation: dict[str, np.ndarray],
    distances: dict[str, np.ndarray],
    graphs: dict[tuple[str, int], np.ndarray],
    anchor_predictions: np.ndarray,
    anchor_residuals: np.ndarray,
) -> np.ndarray:
    observed = split.observed_positions
    query = split.query_positions
    kind = str(spec["kind"])

    if kind == "geo_knn":
        return predict_knn(
            distances[str(spec["distance_name"])],
            y,
            observed,
            query,
            k=int(spec["k"]),
            power=float(spec["distance_power"]),
        )
    if kind == "pls_anchor":
        return np.asarray(anchor_predictions, dtype=float)[list(query)]
    if kind == "graph_direct":
        adjacency = graphs[(str(spec["distance_name"]), int(spec["k"]))]
        return predict_graph_laplacian(
            adjacency,
            y,
            observed,
            query,
            regularization=float(spec["regularization"]),
        )
    if kind == "graph_residual":
        adjacency = graphs[(str(spec["distance_name"]), int(spec["k"]))]
        return predict_residual_graph_correction(
            anchor_predictions,
            anchor_residuals,
            adjacency,
            observed,
            query,
            regularization=float(spec["regularization"]),
        )
    if kind == "local":
        return predict_weighted_local_ridge(
            x_by_representation[str(spec["representation"])],
            distances[str(spec["distance_name"])],
            y,
            observed,
            query,
            k=int(spec["k"]),
            alpha=float(spec["alpha"]),
            distance_power=float(spec["distance_power"]),
        )
    raise ValueError(f"Unsupported 04D.1 method kind: {kind}")


def _prediction_rows(
    split: SplitPositions,
    frame: pd.DataFrame,
    y: np.ndarray,
    spec: pd.Series,
    predicted: np.ndarray,
    support: pd.DataFrame,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, position in enumerate(split.query_positions):
        parcel_id = str(frame.iloc[position][ID_COLUMN])
        support_row = support.loc[parcel_id]
        rows.append(
            {
                "split_id": split.split_id,
                "family": split.family,
                "repeat": split.repeat,
                "method": str(spec["method"]),
                "method_family": str(spec["method_family"]),
                ID_COLUMN: parcel_id,
                "observed": float(y[position]),
                "predicted": float(predicted[index]),
                "error": float(y[position] - predicted[index]),
                "x_support_score": float(support_row["x_support_score"]),
                "x_support_tier": str(support_row["x_support_tier"]),
            }
        )
    return rows


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


def _select_finalists(
    summary: pd.DataFrame,
    manifest: pd.DataFrame,
    config: dict[str, Any],
) -> list[str]:
    merged = summary.merge(
        manifest[["method", "method_family"]],
        on="method",
        how="left",
    )
    count = int(config["finalists"]["top_per_family"])
    selected: list[str] = []
    for family in ["graph_direct", "graph_residual", "local"]:
        selected.extend(
            merged.loc[merged["method_family"].eq(family)]
            .sort_values(["rmse_mean", "rmse_worst", "method"])
            .head(count)["method"]
            .astype(str)
            .tolist()
        )
    selected.extend(map(str, config["finalists"]["always_include"]))
    return list(dict.fromkeys(selected))


def _robustness_table(
    primary_summary: pd.DataFrame,
    stress_summary: pd.DataFrame,
    finalists: list[str],
) -> pd.DataFrame:
    combined = pd.concat(
        [
            primary_summary.loc[primary_summary["method"].isin(finalists)],
            stress_summary.loc[stress_summary["method"].isin(finalists)],
        ],
        ignore_index=True,
    )
    pivot = combined.pivot(
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


def _fit_tier_route(
    primary_predictions: pd.DataFrame,
    finalists: list[str],
    tiers: list[str],
) -> pd.DataFrame:
    frame = primary_predictions.loc[
        primary_predictions["method"].isin(finalists)
    ].copy()
    frame["squared_error"] = np.square(frame["observed"] - frame["predicted"])
    split_tier = (
        frame.groupby(
            ["split_id", "x_support_tier", "method"],
            as_index=False,
        )["squared_error"]
        .mean()
        .assign(rmse=lambda table: np.sqrt(table["squared_error"]))
    )

    rows: list[dict[str, Any]] = []
    for tier in tiers:
        ranking = (
            split_tier.loc[split_tier["x_support_tier"].eq(tier)]
            .groupby("method", as_index=False)["rmse"]
            .mean()
            .sort_values(["rmse", "method"])
        )
        if ranking.empty:
            continue
        rows.append(
            {
                "x_support_tier": str(tier),
                "selected_method": str(ranking.iloc[0]["method"]),
                "selection_rmse": float(ranking.iloc[0]["rmse"]),
            }
        )
    return pd.DataFrame(rows)


def _actual_candidate_rows(
    frame: pd.DataFrame,
    actual_support: pd.DataFrame,
    method_manifest: pd.DataFrame,
    finalists: list[str],
    *,
    y: np.ndarray,
    train_positions: np.ndarray,
    target_positions: np.ndarray,
    x_by_representation: dict[str, np.ndarray],
    distances: dict[str, np.ndarray],
    graphs: dict[tuple[str, int], np.ndarray],
    anchor_predictions: np.ndarray,
    anchor_residuals: np.ndarray,
) -> pd.DataFrame:
    support = actual_support.set_index(ID_COLUMN)
    split = SplitPositions(
        split_id="actual_59",
        family="actual_target",
        repeat=0,
        observed_positions=tuple(map(int, train_positions)),
        query_positions=tuple(map(int, target_positions)),
    )
    rows: list[dict[str, Any]] = []
    for method in finalists:
        spec = method_manifest.loc[method_manifest["method"].eq(method)].iloc[0]
        predicted = _predict_one(
            spec,
            split=split,
            y=y,
            x_by_representation=x_by_representation,
            distances=distances,
            graphs=graphs,
            anchor_predictions=anchor_predictions,
            anchor_residuals=anchor_residuals,
        )
        for index, position in enumerate(target_positions):
            parcel_id = str(frame.iloc[position][ID_COLUMN])
            support_row = support.loc[parcel_id]
            rows.append(
                {
                    ID_COLUMN: parcel_id,
                    "method": method,
                    "method_family": str(spec["method_family"]),
                    "predicted": float(predicted[index]),
                    "x_support_score": float(support_row["x_support_score"]),
                    "x_support_tier": str(support_row["x_support_tier"]),
                }
            )
    return pd.DataFrame(rows)


def _actual_routing(
    candidates: pd.DataFrame,
    route: pd.DataFrame,
) -> pd.DataFrame:
    method_by_tier = dict(
        zip(
            route["x_support_tier"],
            route["selected_method"],
            strict=True,
        )
    )
    rows: list[dict[str, Any]] = []
    for parcel_id, group in candidates.groupby(ID_COLUMN, sort=False):
        tier = str(group.iloc[0]["x_support_tier"])
        method = method_by_tier.get(tier)
        if method is None:
            continue
        selected = group.loc[group["method"].eq(method)]
        if selected.empty:
            raise KeyError(f"Routing method {method!r} missing for target {parcel_id}.")
        row = selected.iloc[0]
        rows.append(
            {
                ID_COLUMN: str(parcel_id),
                "x_support_score": float(row["x_support_score"]),
                "x_support_tier": tier,
                "selected_method": method,
                "candidate_prediction": float(row["predicted"]),
            }
        )
    return pd.DataFrame(rows)


def _md_table(
    frame: pd.DataFrame,
    columns: list[str],
    *,
    max_rows: int | None = None,
) -> list[str]:
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
    primary_summary: pd.DataFrame,
    finalist_manifest: pd.DataFrame,
    robustness: pd.DataFrame,
    nested_summary: pd.DataFrame,
    tier_route: pd.DataFrame,
) -> str:
    lines = [
        "# Checkpoint 04D.1 — Local/graph refinement",
        "",
        f"Generated: {report['generated_at']}",
        f"Git commit: {report.get('git_commit') or 'unknown'}",
        "",
        "## Contract",
        "",
        "04D.1 reuses the exact committed 04B pseudo-target memberships.",
        "All topology and distance construction is X-only and may use the full 197-X universe.",
        "Every y-aware fit sees only the labels visible in the corresponding pseudo split.",
        "",
        "## Primary target-matched search",
        "",
    ]
    lines.extend(
        _md_table(
            primary_summary,
            [
                "method",
                "method_family",
                "rmse_mean",
                "rmse_median",
                "rmse_std",
                "rmse_worst",
                "pooled_mae",
            ],
            max_rows=20,
        )
    )

    lines.extend(["", "## Finalists and stress robustness", ""])
    lines.extend(
        _md_table(
            robustness,
            [
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
            ],
            max_rows=20,
        )
    )

    lines.extend(["", "## Leave-one-split-out validation", ""])
    lines.extend(
        _md_table(
            nested_summary,
            [
                "label",
                "n_splits",
                "n_predictions",
                "rmse_mean",
                "rmse_median",
                "rmse_worst",
                "pooled_rmse",
                "pooled_mae",
            ],
        )
    )

    lines.extend(
        [
            "",
            "The LOSO procedure chooses hyperparameters or support-tier routes using the other",
            "target-matched splits and scores the held-out split only afterward. Because repeated",
            "pseudo-splits reuse parcels, this removes direct same-split tuning optimism but "
            "is not",
            "equivalent to 16 statistically independent experiments.",
            "",
            "## Exploratory route fitted on all primary splits",
            "",
        ]
    )
    lines.extend(
        _md_table(
            tier_route,
            ["x_support_tier", "selected_method", "selection_rmse"],
        )
    )

    lines.extend(
        [
            "",
            "This all-split route is a candidate for later reconstruction, not a hidden-y result.",
            "",
            "## Finalist manifest",
            "",
        ]
    )
    lines.extend(
        _md_table(
            finalist_manifest,
            [
                "method",
                "method_family",
                "representation",
                "distance_name",
                "k",
                "alpha",
                "distance_power",
                "regularization",
            ],
        )
    )

    lines.extend(
        [
            "",
            "## Figures",
            "",
            "![Refinement ranking](figures/refinement_ranking.png)",
            "",
            "![Graph k profiles](figures/graph_k_profiles.png)",
            "",
            "![Local k profile](figures/local_k_profile.png)",
            "",
            "![Nested validation](figures/nested_validation_comparison.png)",
            "",
            "![LOSO selection frequency](figures/loso_selection_frequency.png)",
            "",
            "![Actual candidate spread](figures/actual_candidate_spread.png)",
            "",
            "## Interpretation boundary",
            "",
            "- No hidden FIRA yield is read or scored.",
            "- Actual-target predictions are candidate predictions from validated finalists.",
            "- They are not the final 04F submission.",
            "- SIAP/external localization remains a separate 04E hypothesis.",
            "",
        ]
    )
    return "\n".join(lines)


def run_checkpoint_04d1(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Run exhaustive target-matched local/graph refinement and nested validation."""

    inputs = config["inputs"]
    outputs = config["outputs"]
    output_dir = root / str(outputs["directory"])
    figures_dir = output_dir / str(outputs["figures_directory"])
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("[04D.1 1/8] Loading frozen parcel, embedding and 04B validation artifacts...")
    base_dir = root / str(inputs["base_directory"])
    a04_dir = root / str(inputs["checkpoint04a_directory"])
    b04_dir = root / str(inputs["checkpoint04b_directory"])

    frame = pd.read_csv(base_dir / str(inputs["base_table"]))
    _validate_contract(frame)
    manifest = _load_json(base_dir / str(inputs["base_manifest"]))
    embeddings = pd.read_csv(a04_dir / str(inputs["checkpoint04a_embeddings"]))
    membership = pd.read_csv(b04_dir / str(inputs["checkpoint04b_membership"]))
    pseudo_predictions_04b = pd.read_csv(
        b04_dir / str(inputs["checkpoint04b_pseudo_predictions"]),
        usecols=[
            "split_id",
            ID_COLUMN,
            "x_support_score",
            "x_support_tier",
        ],
    )
    actual_support = pd.read_csv(b04_dir / str(inputs["checkpoint04b_support"]))

    c1 = _embedding_matrix(
        embeddings,
        frame,
        str(config["representations"]["c1_name"]),
    )
    c4 = _embedding_matrix(
        embeddings,
        frame,
        str(config["representations"]["c4_name"]),
    )
    c0, c0_features = _prepare_c0_transductive_matrix(
        frame,
        manifest=manifest,
        config=config,
    )
    x_by_representation = {
        "C1_agronomic": StandardScaler().fit_transform(c1),
        "C4_all_deterministic": StandardScaler().fit_transform(c4),
    }

    print("[04D.1 2/8] Building geographic/agronomic distance topologies and graph cache...")
    latitude = pd.to_numeric(frame["base_centroid_lat"], errors="raise")
    longitude = pd.to_numeric(frame["base_centroid_lon"], errors="raise")
    geographic = haversine_distance_matrix(latitude, longitude)
    agronomic = pairwise_distances(c1, metric="euclidean")
    train_positions = np.flatnonzero(frame[SPLIT_COLUMN].eq(TRAIN_VALUE).to_numpy())
    target_positions = np.flatnonzero(frame[SPLIT_COLUMN].eq(PREDICTION_VALUE).to_numpy())
    distances, distance_scales = _build_distance_matrices(
        config,
        geographic,
        agronomic,
        train_positions,
    )
    method_manifest = _build_method_manifest(config)
    graphs = _graph_cache(method_manifest, distances)
    method_manifest.to_csv(output_dir / "method_manifest.csv", index=False)

    print("[04D.1 3/8] Resolving the exact committed 04B pseudo-competition splits...")
    splits = split_positions_from_membership(
        frame,
        membership,
        train_value=TRAIN_VALUE,
        split_column=SPLIT_COLUMN,
    )
    primary_family = str(config["validation"]["primary_family"])
    primary_splits = [split for split in splits if split.family == primary_family]
    stress_families = set(map(str, config["validation"]["stress_families"]))
    stress_splits = [split for split in splits if split.family in stress_families]
    if len(primary_splits) != 16:
        raise ValueError(f"Expected 16 target-matched splits, found {len(primary_splits)}.")

    support_lookup = _support_lookup(pseudo_predictions_04b)
    y = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce").to_numpy(float)
    strata = frame["meta_estado"].astype(str).to_numpy()

    print(
        f"[04D.1 4/8] Exhaustive primary search: {len(method_manifest)} methods "
        f"x {len(primary_splits)} target-matched splits..."
    )
    primary_rows: list[dict[str, Any]] = []
    start = perf_counter()
    for index, split in enumerate(primary_splits, start=1):
        anchor_predictions, anchor_residuals = fit_pls_anchor_with_cross_fitted_residuals(
            c0,
            y,
            split.observed_positions,
            strata=strata,
            n_components=int(config["baselines"]["pls_anchor"]["n_components"]),
            n_splits=int(config["validation"]["anchor_crossfit_folds"]),
            random_state=int(config["validation"]["random_state"]) + index,
        )
        support = _support_for_split(support_lookup, split, frame)
        for spec in method_manifest.itertuples(index=False):
            spec_series = pd.Series(spec._asdict())
            predicted = _predict_one(
                spec_series,
                split=split,
                y=y,
                x_by_representation=x_by_representation,
                distances=distances,
                graphs=graphs,
                anchor_predictions=anchor_predictions,
                anchor_residuals=anchor_residuals,
            )
            primary_rows.extend(
                _prediction_rows(
                    split,
                    frame,
                    y,
                    spec_series,
                    predicted,
                    support,
                )
            )
        print(
            f"  [{index:02d}/{len(primary_splits):02d}] {split.split_id} complete "
            f"({perf_counter() - start:.1f}s elapsed)",
            flush=True,
        )

    primary_predictions = pd.DataFrame(primary_rows)
    primary_predictions.to_csv(
        output_dir / str(outputs["primary_predictions"]),
        index=False,
    )
    primary_split_metrics = _split_metrics(primary_predictions)
    primary_split_metrics.to_csv(
        output_dir / str(outputs["primary_split_metrics"]),
        index=False,
    )
    primary_summary = summarize_predictions_by_method(primary_predictions)
    primary_summary = primary_summary.merge(
        method_manifest,
        on="method",
        how="left",
    ).sort_values(["rmse_mean", "rmse_worst", "method"]).reset_index(drop=True)
    primary_summary.to_csv(
        output_dir / str(outputs["primary_summary"]),
        index=False,
    )

    print("[04D.1 5/8] Running LOSO hyperparameter selection and support-tier routing...")
    finalists = _select_finalists(primary_summary, method_manifest, config)
    finalist_manifest = method_manifest.loc[
        method_manifest["method"].isin(finalists)
    ].copy()
    finalist_manifest = finalist_manifest.assign(
        primary_rmse=finalist_manifest["method"].map(
            primary_summary.set_index("method")["rmse_mean"]
        )
    ).sort_values(["method_family", "primary_rmse", "method"])
    finalist_manifest.to_csv(
        output_dir / str(outputs["finalist_manifest"]),
        index=False,
    )

    loso_selection, loso_method_predictions = leave_one_split_out_method_selection(
        primary_predictions,
        family=primary_family,
    )
    routing_candidates = method_manifest["method"].astype(str).tolist()
    loso_tier_selection, loso_tier_predictions = leave_one_split_out_tier_routing(
        primary_predictions,
        family=primary_family,
        candidate_methods=routing_candidates,
        tiers=config["nested_selection"]["support_tiers"],
    )
    loso_selection.to_csv(
        output_dir / str(outputs["loso_method_selection"]),
        index=False,
    )
    loso_method_predictions.to_csv(
        output_dir / str(outputs["loso_method_predictions"]),
        index=False,
    )
    loso_tier_selection.to_csv(
        output_dir / str(outputs["loso_tier_selection"]),
        index=False,
    )
    loso_tier_predictions.to_csv(
        output_dir / str(outputs["loso_tier_predictions"]),
        index=False,
    )

    nested_summary = pd.DataFrame(
        [
            summarize_nested_predictions(
                loso_method_predictions,
                label="LOSO_method_selection",
            ),
            summarize_nested_predictions(
                loso_tier_predictions,
                label="LOSO_support_tier_routing",
            ),
        ]
    )
    nested_summary.to_csv(
        output_dir / str(outputs["nested_validation_summary"]),
        index=False,
    )

    print(
        f"[04D.1 6/8] Stress-testing {len(finalists)} finalists on "
        f"{len(stress_splits)} non-primary splits..."
    )
    stress_rows: list[dict[str, Any]] = []
    for index, split in enumerate(stress_splits, start=1):
        anchor_predictions, anchor_residuals = fit_pls_anchor_with_cross_fitted_residuals(
            c0,
            y,
            split.observed_positions,
            strata=strata,
            n_components=int(config["baselines"]["pls_anchor"]["n_components"]),
            n_splits=int(config["validation"]["anchor_crossfit_folds"]),
            random_state=int(config["validation"]["random_state"]) + 1000 + index,
        )
        support = _support_for_split(support_lookup, split, frame)
        for method in finalists:
            spec = method_manifest.loc[method_manifest["method"].eq(method)].iloc[0]
            predicted = _predict_one(
                spec,
                split=split,
                y=y,
                x_by_representation=x_by_representation,
                distances=distances,
                graphs=graphs,
                anchor_predictions=anchor_predictions,
                anchor_residuals=anchor_residuals,
            )
            stress_rows.extend(
                _prediction_rows(
                    split,
                    frame,
                    y,
                    spec,
                    predicted,
                    support,
                )
            )
        print(
            f"  [{index:02d}/{len(stress_splits):02d}] {split.split_id} complete",
            flush=True,
        )

    stress_predictions = pd.DataFrame(stress_rows)
    stress_predictions.to_csv(
        output_dir / str(outputs["stress_predictions"]),
        index=False,
    )
    stress_summary = summarize_predictions_by_method(stress_predictions)
    stress_summary = stress_summary.merge(
        method_manifest,
        on="method",
        how="left",
    )
    stress_summary.to_csv(
        output_dir / str(outputs["stress_summary"]),
        index=False,
    )
    robustness = _robustness_table(
        primary_summary,
        stress_summary,
        finalists,
    )
    robustness.to_csv(
        output_dir / str(outputs["finalist_robustness"]),
        index=False,
    )

    print("[04D.1 7/8] Generating candidate predictions for the fixed 59 targets...")
    actual_anchor, actual_residuals = fit_pls_anchor_with_cross_fitted_residuals(
        c0,
        y,
        train_positions,
        strata=strata,
        n_components=int(config["baselines"]["pls_anchor"]["n_components"]),
        n_splits=int(config["validation"]["anchor_crossfit_folds"]),
        random_state=int(config["validation"]["random_state"]) + 9000,
    )
    actual_candidates = _actual_candidate_rows(
        frame,
        actual_support,
        method_manifest,
        finalists,
        y=y,
        train_positions=train_positions,
        target_positions=target_positions,
        x_by_representation=x_by_representation,
        distances=distances,
        graphs=graphs,
        anchor_predictions=actual_anchor,
        anchor_residuals=actual_residuals,
    )
    actual_candidates.to_csv(
        output_dir / str(outputs["actual_candidate_predictions"]),
        index=False,
    )

    tier_route = _fit_tier_route(
        primary_predictions,
        finalists,
        list(map(str, config["nested_selection"]["support_tiers"])),
    )
    actual_routing = _actual_routing(actual_candidates, tier_route)
    actual_routing.to_csv(
        output_dir / str(outputs["actual_routing_proposal"]),
        index=False,
    )

    print("[04D.1 8/8] Writing figures and canonical reports...")
    dpi = int(config["figures"]["dpi"])
    plot_refinement_ranking(
        primary_summary,
        figures_dir / "refinement_ranking.png",
        top_n=int(config["figures"]["ranking_top_n"]),
        dpi=dpi,
    )
    plot_graph_k_profiles(
        primary_summary,
        method_manifest,
        figures_dir / "graph_k_profiles.png",
        dpi=dpi,
    )
    plot_local_k_profile(
        primary_summary,
        method_manifest,
        figures_dir / "local_k_profile.png",
        dpi=dpi,
    )
    plot_nested_validation_comparison(
        nested_summary,
        figures_dir / "nested_validation_comparison.png",
        dpi=dpi,
    )
    plot_loso_selection_frequency(
        loso_selection,
        figures_dir / "loso_selection_frequency.png",
        dpi=dpi,
    )
    plot_actual_candidate_spread(
        actual_candidates,
        figures_dir / "actual_candidate_spread.png",
        dpi=dpi,
    )

    best_primary = primary_summary.iloc[0]
    best_robust = robustness.iloc[0]
    nested_by_label = nested_summary.set_index("label")
    report = {
        "schema_version": int(config["schema_version"]),
        "generated_at": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(root),
        "stage": str(config["stage"]["name"]),
        "hidden_prediction_targets_used": False,
        "target_covariates_used_transductively": True,
        "primary_family": primary_family,
        "primary_splits": int(len(primary_splits)),
        "stress_splits": int(len(stress_splits)),
        "methods_searched": int(len(method_manifest)),
        "finalists": int(len(finalists)),
        "best_primary_method": str(best_primary["method"]),
        "best_primary_rmse_mean": float(best_primary["rmse_mean"]),
        "best_primary_rmse_worst": float(best_primary["rmse_worst"]),
        "best_robust_method": str(best_robust["method"]),
        "best_robust_worst_family_rmse": float(best_robust["worst_family_rmse"]),
        "loso_method_selection_rmse_mean": float(
            nested_by_label.loc["LOSO_method_selection", "rmse_mean"]
        ),
        "loso_tier_routing_rmse_mean": float(
            nested_by_label.loc["LOSO_support_tier_routing", "rmse_mean"]
        ),
        "distance_scales": distance_scales,
        "c0_features_used": int(len(c0_features)),
        "actual_targets": int(len(target_positions)),
        "actual_candidate_methods": int(actual_candidates["method"].nunique()),
        "actual_route_counts": {
            str(key): int(value)
            for key, value in actual_routing["selected_method"].value_counts().items()
        },
    }
    (output_dir / str(outputs["report_json"])).write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / str(outputs["report_markdown"])).write_text(
        _render_report(
            report,
            primary_summary,
            finalist_manifest,
            robustness,
            nested_summary,
            tier_route,
        )
        + "\n",
        encoding="utf-8",
    )

    print("Checkpoint 04D.1 local/graph refinement: PASS")
    print(f"Methods searched: {len(method_manifest)}")
    print(
        f"Best primary: {report['best_primary_method']} "
        f"RMSE={report['best_primary_rmse_mean']:.4f}"
    )
    print(
        f"Nested method-selection RMSE={report['loso_method_selection_rmse_mean']:.4f}"
    )
    print(
        f"Nested tier-routing RMSE={report['loso_tier_routing_rmse_mean']:.4f}"
    )
    print(f"Inspect: {output_dir / str(outputs['report_markdown'])}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Checkpoint 04D.1 local/graph refinement."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/checkpoint04d1.yaml"),
    )
    parser.add_argument("--root", type=Path, default=None)
    args = parser.parse_args()

    root = (
        args.root.resolve()
        if args.root is not None
        else find_project_root(start=ROOT)
    )
    config_path = args.config if args.config.is_absolute() else root / args.config
    run_checkpoint_04d1(root, _load_yaml(config_path))


if __name__ == "__main__":
    main()
