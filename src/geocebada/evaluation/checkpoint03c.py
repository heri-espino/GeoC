"""Checkpoint 03C representation benchmarking on frozen parcel folds.

The first 03C stage isolates feature-representation effects before broad model tuning.
It compares source, agronomic, empirical and joined representations under the exact
Checkpoint 02 folds using the same fixed Ridge and ExtraTrees baselines.

Target-aware expression discovery is available only through
FoldLocalExpressionAugmenter, which is fitted inside each outer training fold.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from geocebada.evaluation.parcel_modeling import validate_fixed_fold_assignments
from geocebada.features import FoldLocalExpressionAugmenter

ID_COLUMN = "ID_POLIGONO"
TARGET_COLUMN = "RENDIMIENTO_T_HA"
SPLIT_COLUMN = "CONJUNTO"


@dataclass(frozen=True)
class RepresentationSpec:
    """One track-specific feature representation used in Checkpoint 03C."""

    name: str
    track: str
    features: tuple[str, ...]
    layers: tuple[str, ...]
    discovery: bool
    primitive_columns: tuple[str, ...]
    description: str

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation record."""

        return {
            "name": self.name,
            "track": self.track,
            "n_features": len(self.features),
            "layers": list(self.layers),
            "discovery": self.discovery,
            "primitive_columns": list(self.primitive_columns),
            "n_primitives": len(self.primitive_columns),
            "description": self.description,
            "features": list(self.features),
        }


def _manifest_records(manifest: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    records = manifest.get("features")
    if not isinstance(records, list):
        raise ValueError("Feature manifest must contain a list under 'features'.")
    return records


def _allowed_modes(track: str, *, competition_includes_clean: bool) -> set[str]:
    if track == "clean":
        return {"clean"}
    if track == "competition":
        return {"clean", "competition"} if competition_includes_clean else {"competition"}
    raise ValueError(f"Unknown feature track: {track}")


def build_layer_feature_catalog(
    *,
    joined: pd.DataFrame,
    manifests: Mapping[str, Mapping[str, Any]],
    numeric_only: bool = True,
    exclude_sources: Sequence[str] = (),
    exclude_columns: Sequence[str] = (),
) -> pd.DataFrame:
    """Build one cross-layer feature catalog with mode and numeric availability."""

    excluded_sources = set(map(str, exclude_sources))
    excluded_columns = set(map(str, exclude_columns))
    rows: list[dict[str, Any]] = []

    for layer, manifest in manifests.items():
        for record in _manifest_records(manifest):
            column = str(record["column"])
            if column not in joined.columns:
                raise KeyError(f"{layer} manifest feature missing from joined table: {column}")
            if column in excluded_columns:
                continue

            source = str(record.get("source", ""))
            if source in excluded_sources:
                continue

            numeric = bool(pd.api.types.is_numeric_dtype(joined[column]))
            if numeric_only and not numeric:
                continue

            rows.append(
                {
                    "feature": column,
                    "layer": str(layer),
                    "mode": str(record.get("mode", "")),
                    "source": source,
                    "numeric": numeric,
                }
            )

    catalog = pd.DataFrame(rows)
    if catalog.empty:
        raise ValueError("Cross-layer feature catalog is empty.")
    if catalog["feature"].duplicated().any():
        duplicates = (
            catalog.loc[catalog["feature"].duplicated(), "feature"]
            .astype(str)
            .tolist()
        )
        raise ValueError(f"Feature names overlap across layers: {duplicates[:10]}")
    return catalog.reset_index(drop=True)


def build_representation_specs(
    *,
    joined: pd.DataFrame,
    manifests: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
    empirical_config: Mapping[str, Any],
) -> dict[tuple[str, str], RepresentationSpec]:
    """Resolve configured 03C representations into exact numeric feature lists."""

    policy = config["feature_policy"]
    catalog = build_layer_feature_catalog(
        joined=joined,
        manifests=manifests,
        numeric_only=bool(policy.get("numeric_only", True)),
        exclude_sources=tuple(map(str, policy.get("exclude_sources", []))),
        exclude_columns=tuple(map(str, policy.get("exclude_columns", []))),
    )
    mode_by_feature = dict(zip(catalog["feature"], catalog["mode"], strict=True))
    primitive_pool = tuple(
        map(str, empirical_config["supervised_discovery"]["primitive_columns"])
    )

    result: dict[tuple[str, str], RepresentationSpec] = {}
    for track in map(str, config["tracks"]):
        allowed_modes = _allowed_modes(
            track,
            competition_includes_clean=bool(
                policy.get("competition_includes_clean", True)
            ),
        )
        track_catalog = catalog.loc[catalog["mode"].isin(allowed_modes)].copy()

        for name, raw_spec in config["representations"].items():
            layers = tuple(map(str, raw_spec["layers"]))
            selected = track_catalog.loc[track_catalog["layer"].isin(layers), "feature"]
            features = tuple(dict.fromkeys(selected.astype(str).tolist()))
            if not features:
                raise ValueError(
                    f"Representation {name!r} resolved to zero features for track {track!r}."
                )

            discovery = bool(raw_spec.get("discovery", False))
            primitives: tuple[str, ...] = ()
            if discovery:
                feature_set = set(features)
                primitives = tuple(
                    column
                    for column in primitive_pool
                    if column in feature_set
                    and mode_by_feature.get(column) in allowed_modes
                )
                if len(primitives) < 2:
                    raise ValueError(
                        f"Representation {name!r}/{track!r} has fewer than two "
                        "eligible discovery primitives."
                    )

            result[(track, str(name))] = RepresentationSpec(
                name=str(name),
                track=track,
                features=features,
                layers=layers,
                discovery=discovery,
                primitive_columns=primitives,
                description=str(raw_spec.get("description", "")),
            )

    return result


def make_representation_estimator(
    *,
    model_name: str,
    model_config: Mapping[str, Any],
    random_state: int,
    discovery_config: Mapping[str, Any] | None = None,
    primitive_columns: Sequence[str] = (),
) -> Pipeline:
    """Create one fixed baseline pipeline for representation comparison."""

    steps: list[tuple[str, Any]] = []
    if discovery_config is not None:
        steps.append(
            (
                "discover",
                FoldLocalExpressionAugmenter(
                    tuple(map(str, primitive_columns)),
                    top_primitives=int(discovery_config["top_primitives"]),
                    top_expressions=int(discovery_config["top_expressions"]),
                    operations=tuple(map(str, discovery_config["operations"])),
                    epsilon=float(discovery_config["epsilon"]),
                ),
            )
        )

    steps.append(
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                add_indicator=True,
                keep_empty_features=True,
            ),
        )
    )

    kind = str(model_config["kind"])
    if kind == "ridge":
        steps.extend(
            [
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=float(model_config["alpha"]))),
            ]
        )
    elif kind == "extra_trees":
        steps.append(
            (
                "model",
                ExtraTreesRegressor(
                    n_estimators=int(model_config["n_estimators"]),
                    min_samples_leaf=int(model_config["min_samples_leaf"]),
                    max_features=model_config["max_features"],
                    random_state=int(random_state),
                    n_jobs=-1,
                ),
            )
        )
    else:
        raise ValueError(f"Unsupported 03C.1 baseline model kind: {kind}")

    return Pipeline(steps)


def _metric_row(
    observed: pd.Series,
    predicted: np.ndarray,
) -> dict[str, float]:
    y_true = observed.to_numpy(dtype=float)
    y_pred = np.asarray(predicted, dtype=float)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "target_mean": float(np.mean(y_true)),
        "target_std": float(np.std(y_true, ddof=1)) if len(y_true) > 1 else float("nan"),
    }


def evaluate_representation_benchmark(
    *,
    joined: pd.DataFrame,
    folds: pd.DataFrame,
    representations: Mapping[tuple[str, str], RepresentationSpec],
    model_configs: Mapping[str, Mapping[str, Any]],
    empirical_config: Mapping[str, Any],
    protocols: Sequence[str],
    random_state: int,
    id_column: str = ID_COLUMN,
    target_column: str = TARGET_COLUMN,
    split_column: str = SPLIT_COLUMN,
    train_value: str = "ENTRENAMIENTO",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate representations on frozen folds and return scores, OOF rows and discovery logs."""

    validate_fixed_fold_assignments(
        joined,
        folds,
        id_column=id_column,
        split_column=split_column,
        train_value=train_value,
        fold_columns=protocols,
    )
    training = joined.loc[joined[split_column].eq(train_value)].copy()
    training[id_column] = training[id_column].astype(str)
    fold_columns = folds[[id_column, *protocols]].copy()
    fold_columns[id_column] = fold_columns[id_column].astype(str)
    training = training.merge(
        fold_columns,
        on=id_column,
        how="left",
        validate="one_to_one",
    )

    y = pd.to_numeric(training[target_column], errors="raise").astype(float)
    if y.isna().any():
        raise ValueError("Training target contains missing values.")

    fold_rows: list[dict[str, Any]] = []
    oof_rows: list[dict[str, Any]] = []
    discovery_rows: list[dict[str, Any]] = []
    discovery_cfg = empirical_config["supervised_discovery"]

    for (track, representation_name), spec in representations.items():
        missing = sorted(set(spec.features).difference(training.columns))
        if missing:
            raise KeyError(
                f"Representation {representation_name!r}/{track!r} references "
                f"missing columns: {missing[:10]}"
            )
        x = training.loc[:, list(spec.features)].apply(
            pd.to_numeric,
            errors="coerce",
        )

        for model_name, model_config in model_configs.items():
            for protocol in protocols:
                fold_values = sorted(
                    pd.to_numeric(training[protocol], errors="raise")
                    .astype(int)
                    .unique()
                )
                for fold in fold_values:
                    validation_mask = training[protocol].eq(fold)
                    train_mask = ~validation_mask

                    estimator = make_representation_estimator(
                        model_name=str(model_name),
                        model_config=model_config,
                        random_state=int(random_state),
                        discovery_config=discovery_cfg if spec.discovery else None,
                        primitive_columns=spec.primitive_columns,
                    )
                    estimator.fit(x.loc[train_mask], y.loc[train_mask])
                    prediction = estimator.predict(x.loc[validation_mask])
                    observed = y.loc[validation_mask]
                    metrics = _metric_row(observed, prediction)

                    n_discovered = 0
                    if spec.discovery:
                        augmenter = estimator.named_steps["discover"]
                        discovery_report = augmenter.discovery_report()
                        n_discovered = int(len(discovery_report))
                        for row in discovery_report.itertuples(index=False):
                            discovery_rows.append(
                                {
                                    "protocol": str(protocol),
                                    "track": track,
                                    "representation": representation_name,
                                    "model": str(model_name),
                                    "fold": int(fold),
                                    "rank": int(row.rank),
                                    "feature": str(row.feature),
                                    "left": str(row.left),
                                    "right": str(row.right),
                                    "operation": str(row.operation),
                                    "formula": str(row.formula),
                                    "abs_spearman_train": float(
                                        row.abs_spearman_train
                                    ),
                                }
                            )

                    fold_rows.append(
                        {
                            "protocol": str(protocol),
                            "track": track,
                            "representation": representation_name,
                            "model": str(model_name),
                            "fold": int(fold),
                            "n_features": int(len(spec.features)),
                            "n_discovered": n_discovered,
                            "n_train": int(train_mask.sum()),
                            "n_validation": int(validation_mask.sum()),
                            **metrics,
                        }
                    )

                    ids = training.loc[validation_mask, id_column].astype(str)
                    for parcel_id, truth, pred in zip(
                        ids,
                        observed.to_numpy(dtype=float),
                        np.asarray(prediction, dtype=float),
                        strict=True,
                    ):
                        oof_rows.append(
                            {
                                "protocol": str(protocol),
                                "track": track,
                                "representation": representation_name,
                                "model": str(model_name),
                                "fold": int(fold),
                                id_column: parcel_id,
                                "observed": float(truth),
                                "predicted": float(pred),
                                "residual": float(truth - pred),
                            }
                        )

    fold_scores = pd.DataFrame(fold_rows)
    oof = pd.DataFrame(oof_rows)
    discovery = pd.DataFrame(discovery_rows)
    _validate_oof_coverage(
        oof,
        expected_ids=training[id_column],
        id_column=id_column,
    )
    return fold_scores, oof, discovery


def _validate_oof_coverage(
    oof: pd.DataFrame,
    *,
    expected_ids: pd.Series,
    id_column: str,
) -> None:
    expected = set(expected_ids.astype(str))
    grouping = ["protocol", "track", "representation", "model"]
    for key, group in oof.groupby(grouping, sort=False):
        observed = set(group[id_column].astype(str))
        if observed != expected:
            missing = sorted(expected - observed)
            extra = sorted(observed - expected)
            raise ValueError(
                f"OOF coverage mismatch for {key}; "
                f"missing={missing[:5]}, extra={extra[:5]}"
            )
        if group[id_column].duplicated().any():
            raise ValueError(f"Duplicate OOF parcel predictions for {key}.")


def summarize_representation_benchmark(
    fold_scores: pd.DataFrame,
    oof: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize mean-fold and parcel-weighted OOF metrics for each experiment."""

    grouping = ["protocol", "track", "representation", "model"]
    fold_summary = (
        fold_scores.groupby(grouping, as_index=False)
        .agg(
            n_features=("n_features", "first"),
            n_discovered=("n_discovered", "max"),
            fold_rmse_mean=("rmse", "mean"),
            fold_rmse_std=("rmse", "std"),
            fold_mae_mean=("mae", "mean"),
            fold_mae_std=("mae", "std"),
            fold_r2_mean=("r2", "mean"),
            fold_r2_std=("r2", "std"),
            min_validation_rows=("n_validation", "min"),
            max_validation_rows=("n_validation", "max"),
        )
    )

    oof_rows: list[dict[str, Any]] = []
    for key, group in oof.groupby(grouping, sort=False):
        protocol, track, representation, model = key
        observed = group["observed"].to_numpy(dtype=float)
        predicted = group["predicted"].to_numpy(dtype=float)
        oof_rows.append(
            {
                "protocol": protocol,
                "track": track,
                "representation": representation,
                "model": model,
                "oof_rows": int(len(group)),
                "oof_rmse": float(
                    np.sqrt(mean_squared_error(observed, predicted))
                ),
                "oof_mae": float(mean_absolute_error(observed, predicted)),
                "oof_r2": float(r2_score(observed, predicted)),
            }
        )

    summary = fold_summary.merge(
        pd.DataFrame(oof_rows),
        on=grouping,
        how="left",
        validate="one_to_one",
    )

    base = summary.loc[
        summary["representation"].eq("B0_base"),
        [*grouping[:2], "model", "oof_rmse"],
    ].rename(columns={"oof_rmse": "b0_oof_rmse"})
    summary = summary.merge(
        base,
        on=["protocol", "track", "model"],
        how="left",
        validate="many_to_one",
    )
    summary["delta_oof_rmse_vs_B0"] = (
        summary["oof_rmse"] - summary["b0_oof_rmse"]
    )

    all_base = summary.loc[
        summary["representation"].eq("B5_all"),
        [*grouping[:2], "model", "oof_rmse"],
    ].rename(columns={"oof_rmse": "b5_oof_rmse"})
    summary = summary.merge(
        all_base,
        on=["protocol", "track", "model"],
        how="left",
        validate="many_to_one",
    )
    summary["delta_oof_rmse_vs_B5"] = (
        summary["oof_rmse"] - summary["b5_oof_rmse"]
    )

    return summary.sort_values(
        ["protocol", "track", "model", "oof_rmse", "representation"],
        kind="stable",
    ).reset_index(drop=True)
