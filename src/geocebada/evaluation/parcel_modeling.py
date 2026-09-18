"""Feature-space diagnostics, fixed folds and lightweight ablation benchmarks.

This module is intentionally conservative: it diagnoses the frozen 197-row feature table
and evaluates small-sample baselines on precomputed folds. It does not tune final models
and it never uses hidden prediction targets.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin, clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from geocebada.statistics import covariate_shift_screen

ID_COLUMN = "ID_POLIGONO"
TARGET_COLUMN = "RENDIMIENTO_T_HA"
SPLIT_COLUMN = "CONJUNTO"


def classify_feature_family(record: Mapping[str, Any]) -> str:
    """Map one feature-manifest record to a stable modeling family."""

    column = str(record["column"])
    source = str(record.get("source", ""))

    if column.startswith("base_"):
        return "geometry"
    if column.startswith("admin_"):
        return "admin"

    source_map = {
        "sat_basic": "basic",
        "sat_pro": "pro",
        "official_climate": "climate_official",
        "external_chirps_daily": "chirps",
        "external_soilgrids": "soilgrids",
        "official_topography": "topography",
        "external_inegi_cem": "cem",
        "external_siap": "siap",
        "external_wapor": "wapor",
    }
    if source in source_map:
        return source_map[source]

    if source == "official_split+official_parcels+admin":
        return "geometry_admin_other"
    return "other"


def build_feature_catalog(
    manifest: Mapping[str, Any],
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Annotate every manifest feature with family, mode, dtype and basic availability."""

    records = manifest.get("features")
    if not isinstance(records, list):
        raise ValueError("Manifest must contain a list under 'features'.")

    rows: list[dict[str, Any]] = []
    for record in records:
        column = str(record["column"])
        if column not in frame.columns:
            raise KeyError(f"Manifest feature missing from feature table: {column}")
        series = frame[column]
        rows.append(
            {
                "feature": column,
                "family": classify_feature_family(record),
                "source": str(record.get("source", "")),
                "mode": str(record.get("mode", "")),
                "description": str(record.get("description", "")),
                "dtype": str(series.dtype),
                "numeric": bool(pd.api.types.is_numeric_dtype(series)),
                "missing_fraction_all": float(series.isna().mean()),
                "n_unique_non_null": int(series.nunique(dropna=True)),
            }
        )

    catalog = pd.DataFrame(rows)
    if catalog["feature"].duplicated().any():
        duplicates = catalog.loc[catalog["feature"].duplicated(), "feature"].tolist()
        raise ValueError(f"Duplicate feature records in manifest: {duplicates[:10]}")
    return catalog


def summarize_feature_inventory(catalog: pd.DataFrame) -> pd.DataFrame:
    """Summarize feature counts by family, mode and provenance source."""

    required = {"feature", "family", "source", "mode", "numeric"}
    missing = required.difference(catalog.columns)
    if missing:
        raise KeyError(f"Catalog missing column(s): {sorted(missing)}")

    result = (
        catalog.groupby(["family", "mode", "source"], dropna=False, as_index=False)
        .agg(
            feature_count=("feature", "size"),
            numeric_features=("numeric", "sum"),
        )
        .sort_values(["mode", "family", "source"])
        .reset_index(drop=True)
    )
    result["non_numeric_features"] = (
        result["feature_count"] - result["numeric_features"]
    )
    return result


def train_prediction_diagnostics(
    frame: pd.DataFrame,
    *,
    features: Iterable[str],
    split_column: str = SPLIT_COLUMN,
    train_value: str = "ENTRENAMIENTO",
    prediction_value: str = "PREDICCION",
    smd_threshold: float = 0.5,
    adjusted_p_threshold: float = 0.05,
    missing_gap_threshold: float = 0.10,
    outside_support_threshold: float = 0.10,
) -> pd.DataFrame:
    """Compare numeric feature support/distributions between train and prediction rows."""

    feature_list = list(features)
    if not feature_list:
        raise ValueError("At least one feature is required.")

    shift = covariate_shift_screen(
        frame,
        split_column,
        train_value,
        prediction_value,
        features=feature_list,
    ).set_index("feature")

    train_mask = frame[split_column].eq(train_value)
    prediction_mask = frame[split_column].eq(prediction_value)
    rows: list[dict[str, Any]] = []

    quantiles = {
        "q05": 0.05,
        "q25": 0.25,
        "median": 0.50,
        "q75": 0.75,
        "q95": 0.95,
    }

    for feature in feature_list:
        train_raw = pd.to_numeric(frame.loc[train_mask, feature], errors="coerce")
        pred_raw = pd.to_numeric(frame.loc[prediction_mask, feature], errors="coerce")
        train = train_raw.dropna().astype(float)
        pred = pred_raw.dropna().astype(float)

        row: dict[str, Any] = {
            "feature": feature,
            "n_train": int(train.size),
            "n_prediction": int(pred.size),
            "missing_train": float(train_raw.isna().mean()),
            "missing_prediction": float(pred_raw.isna().mean()),
        }

        for prefix, values in (("train", train), ("prediction", pred)):
            row[f"{prefix}_mean"] = (
                float(values.mean()) if len(values) else float("nan")
            )
            row[f"{prefix}_std"] = (
                float(values.std(ddof=1)) if len(values) > 1 else float("nan")
            )
            row[f"{prefix}_min"] = (
                float(values.min()) if len(values) else float("nan")
            )
            row[f"{prefix}_max"] = (
                float(values.max()) if len(values) else float("nan")
            )
            for label, q in quantiles.items():
                row[f"{prefix}_{label}"] = (
                    float(values.quantile(q)) if len(values) else float("nan")
                )

        outside = float("nan")
        if len(train) and len(pred):
            outside = float(
                ((pred < float(train.min())) | (pred > float(train.max()))).mean()
            )
        row["prediction_outside_train_range_fraction"] = outside
        rows.append(row)

    result = pd.DataFrame(rows).set_index("feature")
    common = [
        "smd",
        "abs_smd",
        "ks_statistic",
        "p_value",
        "p_adjusted",
    ]
    result = result.join(shift[common], how="left")
    result["missing_gap_abs"] = (
        result["missing_prediction"] - result["missing_train"]
    ).abs()

    result["flag_smd"] = result["abs_smd"].ge(float(smd_threshold))
    result["flag_ks_fdr"] = result["p_adjusted"].lt(float(adjusted_p_threshold))
    result["flag_missing_gap"] = result["missing_gap_abs"].ge(
        float(missing_gap_threshold)
    )
    result["flag_outside_support"] = result[
        "prediction_outside_train_range_fraction"
    ].ge(float(outside_support_threshold))
    result["flag_any_shift"] = result[
        ["flag_smd", "flag_ks_fdr", "flag_missing_gap", "flag_outside_support"]
    ].any(axis=1)

    return (
        result.reset_index()
        .sort_values(
            [
                "flag_any_shift",
                "abs_smd",
                "prediction_outside_train_range_fraction",
            ],
            ascending=[False, False, False],
            na_position="last",
        )
        .reset_index(drop=True)
    )


def build_fixed_fold_assignments(
    frame: pd.DataFrame,
    *,
    n_splits: int = 5,
    random_state: int = 42,
    id_column: str = ID_COLUMN,
    split_column: str = SPLIT_COLUMN,
    train_value: str = "ENTRENAMIENTO",
    state_column: str = "meta_estado",
    municipality_column: str = "meta_municipio",
) -> pd.DataFrame:
    """Create deterministic state-stratified and municipality-grouped training folds."""

    required = {
        id_column,
        split_column,
        state_column,
        municipality_column,
    }
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Feature table missing fold column(s): {sorted(missing)}")

    train = frame.loc[
        frame[split_column].eq(train_value),
        [id_column, state_column, municipality_column],
    ].copy()
    if train.empty:
        raise ValueError("No training rows available for fold construction.")
    if train[id_column].isna().any() or not train[id_column].is_unique:
        raise ValueError("Training parcel IDs must be unique and non-null.")

    state = train[state_column].astype("string").fillna("__MISSING_STATE__")
    if int(state.value_counts().min()) < int(n_splits):
        raise ValueError("Every state must contain at least n_splits training parcels.")

    result = train.copy()
    result["fold_state_stratified"] = -1

    primary = StratifiedKFold(
        n_splits=int(n_splits),
        shuffle=True,
        random_state=int(random_state),
    )
    for fold, (_, validation_index) in enumerate(
        primary.split(train, state),
        start=1,
    ):
        result.iloc[
            validation_index,
            result.columns.get_loc("fold_state_stratified"),
        ] = fold

    municipality = (
        train[municipality_column]
        .astype("string")
        .fillna("__MISSING_MUNICIPALITY__")
    )
    groups = state.astype(str) + "|" + municipality.astype(str)
    if int(groups.nunique()) < int(n_splits):
        raise ValueError("Need at least n_splits municipality groups.")

    result["fold_municipality_grouped"] = -1
    try:
        spatial = StratifiedGroupKFold(
            n_splits=int(n_splits),
            shuffle=True,
            random_state=int(random_state),
        )
        spatial_splits = list(spatial.split(train, y=state, groups=groups))
    except ValueError:
        fallback = GroupKFold(n_splits=int(n_splits))
        spatial_splits = list(fallback.split(train, groups=groups))

    for fold, (_, validation_index) in enumerate(spatial_splits, start=1):
        result.iloc[
            validation_index,
            result.columns.get_loc("fold_municipality_grouped"),
        ] = fold

    if (result[["fold_state_stratified", "fold_municipality_grouped"]] < 1).any().any():
        raise RuntimeError("Fold assignment left one or more training parcels unassigned.")

    check = pd.DataFrame(
        {
            "group": groups.to_numpy(),
            "fold": result["fold_municipality_grouped"].to_numpy(),
        }
    )
    if check.groupby("group")["fold"].nunique().max() != 1:
        raise RuntimeError("Municipality group leakage detected across grouped folds.")

    return result.reset_index(drop=True)


def validate_fixed_fold_assignments(
    frame: pd.DataFrame,
    folds: pd.DataFrame,
    *,
    id_column: str = ID_COLUMN,
    split_column: str = SPLIT_COLUMN,
    train_value: str = "ENTRENAMIENTO",
    fold_columns: Iterable[str] = (
        "fold_state_stratified",
        "fold_municipality_grouped",
    ),
) -> None:
    """Validate that a frozen fold file covers each training parcel exactly once."""

    expected = set(
        frame.loc[frame[split_column].eq(train_value), id_column].astype(str)
    )
    observed = set(folds[id_column].astype(str))
    if expected != observed:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(
            f"Frozen folds do not match training IDs; missing={missing[:5]}, "
            f"extra={extra[:5]}."
        )
    if folds[id_column].isna().any() or not folds[id_column].is_unique:
        raise ValueError("Frozen fold IDs must be unique and non-null.")

    for column in fold_columns:
        if column not in folds.columns:
            raise KeyError(column)
        values = pd.to_numeric(folds[column], errors="coerce")
        if values.isna().any() or (values < 1).any():
            raise ValueError(f"Invalid fold labels in {column}.")


def resolve_ablation_features(
    catalog: pd.DataFrame,
    spec: Mapping[str, Any],
) -> list[str]:
    """Resolve one declarative ablation specification into concrete feature names."""

    required = {"feature", "family", "mode", "numeric"}
    missing = required.difference(catalog.columns)
    if missing:
        raise KeyError(f"Catalog missing column(s): {sorted(missing)}")

    selected = pd.Series(False, index=catalog.index)

    if bool(spec.get("include_all_clean", False)):
        selected |= catalog["mode"].eq("clean")
    if bool(spec.get("include_all_competition", False)):
        selected |= catalog["mode"].eq("competition")

    clean_families = set(map(str, spec.get("clean_families", [])))
    competition_families = set(map(str, spec.get("competition_families", [])))
    if clean_families:
        selected |= catalog["mode"].eq("clean") & catalog["family"].isin(clean_families)
    if competition_families:
        selected |= catalog["mode"].eq("competition") & catalog["family"].isin(
            competition_families
        )

    if bool(spec.get("numeric_only", True)):
        selected &= catalog["numeric"].astype(bool)

    features = catalog.loc[selected, "feature"].astype(str).tolist()
    if not features:
        raise ValueError(f"Ablation resolved to zero features: {dict(spec)}")
    return features


def initial_regressors(
    *,
    random_state: int = 42,
    ridge_alpha: float = 10.0,
    extra_trees_estimators: int = 300,
    extra_trees_min_samples_leaf: int = 3,
) -> dict[str, RegressorMixin]:
    """Return deliberately untuned baseline regressors for Checkpoint 02."""

    def imputer() -> SimpleImputer:
        return SimpleImputer(
            strategy="median",
            add_indicator=True,
            keep_empty_features=True,
        )

    return {
        "DummyMean": make_pipeline(
            imputer(),
            DummyRegressor(strategy="mean"),
        ),
        "Ridge": make_pipeline(
            imputer(),
            StandardScaler(),
            Ridge(alpha=float(ridge_alpha)),
        ),
        "ExtraTrees": make_pipeline(
            imputer(),
            ExtraTreesRegressor(
                n_estimators=int(extra_trees_estimators),
                min_samples_leaf=int(extra_trees_min_samples_leaf),
                max_features="sqrt",
                random_state=int(random_state),
                n_jobs=-1,
            ),
        ),
    }


def evaluate_fixed_folds(
    frame: pd.DataFrame,
    folds: pd.DataFrame,
    ablations: Mapping[str, Iterable[str]],
    *,
    fold_column: str,
    regressors: Mapping[str, RegressorMixin] | None = None,
    id_column: str = ID_COLUMN,
    target_column: str = TARGET_COLUMN,
    split_column: str = SPLIT_COLUMN,
    train_value: str = "ENTRENAMIENTO",
) -> pd.DataFrame:
    """Evaluate fixed feature ablations and baseline models on one frozen fold protocol."""

    validate_fixed_fold_assignments(frame, folds, id_column=id_column)
    train = frame.loc[frame[split_column].eq(train_value)].copy()
    train[id_column] = train[id_column].astype(str)
    fold_map = folds[[id_column, fold_column]].copy()
    fold_map[id_column] = fold_map[id_column].astype(str)
    train = train.merge(fold_map, on=id_column, how="left", validate="1:1")

    if train[target_column].isna().any():
        raise ValueError("Training target contains missing values.")

    models = dict(regressors or initial_regressors())
    fold_values = sorted(pd.to_numeric(train[fold_column]).astype(int).unique())
    rows: list[dict[str, Any]] = []

    for ablation_name, feature_iterable in ablations.items():
        features = list(feature_iterable)
        missing = set(features).difference(train.columns)
        if missing:
            raise KeyError(
                f"Ablation {ablation_name!r} references missing features: "
                f"{sorted(missing)[:10]}"
            )

        x = train[features].apply(pd.to_numeric, errors="coerce")
        y = pd.to_numeric(train[target_column], errors="raise").astype(float)

        for model_name, estimator in models.items():
            for fold in fold_values:
                validation_mask = train[fold_column].eq(fold)
                train_mask = ~validation_mask
                fitted = clone(estimator)
                fitted.fit(x.loc[train_mask], y.loc[train_mask])
                prediction = fitted.predict(x.loc[validation_mask])
                observed = y.loc[validation_mask]

                rows.append(
                    {
                        "protocol": fold_column,
                        "ablation": str(ablation_name),
                        "model": str(model_name),
                        "fold": int(fold),
                        "n_features": int(len(features)),
                        "n_train": int(train_mask.sum()),
                        "n_validation": int(validation_mask.sum()),
                        "rmse": float(
                            np.sqrt(mean_squared_error(observed, prediction))
                        ),
                        "mae": float(mean_absolute_error(observed, prediction)),
                        "r2": float(r2_score(observed, prediction)),
                    }
                )

    return pd.DataFrame(rows)


def summarize_fixed_fold_scores(scores: pd.DataFrame) -> pd.DataFrame:
    """Summarize fixed-fold benchmark scores without declaring a model winner."""

    required = {
        "protocol",
        "ablation",
        "model",
        "n_features",
        "rmse",
        "mae",
        "r2",
    }
    missing = required.difference(scores.columns)
    if missing:
        raise KeyError(f"Score table missing column(s): {sorted(missing)}")

    summary = (
        scores.groupby(
            ["protocol", "ablation", "model", "n_features"],
            as_index=False,
        )
        .agg(
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
        )
        .sort_values(["protocol", "ablation", "rmse_mean", "model"])
        .reset_index(drop=True)
    )
    return summary
