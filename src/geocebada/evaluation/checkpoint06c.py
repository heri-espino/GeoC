"""Checkpoint 06C low-capacity tail calibration.

This stage is entered only after 06A/06B supported the tail hypothesis.
Calibrators are fit on honest cross-fitted predictions for each pseudo-train
set and scored only on that split's untouched pseudo-target predictions.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from geocebada.evaluation.checkpoint06 import ID_COLUMN


def _rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y, p)))


def _ridge_pipeline(alpha: float) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=float(alpha))),
        ]
    )


def candidate_names(config: Mapping[str, Any]) -> list[str]:
    """Return the frozen 06C candidate universe in deterministic order."""

    names = ["Local04D"]
    for stretch in config["calibration"]["stretch_values"]:
        token = str(float(stretch)).replace(".", "p")
        names.append(f"LocalStretch_s{token}")
    for alpha in config["calibration"]["ridge_alphas"]:
        token = str(float(alpha)).replace(".", "p")
        names.extend(
            [
                f"LocalResidualRidge_a{token}",
                f"LocalPiecewiseRidge_a{token}",
                f"LocalCatResidualRidge_a{token}",
            ]
        )
    return names


def fit_predict_calibrators(
    train_oof: pd.DataFrame,
    pseudo_target: pd.DataFrame,
    *,
    config: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    """Fit 06C calibrators on pseudo-train OOF predictions and predict pseudo-targets."""

    required = {"observed", "Local04D", "CatBoost_C1"}
    missing_train = required.difference(train_oof.columns)
    missing_test = {"Local04D", "CatBoost_C1"}.difference(pseudo_target.columns)
    if missing_train:
        raise KeyError(f"06C pseudo-train OOF missing columns: {sorted(missing_train)}")
    if missing_test:
        raise KeyError(f"06C pseudo-target predictions missing columns: {sorted(missing_test)}")

    y = pd.to_numeric(train_oof["observed"], errors="raise").to_numpy(float)
    local_train = pd.to_numeric(train_oof["Local04D"], errors="raise").to_numpy(float)
    cat_train = pd.to_numeric(train_oof["CatBoost_C1"], errors="raise").to_numpy(float)
    local_test = pd.to_numeric(pseudo_target["Local04D"], errors="raise").to_numpy(float)
    cat_test = pd.to_numeric(pseudo_target["CatBoost_C1"], errors="raise").to_numpy(float)

    result: dict[str, np.ndarray] = {"Local04D": local_test.copy()}
    center = float(np.mean(y))

    for stretch in config["calibration"]["stretch_values"]:
        stretch = float(stretch)
        token = str(stretch).replace(".", "p")
        result[f"LocalStretch_s{token}"] = center + stretch * (local_test - center)

    residual = y - local_train
    low_q = float(config["calibration"]["piecewise_low_prediction_quantile"])
    high_q = float(config["calibration"]["piecewise_high_prediction_quantile"])
    low_knot = float(np.quantile(local_train, low_q))
    high_knot = float(np.quantile(local_train, high_q))

    for alpha in config["calibration"]["ridge_alphas"]:
        alpha = float(alpha)
        token = str(alpha).replace(".", "p")

        x_train = (local_train - center).reshape(-1, 1)
        x_test = (local_test - center).reshape(-1, 1)
        model = _ridge_pipeline(alpha)
        model.fit(x_train, residual)
        result[f"LocalResidualRidge_a{token}"] = local_test + model.predict(x_test)

        piece_train = np.column_stack(
            [
                local_train - center,
                np.maximum(low_knot - local_train, 0.0),
                np.maximum(local_train - high_knot, 0.0),
            ]
        )
        piece_test = np.column_stack(
            [
                local_test - center,
                np.maximum(low_knot - local_test, 0.0),
                np.maximum(local_test - high_knot, 0.0),
            ]
        )
        piece_model = _ridge_pipeline(alpha)
        piece_model.fit(piece_train, residual)
        result[f"LocalPiecewiseRidge_a{token}"] = (
            local_test + piece_model.predict(piece_test)
        )

        expert_train = np.column_stack(
            [
                local_train - center,
                cat_train - local_train,
                np.abs(cat_train - local_train),
            ]
        )
        expert_test = np.column_stack(
            [
                local_test - center,
                cat_test - local_test,
                np.abs(cat_test - local_test),
            ]
        )
        expert_model = _ridge_pipeline(alpha)
        expert_model.fit(expert_train, residual)
        result[f"LocalCatResidualRidge_a{token}"] = (
            local_test + expert_model.predict(expert_test)
        )

    expected = set(candidate_names(config))
    if set(result) != expected:
        raise RuntimeError("06C candidate generation does not match frozen candidate names.")
    return result


def build_calibration_predictions(
    base_oof: pd.DataFrame,
    pseudo_targets: pd.DataFrame,
    *,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Build honest target-matched predictions for every frozen 06C candidate."""

    target_oof = base_oof.loc[base_oof["family"].eq("target_matched")].copy()
    target_test = pseudo_targets.loc[pseudo_targets["family"].eq("target_matched")].copy()

    rows: list[dict[str, Any]] = []
    for split_id, test in target_test.groupby("split_id", sort=True):
        train = target_oof.loc[target_oof["split_id"].eq(split_id)].copy()
        if train.empty or test.empty:
            raise ValueError(f"{split_id}: missing 06C pseudo-train or pseudo-target rows.")
        if train[ID_COLUMN].duplicated().any() or test[ID_COLUMN].duplicated().any():
            raise ValueError(f"{split_id}: duplicated parcel IDs in 06C inputs.")

        predictions = fit_predict_calibrators(train, test, config=config)
        y_train = pd.to_numeric(train["observed"], errors="raise")
        low = float(y_train.quantile(float(config["tails"]["primary_low_quantile"])))
        high = float(y_train.quantile(float(config["tails"]["primary_high_quantile"])))
        observed = pd.to_numeric(test["observed"], errors="raise").to_numpy(float)
        regimes = np.where(
            observed <= low,
            "low",
            np.where(observed >= high, "high", "center"),
        )

        for method, pred in predictions.items():
            for parcel_id, y, p, regime in zip(
                test[ID_COLUMN].astype(str),
                observed,
                np.asarray(pred, dtype=float),
                regimes,
                strict=True,
            ):
                rows.append(
                    {
                        "family": "target_matched",
                        "split_id": split_id,
                        ID_COLUMN: parcel_id,
                        "method": method,
                        "observed": float(y),
                        "predicted": float(p),
                        "residual": float(y - p),
                        "squared_error": float((y - p) ** 2),
                        "absolute_error": float(abs(y - p)),
                        "tail_regime": str(regime),
                        "tail_low_threshold": low,
                        "tail_high_threshold": high,
                    }
                )
    return pd.DataFrame(rows)


def summarize_calibration_predictions(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return per-split and pooled/development summaries for 06C candidates."""

    split_rows: list[dict[str, Any]] = []
    for (method, split_id), group in predictions.groupby(["method", "split_id"], sort=True):
        y = group["observed"].to_numpy(float)
        p = group["predicted"].to_numpy(float)
        tail = group["tail_regime"].ne("center")
        center = ~tail
        low = group["tail_regime"].eq("low")
        high = group["tail_regime"].eq("high")
        split_rows.append(
            {
                "method": method,
                "split_id": split_id,
                "n": int(len(group)),
                "rmse": _rmse(y, p),
                "tail_rmse": _rmse(y[tail], p[tail]) if tail.any() else np.nan,
                "center_rmse": _rmse(y[center], p[center]) if center.any() else np.nan,
                "low_rmse": _rmse(y[low], p[low]) if low.any() else np.nan,
                "high_rmse": _rmse(y[high], p[high]) if high.any() else np.nan,
                "low_bias": float(np.mean(y[low] - p[low])) if low.any() else np.nan,
                "high_bias": float(np.mean(y[high] - p[high])) if high.any() else np.nan,
            }
        )
    split = pd.DataFrame(split_rows)

    summary_rows: list[dict[str, Any]] = []
    for method, group in predictions.groupby("method", sort=True):
        metrics = split.loc[split["method"].eq(method)]
        y = group["observed"].to_numpy(float)
        p = group["predicted"].to_numpy(float)
        tail = group["tail_regime"].ne("center")
        center = ~tail
        low = group["tail_regime"].eq("low")
        high = group["tail_regime"].eq("high")
        summary_rows.append(
            {
                "method": method,
                "n_splits": int(metrics["split_id"].nunique()),
                "n_predictions": int(len(group)),
                "rmse_mean": float(metrics["rmse"].mean()),
                "rmse_median": float(metrics["rmse"].median()),
                "rmse_worst": float(metrics["rmse"].max()),
                "pooled_rmse": _rmse(y, p),
                "tail_pooled_rmse": _rmse(y[tail], p[tail]),
                "center_pooled_rmse": _rmse(y[center], p[center]),
                "low_pooled_rmse": _rmse(y[low], p[low]),
                "high_pooled_rmse": _rmse(y[high], p[high]),
                "low_bias": float(np.mean(y[low] - p[low])),
                "high_bias": float(np.mean(y[high] - p[high])),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values(
        ["rmse_mean", "pooled_rmse"], ascending=True
    ).reset_index(drop=True)
    return split, summary


def _summary_for_splits(
    predictions: pd.DataFrame,
    split_metrics: pd.DataFrame,
    method: str,
    split_ids: set[str],
) -> dict[str, float]:
    p = predictions.loc[
        predictions["method"].eq(method) & predictions["split_id"].isin(split_ids)
    ]
    s = split_metrics.loc[
        split_metrics["method"].eq(method) & split_metrics["split_id"].isin(split_ids)
    ]
    y = p["observed"].to_numpy(float)
    pred = p["predicted"].to_numpy(float)
    tail = p["tail_regime"].ne("center").to_numpy()
    center = ~tail
    return {
        "rmse_mean": float(s["rmse"].mean()),
        "pooled_rmse": _rmse(y, pred),
        "tail_rmse": _rmse(y[tail], pred[tail]),
        "center_rmse": _rmse(y[center], pred[center]),
    }


def leave_one_split_out_calibration_selection(
    predictions: pd.DataFrame,
    split_metrics: pd.DataFrame,
    *,
    center_tolerance_absolute: float,
    center_tolerance_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Select a calibration on other splits, then score it on the held-out split."""

    all_splits = sorted(predictions["split_id"].unique())
    methods = sorted(set(predictions["method"]) - {"Local04D"})
    selection_rows: list[dict[str, Any]] = []
    holdout_rows: list[pd.DataFrame] = []

    for holdout in all_splits:
        train_splits = set(all_splits) - {holdout}
        incumbent = _summary_for_splits(
            predictions, split_metrics, "Local04D", train_splits
        )
        center_tol = max(
            float(center_tolerance_absolute),
            float(center_tolerance_fraction) * incumbent["center_rmse"],
        )
        eligible: list[tuple[str, dict[str, float]]] = []
        for method in methods:
            metrics = _summary_for_splits(
                predictions, split_metrics, method, train_splits
            )
            passes = (
                metrics["rmse_mean"] < incumbent["rmse_mean"]
                and metrics["pooled_rmse"] < incumbent["pooled_rmse"]
                and metrics["tail_rmse"] < incumbent["tail_rmse"]
                and metrics["center_rmse"] <= incumbent["center_rmse"] + center_tol
            )
            if passes:
                eligible.append((method, metrics))

        if eligible:
            selected, selected_metrics = min(
                eligible,
                key=lambda item: (item[1]["rmse_mean"], item[1]["pooled_rmse"]),
            )
        else:
            selected = "Local04D"
            selected_metrics = incumbent

        selected_holdout = predictions.loc[
            predictions["split_id"].eq(holdout)
            & predictions["method"].eq(selected)
        ].copy()
        selected_holdout["selected_method"] = selected
        holdout_rows.append(selected_holdout)
        selection_rows.append(
            {
                "holdout_split": holdout,
                "selected_method": selected,
                "n_eligible": int(len(eligible)),
                "train_rmse_mean": selected_metrics["rmse_mean"],
                "train_pooled_rmse": selected_metrics["pooled_rmse"],
                "train_tail_rmse": selected_metrics["tail_rmse"],
                "train_center_rmse": selected_metrics["center_rmse"],
                "incumbent_train_rmse_mean": incumbent["rmse_mean"],
                "incumbent_train_pooled_rmse": incumbent["pooled_rmse"],
                "incumbent_train_tail_rmse": incumbent["tail_rmse"],
                "incumbent_train_center_rmse": incumbent["center_rmse"],
                "center_tolerance": center_tol,
            }
        )

    selected_predictions = pd.concat(holdout_rows, ignore_index=True)
    selections = pd.DataFrame(selection_rows)

    y = selected_predictions["observed"].to_numpy(float)
    p = selected_predictions["predicted"].to_numpy(float)
    tail = selected_predictions["tail_regime"].ne("center").to_numpy()
    center = ~tail

    incumbent_rows = predictions.loc[predictions["method"].eq("Local04D")].copy()
    iy = incumbent_rows["observed"].to_numpy(float)
    ip = incumbent_rows["predicted"].to_numpy(float)
    itail = incumbent_rows["tail_regime"].ne("center").to_numpy()
    icenter = ~itail

    selected_split_rmse = pd.Series(
        {
            split_id: _rmse(
                group["observed"].to_numpy(float),
                group["predicted"].to_numpy(float),
            )
            for split_id, group in selected_predictions.groupby("split_id", sort=True)
        },
        dtype=float,
    )
    incumbent_split_rmse = pd.Series(
        {
            split_id: _rmse(
                group["observed"].to_numpy(float),
                group["predicted"].to_numpy(float),
            )
            for split_id, group in incumbent_rows.groupby("split_id", sort=True)
        },
        dtype=float,
    )

    summary = {
        "n_splits": int(len(all_splits)),
        "selected_rmse_mean": float(selected_split_rmse.mean()),
        "selected_pooled_rmse": _rmse(y, p),
        "selected_tail_rmse": _rmse(y[tail], p[tail]),
        "selected_center_rmse": _rmse(y[center], p[center]),
        "incumbent_rmse_mean": float(incumbent_split_rmse.mean()),
        "incumbent_pooled_rmse": _rmse(iy, ip),
        "incumbent_tail_rmse": _rmse(iy[itail], ip[itail]),
        "incumbent_center_rmse": _rmse(iy[icenter], ip[icenter]),
        "rmse_mean_improvement": float(
            incumbent_split_rmse.mean() - selected_split_rmse.mean()
        ),
        "pooled_rmse_improvement": float(_rmse(iy, ip) - _rmse(y, p)),
        "tail_rmse_improvement": float(
            _rmse(iy[itail], ip[itail]) - _rmse(y[tail], p[tail])
        ),
        "center_rmse_change": float(
            _rmse(y[center], p[center]) - _rmse(iy[icenter], ip[icenter])
        ),
        "selection_counts": {
            str(key): int(value)
            for key, value in selections["selected_method"].value_counts().to_dict().items()
        },
    }
    return selections, selected_predictions, summary
