"""Checkpoint 06F selective center expert with abstention.

Checkpoint 06E found a center-only residual signal but its soft gate leaked enough
correction into tail cases that the gain did not survive split-excluded selection.
06F tests one narrower post-hoc hypothesis: apply the frozen G1-PCA Ridge center
residual expert only when both supervised tail scores are below a conservative
cutoff; otherwise abstain and return Local04D unchanged.

Every supervised quantity is fit only on the visible pseudo-train labels for each
target-matched split. Pseudo-target y is used only after predictions are frozen.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from geocebada.evaluation.checkpoint06 import ID_COLUMN
from geocebada.evaluation.checkpoint06e import (
    fit_center_residual_expert,
    fit_tail_gates,
    summarize_mixture_predictions,
)


def _rmse(observed: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(observed, predicted)))


def _cutoff_token(cutoff: float) -> str:
    return str(float(cutoff)).replace(".", "p")


def _candidate_names(config: Mapping[str, Any]) -> list[str]:
    return [
        "Local04D",
        *[
            f"SelectiveCenter_t{_cutoff_token(float(cutoff))}"
            for cutoff in config["gate"]["tail_score_cutoffs"]
        ],
    ]


def _safe_center_mask(
    p_low_raw: np.ndarray,
    p_high_raw: np.ndarray,
    *,
    cutoff: float,
) -> np.ndarray:
    """Return rows whose two tail scores are both below the frozen cutoff."""

    p_low = np.asarray(p_low_raw, dtype=float)
    p_high = np.asarray(p_high_raw, dtype=float)
    if p_low.shape != p_high.shape:
        raise ValueError("06F low/high tail scores must have identical shape.")
    if not 0.0 < float(cutoff) < 1.0:
        raise ValueError("06F tail-score cutoff must lie strictly between 0 and 1.")
    return (p_low < float(cutoff)) & (p_high < float(cutoff))


def _apply_selective_center_correction(
    local_prediction: np.ndarray,
    center_delta: np.ndarray,
    p_low_raw: np.ndarray,
    p_high_raw: np.ndarray,
    *,
    cutoff: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply center residual correction only to high-confidence center rows."""

    local = np.asarray(local_prediction, dtype=float)
    delta = np.asarray(center_delta, dtype=float)
    if local.shape != delta.shape:
        raise ValueError("06F Local04D prediction and center delta must align.")
    route = _safe_center_mask(p_low_raw, p_high_raw, cutoff=float(cutoff))
    return local + route.astype(float) * delta, route


def _build_selective_center_predictions(
    base_oof: pd.DataFrame,
    pseudo_targets: pd.DataFrame,
    agronomic: pd.DataFrame,
    manifest: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build honest target-matched hard-routing predictions for Checkpoint 06F."""

    train_all = base_oof.loc[base_oof["family"].eq("target_matched")].copy()
    test_all = pseudo_targets.loc[
        pseudo_targets["family"].eq("target_matched")
    ].copy()
    expected = set(_candidate_names(config))
    rows: list[dict[str, Any]] = []
    gate_rows: list[pd.DataFrame] = []

    for split_id, test in test_all.groupby("split_id", sort=True):
        train = train_all.loc[train_all["split_id"].eq(split_id)].copy()
        if train[ID_COLUMN].duplicated().any() or test[ID_COLUMN].duplicated().any():
            raise ValueError(f"{split_id}: duplicated 06F parcel rows.")

        weights, thresholds = fit_tail_gates(
            train,
            agronomic,
            manifest,
            test[ID_COLUMN].astype(str).tolist(),
            low_family=str(config["gate"]["low_family"]),
            high_family=str(config["gate"]["high_family"]),
            c_value=float(config["gate"]["logistic_c"]),
            low_quantile=float(config["tails"]["low_quantile"]),
            high_quantile=float(config["tails"]["high_quantile"]),
            random_state=int(config["runtime"]["random_state"]),
        )
        weights["split_id"] = split_id
        gate_rows.append(weights)

        test_indexed = test.copy()
        test_indexed[ID_COLUMN] = test_indexed[ID_COLUMN].astype(str)
        test_indexed = test_indexed.set_index(ID_COLUMN)
        weights_indexed = weights.set_index(ID_COLUMN).loc[test_indexed.index]

        local = pd.to_numeric(
            test_indexed["Local04D"], errors="raise"
        ).to_numpy(float)
        observed = pd.to_numeric(
            test_indexed["observed"], errors="raise"
        ).to_numpy(float)
        center_delta = fit_center_residual_expert(
            train,
            agronomic,
            manifest,
            test_indexed.index.tolist(),
            representation=str(config["center"]["representation"]),
            alpha=float(config["center"]["ridge_alpha"]),
            pca_variance=float(config["center"]["pca_variance"]),
            low_threshold=float(thresholds["low_threshold"]),
            high_threshold=float(thresholds["high_threshold"]),
        )

        p_low = weights_indexed["p_low_raw"].to_numpy(float)
        p_high = weights_indexed["p_high_raw"].to_numpy(float)
        predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {
            "Local04D": (local.copy(), np.zeros(len(local), dtype=bool))
        }
        for cutoff in config["gate"]["tail_score_cutoffs"]:
            cutoff = float(cutoff)
            predicted, routed = _apply_selective_center_correction(
                local,
                center_delta,
                p_low,
                p_high,
                cutoff=cutoff,
            )
            predictions[f"SelectiveCenter_t{_cutoff_token(cutoff)}"] = (
                predicted,
                routed,
            )

        if set(predictions) != expected:
            raise RuntimeError("06F generated candidate set differs from frozen config.")

        low_threshold = float(thresholds["low_threshold"])
        high_threshold = float(thresholds["high_threshold"])
        regimes = np.where(
            observed <= low_threshold,
            "low",
            np.where(observed >= high_threshold, "high", "center"),
        )

        for method, (predicted, routed) in predictions.items():
            for parcel_id, y, p, regime, use_expert, low_score, high_score, delta in zip(
                test_indexed.index,
                observed,
                predicted,
                regimes,
                routed,
                p_low,
                p_high,
                center_delta,
                strict=True,
            ):
                rows.append(
                    {
                        "family": "target_matched",
                        "split_id": split_id,
                        ID_COLUMN: str(parcel_id),
                        "method": method,
                        "observed": float(y),
                        "predicted": float(p),
                        "residual": float(y - p),
                        "absolute_error": float(abs(y - p)),
                        "squared_error": float((y - p) ** 2),
                        "tail_regime": str(regime),
                        "tail_low_threshold": low_threshold,
                        "tail_high_threshold": high_threshold,
                        "center_expert_used": bool(use_expert),
                        "p_low_raw": float(low_score),
                        "p_high_raw": float(high_score),
                        "center_delta": float(delta),
                    }
                )

    return pd.DataFrame(rows), pd.concat(gate_rows, ignore_index=True)


def _summarize_selective_routing(predictions: pd.DataFrame) -> pd.DataFrame:
    """Summarize abstention coverage and true-regime routing after scoring."""

    rows: list[dict[str, Any]] = []
    for method, group in predictions.groupby("method", sort=True):
        used = group["center_expert_used"].astype(bool).to_numpy()
        center = group["tail_regime"].eq("center").to_numpy()
        tail = ~center
        n_used = int(used.sum())
        rows.append(
            {
                "method": method,
                "n_predictions": int(len(group)),
                "n_center_expert_used": n_used,
                "expert_use_rate": float(np.mean(used)),
                "true_center_use_rate": float(np.mean(used[center])),
                "true_tail_use_rate": float(np.mean(used[tail])),
                "routed_center_precision": (
                    float(np.mean(center[used])) if n_used else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("method").reset_index(drop=True)


def _subset_metrics(
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
    center = p["tail_regime"].eq("center").to_numpy()
    tail = ~center
    return {
        "rmse_mean": float(s["rmse"].mean()),
        "pooled_rmse": _rmse(y, pred),
        "center_rmse": _rmse(y[center], pred[center]),
        "tail_rmse": _rmse(y[tail], pred[tail]),
    }


def _leave_one_split_out_selective_selection(
    predictions: pd.DataFrame,
    split_metrics: pd.DataFrame,
    *,
    min_center_improvement: float,
    tail_tolerance_absolute: float,
    tail_tolerance_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Select a frozen hard gate on 15 splits and score it on the excluded split."""

    all_splits = sorted(predictions["split_id"].unique())
    candidates = sorted(set(predictions["method"]) - {"Local04D"})
    selection_rows: list[dict[str, Any]] = []
    holdout_rows: list[pd.DataFrame] = []

    for holdout in all_splits:
        train_splits = set(all_splits) - {holdout}
        incumbent = _subset_metrics(
            predictions,
            split_metrics,
            "Local04D",
            train_splits,
        )
        tail_tolerance = max(
            float(tail_tolerance_absolute),
            float(tail_tolerance_fraction) * incumbent["tail_rmse"],
        )
        eligible: list[tuple[str, dict[str, float]]] = []
        for method in candidates:
            metrics = _subset_metrics(
                predictions,
                split_metrics,
                method,
                train_splits,
            )
            passes = (
                metrics["rmse_mean"] < incumbent["rmse_mean"]
                and metrics["pooled_rmse"] < incumbent["pooled_rmse"]
                and metrics["center_rmse"]
                <= incumbent["center_rmse"] - float(min_center_improvement)
                and metrics["tail_rmse"]
                <= incumbent["tail_rmse"] + tail_tolerance
            )
            if passes:
                eligible.append((method, metrics))

        if eligible:
            selected, selected_metrics = min(
                eligible,
                key=lambda item: (
                    item[1]["rmse_mean"],
                    item[1]["center_rmse"],
                    item[1]["pooled_rmse"],
                ),
            )
        else:
            selected = "Local04D"
            selected_metrics = incumbent

        holdout_prediction = predictions.loc[
            predictions["split_id"].eq(holdout)
            & predictions["method"].eq(selected)
        ].copy()
        holdout_prediction["selected_method"] = selected
        holdout_rows.append(holdout_prediction)
        selection_rows.append(
            {
                "holdout_split": holdout,
                "selected_method": selected,
                "n_eligible": int(len(eligible)),
                "train_rmse_mean": selected_metrics["rmse_mean"],
                "train_pooled_rmse": selected_metrics["pooled_rmse"],
                "train_center_rmse": selected_metrics["center_rmse"],
                "train_tail_rmse": selected_metrics["tail_rmse"],
                "incumbent_train_rmse_mean": incumbent["rmse_mean"],
                "incumbent_train_pooled_rmse": incumbent["pooled_rmse"],
                "incumbent_train_center_rmse": incumbent["center_rmse"],
                "incumbent_train_tail_rmse": incumbent["tail_rmse"],
                "tail_tolerance": tail_tolerance,
            }
        )

    selections = pd.DataFrame(selection_rows)
    selected_predictions = pd.concat(holdout_rows, ignore_index=True)
    _, selected_summary = summarize_mixture_predictions(
        selected_predictions.assign(method="LOSO_Selected")
    )
    incumbent_rows = predictions.loc[predictions["method"].eq("Local04D")].copy()
    _, incumbent_summary = summarize_mixture_predictions(incumbent_rows)

    selected = selected_summary.iloc[0]
    incumbent = incumbent_summary.iloc[0]
    payload = {
        "n_splits": int(len(all_splits)),
        "selected_rmse_mean": float(selected["rmse_mean"]),
        "selected_pooled_rmse": float(selected["pooled_rmse"]),
        "selected_center_rmse": float(selected["center_pooled_rmse"]),
        "selected_tail_rmse": float(selected["tail_pooled_rmse"]),
        "incumbent_rmse_mean": float(incumbent["rmse_mean"]),
        "incumbent_pooled_rmse": float(incumbent["pooled_rmse"]),
        "incumbent_center_rmse": float(incumbent["center_pooled_rmse"]),
        "incumbent_tail_rmse": float(incumbent["tail_pooled_rmse"]),
        "rmse_mean_improvement": float(
            incumbent["rmse_mean"] - selected["rmse_mean"]
        ),
        "pooled_rmse_improvement": float(
            incumbent["pooled_rmse"] - selected["pooled_rmse"]
        ),
        "center_rmse_improvement": float(
            incumbent["center_pooled_rmse"] - selected["center_pooled_rmse"]
        ),
        "tail_rmse_change": float(
            selected["tail_pooled_rmse"] - incumbent["tail_pooled_rmse"]
        ),
        "selection_counts": {
            str(key): int(value)
            for key, value in selections["selected_method"].value_counts().to_dict().items()
        },
    }
    return selections, selected_predictions, payload
