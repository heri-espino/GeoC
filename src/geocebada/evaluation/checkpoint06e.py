"""Checkpoint 06E supervised center/tail mixture.

Local04D remains the anchor. Two supervised X-only tail gates estimate
P(low|X) and P(high|X). A center specialist learns honest Local04D residuals
only on pseudo-train center parcels. Low/high experts are deliberately simple:
strongly shrunken mean residual corrections from the corresponding pseudo-train
tail.

For each target-matched split, every supervised quantity is fit only on the 97
visible pseudo-train labels and their cross-fitted base predictions. The 41
pseudo-target rows are scored only after the mixture is frozen for that split.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from geocebada.evaluation.checkpoint06 import ID_COLUMN, _numeric_feature_families


def _rmse(observed: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(observed, predicted)))


def _gate_pipeline(c_value: float, random_state: int) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=float(c_value),
                    class_weight="balanced",
                    max_iter=3000,
                    solver="liblinear",
                    random_state=int(random_state),
                ),
            ),
        ]
    )


def _center_pipeline(
    *,
    alpha: float,
    representation: str,
    pca_variance: float,
) -> Pipeline:
    steps: list[tuple[str, Any]] = [
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler()),
    ]
    if representation == "all_agronomic_pca":
        steps.append(
            (
                "pca",
                PCA(
                    n_components=float(pca_variance),
                    svd_solver="full",
                ),
            )
        )
    elif representation != "cross_domain":
        raise ValueError(f"Unknown center representation: {representation}")
    steps.append(("ridge", Ridge(alpha=float(alpha))))
    return Pipeline(steps)


def normalize_gate_probabilities(
    p_low: np.ndarray,
    p_high: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert separate low/high probabilities into normalized low/center/high weights."""

    p_low = np.clip(np.asarray(p_low, dtype=float), 0.0, 1.0)
    p_high = np.clip(np.asarray(p_high, dtype=float), 0.0, 1.0)
    raw_center = (1.0 - p_low) * (1.0 - p_high)
    denominator = p_low + p_high + raw_center
    denominator = np.where(denominator <= 1.0e-12, 1.0, denominator)
    return (
        p_low / denominator,
        raw_center / denominator,
        p_high / denominator,
    )


def fit_tail_gates(
    train_oof: pd.DataFrame,
    agronomic: pd.DataFrame,
    manifest: Mapping[str, Any],
    test_ids: Sequence[str],
    *,
    low_family: str,
    high_family: str,
    c_value: float,
    low_quantile: float,
    high_quantile: float,
    random_state: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Fit fold-valid low/high classifiers and return soft regime weights."""

    families = _numeric_feature_families(agronomic, manifest)
    if low_family not in families or high_family not in families:
        raise KeyError("Configured 06E gate family not found in agronomic manifest.")

    train = train_oof.copy()
    train[ID_COLUMN] = train[ID_COLUMN].astype(str)
    feature_frame = agronomic.copy()
    feature_frame[ID_COLUMN] = feature_frame[ID_COLUMN].astype(str)
    feature_frame = feature_frame.set_index(ID_COLUMN)

    train_ids = train[ID_COLUMN].tolist()
    observed = pd.to_numeric(train["observed"], errors="raise").to_numpy(float)
    low_threshold = float(np.quantile(observed, float(low_quantile)))
    high_threshold = float(np.quantile(observed, float(high_quantile)))
    y_low = (observed <= low_threshold).astype(int)
    y_high = (observed >= high_threshold).astype(int)

    low_model = _gate_pipeline(float(c_value), int(random_state))
    high_model = _gate_pipeline(float(c_value), int(random_state) + 1)
    low_model.fit(feature_frame.loc[train_ids, families[low_family]], y_low)
    high_model.fit(feature_frame.loc[train_ids, families[high_family]], y_high)

    test_ids = [str(value) for value in test_ids]
    p_low_raw = low_model.predict_proba(
        feature_frame.loc[test_ids, families[low_family]]
    )[:, 1]
    p_high_raw = high_model.predict_proba(
        feature_frame.loc[test_ids, families[high_family]]
    )[:, 1]
    p_low, p_center, p_high = normalize_gate_probabilities(p_low_raw, p_high_raw)

    weights = pd.DataFrame(
        {
            ID_COLUMN: test_ids,
            "p_low_raw": p_low_raw,
            "p_high_raw": p_high_raw,
            "w_low": p_low,
            "w_center": p_center,
            "w_high": p_high,
        }
    )
    return weights, {
        "low_threshold": low_threshold,
        "high_threshold": high_threshold,
        "n_low_train": int(y_low.sum()),
        "n_center_train": int(
            np.sum((observed > low_threshold) & (observed < high_threshold))
        ),
        "n_high_train": int(y_high.sum()),
    }


def fit_center_residual_expert(
    train_oof: pd.DataFrame,
    agronomic: pd.DataFrame,
    manifest: Mapping[str, Any],
    test_ids: Sequence[str],
    *,
    representation: str,
    alpha: float,
    pca_variance: float,
    low_threshold: float,
    high_threshold: float,
) -> np.ndarray:
    """Fit a center-only Ridge residual expert from honest Local04D pseudo-train residuals."""

    families = _numeric_feature_families(agronomic, manifest)
    feature_frame = agronomic.copy()
    feature_frame[ID_COLUMN] = feature_frame[ID_COLUMN].astype(str)
    feature_frame = feature_frame.set_index(ID_COLUMN)

    train = train_oof.copy()
    train[ID_COLUMN] = train[ID_COLUMN].astype(str)
    observed = pd.to_numeric(train["observed"], errors="raise").to_numpy(float)
    local = pd.to_numeric(train["Local04D"], errors="raise").to_numpy(float)
    center_mask = (observed > float(low_threshold)) & (observed < float(high_threshold))
    if int(center_mask.sum()) < 20:
        raise ValueError("06E center expert has too few pseudo-train center rows.")

    if representation == "cross_domain":
        columns = families["cross_domain"]
    elif representation == "all_agronomic_pca":
        columns = families["all_agronomic"]
    else:
        raise ValueError(f"Unknown center representation: {representation}")

    center_ids = train.loc[center_mask, ID_COLUMN].tolist()
    residual = observed[center_mask] - local[center_mask]
    model = _center_pipeline(
        alpha=float(alpha),
        representation=str(representation),
        pca_variance=float(pca_variance),
    )
    model.fit(feature_frame.loc[center_ids, columns], residual)
    return np.asarray(
        model.predict(feature_frame.loc[[str(value) for value in test_ids], columns]),
        dtype=float,
    )


def _shrunken_tail_offsets(
    train_oof: pd.DataFrame,
    *,
    low_threshold: float,
    high_threshold: float,
    shrink: float,
) -> tuple[float, float]:
    observed = pd.to_numeric(train_oof["observed"], errors="raise").to_numpy(float)
    local = pd.to_numeric(train_oof["Local04D"], errors="raise").to_numpy(float)
    residual = observed - local
    low = residual[observed <= float(low_threshold)]
    high = residual[observed >= float(high_threshold)]
    if len(low) < 5 or len(high) < 5:
        raise ValueError("06E tail correction has too few pseudo-train tail rows.")
    return float(shrink * np.mean(low)), float(shrink * np.mean(high))


def candidate_names(config: Mapping[str, Any]) -> list[str]:
    """Return the frozen supervised mixture candidate universe."""

    names = ["Local04D"]
    for shrink in config["mixture"]["tail_shrink_values"]:
        token = str(float(shrink)).replace(".", "p")
        names.append(f"TailGate_s{token}")
    for representation in config["center"]["representations"]:
        rep_token = "Cross" if representation == "cross_domain" else "G1PCA"
        for alpha in config["center"]["ridge_alphas"]:
            alpha_token = str(float(alpha)).replace(".", "p")
            names.append(f"Center_{rep_token}_a{alpha_token}")
            for shrink in config["mixture"]["tail_shrink_values"]:
                shrink_token = str(float(shrink)).replace(".", "p")
                names.append(
                    f"CenterTail_{rep_token}_a{alpha_token}_s{shrink_token}"
                )
    return names


def build_supervised_mixture_predictions(
    base_oof: pd.DataFrame,
    pseudo_targets: pd.DataFrame,
    agronomic: pd.DataFrame,
    manifest: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build honest target-matched 06E predictions and gate diagnostics."""

    train_all = base_oof.loc[base_oof["family"].eq("target_matched")].copy()
    test_all = pseudo_targets.loc[pseudo_targets["family"].eq("target_matched")].copy()
    rows: list[dict[str, Any]] = []
    gate_rows: list[pd.DataFrame] = []
    expected_names = set(candidate_names(config))

    for split_id, test in test_all.groupby("split_id", sort=True):
        train = train_all.loc[train_all["split_id"].eq(split_id)].copy()
        if train[ID_COLUMN].duplicated().any() or test[ID_COLUMN].duplicated().any():
            raise ValueError(f"{split_id}: duplicated 06E parcel rows.")

        weights, thresholds = fit_tail_gates(
            train,
            agronomic,
            manifest,
            test[ID_COLUMN].astype(str).tolist(),
            low_family=str(config["gates"]["low_family"]),
            high_family=str(config["gates"]["high_family"]),
            c_value=float(config["gates"]["logistic_c"]),
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
        local_test = pd.to_numeric(
            test_indexed["Local04D"], errors="raise"
        ).to_numpy(float)
        observed = pd.to_numeric(
            test_indexed["observed"], errors="raise"
        ).to_numpy(float)

        predictions: dict[str, np.ndarray] = {"Local04D": local_test.copy()}
        tail_offsets: dict[float, tuple[float, float]] = {}
        for shrink in config["mixture"]["tail_shrink_values"]:
            shrink = float(shrink)
            low_offset, high_offset = _shrunken_tail_offsets(
                train,
                low_threshold=thresholds["low_threshold"],
                high_threshold=thresholds["high_threshold"],
                shrink=shrink,
            )
            tail_offsets[shrink] = (low_offset, high_offset)
            token = str(shrink).replace(".", "p")
            predictions[f"TailGate_s{token}"] = (
                local_test
                + weights_indexed["w_low"].to_numpy(float) * low_offset
                + weights_indexed["w_high"].to_numpy(float) * high_offset
            )

        for representation in config["center"]["representations"]:
            rep_token = "Cross" if representation == "cross_domain" else "G1PCA"
            for alpha in config["center"]["ridge_alphas"]:
                alpha = float(alpha)
                alpha_token = str(alpha).replace(".", "p")
                center_delta = fit_center_residual_expert(
                    train,
                    agronomic,
                    manifest,
                    test_indexed.index.tolist(),
                    representation=str(representation),
                    alpha=alpha,
                    pca_variance=float(config["center"]["pca_variance"]),
                    low_threshold=thresholds["low_threshold"],
                    high_threshold=thresholds["high_threshold"],
                )
                predictions[f"Center_{rep_token}_a{alpha_token}"] = (
                    local_test
                    + weights_indexed["w_center"].to_numpy(float) * center_delta
                )
                for shrink, (low_offset, high_offset) in tail_offsets.items():
                    shrink_token = str(shrink).replace(".", "p")
                    predictions[
                        f"CenterTail_{rep_token}_a{alpha_token}_s{shrink_token}"
                    ] = (
                        local_test
                        + weights_indexed["w_center"].to_numpy(float) * center_delta
                        + weights_indexed["w_low"].to_numpy(float) * low_offset
                        + weights_indexed["w_high"].to_numpy(float) * high_offset
                    )

        if set(predictions) != expected_names:
            raise RuntimeError("06E generated candidate set differs from frozen config.")

        low_threshold = float(thresholds["low_threshold"])
        high_threshold = float(thresholds["high_threshold"])
        regimes = np.where(
            observed <= low_threshold,
            "low",
            np.where(observed >= high_threshold, "high", "center"),
        )
        for method, predicted in predictions.items():
            for parcel_id, y, p, regime in zip(
                test_indexed.index,
                observed,
                predicted,
                regimes,
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
                    }
                )

    return pd.DataFrame(rows), pd.concat(gate_rows, ignore_index=True)


def summarize_mixture_predictions(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize split-level and pooled 06E performance."""

    split_rows: list[dict[str, Any]] = []
    for (method, split_id), group in predictions.groupby(
        ["method", "split_id"], sort=True
    ):
        y = group["observed"].to_numpy(float)
        p = group["predicted"].to_numpy(float)
        low = group["tail_regime"].eq("low").to_numpy()
        high = group["tail_regime"].eq("high").to_numpy()
        center = group["tail_regime"].eq("center").to_numpy()
        tail = low | high
        split_rows.append(
            {
                "method": method,
                "split_id": split_id,
                "rmse": _rmse(y, p),
                "center_rmse": _rmse(y[center], p[center]),
                "tail_rmse": _rmse(y[tail], p[tail]),
                "low_rmse": _rmse(y[low], p[low]),
                "high_rmse": _rmse(y[high], p[high]),
                "center_bias": float(np.mean(y[center] - p[center])),
                "low_bias": float(np.mean(y[low] - p[low])),
                "high_bias": float(np.mean(y[high] - p[high])),
            }
        )
    split = pd.DataFrame(split_rows)

    summary_rows: list[dict[str, Any]] = []
    for method, group in predictions.groupby("method", sort=True):
        metrics = split.loc[split["method"].eq(method)]
        y = group["observed"].to_numpy(float)
        p = group["predicted"].to_numpy(float)
        low = group["tail_regime"].eq("low").to_numpy()
        high = group["tail_regime"].eq("high").to_numpy()
        center = group["tail_regime"].eq("center").to_numpy()
        tail = low | high
        summary_rows.append(
            {
                "method": method,
                "n_splits": int(metrics["split_id"].nunique()),
                "n_predictions": int(len(group)),
                "rmse_mean": float(metrics["rmse"].mean()),
                "rmse_median": float(metrics["rmse"].median()),
                "rmse_worst": float(metrics["rmse"].max()),
                "pooled_rmse": _rmse(y, p),
                "center_pooled_rmse": _rmse(y[center], p[center]),
                "tail_pooled_rmse": _rmse(y[tail], p[tail]),
                "low_pooled_rmse": _rmse(y[low], p[low]),
                "high_pooled_rmse": _rmse(y[high], p[high]),
                "center_bias": float(np.mean(y[center] - p[center])),
                "low_bias": float(np.mean(y[low] - p[low])),
                "high_bias": float(np.mean(y[high] - p[high])),
            }
        )
    return split, pd.DataFrame(summary_rows).sort_values(
        ["rmse_mean", "pooled_rmse"], ascending=True
    ).reset_index(drop=True)


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


def leave_one_split_out_mixture_selection(
    predictions: pd.DataFrame,
    split_metrics: pd.DataFrame,
    *,
    min_center_improvement: float,
    tail_tolerance_absolute: float,
    tail_tolerance_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Select 06E candidates on 15 splits and score the selected rule on the 16th."""

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
