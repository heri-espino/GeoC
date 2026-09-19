"""Checkpoint 03C.3 finalist equal-weight ensemble evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _metric_row(observed: pd.Series, predicted: pd.Series) -> dict[str, float]:
    y_true = observed.to_numpy(dtype=float)
    y_pred = predicted.to_numpy(dtype=float)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def build_finalist_ensemble_predictions(
    oof_predictions: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Build fixed finalist and equal-weight ensemble predictions from 03C.2 OOF rows."""

    identity = config["identity"]
    protocol_col = str(identity["protocol_column"])
    fold_col = str(identity["fold_column"])
    id_col = str(identity["id_column"])
    observed_col = str(identity["observed_column"])
    predicted_col = str(identity["predicted_column"])
    representation_col = str(identity["representation_column"])
    model_col = str(identity["model_column"])

    required = {
        protocol_col,
        fold_col,
        id_col,
        observed_col,
        predicted_col,
        representation_col,
        model_col,
    }
    missing = sorted(required - set(oof_predictions.columns))
    if missing:
        raise ValueError(f"03C.2 OOF table is missing required columns: {missing}")

    base_frames: list[pd.DataFrame] = []
    base_names = list(map(str, config["base_candidates"].keys()))
    for candidate_name, spec in config["base_candidates"].items():
        subset = oof_predictions.loc[
            oof_predictions[representation_col].eq(str(spec["representation"]))
            & oof_predictions[model_col].eq(str(spec["model"])),
            [protocol_col, fold_col, id_col, observed_col, predicted_col],
        ].copy()
        if subset.empty:
            raise ValueError(f"No OOF rows found for finalist {candidate_name!r}.")
        if subset.duplicated([protocol_col, id_col]).any():
            raise ValueError(
                f"Finalist {candidate_name!r} has duplicate protocol/parcel OOF rows."
            )
        subset["candidate"] = str(candidate_name)
        subset["candidate_type"] = "base"
        base_frames.append(subset)

    base_long = pd.concat(base_frames, ignore_index=True)
    consistency = base_long.groupby([protocol_col, id_col], observed=True).agg(
        observed_nunique=(observed_col, "nunique"),
        fold_nunique=(fold_col, "nunique"),
        candidate_count=("candidate", "nunique"),
    )
    if not consistency["observed_nunique"].eq(1).all():
        raise ValueError("Observed target differs across finalist OOF rows.")
    if not consistency["fold_nunique"].eq(1).all():
        raise ValueError("Outer fold differs across finalist OOF rows.")
    if not consistency["candidate_count"].eq(len(base_names)).all():
        raise ValueError("At least one finalist lacks OOF coverage for a protocol/parcel.")

    wide = (
        base_long.pivot(
            index=[protocol_col, fold_col, id_col, observed_col],
            columns="candidate",
            values=predicted_col,
        )
        .reset_index()
        .rename_axis(columns=None)
    )

    output_frames: list[pd.DataFrame] = []
    for candidate_name in base_names:
        frame = wide[[protocol_col, fold_col, id_col, observed_col]].copy()
        frame["candidate"] = candidate_name
        frame["candidate_type"] = "base"
        frame[predicted_col] = pd.to_numeric(wide[candidate_name], errors="raise")
        output_frames.append(frame)

    for ensemble_name, spec in config["ensembles"].items():
        members = list(map(str, spec["members"]))
        weights = np.asarray(spec["weights"], dtype=float)
        if len(members) != len(weights) or len(members) < 2:
            raise ValueError(
                f"Ensemble {ensemble_name!r} needs matching member/weight lists."
            )
        unknown = sorted(set(members) - set(base_names))
        if unknown:
            raise ValueError(
                f"Ensemble {ensemble_name!r} references unknown finalists: {unknown}"
            )
        if not np.isfinite(weights).all() or np.any(weights < 0):
            raise ValueError(f"Ensemble {ensemble_name!r} has invalid weights.")
        if not np.isclose(weights.sum(), 1.0, atol=1.0e-9):
            raise ValueError(f"Ensemble {ensemble_name!r} weights must sum to one.")

        matrix = wide.loc[:, members].to_numpy(dtype=float)
        frame = wide[[protocol_col, fold_col, id_col, observed_col]].copy()
        frame["candidate"] = str(ensemble_name)
        frame["candidate_type"] = "ensemble"
        frame[predicted_col] = matrix @ weights
        output_frames.append(frame)

    result = pd.concat(output_frames, ignore_index=True)
    result["residual"] = (
        pd.to_numeric(result[observed_col], errors="raise")
        - pd.to_numeric(result[predicted_col], errors="raise")
    )
    validate_finalist_oof_coverage(result, config)
    return result.sort_values(
        [protocol_col, "candidate", fold_col, id_col],
        kind="stable",
    ).reset_index(drop=True)


def validate_finalist_oof_coverage(
    predictions: pd.DataFrame,
    config: Mapping[str, Any],
) -> None:
    """Require complete one-row-per-parcel OOF coverage for every 03C.3 candidate."""

    identity = config["identity"]
    protocol_col = str(identity["protocol_column"])
    id_col = str(identity["id_column"])
    expected_rows = int(config["validation"]["expected_training_rows"])
    expected_candidates = {
        *map(str, config["base_candidates"].keys()),
        *map(str, config["ensembles"].keys()),
    }
    expected_protocols = {
        str(config["validation"]["state_protocol"]),
        str(config["validation"]["grouped_protocol"]),
    }

    if set(predictions["candidate"]) != expected_candidates:
        raise ValueError("03C.3 candidate set does not match the frozen config.")
    if set(predictions[protocol_col]) != expected_protocols:
        raise ValueError("03C.3 protocol set does not match the frozen config.")
    if predictions.duplicated([protocol_col, "candidate", id_col]).any():
        raise ValueError("Duplicate protocol/candidate/parcel OOF prediction detected.")

    counts = predictions.groupby(
        [protocol_col, "candidate"],
        observed=True,
    )[id_col].nunique()
    if not counts.eq(expected_rows).all():
        raise ValueError(
            "Every 03C.3 protocol/candidate must cover exactly "
            f"{expected_rows} training parcels."
        )


def summarize_finalist_ensemble_results(
    predictions: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Summarize fold, protocol, robustness and residual-correlation evidence."""

    identity = config["identity"]
    protocol_col = str(identity["protocol_column"])
    fold_col = str(identity["fold_column"])
    observed_col = str(identity["observed_column"])
    predicted_col = str(identity["predicted_column"])
    state_protocol = str(config["validation"]["state_protocol"])
    grouped_protocol = str(config["validation"]["grouped_protocol"])

    fold_rows: list[dict[str, Any]] = []
    protocol_rows: list[dict[str, Any]] = []

    for (protocol, candidate), subset in predictions.groupby(
        [protocol_col, "candidate"],
        observed=True,
        sort=True,
    ):
        candidate_type = str(subset["candidate_type"].iloc[0])
        fold_rmse: list[float] = []
        for fold, fold_subset in subset.groupby(fold_col, observed=True, sort=True):
            metrics = _metric_row(
                fold_subset[observed_col],
                fold_subset[predicted_col],
            )
            fold_rmse.append(metrics["rmse"])
            fold_rows.append(
                {
                    "protocol": str(protocol),
                    "candidate": str(candidate),
                    "candidate_type": candidate_type,
                    "fold": int(fold),
                    "n_rows": int(len(fold_subset)),
                    **metrics,
                }
            )

        pooled = _metric_row(subset[observed_col], subset[predicted_col])
        protocol_rows.append(
            {
                "protocol": str(protocol),
                "candidate": str(candidate),
                "candidate_type": candidate_type,
                "n_rows": int(len(subset)),
                "oof_rmse": pooled["rmse"],
                "oof_mae": pooled["mae"],
                "oof_r2": pooled["r2"],
                "fold_rmse_mean": float(np.mean(fold_rmse)),
                "fold_rmse_std": float(np.std(fold_rmse, ddof=1)),
            }
        )

    fold_metrics = pd.DataFrame(fold_rows).sort_values(
        ["protocol", "candidate", "fold"],
        kind="stable",
    )
    protocol_summary = pd.DataFrame(protocol_rows).sort_values(
        ["protocol", "oof_rmse", "candidate"],
        kind="stable",
    )

    state = protocol_summary.loc[
        protocol_summary["protocol"].eq(state_protocol),
        ["candidate", "candidate_type", "oof_rmse"],
    ].rename(columns={"oof_rmse": "state_oof_rmse"})
    grouped = protocol_summary.loc[
        protocol_summary["protocol"].eq(grouped_protocol),
        ["candidate", "oof_rmse"],
    ].rename(columns={"oof_rmse": "grouped_oof_rmse"})
    robustness = state.merge(grouped, on="candidate", how="inner", validate="one_to_one")
    robustness["mean_oof_rmse"] = (
        robustness["state_oof_rmse"] + robustness["grouped_oof_rmse"]
    ) / 2.0
    robustness["worst_protocol_rmse"] = robustness[
        ["state_oof_rmse", "grouped_oof_rmse"]
    ].max(axis=1)
    robustness["protocol_gap"] = (
        robustness["state_oof_rmse"] - robustness["grouped_oof_rmse"]
    ).abs()
    robustness = robustness.sort_values(
        ["worst_protocol_rmse", "mean_oof_rmse", "candidate"],
        kind="stable",
    ).reset_index(drop=True)

    base_names = list(map(str, config["base_candidates"].keys()))
    correlation_rows: list[dict[str, Any]] = []
    for protocol, subset in predictions.loc[
        predictions["candidate"].isin(base_names)
    ].groupby(protocol_col, observed=True, sort=True):
        pivot = subset.pivot(
            index=str(identity["id_column"]),
            columns="candidate",
            values="residual",
        )
        corr = pivot.loc[:, base_names].corr()
        for index, candidate_a in enumerate(base_names):
            for candidate_b in base_names[index + 1 :]:
                correlation_rows.append(
                    {
                        "protocol": str(protocol),
                        "candidate_a": candidate_a,
                        "candidate_b": candidate_b,
                        "residual_correlation": float(
                            corr.loc[candidate_a, candidate_b]
                        ),
                    }
                )
    residual_correlation = pd.DataFrame(correlation_rows).sort_values(
        ["protocol", "candidate_a", "candidate_b"],
        kind="stable",
    )

    return fold_metrics, protocol_summary, robustness, residual_correlation
