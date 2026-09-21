"""Finalization helpers for Checkpoint 04F."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from geocebada.evaluation.checkpoint04b import regression_metrics

ID_COLUMN = "ID_POLIGONO"


def blend_method_name(graph_weight: float) -> str:
    """Return the canonical Local/Graph blend label for one graph weight."""

    weight = float(graph_weight)
    return f"LocalGraph_g{weight:.2f}".replace(".", "p")


def build_local_graph_blends(
    predictions: pd.DataFrame,
    *,
    family: str,
    graph_weights: Sequence[float],
    local_method: str = "Baseline_Local04D",
    graph_method: str = "Baseline_Graph04D",
) -> pd.DataFrame:
    """Build fixed Local04D/Graph04D blends for one validation family."""

    frame = predictions.loc[
        predictions["family"].eq(str(family))
        & predictions["method"].isin([str(local_method), str(graph_method)])
    ].copy()
    required = {"family", "split_id", "method", ID_COLUMN, "observed", "predicted"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"Missing prediction columns: {missing}")

    rows: list[pd.DataFrame] = []
    for split_id, split_frame in frame.groupby("split_id", sort=False):
        local = split_frame.loc[
            split_frame["method"].eq(str(local_method)),
            [ID_COLUMN, "observed", "predicted"],
        ].copy()
        graph = split_frame.loc[
            split_frame["method"].eq(str(graph_method)),
            [ID_COLUMN, "observed", "predicted"],
        ].copy()
        local[ID_COLUMN] = local[ID_COLUMN].astype(str)
        graph[ID_COLUMN] = graph[ID_COLUMN].astype(str)
        merged = local.merge(
            graph,
            on=[ID_COLUMN, "observed"],
            how="inner",
            validate="one_to_one",
            suffixes=("_local", "_graph"),
        )
        if len(merged) != len(local) or len(merged) != len(graph):
            raise ValueError(f"Local/Graph row mismatch for split {split_id}.")

        for raw_weight in graph_weights:
            weight = float(raw_weight)
            if not 0.0 <= weight <= 1.0:
                raise ValueError("Graph blend weights must lie in [0, 1].")
            rows.append(
                pd.DataFrame(
                    {
                        "family": str(family),
                        "split_id": str(split_id),
                        "method": blend_method_name(weight),
                        ID_COLUMN: merged[ID_COLUMN].to_numpy(),
                        "observed": merged["observed"].to_numpy(dtype=float),
                        "predicted": (
                            (1.0 - weight)
                            * merged["predicted_local"].to_numpy(dtype=float)
                            + weight
                            * merged["predicted_graph"].to_numpy(dtype=float)
                        ),
                        "graph_weight": weight,
                    }
                )
            )

    if not rows:
        raise ValueError("No Local/Graph blend rows were generated.")
    return pd.concat(rows, ignore_index=True)


def summarize_blend_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Summarize fixed blend candidates using equal-weight split RMSE."""

    rows: list[dict[str, Any]] = []
    for method, method_frame in predictions.groupby("method", sort=False):
        split_rmses: list[float] = []
        for _, split_frame in method_frame.groupby("split_id", sort=False):
            split_rmses.append(
                regression_metrics(
                    split_frame["observed"],
                    split_frame["predicted"],
                )["rmse"]
            )
        pooled = regression_metrics(
            method_frame["observed"],
            method_frame["predicted"],
        )
        weight_values = method_frame["graph_weight"].dropna().unique()
        graph_weight = (
            float(weight_values[0])
            if len(weight_values) == 1
            else float("nan")
        )
        rows.append(
            {
                "method": str(method),
                "graph_weight": graph_weight,
                "n_splits": int(method_frame["split_id"].nunique()),
                "n_predictions": int(len(method_frame)),
                "rmse_mean": float(np.mean(split_rmses)),
                "rmse_median": float(np.median(split_rmses)),
                "rmse_worst": float(np.max(split_rmses)),
                "pooled_rmse": pooled["rmse"],
                "pooled_mae": pooled["mae"],
                "pooled_r2": pooled["r2"],
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["rmse_mean", "graph_weight"],
    ).reset_index(drop=True)


def leave_one_split_out_blend_selection(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select a fixed blend weight on all other splits and score the holdout."""

    split_ids = list(dict.fromkeys(predictions["split_id"].astype(str)))
    selection_rows: list[dict[str, Any]] = []
    selected_blocks: list[pd.DataFrame] = []

    for holdout in split_ids:
        training = predictions.loc[
            ~predictions["split_id"].eq(holdout)
        ].copy()
        validation = predictions.loc[
            predictions["split_id"].eq(holdout)
        ].copy()

        ranking = summarize_blend_predictions(training)
        if ranking.empty:
            raise ValueError("No blend candidates available for LOSO selection.")
        selected_method = str(ranking.iloc[0]["method"])
        selected_weight = float(ranking.iloc[0]["graph_weight"])
        selected = validation.loc[
            validation["method"].eq(selected_method)
        ].copy()
        metrics = regression_metrics(selected["observed"], selected["predicted"])
        selection_rows.append(
            {
                "holdout_split": str(holdout),
                "selected_method": selected_method,
                "selected_graph_weight": selected_weight,
                "selection_rmse": float(ranking.iloc[0]["rmse_mean"]),
                **{f"holdout_{key}": value for key, value in metrics.items()},
            }
        )
        selected_blocks.append(selected)

    return pd.DataFrame(selection_rows), pd.concat(
        selected_blocks,
        ignore_index=True,
    )


def build_final_prediction_table(
    candidates: pd.DataFrame,
    *,
    selected_method: str,
    target_column: str,
    expected_rows: int,
) -> pd.DataFrame:
    """Return the canonical two-column final prediction table."""

    selected = candidates.loc[
        candidates["method"].eq(str(selected_method)),
        [ID_COLUMN, "predicted"],
    ].copy()
    selected[ID_COLUMN] = selected[ID_COLUMN].astype(str)
    if len(selected) != int(expected_rows):
        raise ValueError(
            f"Expected {expected_rows} final predictions, found {len(selected)}."
        )
    if selected[ID_COLUMN].duplicated().any():
        raise ValueError("Final prediction candidate contains duplicate IDs.")
    if not np.isfinite(selected["predicted"].to_numpy(dtype=float)).all():
        raise ValueError("Final prediction candidate contains non-finite values.")
    return selected.rename(columns={"predicted": str(target_column)}).sort_values(
        ID_COLUMN
    ).reset_index(drop=True)


def build_final_prediction_diagnostics(
    candidates: pd.DataFrame,
    *,
    selected_method: str,
    secondary_method: str,
    diagnostic_graph_weight: float,
) -> pd.DataFrame:
    """Build support and Local-vs-Graph disagreement diagnostics for the 59 targets."""

    columns = [
        ID_COLUMN,
        "predicted",
        "x_support_score",
        "x_support_tier",
    ]
    local = candidates.loc[
        candidates["method"].eq(str(selected_method)),
        columns,
    ].copy()
    graph = candidates.loc[
        candidates["method"].eq(str(secondary_method)),
        [ID_COLUMN, "predicted"],
    ].copy()
    local[ID_COLUMN] = local[ID_COLUMN].astype(str)
    graph[ID_COLUMN] = graph[ID_COLUMN].astype(str)

    merged = local.merge(
        graph,
        on=ID_COLUMN,
        how="inner",
        validate="one_to_one",
        suffixes=("_final", "_graph"),
    )
    if len(merged) != len(local) or len(merged) != len(graph):
        raise ValueError("Final and secondary actual-target candidates do not align.")

    weight = float(diagnostic_graph_weight)
    merged["local_graph_abs_disagreement"] = np.abs(
        merged["predicted_final"] - merged["predicted_graph"]
    )
    merged["diagnostic_local_graph_blend"] = (
        (1.0 - weight) * merged["predicted_final"]
        + weight * merged["predicted_graph"]
    )
    return merged.sort_values(ID_COLUMN).reset_index(drop=True)
