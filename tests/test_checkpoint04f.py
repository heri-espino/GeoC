from __future__ import annotations

import pandas as pd
import pytest

from geocebada.evaluation.checkpoint04f import (
    build_final_prediction_diagnostics,
    build_final_prediction_table,
    build_local_graph_blends,
    leave_one_split_out_blend_selection,
    summarize_blend_predictions,
)


def _pseudo_predictions() -> pd.DataFrame:
    rows = []
    for split_id, observed, local, graph in [
        ("s1", [1.0, 2.0], [1.1, 1.9], [0.9, 2.1]),
        ("s2", [1.5, 2.5], [1.6, 2.4], [1.4, 2.6]),
        ("s3", [2.0, 3.0], [2.1, 2.9], [1.9, 3.1]),
    ]:
        for parcel_id, y, lp, gp in zip(
            ["A", "B"],
            observed,
            local,
            graph,
            strict=True,
        ):
            rows.append(
                {
                    "family": "target_matched",
                    "split_id": split_id,
                    "method": "Baseline_Local04D",
                    "ID_POLIGONO": parcel_id,
                    "observed": y,
                    "predicted": lp,
                }
            )
            rows.append(
                {
                    "family": "target_matched",
                    "split_id": split_id,
                    "method": "Baseline_Graph04D",
                    "ID_POLIGONO": parcel_id,
                    "observed": y,
                    "predicted": gp,
                }
            )
    return pd.DataFrame(rows)


def test_local_graph_blends_and_loso_cover_every_holdout() -> None:
    blends = build_local_graph_blends(
        _pseudo_predictions(),
        family="target_matched",
        graph_weights=[0.0, 0.5, 1.0],
    )
    summary = summarize_blend_predictions(blends)
    selection, selected = leave_one_split_out_blend_selection(blends)

    assert set(summary["graph_weight"]) == {0.0, 0.5, 1.0}
    assert len(selection) == 3
    assert selected["split_id"].nunique() == 3
    assert len(selected) == 6


def test_final_prediction_table_requires_exact_row_count() -> None:
    candidates = pd.DataFrame(
        {
            "ID_POLIGONO": ["B", "A", "A", "B"],
            "method": ["final", "final", "other", "other"],
            "predicted": [2.0, 1.0, 9.0, 9.0],
            "x_support_score": [0.2, 0.1, 0.1, 0.2],
            "x_support_tier": ["mid", "low", "low", "mid"],
        }
    )
    result = build_final_prediction_table(
        candidates,
        selected_method="final",
        target_column="RENDIMIENTO_T_HA",
        expected_rows=2,
    )

    assert list(result["ID_POLIGONO"]) == ["A", "B"]
    assert list(result["RENDIMIENTO_T_HA"]) == [1.0, 2.0]

    with pytest.raises(ValueError):
        build_final_prediction_table(
            candidates,
            selected_method="final",
            target_column="RENDIMIENTO_T_HA",
            expected_rows=3,
        )


def test_final_diagnostics_preserve_support_and_compute_disagreement() -> None:
    candidates = pd.DataFrame(
        {
            "ID_POLIGONO": ["A", "B", "A", "B"],
            "method": [
                "Baseline_Local04D",
                "Baseline_Local04D",
                "Baseline_Graph04D",
                "Baseline_Graph04D",
            ],
            "predicted": [1.0, 2.0, 1.4, 1.8],
            "x_support_score": [0.1, 0.8, 0.1, 0.8],
            "x_support_tier": ["low", "high", "low", "high"],
        }
    )

    result = build_final_prediction_diagnostics(
        candidates,
        selected_method="Baseline_Local04D",
        secondary_method="Baseline_Graph04D",
        diagnostic_graph_weight=0.25,
    )

    assert list(result["x_support_tier"]) == ["low", "high"]
    assert list(result["local_graph_abs_disagreement"]) == pytest.approx([0.4, 0.2])
    assert list(result["diagnostic_local_graph_blend"]) == pytest.approx([1.1, 1.95])
