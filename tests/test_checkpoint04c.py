from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from geocebada.evaluation.checkpoint04c import (
    blend_anchor_predictions,
    residual_correlation_table,
    resolve_agronomic_competition_features,
)


def test_resolve_agronomic_competition_features_filters_modes_and_types() -> None:
    frame = pd.DataFrame(
        {
            "keep_clean": [1.0, 2.0],
            "keep_comp": [3.0, 4.0],
            "drop_mode": [5.0, 6.0],
            "drop_text": ["a", "b"],
        }
    )
    manifest = {
        "features": [
            {"column": "keep_clean", "mode": "clean"},
            {"column": "keep_comp", "mode": "competition"},
            {"column": "drop_mode", "mode": "diagnostic"},
            {"column": "drop_text", "mode": "competition"},
        ]
    }

    result = resolve_agronomic_competition_features(frame, manifest)

    assert result == ("keep_clean", "keep_comp")


def test_blend_anchor_predictions_is_convex() -> None:
    baseline = np.array([2.0, 4.0])
    anchor = np.array([4.0, 8.0])

    result = blend_anchor_predictions(
        baseline,
        anchor,
        anchor_weight=0.25,
    )

    np.testing.assert_allclose(result, [2.5, 5.0])
    with pytest.raises(ValueError):
        blend_anchor_predictions(baseline, anchor, anchor_weight=1.1)


def test_residual_correlation_table_uses_matching_pseudo_rows() -> None:
    frame = pd.DataFrame(
        {
            "family": ["target_matched"] * 6,
            "split_id": ["s1"] * 6,
            "ID_POLIGONO": ["A", "B", "A", "B", "A", "B"],
            "method": ["m1", "m1", "m2", "m2", "m3", "m3"],
            "observed": [1.0, 2.0] * 3,
            "predicted": [0.8, 2.2, 0.9, 2.1, 1.2, 1.8],
        }
    )

    result = residual_correlation_table(
        frame,
        family="target_matched",
        methods=["m1", "m2", "m3"],
    )

    assert len(result) == 3
    assert set(result["n_pairs"]) == {2}
    assert set(result["method_a"]).union(result["method_b"]) == {"m1", "m2", "m3"}
