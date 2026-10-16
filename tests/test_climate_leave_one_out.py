"""One-heldout-parcel validation of the frozen geographic climate models."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from geocebada.evaluation.area_weighted_climate_comparison import Features
from geocebada.evaluation.checkpoint10_pca import PcaData
from geocebada.evaluation.climate_leave_one_out import (
    BASE, MODELS, TEMPORAL, TOUCHED, WEIGHTED,
    leave_one_out_predictions, leave_one_out_summary,
)

ID = "ID_POLIGONO"


@pytest.fixture(scope="module")
def example_features():
    ids = [f"p{i:03d}" for i in range(197)]
    rng = np.random.default_rng(78)
    x = rng.normal(size=(197, 20))
    positions = np.arange(197, dtype=float)
    distances = np.abs(positions[:, None] - positions[None, :])
    y = pd.Series(4 + rng.normal(size=138), index=ids[:138])
    data = PcaData(
        ids=ids,
        index={pid: i for i, pid in enumerate(ids)},
        y=y,
        geographic=distances,
        x={20: x},
        variance={20: 0.7},
        folds={},
        parity_max_diff=0.,
    )
    spaces = {
        BASE: x,
        TOUCHED: np.column_stack([x, rng.normal(size=(197, 6))]),
        WEIGHTED: np.column_stack([x, rng.normal(size=(197, 6))]),
        TEMPORAL: np.column_stack([x, rng.normal(size=(197, 12))]),
    }
    return Features(baseline=data, spaces=spaces)


def test_loo_each_parcel_once_all_four_models(example_features):
    predictions = leave_one_out_predictions(example_features)
    assert len(predictions) == 138 * 4
    assert predictions.groupby(ID)["model"].nunique().eq(4).all()
    assert predictions["outer_train"].eq(137).all()
    assert predictions["outer_heldout"].eq(1).all()
    scores, paired = leave_one_out_summary(predictions, bootstrap_count=100)
    assert len(scores) == 4
    assert len(paired) == 3
    assert scores["r2"].notna().all()
    assert set(scores["model"]) == set(MODELS)


def test_loo_no_heldout_y_leakage(example_features):
    pid = str(example_features.baseline.y.index[0])
    before = leave_one_out_predictions(example_features, holdout_ids=[pid])
    original = float(example_features.baseline.y.loc[pid])
    example_features.baseline.y.loc[pid] = original + 100000
    try:
        after = leave_one_out_predictions(example_features, holdout_ids=[pid])
    finally:
        example_features.baseline.y.loc[pid] = original
    np.testing.assert_allclose(before["predicted"], after["predicted"], atol=0, rtol=0)
    assert not np.array_equal(before["observed"], after["observed"])


def test_loo_summary_rejects_missing_or_repeated_predictions(example_features):
    one = leave_one_out_predictions(
        example_features, holdout_ids=[str(example_features.baseline.y.index[0])],
    )
    with pytest.raises(ValueError, match="138 parcels"):
        leave_one_out_summary(one, bootstrap_count=100)
    duplicated = pd.concat([one, one], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate"):
        leave_one_out_summary(duplicated, bootstrap_count=100)


def test_loo_rejects_unknown_test_parcel(example_features):
    with pytest.raises(ValueError, match="unique and have observed"):
        leave_one_out_predictions(example_features, holdout_ids=["p196"])
