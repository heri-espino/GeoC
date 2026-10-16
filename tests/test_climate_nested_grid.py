"""Nested grid search tests: coverage, fair comparisons, and label isolation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from geocebada.evaluation.area_weighted_climate_comparison import Features
from geocebada.evaluation.checkpoint10_pca import PcaData
from geocebada.evaluation.climate_leave_one_out import MODELS
from geocebada.evaluation.climate_nested_grid import (
    FROZEN,
    TUNED_SUFFIX,
    grid_candidates,
    summarize_nested_grid,
    tune_one_outer_parcel,
)

ID = "ID_POLIGONO"


@pytest.fixture(scope="module")
def features():
    ids = [f"p{i:03d}" for i in range(197)]
    rng = np.random.default_rng(128)
    x = rng.normal(size=(197, 20))
    geo = np.abs(np.arange(197)[:, None] - np.arange(197)[None, :]).astype(float)
    y = pd.Series(3.5 + rng.normal(0, 0.5, 138), index=ids[:138])
    data = PcaData(
        ids=ids,
        index={pid: i for i, pid in enumerate(ids)},
        y=y,
        geographic=geo,
        x={20: x},
        variance={20: .6},
        folds={},
        parity_max_diff=0.,
    )
    spaces = {
        MODELS[0]: x,
        MODELS[1]: np.column_stack((x, rng.normal(size=(197, 6)))),
        MODELS[2]: np.column_stack((x, rng.normal(size=(197, 6)))),
        MODELS[3]: np.column_stack((x, rng.normal(size=(197, 12)))),
    }
    return Features(baseline=data, spaces=spaces)


def test_grid_size_and_frozen_control():
    focused = grid_candidates()
    wide = grid_candidates(wide=True)
    assert len(focused) == 60
    assert len(wide) == 168
    assert focused[0] == wide[0] == FROZEN
    assert len(set(focused)) == len(focused)
    assert len(set(wide)) == len(wide)


def test_one_outer_has_frozen_and_four_tuned_predictions(features):
    held = features.baseline.y.index[0]
    predictions, selections = tune_one_outer_parcel(
        features, held, candidates=[FROZEN, (12, 10., 1.)],
    )
    assert len(predictions) == 5
    assert len(selections) == 4
    assert set(predictions["model"]) == {
        MODELS[0], *(m + TUNED_SUFFIX for m in MODELS)
    }
    assert set(selections["model"]) == set(MODELS)
    assert selections["inner_predictions"].eq(137).all()
    assert selections["inner_folds"].eq(5).all()
    assert predictions["outer_train"].eq(137).all()
    assert all(
        (int(row.k), float(row.alpha), float(row.power))
        in {FROZEN, (12, 10., 1.)}
        for row in selections.itertuples()
    )


def test_outer_y_never_controls_inner_selection_or_prediction(features):
    held = features.baseline.y.index[1]
    before, selections = tune_one_outer_parcel(
        features, held, candidates=[FROZEN, (12, 10., 1.)],
        models=(MODELS[0],),
    )
    original = float(features.baseline.y.loc[held])
    features.baseline.y.loc[held] = original + 4000.
    try:
        after, selections_changed = tune_one_outer_parcel(
            features, held, candidates=[FROZEN, (12, 10., 1.)],
            models=(MODELS[0],),
        )
    finally:
        features.baseline.y.loc[held] = original
    np.testing.assert_allclose(before["predicted"], after["predicted"], rtol=0, atol=0)
    pd.testing.assert_frame_equal(selections, selections_changed)
    assert (before["observed"] != after["observed"]).all()


def test_complete_paired_summary_and_incomplete_rejection():
    ids = [f"p{i:03d}" for i in range(138)]
    truth = np.linspace(1., 7., 138)
    rows, selections = [], []
    for i, pid in enumerate(ids):
        for j, model in enumerate([MODELS[0], *(m + TUNED_SUFFIX for m in MODELS)]):
            rows.append({
                ID: pid, "model": model,
                "observed": float(truth[i]),
                "predicted": float(truth[i] + 0.1 * (j + 1)),
                "outer_train": 137, "outer_heldout": 1,
            })
        for model in MODELS:
            selections.append({
                ID: pid, "model": model,
                "k": 24, "alpha": 30., "power": 1.,
                "inner_cv_rmse": 0.3, "frozen_inner_cv_rmse": 0.3,
                "inner_folds": 5, "inner_predictions": 137,
                "candidates": 60, "selected_frozen": True,
            })
    prediction = pd.DataFrame(rows)
    selection = pd.DataFrame(selections)
    scores, paired, counts = summarize_nested_grid(prediction, selection, bootstrap_count=100)
    assert len(scores) == 5
    assert len(paired) == 4
    assert len(counts) == 4
    assert scores.iloc[0]["model"] == MODELS[0]
    assert scores.iloc[0]["rmse"] == pytest.approx(.1)
    with pytest.raises(ValueError, match="138"):
        summarize_nested_grid(prediction.iloc[:-5], selection, bootstrap_count=100)
    with pytest.raises(ValueError, match="Duplicate"):
        summarize_nested_grid(
            pd.concat([prediction, prediction.iloc[:1]]),
            selection, bootstrap_count=100,
        )
