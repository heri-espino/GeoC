"""Fast synthetic tests for widened Local04D Checkpoint 11 CV and safeguards."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from geocebada.evaluation.checkpoint10_pca import FROZEN, PcaData
from geocebada.evaluation.checkpoint11_wide_pca import (
    ALPHAS,
    DIMS,
    NEIGHBORS,
    POWERS,
    choose_nested,
    enlarge_c4,
    make_grid,
    models,
    score_grid,
    shortlist,
    summarize,
)

ID = "ID_POLIGONO"


def sample() -> PcaData:
    rng = np.random.default_rng(1711)
    ids = [f"A{i:03d}" for i in range(138)]
    x160 = rng.normal(size=(138, 160))
    dist = np.abs(np.arange(138)[:, None] - np.arange(138)[None, :]) + 0.2
    return PcaData(
        ids=ids,
        index={v: i for i, v in enumerate(ids)},
        y=pd.Series(rng.normal(3.0, 0.5, size=138), index=ids),
        geographic=dist.astype(float),
        x={160: x160},
        variance={20: 0.8},
        folds={"confirm_01": (ids[:97], ids[97:])},
        parity_max_diff=0.0,
    )


def test_full_wide_grid_and_baseline_inclusion() -> None:
    assert len(DIMS) == 8
    assert len(NEIGHBORS) == 7
    assert len(ALPHAS) == 7
    assert len(POWERS) == 5
    assert len(make_grid(20)) * len(DIMS) == 1960
    assert len({(q["k"], q["alpha"], q["power"])
                for q in make_grid(20)}) == len(make_grid(20))
    assert {"components": 20, **FROZEN} in make_grid(20)
    assert len(make_grid(20, smoke=True)) == 8


def test_prefixes_and_train_only_selection() -> None:
    data = enlarge_c4(sample(), (20, 40, 160))
    assert data.x[20].shape == (138, 20)
    np.testing.assert_array_equal(data.x[20], data.x[160][:, :20])
    train, test = data.folds["confirm_01"]
    candidates = make_grid(20, smoke=True)
    first = score_grid(
        data, train, candidates, n_folds=3, n_repeats=2, seed=17
    )
    assert first["inner_predictions"].eq(194).all()
    assert first["inner_folds"].eq(6).all()
    data.y.loc[test] = 50_000
    second = score_grid(
        data, train, candidates, n_folds=3, n_repeats=2, seed=17
    )
    pd.testing.assert_frame_equal(first, second)


def test_shortlist_and_outer_pairing() -> None:
    data = enlarge_c4(sample(), (20, 40))
    tr, test = data.folds["confirm_01"]
    coarse = {}
    for n in (20, 40):
        coarse[n] = score_grid(
            data, tr, make_grid(n, smoke=True), n_folds=2, seed=45
        )
    candidates = shortlist(coarse, top_per_dim=2)
    assert {"components": 20, **FROZEN} in candidates
    assert {"components": 40, **FROZEN} in candidates
    assert len(candidates) <= 6
    reranked = score_grid(
        data, tr, candidates, n_folds=3, n_repeats=2, seed=56
    )
    predictions, selected = choose_nested(
        data, "confirm_01", reranked, dimensions=(20, 40)
    )
    assert len(predictions) == 4 * len(test)
    assert set(predictions["model"]) == models((20, 40))
    assert predictions.groupby(ID)["model"].nunique().eq(4).all()
    assert selected["joint_best"]["components"] in (20, 40)
    assert selected["hidden_y_accessed"] is False


def test_summary_requires_complete_paired_splits() -> None:
    data = enlarge_c4(sample(), (20, 40))
    tr, _ = data.folds["confirm_01"]
    coarse = {
        n: score_grid(data, tr, make_grid(n, smoke=True), n_folds=2, seed=55)
        for n in (20, 40)
    }
    reranked = score_grid(
        data, tr, shortlist(coarse, top_per_dim=2), n_folds=3, seed=66
    )
    predictions, _ = choose_nested(
        data, "confirm_01", reranked, dimensions=(20, 40)
    )
    with pytest.raises(ValueError, match="Incomplete paired"):
        summarize(predictions, dimensions=(20, 40))


def test_clustered_bootstrap_on_complete_synthetic_predictions() -> None:
    rng = np.random.default_rng(71)
    rows = []
    dimensions = (20, 40)
    for i in range(16):
        for j in range(41):
            pid = f"S{(i * 3 + j) % 75:03d}"
            value = rng.normal(3, 0.2)
            for model in models(dimensions):
                rows.append({
                    "split_id": f"confirm_{i + 1:02d}", ID: pid,
                    "model": model, "observed": value,
                    "predicted": value + 0.15,
                })
    scores, ci = summarize(
        pd.DataFrame(rows), dimensions=dimensions,
        bootstraps=30,
    )
    assert len(scores) == 4
    assert scores["repeated_predictions"].eq(656).all()
    assert scores["unique_parcels"].eq(75).all()
    assert len(ci) == 3
    assert ci["delta_mse"].abs().max() < 1e-10
    assert ci["ci95_low"].abs().max() < 1e-10
    assert ci["ci95_high"].abs().max() < 1e-10
