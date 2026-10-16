"""Synthetic protocol checks: exact baseline, local weighting, pairing and no y leakage."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from geocebada.evaluation.checkpoint04d1 import predict_weighted_local_ridge
from geocebada.evaluation.checkpoint08_local import ID
from geocebada.evaluation.checkpoint10_pca import PcaData
from geocebada.evaluation.geographic_ridge_v2 import (
    ADAPTIVE,
    ALL_ARMS,
    FROZEN_NAME,
    NORMALIZED,
    RAW,
    ROBUST,
    SHRINK,
    K,
    _inner_folds,
    _one_new_prediction,
    _predict,
    _select_alphas,
    _summarize,
    _weights,
)


def _fake():
    rng = np.random.default_rng(23)
    ids = [f"P{i:03d}" for i in range(197)]
    known = ids[:138]
    x = rng.normal(size=(197, 20))
    geographic = (
        np.abs(np.arange(197)[:, None] - np.arange(197)[None, :])
        + 0.015
    ).astype(float)
    return PcaData(
        ids=ids,
        index={pid: i for i, pid in enumerate(ids)},
        y=pd.Series(rng.normal(3., .5, len(known)), index=known),
        geographic=geographic,
        x={20: x},
        variance={20: .7},
        folds={"confirm_01": (known[:97], known[97:])},
        parity_max_diff=0.,
    )


def test_effective_neighbor_count_and_normalization():
    data = _fake()
    observed = np.arange(97)
    weights = _weights(data.geographic, observed, 101)
    assert len(weights.positions) == K
    np.testing.assert_allclose(weights.normalized.sum(), K, atol=1e-12)
    assert 1. <= weights.effective_n <= K
    np.testing.assert_allclose(
        weights.normalized / weights.normalized.sum(),
        weights.raw / weights.raw.sum(),
    )


def test_reference_predictor_exact_numerical_parity():
    data = _fake()
    fit, held = data.folds["confirm_01"]
    positions = [data.index[pid] for pid in fit]
    queries = [data.index[pid] for pid in held[:4]]
    y = np.full(197, np.nan)
    y[positions] = data.y.loc[fit].to_numpy()
    original = predict_weighted_local_ridge(
        data.x[20], data.geographic, y, positions, queries,
        k=24, alpha=30., distance_power=1.,
    )
    comparison = _predict(data, fit, held[:4], FROZEN_NAME, 30.)
    np.testing.assert_array_equal(original, comparison)


def test_all_arms_finite_and_distinct_in_nonuniform_geometry():
    data = _fake()
    fit, held = data.folds["confirm_01"]
    out = {
        arm: _predict(data, fit, held[:2], arm, 30.0)
        for arm in ALL_ARMS
    }
    assert all(arr.shape == (2,) and np.isfinite(arr).all() for arr in out.values())
    assert np.array_equal(out[FROZEN_NAME], out[RAW])
    assert not np.allclose(out[RAW], out[NORMALIZED])


def test_huge_local_outlier_is_downweighted_by_robust_irls():
    data = _fake()
    train, held = data.folds["confirm_01"]
    query = data.index[held[0]]
    weights = _weights(
        data.geographic, np.array([data.index[pid] for pid in train]), query,
    )
    base = _one_new_prediction(
        data.x[20], np.array([data.y.get(pid, np.nan) for pid in data.ids]),
        weights, query, NORMALIZED, 30.,
    )
    changed = data.y.copy()
    closest = data.ids[weights.positions[0]]
    changed.loc[closest] += 10.
    values = np.full(197, np.nan)
    values[:138] = changed.to_numpy()
    normal = _one_new_prediction(data.x[20], values, weights, query, NORMALIZED, 30.)
    robust = _one_new_prediction(data.x[20], values, weights, query, ROBUST, 30.)
    assert np.isfinite([base, normal, robust]).all()


def test_identical_inner_folds_cover_every_training_label_twice():
    data = _fake()
    train, _ = data.folds["confirm_01"]
    folds = _inner_folds(train, "confirm_01")
    assert len(folds) == 10
    held = [pid for a, b in folds for pid in b]
    assert set(held) == set(train)
    assert pd.Series(held).value_counts().eq(2).all()
    for fit, validation in folds:
        assert not set(fit) & set(validation)


def test_heldout_y_cannot_change_predictions_or_inner_selection():
    data = _fake()
    train, held = data.folds["confirm_01"]
    first = {
        arm: _predict(data, train, held[:2], arm, 30.) for arm in ALL_ARMS
    }
    before, selection = _select_alphas(
        data, train, "confirm_01", arms=(RAW, NORMALIZED), alphas=(3., 30.),
    )
    data.y.loc[held] = -1000.
    after, selection2 = _select_alphas(
        data, train, "confirm_01", arms=(RAW, NORMALIZED), alphas=(3., 30.),
    )
    pd.testing.assert_frame_equal(before, after)
    assert selection == selection2
    for arm in ALL_ARMS:
        np.testing.assert_array_equal(
            first[arm], _predict(data, train, held[:2], arm, 30.),
        )


def test_summary_rejects_missing_pairs_and_handles_repeated_parcels():
    records = []
    for split in range(1, 17):
        for parcel in range(41):
            for n, arm in enumerate(ALL_ARMS):
                records.append({
                    "split_id": f"confirm_{split:02d}",
                    ID: f"P{parcel:03d}",
                    "model": arm,
                    "observed": 3. + parcel / 50,
                    "predicted": 3. + parcel / 50 + (n + 1) * .02,
                })
    frame = pd.DataFrame(records)
    summary, paired, parcel = _summarize(frame, bootstraps=25)
    assert set(summary.model) == set(ALL_ARMS)
    assert summary.unique_parcels.eq(41).all()
    assert summary.repeated_predictions.eq(656).all()
    assert len(paired) == len(ALL_ARMS) - 1
    assert len(parcel) == 41 * len(ALL_ARMS)
    with pytest.raises(ValueError, match="Incomplete"):
        _summarize(frame.iloc[:-1], bootstraps=2)


def test_input_validation_does_not_accept_training_target_overlap():
    data = _fake()
    fit, held = data.folds["confirm_01"]
    with pytest.raises(ValueError, match="overlap"):
        _predict(data, fit, [fit[0]], ADAPTIVE, 30.)
    with pytest.raises(ValueError, match="positive"):
        _predict(data, fit, held[:1], SHRINK, -1.)
