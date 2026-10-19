"""Lightweight QA for a three-class learned router and two small specialists."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


def load(monkeypatch):
    folder = Path(__file__).resolve().parents[1] / "tools"
    monkeypatch.syspath_prepend(str(folder))
    spec = importlib.util.spec_from_file_location(
        "run_three_regime_classifier",
        folder / "run_three_regime_classifier.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def dataset():
    rng = np.random.default_rng(13)
    x = rng.normal(0, 1, (197, 5))
    x[0], x[1] = np.array([4,4,4,4,4]), np.array([4.2,4,4,4,4])
    for j in range(2, 8):
        x[j] = np.array([-4 + 0.08*j,-4,-4,-4,-4])
    y = 3+rng.normal(0, .3, 138)
    y[:2] = [1., 1.2]
    y[2:8] = [6., 6., 6.2, 6.2, 6.3, 6.3]
    return x, y, (0,1), tuple(range(2,8))


def test_default_generation_is_small(monkeypatch):
    m = load(monkeypatch)
    z, y, a, b = dataset()
    train = np.arange(138)
    part = m.fit_fold(
        z, np.eye(z.shape[1]), y, train,
        np.array([150]), a, b, sigma=.05,
        paths=3, steps=7, alpha=5., trees=20,
        seed=1, synthetic_weight=1.,
    )
    assert part["samples_a"] == 21
    assert part["samples_b"] == 105
    assert part["real_train"] == 138
    assert part["active_experts"] == [1,2]
    np.testing.assert_allclose(
        part["probabilities"].sum(axis=1), [1.], atol=1e-8
    )


def test_classifier_has_three_distinct_training_classes(monkeypatch):
    m = load(monkeypatch)
    z, y, a, b = dataset()
    train = np.arange(138)
    groups = m.real_group_positions(train, a, b)
    assert (groups == 0).sum() == 130
    assert (groups == 1).sum() == 2
    assert (groups == 2).sum() == 6
    clouds = {1: z[[0,1]], 2: z[[2,3,4]]}
    x, labels, w = m.build_classifier_training(
        z, train, groups, clouds
    )
    assert len(x) == 143
    assert set(labels) == {0,1,2}
    assert np.all(np.isfinite(w))
    assert w[138:140].sum() == pytest.approx(1.)


def test_when_a_anchor_heldout_no_a_specialist_and_no_leakage(monkeypatch):
    m = load(monkeypatch)
    z, y, a, b = dataset()
    train = np.delete(np.arange(138), 0)
    part = m.fit_fold(
        z, np.eye(z.shape[1]), y, train, np.array([0]),
        a, b, sigma=.05, paths=2, steps=7,
        alpha=5., trees=20, seed=1, synthetic_weight=1.,
    )
    assert part["samples_a"] == 0
    assert part["samples_b"] == 70
    assert part["active_experts"] == [2]
    assert part["probabilities"][0,1] == 0.
    altered_y = y.copy()
    altered_y[0] = 10000000000.
    other = m.fit_fold(
        z, np.eye(z.shape[1]), altered_y, train,
        np.array([0]), a, b, sigma=.05, paths=2, steps=7,
        alpha=5., trees=20, seed=1, synthetic_weight=1.,
    )
    np.testing.assert_array_equal(part["probabilities"], other["probabilities"])
    np.testing.assert_allclose(
        part["experts"][:,2], other["experts"][:,2]
    )


def test_b_anchor_heldout_uses_five_remaining_real_group_b(monkeypatch):
    m = load(monkeypatch)
    z, y, a, b = dataset()
    train = np.delete(np.arange(138), 5)
    part = m.fit_fold(
        z, np.eye(z.shape[1]), y, train,
        np.array([5]), a, b, sigma=.05, paths=2,
        steps=7, alpha=5., trees=20, seed=2,
        synthetic_weight=1.,
    )
    assert part["samples_a"] == 14
    assert part["samples_b"] == 4 * 2 * 7
    assert part["real_class_counts"] == {"0":130, "1":2, "2":5}
    altered_y = y.copy()
    altered_y[5] = -12345.
    other = m.fit_fold(
        z, np.eye(z.shape[1]), altered_y, train,
        np.array([5]), a, b, sigma=.05, paths=2,
        steps=7, alpha=5., trees=20, seed=2,
        synthetic_weight=1.,
    )
    np.testing.assert_allclose(
        part["experts"][:,1:], other["experts"][:,1:],
        equal_nan=True
    )


def test_classifier_route_picks_only_rare_prediction(monkeypatch):
    m = load(monkeypatch)
    reference = np.array([4.,4.,4.,4.])
    fit = {
        "probabilities": np.array([
            [.7,.2,.1], [.1,.8,.1], [.1,.1,.8], [.35,.34,.31]
        ]),
        "experts": np.array([
            [np.nan, 2.,6.], [np.nan,2.,6.],
            [np.nan,2.,6.], [np.nan,2.,6.]
        ]),
    }
    pred, route, confidence = m.route_predictions(
        reference, fit, min_prob=.6, blend=.5,
        max_change=.3
    )
    np.testing.assert_array_equal(route, [0,1,2,0])
    np.testing.assert_allclose(pred, [4.,3.7,4.3,4.])
    np.testing.assert_allclose(confidence, [.7,.8,.8,.35])
    same, labels, conf = m.route_predictions(
        reference, fit, min_prob=.95, blend=1., max_change=.3
    )
    np.testing.assert_array_equal(same, reference)
    np.testing.assert_array_equal(labels, [0,0,0,0])


def test_invalid_and_test_in_training_rejected(monkeypatch):
    m = load(monkeypatch)
    z, y, a, b = dataset()
    with pytest.raises(ValueError, match="overlap"):
        m.fit_fold(
            z, np.eye(z.shape[1]), y, np.arange(138),
            np.array([1]), a, b, sigma=.1, paths=2,
            steps=7, alpha=5., trees=20,
            seed=1, synthetic_weight=1.
        )
    with pytest.raises(ValueError, match="Bad classifier"):
        m.route_predictions(
            np.array([4.]),
            {"probabilities":np.array([[.1,.8,.1]]),
             "experts":np.array([[np.nan,2.,6.]])},
            min_prob=1.3, blend=.2, max_change=.3
        )
