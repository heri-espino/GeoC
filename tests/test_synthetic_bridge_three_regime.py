"""Lightweight synthetic QA of Gaussian-bridge 3-regime learner; no real yields."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def load(monkeypatch):
    folder = Path(__file__).resolve().parents[1] / "tools"
    monkeypatch.syspath_prepend(str(folder))
    spec = importlib.util.spec_from_file_location(
        "run_synthetic_bridge_three_regime",
        folder / "run_synthetic_bridge_three_regime.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_features():
    rng = np.random.default_rng(42)
    frame = pd.DataFrame(rng.normal(size=(197, 5)), columns=list("abcde"))
    frame.loc[2, "a"] = np.nan
    return frame


def test_representation_is_x_only_and_finite(monkeypatch):
    m = load(monkeypatch)
    frame = fake_features()
    z, root, cols = m.representation(frame, list("abcde"))
    assert z.shape == (197, 5)
    assert root.shape == (5, 5)
    assert cols == list("abcde")
    assert np.isfinite(z).all()


def test_brownian_bridge_zero_noise_at_endpoints_and_variable_midpoint(monkeypatch):
    m = load(monkeypatch)
    a, b = np.array([0., 0., 0.]), np.array([4., 2., 1.])
    root = np.eye(3)
    result = m.brownian_bridge_samples(
        a, b, 2., 5., root, sigma=0.6, paths=400, steps=9, seed=101,
    )
    left_x, left_y, right_x, right_y = result
    # First sample of each path is exact A, last in second cloud is exact B.
    np.testing.assert_allclose(left_x[::5], np.broadcast_to(a, left_x[::5].shape), atol=1e-12)
    np.testing.assert_allclose(right_x[4::5], np.broadcast_to(b, right_x[4::5].shape), atol=1e-12)
    np.testing.assert_allclose(left_y[::5], np.full_like(left_y[::5], 2.), atol=1e-12)
    np.testing.assert_allclose(right_y[4::5], np.full_like(right_y[4::5], 5.), atol=1e-12)
    # At midpoint the bridge has Var ~= sigma², after removing the mean line.
    midpoint = left_x[4::5] - (a+b)/2
    assert np.all(np.var(midpoint, axis=0) > 0.20)
    assert np.all(np.var(midpoint, axis=0) < 0.55)
    repeated = m.brownian_bridge_samples(
        a, b, 2., 5., root, sigma=0.6, paths=400, steps=9, seed=101,
    )
    for first, second in zip(result, repeated):
        np.testing.assert_array_equal(first, second)


def test_zero_noise_is_linear_and_specialists_split_at_midpoint(monkeypatch):
    m = load(monkeypatch)
    result = m.brownian_bridge_samples(
        np.array([0., 0.]), np.array([2., 4.]), 1., 3.,
        np.eye(2), sigma=0., paths=3, steps=9, seed=42,
    )
    left_x, left_y, right_x, right_y = result
    assert left_x.shape == (15, 2)
    assert right_x.shape == (15, 2)
    np.testing.assert_allclose(left_y[:5], 1.+left_x[:5, 0])
    np.testing.assert_allclose(right_y[:5], 1.+right_x[:5, 0])


def test_no_synthetic_labels_from_heldout_anchor(monkeypatch):
    m = load(monkeypatch)
    z = np.arange(197*3, dtype=float).reshape(197, 3) / 100.
    y = np.full(138, 4.)
    # If either anchor is withheld, the generated component MUST be absent.
    assert m.specialist_prediction(
        z, np.eye(3), y, np.arange(1, 138), np.array([0]),
        (0, 20), sigma=0.2, paths=4, steps=9, alpha=2., seed=1
    ) is None
    assert m.specialist_prediction(
        z, np.eye(3), y, np.delete(np.arange(138), 20),
        np.array([20]), (0, 20), sigma=0.2, paths=4, steps=9,
        alpha=2., seed=1
    ) is None


def test_three_routes_protect_default_and_disallow_train_query_overlap(monkeypatch):
    m = load(monkeypatch)
    base = np.array([4., 4.])
    result, mask = m.apply_gate(base, None, ratio=1., radius=1.5, blend=1.)
    np.testing.assert_array_equal(result, base)
    assert not mask.any()
    component = {
        "expert": np.array([1.2, 6.]),
        "synthetic_distance": np.array([0.01, 1.5]),
        "ordinary_distance": np.array([0.5, 0.1]),
        "support": 1.,
        "projection": np.array([0.2, 0.8]),
        "expert_group": np.array([1, 2]),
    }
    answer, mask = m.apply_gate(base, component, ratio=1., radius=1., blend=.5)
    np.testing.assert_array_equal(mask, [True, False])
    np.testing.assert_allclose(answer, [2.6, 4.])
    z = np.random.default_rng(1).normal(size=(197, 3))
    with pytest.raises(ValueError, match="also a supervised training"):
        m.specialist_prediction(
            z, np.eye(3), np.ones(138), np.arange(138),
            np.array([5]), (0, 20),
            sigma=0.1, paths=2, steps=9, alpha=1., seed=1
        )


def test_real_two_anchor_fit_uses_only_training_labels(monkeypatch):
    m = load(monkeypatch)
    rng = np.random.default_rng(100)
    z = rng.normal(size=(197, 3))
    z[0] = np.array([4., 4., 4.])
    z[1] = np.array([5., 5., 5.])
    z[5] = np.array([4.5, 4.5, 4.5])
    y = np.full(138, 3.)
    y[:2] = [1.5, 2.0]
    train = np.delete(np.arange(138), 5)
    component = m.specialist_prediction(
        z, np.eye(3), y, train, np.array([5]), (0, 1),
        sigma=0.1, paths=4, steps=9, alpha=1., seed=3
    )
    assert component is not None
    assert np.isfinite(component["expert"]).all()
    assert component["expert_group"].shape == (1,)
    assert component["ordinary_distance"].shape == (1,)
    assert component["support"] > 0
