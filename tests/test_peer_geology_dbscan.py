"""Unit tests for the target-safe, DBSCAN geological gate (no private data)."""
import numpy as np
import pytest

from geocebada.evaluation.peer_geology_dbscan import (
    DBSCAN_GRID,
    GeologicalDBSCANGate,
    adaptive_radius,
)


def _clouds():
    r = np.random.default_rng(3026)
    left = r.normal(loc=(-4, 0), scale=.15, size=(18, 2))
    right = r.normal(loc=(4, 0), scale=.15, size=(18, 2))
    extra = np.array([[40., 40.]])
    return np.vstack([left, right, extra])


def test_density_scales_cover_fixed_predeclared_grid():
    assert len(DBSCAN_GRID) == 8
    z = _clouds()
    assert adaptive_radius(z, 3, .75) < adaptive_radius(z, 3, 3.0)
    with pytest.raises(ValueError):
        adaptive_radius(z, 50, 1)
    with pytest.raises(ValueError):
        adaptive_radius(z, 3, 0)


def test_query_outside_training_cores_falls_back_to_global_weight():
    z = _clouds()
    ct = np.zeros(len(z))
    expert = np.ones(len(z))
    y = np.array([1.] * 18 + [0.] * 18 + [.5])
    gate = GeologicalDBSCANGate(3, 2).fit(z, y, ct, expert)
    assert gate.n_clusters_ == 2
    assert gate.noise_fraction_ > 0
    left, right, far = gate.weights(np.array([[-4., 0], [4., 0], [100., -100.]]))
    assert left > right
    assert abs(far - gate.global_) < 1e-12
    assert np.all(np.array([left, right, far]) >= 0)
    assert np.all(np.array([left, right, far]) <= 1)
    np.testing.assert_allclose(
        gate.predict(np.array([[100., -100.]]), [0.], [1.]),
        [gate.global_],
    )


def test_all_noise_falls_back_to_global_without_extrapolating():
    z = np.column_stack([np.arange(16) * 100., np.zeros(16)])
    y = np.linspace(0, 1, len(z))
    ct = np.zeros(len(z))
    expert = np.ones(len(z))
    gate = GeologicalDBSCANGate(5, .75).fit(z, y, ct, expert)
    assert gate.n_clusters_ == 0
    assert gate.noise_fraction_ == 1.
    np.testing.assert_allclose(
        gate.weights(np.array([[1e6, 1e6]])), [gate.global_],
    )


def test_duplicate_geology_and_invalid_features():
    z = np.zeros((10, 10))
    ct = np.zeros(10)
    expert = np.ones(10)
    y = np.linspace(0, 1, 10)
    gate = GeologicalDBSCANGate(3, .75).fit(z, y, ct, expert)
    assert gate.n_clusters_ == 1
    np.testing.assert_allclose(gate.weights(np.zeros((3, 10))), gate.weights_[0])
    with pytest.raises(ValueError):
        gate.weights(np.full((1, 10), np.nan))
