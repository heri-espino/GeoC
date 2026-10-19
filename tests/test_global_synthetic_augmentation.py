"""Pure synthetic-data QA; needs no private champion bundle or parcel labels."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


def load(monkeypatch):
    folder = Path(__file__).resolve().parents[1] / "tools"
    monkeypatch.syspath_prepend(str(folder))
    spec = importlib.util.spec_from_file_location(
        "run_global_synthetic_augmentation",
        folder / "run_global_synthetic_augmentation.py",
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def data():
    rng = np.random.default_rng(121)
    z = rng.normal(size=(197, 5))
    z[0] = [3., 3., 3., 3., 3.]
    z[1] = [4., 4., 4., 4., 4.]
    y = 3 + rng.normal(size=138)
    y[0], y[1] = 6.2, 6.8
    return z, y, (0, 1)


def draw(m, z, y, anchors, mode, count, train, sigma=0.1):
    roots = m.local_roots(z, anchors, neighbors=20)
    return m.augmented(
        z, y, train, anchors, np.eye(z.shape[1]), roots,
        mode=mode, count=count, sigma=sigma, seed=20261019
    )


@pytest.mark.parametrize("method", ["repeat", "local", "bridge", "mixed"])
def test_all_four_augmentations_have_exact_count_and_are_repeatable(monkeypatch, method):
    m = load(monkeypatch)
    z, y, anchors = data()
    first = draw(m, z, y, anchors, method, 257, np.arange(138))
    second = draw(m, z, y, anchors, method, 257, np.arange(138))
    assert first[0].shape == (257, z.shape[1])
    assert first[1].shape == (257,)
    assert np.isfinite(first[0]).all()
    assert np.isfinite(first[1]).all()
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])


@pytest.mark.parametrize("method", ["repeat", "local", "bridge", "mixed"])
def test_heldout_label_does_not_enter_pseudo_y(monkeypatch, method):
    m = load(monkeypatch)
    z, y, anchors = data()
    train = np.delete(np.arange(138), 0)
    generated = draw(m, z, y, anchors, method, 256, train)
    modified = y.copy()
    modified[0] = -100000
    replay = draw(m, z, modified, anchors, method, 256, train)
    np.testing.assert_array_equal(generated[0], replay[0])
    np.testing.assert_array_equal(generated[1], replay[1])
    # One anchor available -> no bridge involving the excluded anchor
    assert np.allclose(generated[1], y[1])
    if method in {"bridge", "mixed"}:
        assert "fallback_local" in generated[2]


def test_bridge_interpolated_pseudo_labels_no_noise(monkeypatch):
    m = load(monkeypatch)
    z, y, anchors = data()
    xs, pseudo, mode = draw(
        m, z, y, anchors, "bridge", 1024, np.arange(138), sigma=0
    )
    t = (xs[:, 0]-z[0,0])/(z[1,0]-z[0,0])
    np.testing.assert_allclose(pseudo, (1-t)*y[0]+t*y[1], atol=1e-10)
    assert mode == "bridge"
    assert -1e-12 <= t.min() and t.max() <= 1+1e-12


def test_training_with_augmented_data_is_a_single_global_regressor(monkeypatch):
    m = load(monkeypatch)
    z, y, anchors = data()
    train = np.arange(1, 138)
    query = np.array([0])
    syn = draw(m, z, y, anchors, "local", 256, train)
    for model in ("ridge", "extra_trees"):
        kw = dict(kind=model, alpha=30., trees=12, leaf=2, seed=42)
        baseline = m.train_predict(z, y, train, query, **kw)
        changed = m.train_predict(
            z, y, train, query, synthetic=syn[:2], mass="full", **kw
        )
        assert np.isfinite(baseline).all() and np.isfinite(changed).all()
        assert baseline.shape == (1,) and changed.shape == (1,)
        altered_y = y.copy()
        altered_y[0] = 1234567.
        replay = m.train_predict(
            z, altered_y, train, query, synthetic=syn[:2],
            mass="full", **kw
        )
        np.testing.assert_array_equal(changed, replay)
        with pytest.raises(ValueError, match="Heldout"):
            m.train_predict(z, y, np.arange(138), query, **kw)


def test_summary_reports_common_138_and_each_plain_control(monkeypatch):
    m = load(monkeypatch)
    y = np.array([2., 3., 4., 5.])
    t = pd.DataFrame({
        "ID_POLIGONO": list("abcd"), "observed": y,
        "Reference26_mean": y+.5,
        "plain__ridge": y+.4,
        "aug__ridge__local__n256__mfull": y+.3
    })
    settings = (("local",), (256,), ("full",), ("ridge",))
    out = m.metrics(t, ("a", "b"), settings)
    candidate = out[out["model"].str.startswith("aug__")].iloc[0]
    assert candidate["rmse"] == pytest.approx(.3)
    assert candidate["rmse_plain_same_learner"] == pytest.approx(.4)
    assert candidate["rmse_other136"] == pytest.approx(.3)
    assert candidate["n_cv_holdouts"] == 4


def test_reject_invalid_grid(monkeypatch):
    m = load(monkeypatch)
    args = SimpleNamespace(
        methods="local,bridge", sizes="256,1024", masses="32,full",
        models="ridge,extra_trees", sigma=.12, local_neighbors=20,
        ridge_alpha=30, hard_centers=8, trees=64, leaf=3, jobs=2
    )
    assert m.parse_settings(args)[1] == (256, 1024)
    args.methods = "local,imaginary"
    with pytest.raises(ValueError, match="Invalid"):
        m.parse_settings(args)


def test_fold_local_press_selects_only_training_observations(monkeypatch):
    m = load(monkeypatch)
    z, y, anchors = data()
    train = np.delete(np.arange(138), 1)
    centers, scores = m.fold_hard_centers(
        z, y, train, count=8, alpha=30.
    )
    assert len(centers) == 8 and len(np.unique(centers)) == 8
    assert 1 not in centers
    assert np.isfinite(scores).all()
    # Replace the HELD-OUT y with an arbitrary value; selection is invariant.
    changed = y.copy()
    changed[1] = 99999.
    repeat, repeated_scores = m.fold_hard_centers(
        z, changed, train, count=8, alpha=30.
    )
    np.testing.assert_array_equal(centers, repeat)
    np.testing.assert_array_equal(scores, repeated_scores)


def test_fold_local_press_matches_bruteforce_ridge_leave_one_out(monkeypatch):
    m = load(monkeypatch)
    rng = np.random.default_rng(123)
    z = rng.normal(size=(197, 4))
    y = rng.normal(size=138)
    train = np.arange(20)
    alpha = 10.
    centers, residual = m.fold_hard_centers(
        z, y, train, count=20, alpha=alpha
    )
    from sklearn.linear_model import Ridge
    brute = {}
    for position in train:
        others = train[train != position]
        reg = Ridge(alpha=alpha).fit(z[others], y[others])
        brute[int(position)] = y[position] - reg.predict(z[[position]])[0]
    np.testing.assert_allclose(
        residual, [brute[int(v)] for v in centers], atol=1e-10
    )


def test_gaussian_hard_centers_can_be_8_and_heldout_y_never_enters(monkeypatch):
    m = load(monkeypatch)
    z, y, anchors = data()
    train = np.delete(np.arange(138), 0)
    centers, press = m.fold_hard_centers(
        z, y, train, count=8, alpha=30.
    )
    cov_roots = m.local_roots(z, tuple(map(int, centers)), neighbors=20)
    kwargs = dict(
        mode="gaussian_hard", count=1024, sigma=0.15,
        seed=20261019, hard_positions=centers, hard_roots=cov_roots,
    )
    global_cov = np.eye(z.shape[1])
    regular_roots = m.local_roots(z, anchors, neighbors=20)
    a = m.augmented(
        z, y, train, anchors, global_cov, regular_roots, **kwargs
    )
    assert a[0].shape == (1024, z.shape[1])
    assert a[1].shape == (1024,)
    assert set(np.unique(a[1])).issubset(set(y[centers]))
    assert np.isfinite(a[0]).all()
    tampered = y.copy()
    tampered[0] = 1e9
    b = m.augmented(
        z, tampered, train, anchors, global_cov, regular_roots, **kwargs
    )
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])
    # At sigma=0, every generated X must exactly match some hard real X.
    zero = m.augmented(
        z, y, train, anchors, global_cov, regular_roots,
        mode="gaussian_hard", count=256, sigma=0.,
        seed=20261019, hard_positions=centers, hard_roots=cov_roots,
    )
    for vector in zero[0]:
        assert any(np.allclose(vector, z[v]) for v in centers)


def test_hard_center_generator_rejects_a_heldout_center(monkeypatch):
    m = load(monkeypatch)
    z, y, anchors = data()
    train = np.arange(1, 138)
    with pytest.raises(ValueError, match="training labels"):
        m.gaussian_hard_samples(
            z, y, train, positions=np.array([0, 3]),
            roots={0: np.eye(z.shape[1]), 3: np.eye(z.shape[1])},
            count=100, sigma=0.1, seed=42,
        )
