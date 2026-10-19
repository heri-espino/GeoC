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
    assert 0 <= t.min() and t.max() <= 1


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
        ridge_alpha=30, trees=64, leaf=3, jobs=2
    )
    assert m.parse_settings(args)[1] == (256, 1024)
    args.methods = "local,imaginary"
    with pytest.raises(ValueError, match="Invalid"):
        m.parse_settings(args)
