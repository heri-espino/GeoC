"""Synthetic tests: two actual bridge groups, 2/6 anchors and safe LOOCV."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


def load(monkeypatch):
    folder = Path(__file__).resolve().parents[1] / "tools"
    monkeypatch.syspath_prepend(str(folder))
    spec = importlib.util.spec_from_file_location(
        "run_two_bridge_three_regime",
        folder / "run_two_bridge_three_regime.py",
    )
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def synthetic():
    rng = np.random.default_rng(23)
    z = rng.normal(size=(197, 5))
    z[:8] = np.array([
        [0, 0, 0, 0, 0], [1, 0, 0, 0, 0],
        [4, 0, 0, 0, 0], [5, 0, 0, 0, 0],
        [6, 0, 0, 0, 0], [7, 0, 0, 0, 0],
        [8, 0, 0, 0, 0], [9, 0, 0, 0, 0],
    ], float)
    y = 3 + rng.normal(size=138)
    return z, y, (0, 1), (2, 3, 4, 5, 6, 7)


def test_exact_two_plus_six_consensus_groups(monkeypatch, tmp_path):
    m = load(monkeypatch)
    ids = [f"Parcel{i:03d}" for i in range(138)]
    manifest = {
        "intersection_top2_across_families_private": [ids[0], ids[1]],
        "intersection_top10_across_families_private": [
            ids[i] for i in range(8)
        ],
    }
    p = tmp_path/"private.json"
    p.write_text(json.dumps(manifest), encoding="utf-8")
    a, b = m.read_private_groups(p, ids)
    assert len(a) == 2 and len(b) == 6
    assert set(a) == {0, 1} and set(b) == set(range(2, 8))
    manifest["intersection_top10_across_families_private"].append(ids[8])
    p.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="Expected two"):
        m.read_private_groups(p, ids)


def test_bridge_chain_covers_six_and_has_five_edges(monkeypatch):
    m = load(monkeypatch)
    z, y, a, b = synthetic()
    edges = m.chain_pairs(z, b)
    assert len(edges) == 5
    assert len(set(tuple(sorted(edge)) for edge in edges)) == 5
    assert set(node for edge in edges for node in edge) == set(b)
    assert m.chain_pairs(z, (0,)) == []


def test_unique_bridge_has_exactly_specified_samples(monkeypatch):
    m = load(monkeypatch)
    z, y, a, b = synthetic()
    paths, steps = 8, 17
    x, pseudo = m.unique_bridge(
        z[0], z[1], y[0], y[1], np.eye(z.shape[1]),
        sigma=0.1, paths=paths, steps=steps, seed=42
    )
    assert x.shape == (136, z.shape[1])
    assert pseudo.shape == (136,)
    assert np.isfinite(x).all() and np.isfinite(pseudo).all()


def test_group_b_still_fits_when_one_of_six_is_held_out(monkeypatch):
    m = load(monkeypatch)
    z, y, a, b = synthetic()
    root = np.eye(z.shape[1])
    full = np.arange(138)
    xa, ya, links_a = m.synthetic_group(
        z, root, y, full, a, sigma=0.1,
        paths=2, steps=9, seed=101,
    )
    xb, yb, links_b = m.synthetic_group(
        z, root, y, full, b, sigma=0.1,
        paths=2, steps=9, seed=101,
    )
    assert (len(xa), len(xb), links_a, links_b) == (18, 90, 1, 5)
    # Heldout original label is inaccessible to pseudo-y augmentation.
    missing_a = np.delete(full, 0)
    x_m, y_m, l_m = m.synthetic_group(
        z, root, y, missing_a, a, sigma=0.1,
        paths=2, steps=9, seed=101,
    )
    assert (len(x_m), l_m) == (0, 0)
    missing_b = np.delete(full, 5)
    x_m, y_m, l_m = m.synthetic_group(
        z, root, y, missing_b, b, sigma=0.1,
        paths=2, steps=9, seed=101,
    )
    assert (len(x_m), l_m) == (72, 4)
    changed_y = y.copy()
    changed_y[5] = 1e8
    x_unchanged, pseudo_unchanged, _ = m.synthetic_group(
        z, root, changed_y, missing_b, b, sigma=0.1,
        paths=2, steps=9, seed=101,
    )
    np.testing.assert_array_equal(x_m, x_unchanged)
    np.testing.assert_array_equal(y_m, pseudo_unchanged)


def test_heldout_a_may_be_routed_by_b_but_not_trained_on(monkeypatch):
    m = load(monkeypatch)
    z, y, a, b = synthetic()
    train = np.delete(np.arange(138), 0)
    part = m.fit_predict(
        z, np.eye(z.shape[1]), y, train, np.array([0]), a, b,
        sigma=0.03, paths=2, steps=9, alpha=2., seed=1
    )
    assert part["samples_a"] == 0
    assert part["samples_b"] == 5 * 2 * 9
    assert np.isinf(part["synthetic_distance"]).sum() == 0
    assert part["route"].tolist() == [2]
    with pytest.raises(ValueError, match="overlap"):
        m.fit_predict(
            z, np.eye(z.shape[1]), y, np.arange(138),
            np.array([0]), a, b, sigma=0.1,
            paths=2, steps=9, alpha=1., seed=1
        )


def test_gate_caps_change_and_never_changes_other_route(monkeypatch):
    m = load(monkeypatch)
    component = {
        "route": np.array([1, 2, 1]),
        "expert": np.array([9., 10., 7.]),
        "synthetic_distance": np.array([0.01, 0.02, np.inf]),
        "ordinary_distance": np.array([1., 1., 1.]),
        "support": 1.,
    }
    original = np.array([4., 4., 4.])
    result, routes = m.gated(
        original, component, ratio=1., radius=1.,
        blend=0.5, max_change=0.3,
    )
    np.testing.assert_allclose(result, [4.3, 4.3, 4.])
    np.testing.assert_array_equal(routes, [1, 2, 0])


def test_cv_subgroups_and_baseline_protected_when_no_routes(monkeypatch):
    m = load(monkeypatch)
    y = np.array([2., 3., 4., 4.5])
    ref = np.array([2.2, 3.1, 3.9, 4.4])
    unchanged = m.scored(y, ref, ref, np.zeros(4), np.array([1, 2, 0, 0]))
    assert unchanged["n_changed"] == 0
    assert unchanged["delta_mse"] == pytest.approx(0)
    assert unchanged["baseline_rmse_a2"] == pytest.approx(unchanged["rmse_a2"])
    assert unchanged["baseline_rmse_ordinary130"] == pytest.approx(
        unchanged["rmse_ordinary130"]
    )
