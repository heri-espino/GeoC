"""Synthetic unit tests for deliberately invalid 70/30 diagnostic protocols."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def load(monkeypatch):
    tools = Path(__file__).resolve().parents[1]/"tools"
    monkeypatch.syspath_prepend(str(tools))
    spec = importlib.util.spec_from_file_location(
        "run_intentionally_leaky_7030", tools/"run_intentionally_leaky_7030.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture():
    rng = np.random.default_rng(808)
    z = rng.normal(size=(197, 5))
    y = 2.2 + z[:138, 0]*0.3 + rng.normal(scale=.4, size=138)
    ids = [f"TEST{i:03d}" for i in range(138)]
    roots = {i:np.eye(5) for i in range(138)}
    cfg = {"seed":20261019, "synthetic_count":84, "sigma":.05,
           "mass":"full", "hard_centers":8, "alpha":30.,
           "trees":12, "leaf":2, "models":["ridge", "extra_trees"]}
    return z, y, ids, roots, cfg


def test_100_repeat_7030_split_is_reproducible(monkeypatch):
    m = load(monkeypatch)
    for r in range(100):
        tr, te = m.split_indices(138, seed=20261019+r)
        assert len(tr) == 96 and len(te) == 42
        assert not set(tr)&set(te)
        np.testing.assert_array_equal(
            te, m.split_indices(138, seed=20261019+r)[1]
        )
    assert not np.array_equal(
        m.split_indices(138, seed=20261019)[1],
        m.split_indices(138, seed=20261020)[1],
    )


def test_gaussian_children_all_test_origins_and_exact_copy(monkeypatch):
    m = load(monkeypatch)
    z,y,ids,roots,cfg = fixture()
    test = np.arange(42,84)
    xs, ys, origins = m.test_origin_children(
        z,y,test,roots,count=84,sigma=0.,seed=5
    )
    assert xs.shape == (84,5)
    assert len(np.unique(origins)) == len(test)
    np.testing.assert_array_equal(xs, z[origins])
    np.testing.assert_array_equal(ys, y[origins])
    modified = y.copy()
    modified[test] += 900
    other_x, other_y, other_origins = m.test_origin_children(
        z,modified,test,roots,count=84,sigma=0.,seed=5
    )
    np.testing.assert_array_equal(other_x, xs)
    np.testing.assert_array_equal(other_origins, origins)
    np.testing.assert_allclose(other_y,ys+900)


def test_clean_models_do_not_learn_test_labels_but_leaky_model_does(monkeypatch):
    m = load(monkeypatch)
    z,y,ids,roots,cfg = fixture()
    _,te = m.split_indices(138,seed=cfg["seed"])
    first = m.run_one(0,(z,y,ids,roots,cfg))
    changed = y.copy()
    changed[te] += 10.0
    second = m.run_one(0,(z,changed,ids,roots,cfg))
    r1,r2 = first[1],second[1]
    left = r1.loc[r1["model"]=="ridge"].sort_values(m.ID)
    right = r2.loc[r2["model"]=="ridge"].sort_values(m.ID)
    for col in ("clean_real_test","clean_training_augmented_real_test"):
        np.testing.assert_allclose(left[col],right[col],atol=1e-10)
    assert not np.allclose(
        left["leaky_test_exact_copies_training_real_test"],
        right["leaky_test_exact_copies_training_real_test"],
    )


def test_each_split_has_explicit_leakage_flags_and_no_fictitious_unique_test_parents(monkeypatch):
    m=load(monkeypatch)
    z,y,ids,roots,cfg=fixture()
    scores,real,fake = m.run_one(2,(z,y,ids,roots,cfg))
    assert len(scores)==len(m.MODELS)*len(m.SCENARIOS)
    assert len(real)==42*len(m.MODELS)
    assert len(fake)==84*len(m.MODELS)
    assert set(scores["n_real_train"])=={96}
    assert set(scores["n_real_test"])=={42}
    assert set(scores["n_unique_real_test_parents"])=={42}
    assert set(scores["n_evaluated"])=={42,126}
    for row in scores.itertuples(index=False):
        assert row.accessed_heldout_y_for_training == row.scenario.startswith("leaky_test_")
        assert row.evaluated_synthetic_test_labels == ("expanded_test" in row.scenario)
    assert np.isfinite(scores["rmse_t_ha"]).all()


def test_fake_test_expansion_is_dependence_not_more_real_observations(monkeypatch):
    m=load(monkeypatch)
    z,y,ids,roots,cfg=fixture()
    cfg["sigma"] = 0.
    scores, _, fake = m.run_one(0,(z,y,ids,roots,cfg))
    assert len(fake)==168
    clean = scores.query(
        "model == 'ridge' and scenario == 'clean_real_test'"
    )["rmse_t_ha"].iloc[0]
    fake_test = scores.query(
        "model == 'ridge' and scenario == 'clean_model_synthetic_expanded_test'"
    )["rmse_t_ha"].iloc[0]
    # 42 test parents each repeated exactly twice, same X and copied y:
    # the apparent test size triples but the RMSE stays exactly identical.
    assert clean == pytest.approx(fake_test, abs=1e-10)


def test_corrupt_inputs_rejected(monkeypatch):
    m = load(monkeypatch)
    z,y,ids,roots,cfg=fixture()
    with pytest.raises(ValueError,match=">=1 child"):
        m.test_origin_children(
            z,y,np.arange(42),roots,count=41,sigma=.1,seed=1
        )
    with pytest.raises(ValueError,match="Test fraction"):
        m.split_indices(138,seed=1,test_fraction=1.)
