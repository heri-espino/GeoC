"""Contracts for original Reference26 250-config full LOOCV Ridge search."""
import importlib.util
import sys
import types
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from geocebada.evaluation.peer_reference26_ridge_grid import (
    ALPHAS,
    DEFAULT_PAIR,
    GRID,
    NEIGHBORS,
    share_climate_fits_between_grid_candidates,
    tuned_local_ridge,
)


def runner(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "tools"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        "run_reference26_ridge250_loo", scripts / "run_reference26_ridge250_loo.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic(module):
    ids = [f"P{i:03d}" for i in range(138)]
    y = np.linspace(2.1, 5.5, 138)
    frame = pd.DataFrame({module.ID: ids, module.TARGET: y})
    original = pd.DataFrame({
        module.ID: ids, "observed": y,
        "Reference26": y + .2,
        "Compact14": y + .3,
    })
    return ids, y, frame, original


def test_250_distinct_configs_contain_original():
    assert len(NEIGHBORS) == 10
    assert len(ALPHAS) == 25
    assert len(GRID) == len(set(GRID)) == 250
    assert DEFAULT_PAIR in GRID
    assert GRID[0] == (8, .1)


def test_fixed_pair_names_and_unrestricted_jobs(monkeypatch):
    module = runner(monkeypatch)
    assert module.candidate_name(24, 30.0) == "Reference26_k24_alpha30"
    assert module.candidate_slug(24, 30.0) == "k24_a30"
    assert module.parse_candidates("8,24", "10,30") == (
        (8, 10.), (8, 30.), (24, 10.), (24, 30.)
    )
    with pytest.raises(ValueError, match="duplicated"):
        module.parse_candidates("8,8", "10")
    ids, y, frame, _orig = synthetic(module)
    monkeypatch.setattr(
        module, "load_inputs",
        lambda *_args: (frame, None, None, None, ids, None, None),
    )
    monkeypatch.setattr(
        module, "original_results",
        lambda *_args: pd.DataFrame({
            module.ID: ids, "observed": y, module.NAME: y+.2,
        }),
    )
    monkeypatch.setattr(
        sys, "argv",
        ["run_reference26_ridge250_loo.py", "--check", "--jobs", "300"],
    )
    assert module.main() == 0


def test_original_138_fold_baseline_alignment(monkeypatch, tmp_path):
    module = runner(monkeypatch)
    ids, y, _frame, orig = synthetic(module)
    file = tmp_path / "original.csv"
    module.write_csv(file, orig.iloc[::-1])
    output = module.original_results(file, ids, y)
    assert len(output) == 138
    assert output[module.ID].tolist() == ids
    orig.loc[4, "observed"] += 1.
    module.write_csv(file, orig)
    with pytest.raises(ValueError, match="differ"):
        module.original_results(file, ids, y)


def test_local_ridge_patch_changes_both_training_and_prediction(monkeypatch):
    original = lambda *a: np.array([-99.0])
    fake = {}
    parent = types.ModuleType("src")
    parent.__path__ = []
    monkeypatch.setitem(sys.modules, "src", parent)
    for short in (
        "geoc_bridge", "geoc_signal_research", "conditional_local_phases",
        "current_champions", "geoc_local04d_snapshot",
        "local_ridge_sufficient_statistics",
    ):
        module = types.ModuleType(f"src.{short}")
        fake[short] = module
        monkeypatch.setitem(sys.modules, f"src.{short}", module)
        setattr(parent, short, module)

    for short in ("geoc_bridge", "geoc_signal_research", "conditional_local_phases"):
        fake[short].local_prediction = original
    fake["current_champions"].predict_local_ridge = original
    fake["geoc_local04d_snapshot"].predict_weighted_local_ridge = (
        lambda *a, **kw: np.array([kw["k"] + kw["alpha"]])
    )
    fake["local_ridge_sufficient_statistics"].predict_local_ridge = (
        lambda *a, **kw: np.array([kw["k"] + kw["alpha"]])
    )
    xs = np.zeros((3, 2))
    distances = np.ones((3, 3))
    with tuned_local_ridge(12, 100.):
        assert fake["geoc_signal_research"].local_prediction(
            xs, distances, [0, 1], [2], [2., 3.]
        ) == pytest.approx([112])
        assert fake["current_champions"].predict_local_ridge(
            xs, distances, [0, 1], [2], [2., 3.]
        ) == pytest.approx([112])
    assert fake["geoc_signal_research"].local_prediction is original
    assert fake["current_champions"].predict_local_ridge is original


def test_shared_climate_cache_uses_train_positions_only(monkeypatch):
    fake = types.ModuleType("src.current_champions")
    calls = []

    class ColdBases:
        def ct3(self, tr):
            calls.append(tuple(tr))
            return {"model": len(calls)}

    fake.ColdBases = ColdBases
    parent = types.ModuleType("src")
    parent.__path__ = []
    monkeypatch.setitem(sys.modules, "src", parent)
    monkeypatch.setitem(sys.modules, "src.current_champions", fake)
    parent.current_champions = fake
    with share_climate_fits_between_grid_candidates():
        one = ColdBases().ct3([0, 1, 2])
        two = ColdBases().ct3([0, 1, 2])
        assert one is two
    assert calls == [(0, 1, 2)]
    assert ColdBases().ct3([0, 1, 2]) is not one


def test_single_candidate_fits_native_ref_and_masks_y(monkeypatch, tmp_path):
    module = runner(monkeypatch)
    ids, y, frame, _orig = synthetic(module)
    fitting = []

    class ColdExperts:
        def __init__(self, masked, groups, distance, geology, protocol, tr, inner, cache):
            assert len(tr) == 137 and 0 not in tr and len(inner) == 4
            assert np.isnan(masked.iloc[0][module.TARGET])
            assert np.isfinite(masked.iloc[tr][module.TARGET].to_numpy(float)).all()
            fitting.append("native full fit")

        def full_fit(self, tr):
            return {"train_positions": tr}

    def native_predict(pack, query, masked, distances):
        assert query.tolist() == [0]
        return {"Reference26": [3.0]}

    monkeypatch.setattr(
        module, "_WORKER",
        (frame, pd.DataFrame(index=ids), np.zeros((138, 138)),
         {"geology": {}}, ids, ColdExperts, native_predict),
    )
    monkeypatch.setattr(module, "tuned_local_ridge", lambda *_a: nullcontext())
    prediction, cache_dir = module.fit_one(0, (24, 30.), tmp_path)
    assert prediction == 3.0
    assert cache_dir.name == "k24_a30"
    assert fitting == ["native full fit"]


def test_baseline_parity_fail_closed(monkeypatch, tmp_path):
    module = runner(monkeypatch)
    ids, y, frame, _orig = synthetic(module)
    monkeypatch.setattr(
        module, "_WORKER",
        (frame, None, None, None, ids, None, None),
    )
    monkeypatch.setattr(
        module, "fit_one", lambda *_args: (2.0, tmp_path / "no_cache"),
    )
    orig = pd.DataFrame({
        module.ID: ids, "observed": y, module.NAME: np.full(138, 4.0)
    })
    with pytest.raises(RuntimeError, match="parity FAILED"):
        module.parity_baseline(tmp_path, tmp_path, tmp_path, orig)


def test_parcel_level_outputs_only_under_private_folder(monkeypatch, tmp_path):
    module = runner(monkeypatch)
    ids, y, _frame, _orig = synthetic(module)
    baseline = pd.DataFrame({
        module.ID: ids, "observed": y, module.NAME: y+.1,
    })
    ks = ((24, 30.), (12, 1.0))
    for pos in range(138):
        for config in ks:
            dest = module.fold_file(tmp_path, pos, config)
            module.write_csv(dest, pd.DataFrame([{
                module.ID: ids[pos],
                "observed": y[pos],
                module.candidate_name(*config): y[pos] + .1,
            }]))
    module.summarize(tmp_path, baseline, ks)
    result = pd.read_csv(tmp_path / "loo138_summary_250.csv")
    assert len(result) == 3
    assert set(result.model) == {
        "Reference26_original", *[module.candidate_name(*c) for c in ks],
    }
    assert (tmp_path / "loo138_predictions_private.csv").is_file()
