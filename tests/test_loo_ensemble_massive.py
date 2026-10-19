"""Synthetic tests for million-weight quadratic ensemble search."""
import importlib.util
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def load(monkeypatch):
    folder = Path(__file__).resolve().parents[1] / "tools"
    monkeypatch.syspath_prepend(str(folder))
    spec = importlib.util.spec_from_file_location(
        "run_loo_ensemble_massive", folder / "run_loo_ensemble_massive.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_data(module, m=3):
    n = 138
    y = 2.5 + np.linspace(0, 1, n)
    noise = np.sin(np.arange(n) / 3)
    cols = {
        module.ID: [f"Parcel_{i:03d}" for i in range(n)],
        module.TARGET: y,
        module.ANCHOR: y + noise*0.3 + 0.07,
    }
    if m > 1:
        cols["Compact14"] = y - noise*0.25 + 0.02
    if m > 2:
        cols["Local04D"] = y + 0.2*np.cos(np.arange(n)/7)
    if m > 3:
        cols["Reference26_original"] = y + noise*.23 + .08
    if m > 4:
        cols["Reference26_DBSCAN"] = y + noise*.29 + .03
    return pd.DataFrame(cols)


def test_combinatorial_grid_count_and_streaming_no_duplicates(monkeypatch):
    mod = load(monkeypatch)
    assert mod.count_simplex_weights(5, 100) == math.comb(104, 4)
    assert mod.count_simplex_weights(5, 50) == math.comb(54, 4)
    assert mod.discrete_steps(.01) == 100
    with pytest.raises(ValueError, match="dividing 1"):
        mod.discrete_steps(.03)
    for m, steps in itertools.product((2, 3, 5), (2, 3)):
        rows = np.vstack(list(mod.simplex_batches(m, steps, 2)))
        assert len(rows) == mod.count_simplex_weights(m, steps)
        assert np.allclose(rows.sum(axis=1), 1)
        assert np.all(rows >= 0)
        assert len({tuple(x) for x in rows}) == len(rows)


def test_gram_quadratic_equals_direct_138_prediction_rmse(monkeypatch):
    mod = load(monkeypatch)
    table = fake_data(mod, m=5)
    names, y, p, err, gram = mod.gram_from_predictions(table)
    assert len(names) == 5
    weights = np.array([
        [.7, .1, .1, .05, .05],
        [0, 0, 1, 0, 0],
        [.2, .2, .2, .2, .2],
    ])
    direct = np.mean((p @ weights.T - y[:, None])**2, axis=0)
    fast = mod.gram_mse(weights, gram)
    np.testing.assert_allclose(fast, direct, rtol=1e-12, atol=1e-12)
    assert mod.gram_mse(weights[0], gram) == pytest.approx(direct[0])


def test_exact_grid_best_and_continuous_optimum_no_worse(monkeypatch):
    mod = load(monkeypatch)
    table = fake_data(mod)
    names, y, p, err, gram = mod.gram_from_predictions(table)
    result, count = mod.exhaustive_grid(
        gram, step=.1, chunk=6, keep=7, max_evals=1000
    )
    assert count == math.comb(12, 2)
    assert len(result) == 7
    assert np.all(np.diff(result.mse) >= -1e-12)
    optimal = mod.constrained_optimum(gram)
    assert mod.gram_mse(optimal, gram) <= result.iloc[0].mse + 1e-8
    assert np.all(optimal >= -1e-9)
    assert np.sum(optimal) == pytest.approx(1.)
    anchor = mod.constrained_optimum(gram, anchor_min=.7)
    assert anchor[0] >= .7-1e-8
    assert sum(anchor) == pytest.approx(1.)


def test_regularization_shrinks_to_anchor_and_subsets(monkeypatch):
    mod = load(monkeypatch)
    table = fake_data(mod, m=3)
    names, y, p, err, gram = mod.gram_from_predictions(table)
    regularized = mod.constrained_optimum(gram, penalty=1.)
    assert regularized[0] > .95
    only_anchor = mod.constrained_optimum(gram, active=(0,))
    np.testing.assert_allclose(only_anchor, [1, 0, 0])
    only_other = mod.constrained_optimum(gram, active=(1,))
    np.testing.assert_allclose(only_other, [0, 1, 0])
    with pytest.raises(ValueError, match="Anchor must"):
        mod.constrained_optimum(gram, anchor_min=.5, active=(1, 2))
    all_scores = mod.optimizer_experiments(gram, names)
    assert len(all_scores) > 15
    assert "reference26_at_least_half" in set(all_scores.family)
    assert (all_scores.nonzero_models <= 3).all()


def test_no_million_fit_false_claim_and_guard(monkeypatch):
    mod = load(monkeypatch)
    _, _, _, _, gram = mod.gram_from_predictions(fake_data(mod, 5))
    with pytest.raises(ValueError, match="exceed --max-evals"):
        mod.exhaustive_grid(
            gram, step=.01, chunk=100, keep=5, max_evals=100,
        )


def test_naive_bootstrap_explicitly_diagnostic_and_one_parcel_sensitivity(
    monkeypatch,
):
    mod = load(monkeypatch)
    _, _, _, errors, _ = mod.gram_from_predictions(fake_data(mod, 3))
    w = np.array([.7, .3, 0.])
    output = mod.paired_bootstrap(errors, w, reps=300, seed=42)
    assert output["bootstrap_reps"] == 300
    assert 0 <= output["fraction_parcels_improved"] <= 1
    assert output["naive_paired_bootstrap_lo"] <= (
        output["naive_paired_bootstrap_hi"]
    )
    assert output["delta_mse_after_removing_best_1"] >= (
        output["delta_mse"]-1e-12
    )


def test_main_writes_only_aggregate_csv_no_138_targets(monkeypatch, tmp_path):
    mod = load(monkeypatch)
    monkeypatch.setattr(mod, "load_table", lambda *a: fake_data(mod, 3))
    monkeypatch.setattr(sys, "argv", [
        "run_loo_ensemble_massive.py", "--run", "--step", ".1",
        "--chunk-size", "7", "--bootstrap", "100",
        "--output", str(tmp_path),
    ])
    assert mod.main() == 0
    assert sorted(f.name for f in tmp_path.glob("*.csv")) == [
        "grid_best.csv", "optimized_subsets.csv", "robustness.csv"
    ]
    for path in tmp_path.glob("*.csv"):
        df = pd.read_csv(path)
        assert mod.ID not in df
        assert mod.TARGET not in df
    report = json.loads((tmp_path / "run_spec.json").read_text())
    assert report["grid_candidates"] == math.comb(12, 2)
    assert report["no_base_model_training"] is True
