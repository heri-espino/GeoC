"""Synthetic QA for cheap 138-fold convex-blend stability diagnostics."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def load(monkeypatch):
    folder = Path(__file__).resolve().parents[1] / "tools"
    monkeypatch.syspath_prepend(str(folder))
    spec = importlib.util.spec_from_file_location(
        "run_loo_ensemble_stability", folder / "run_loo_ensemble_stability.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sample(mod):
    y = np.linspace(2.0, 5.0, mod.N if hasattr(mod, "N") else 138)
    err = .2 + .05 * np.sin(np.arange(len(y)))
    frame = pd.DataFrame({
        mod.ID: [f"Parcel_{i:03d}" for i in range(len(y))],
        mod.TARGET: y,
        mod.ANCHOR: y+err,
        "Compact14": y-err,
    })
    return frame


def test_exact_convex_50_50_cancels_complementary_errors(monkeypatch):
    mod = load(monkeypatch)
    df = sample(mod)
    y = df[mod.TARGET].to_numpy(float)
    a = df[mod.ANCHOR].to_numpy(float)
    b = df["Compact14"].to_numpy(float)
    assert mod.optimal_convex_weight(y, a, b) == pytest.approx(.5)
    grid = mod.summarize_weight_grid(y, a, b)
    best = grid.loc[grid.weight_candidate.eq(.5)]
    assert len(best) == 1
    assert best.iloc[0].rmse < 1e-12
    assert grid.loc[grid.weight_candidate.eq(0), "rmse"].iloc[0] > .1


def test_30pct_blend_improves_residuals_bootstrap_and_outlier_pruning(monkeypatch):
    mod = load(monkeypatch)
    df = sample(mod)
    y = df[mod.TARGET].to_numpy(float)
    a = df[mod.ANCHOR].to_numpy(float)
    b = df["Compact14"].to_numpy(float)
    s, influence = mod.paired_diagnostics(
        y, a, b, .3, bootstrap=500, seed=20261019
    )
    assert s["n_heldouts"] == 138
    assert s["rmse_blend"] < s["rmse_reference26mean"]
    assert s["improved_parcels"] == 138
    assert s["worsened_parcels"] == 0
    assert s["bootstrap_delta_mse_ci_upper"] < 0
    assert influence["largest_improvements_removed"].tolist() == [
        0, 1, 2, 5, 10,
    ]
    assert influence["remaining_parcels"].tolist() == [
        138, 137, 136, 133, 128,
    ]


def test_no_diversity_gives_no_gain(monkeypatch):
    mod = load(monkeypatch)
    df = sample(mod)
    y = df[mod.TARGET].to_numpy(float)
    a = df[mod.ANCHOR].to_numpy(float)
    assert mod.optimal_convex_weight(y, a, a) == 0
    s, _ = mod.paired_diagnostics(
        y, a, a, .3, bootstrap=50, seed=5
    )
    assert s["unchanged_parcels"] == 138
    assert s["rmse_reference26mean"] == pytest.approx(s["rmse_blend"])
    with pytest.raises(ValueError, match="bootstrap"):
        mod.paired_diagnostics(y, a, a, .3, bootstrap=0, seed=5)


def test_report_writes_only_aggregate_rows_not_individual_y_or_id(
    monkeypatch, tmp_path,
):
    mod = load(monkeypatch)
    df = sample(mod)
    mod.report(
        df, "Compact14", .3, bootstrap=100, seed=19, output=tmp_path
    )
    expected = sorted([
        "stability_compact14_weight_grid.csv",
        "stability_compact14_influence.csv",
        "stability_compact14_summary.csv",
    ])
    assert sorted(f.name for f in tmp_path.glob("*.csv")) == expected
    for path in tmp_path.glob("*.csv"):
        recorded = pd.read_csv(path)
        assert mod.ID not in recorded.columns
        assert mod.TARGET not in recorded.columns


def test_summary_is_read_only_and_does_not_require_private_data(
    monkeypatch, tmp_path,
):
    mod = load(monkeypatch)
    df = sample(mod)
    mod.report(df, "Compact14", .3, bootstrap=20, seed=1, output=tmp_path)
    monkeypatch.setattr(
        sys, "argv",
        ["run_loo_ensemble_stability.py", "--summary",
         "--output", str(tmp_path)],
    )
    assert mod.main() == 0
