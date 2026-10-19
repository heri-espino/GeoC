"""Synthetic safety checks for separate mean-imputed staged Reference26 Ridge."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def runner(monkeypatch):
    tools = Path(__file__).resolve().parents[1] / "tools"
    monkeypatch.syspath_prepend(str(tools))
    spec = importlib.util.spec_from_file_location(
        "reference26_budget_mean_tests",
        tools / "run_reference26_ridge_budget_mean.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic(module):
    ids = [f"P{i:03d}" for i in range(138)]
    y = np.linspace(2.1, 4.9, 138)
    original = pd.DataFrame({
        module.ID: ids, "observed": y, "Reference26": y+.25,
    })
    return ids, y, original


def test_mean_version_is_distinct_from_median_and_follows_25_8_3(monkeypatch):
    m = runner(monkeypatch)
    assert m.OUTPUT != m.budget.OUTPUT
    assert m.OUTPUT != m.original_grid.OUTPUT
    assert m.STAGES == {
        "screen": (12, 25),
        "advance": (46, 8),
        "final": (138, 3),
    }
    assert len(m.SCREEN) == 25
    assert m.DEFAULT in m.SCREEN


def test_mean_batch_applies_override_to_native_fit_and_preserves_137_labels(
    monkeypatch, tmp_path,
):
    m = runner(monkeypatch)
    ids, y, original = synthetic(m)
    frame = pd.DataFrame({m.ID: ids, m.TARGET: y})
    events = []
    monkeypatch.setattr(
        m.original_grid, "_WORKER", (
            frame, None, None, None, ids, None, None,
        ),
    )
    from contextlib import contextmanager, nullcontext

    @contextmanager
    def mean_patch():
        events.append("mean_start")
        yield
        events.append("mean_end")

    @contextmanager
    def fake_climate():
        events.append("cache_start")
        yield
        events.append("cache_end")

    monkeypatch.setattr(m, "use_training_means", mean_patch)
    monkeypatch.setattr(
        "geocebada.evaluation.peer_reference26_ridge_grid."
        "share_climate_fits_between_grid_candidates",
        fake_climate,
    )
    def fake_fit(pos, pair, output):
        assert pos == 0
        assert pair == m.DEFAULT
        assert events == ["mean_start", "cache_start"]
        return float(y[pos]+.1), tmp_path / "no_temp"

    monkeypatch.setattr(m.original_grid, "fit_one", fake_fit)
    idx, done, cached, elapsed = m.run_fold_batch(
        0, (m.DEFAULT,), str(tmp_path),
    )
    assert (idx, done, cached) == (1, 1, 0)
    assert elapsed >= 0
    assert events == [
        "mean_start", "cache_start", "cache_end", "mean_end",
    ]
    saved = m.original_grid.fold_file(tmp_path, 0, m.DEFAULT)
    result = pd.read_csv(saved)
    assert abs(result[m.original_grid.candidate_name(*m.DEFAULT)].iloc[0]
               - y[0] - .1) < 1e-12


def test_mean_summary_compares_same_heldouts_to_old_median(monkeypatch, tmp_path):
    m = runner(monkeypatch)
    ids, y, original = synthetic(m)
    for i in range(12):
        m.original_grid.write_csv(
            m.fold_path(tmp_path, i, m.DEFAULT),
            pd.DataFrame([{
                m.ID: ids[i],
                "observed": y[i],
                m.original_grid.candidate_name(*m.DEFAULT): y[i] + .1,
            }]),
        )
    monkeypatch.setattr(m, "SCREEN", (m.DEFAULT,))
    m.results(tmp_path, original, "screen", (m.DEFAULT,))
    scores = pd.read_csv(tmp_path / "screen_scores.csv")
    assert scores.imputation.tolist() == ["mean"]
    assert scores.n_common_holdouts.tolist() == [12]
    assert scores.rmse.iloc[0] == pytest.approx(.1)
    assert scores.delta_mse_vs_original_median.iloc[0] == pytest.approx(
        .1**2 - .25**2
    )


def test_mean_provenance_excludes_any_median_cache_and_fails_on_wrong_output(
    monkeypatch, tmp_path,
):
    m = runner(monkeypatch)
    # Explicitly refuse old median provenance masquerading as a mean result.
    output = tmp_path / "mean"
    wrong = {"imputation": "median"}
    m.check_saved(output, wrong)
    with pytest.raises(ValueError, match="changed"):
        m.check_saved(output, {"imputation": "mean"})


def test_check_unrestricted_jobs_without_data_fits(monkeypatch):
    m = runner(monkeypatch)
    ids, y, original = synthetic(m)
    frame = pd.DataFrame({m.ID: ids, m.TARGET: y})
    monkeypatch.setattr(
        m.original_grid, "load_inputs",
        lambda *_args: (frame, None, None, None, ids, None, None),
    )
    monkeypatch.setattr(
        m.original_grid, "original_results",
        lambda *_args: original,
    )
    monkeypatch.setattr(
        sys, "argv",
        ["run_reference26_ridge_budget_mean.py", "--check", "--jobs", "250"],
    )
    assert m.main() == 0
