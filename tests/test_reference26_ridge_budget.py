"""Safety tests for staged, resumable 25 -> 8 -> 3 Ridge search."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def budget(monkeypatch):
    tools = Path(__file__).resolve().parents[1] / "tools"
    monkeypatch.syspath_prepend(str(tools))
    spec = importlib.util.spec_from_file_location(
        "reference26_ridge_budget_tests",
        tools / "run_reference26_ridge_budget.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture_data(mod):
    ids = [f"P{i:03d}" for i in range(138)]
    y = np.linspace(2, 6, 138)
    baseline = pd.DataFrame({
        mod.ID: ids,
        "observed": y,
        "Reference26": y + .30,
    })
    return ids, y, baseline


def test_budget_fixed_25_and_fidelity_schedule(monkeypatch):
    mod = budget(monkeypatch)
    assert len(mod.SCREEN) == len(set(mod.SCREEN)) == 25
    assert mod.DEFAULT_PAIR in mod.SCREEN
    assert mod.STAGES == {
        "screen": (12, 25), "advance": (46, 8), "final": (138, 3),
    }
    assert mod.selected_pairs(Path("missing"), "screen") == mod.SCREEN


def test_safe_copy_reuses_old_completed_pair_without_changing_source(
    monkeypatch, tmp_path,
):
    mod = budget(monkeypatch)
    ids, y, _baseline = fixture_data(mod)
    src, dest = tmp_path / "old", tmp_path / "new"
    pair = (20, 10.)
    old_path = mod.original_grid.fold_file(src, 0, pair)
    mod.original_grid.write_csv(old_path, pd.DataFrame([{
        mod.ID: ids[0], "observed": y[0],
        mod.original_grid.candidate_name(*pair): y[0] + .15,
    }]))
    before = old_path.read_bytes()
    found, copied = mod.cached_old(src, dest, 0, pair, ids, y)
    assert (found, copied) == (True, True)
    assert old_path.read_bytes() == before
    found, copied = mod.cached_old(src, dest, 0, pair, ids, y)
    assert (found, copied) == (True, False)
    assert mod.original_grid.fold_file(dest, 0, pair).is_file()


def test_bad_y_in_old_cache_rejected(monkeypatch, tmp_path):
    mod = budget(monkeypatch)
    ids, y, _baseline = fixture_data(mod)
    pair = (20, 10.)
    file = mod.original_grid.fold_file(tmp_path / "old", 0, pair)
    mod.original_grid.write_csv(file, pd.DataFrame([{
        mod.ID: ids[0], "observed": y[0] + 1.,
        mod.original_grid.candidate_name(*pair): y[0] + .15,
    }]))
    with pytest.raises(ValueError, match="Incorrect cached"):
        mod.cached_old(
            tmp_path / "old", tmp_path / "new", 0, pair, ids, y
        )


def test_stage_scoring_and_shortlists_share_identical_first_parcels(
    monkeypatch, tmp_path,
):
    mod = budget(monkeypatch)
    ids, y, baseline = fixture_data(mod)
    for pos in range(12):
        for i, pair in enumerate(mod.SCREEN):
            # A good static candidate has RMSE ~ .1, default ~ .15.
            diff = 0.15 if pair == mod.DEFAULT_PAIR else 0.1 + i * .002
            mod.original_grid.write_csv(
                mod.original_grid.fold_file(tmp_path, pos, pair),
                pd.DataFrame([{
                    mod.ID: ids[pos], "observed": y[pos],
                    mod.original_grid.candidate_name(*pair): y[pos] + diff,
                }]),
            )
    mod.scores_for_stage(
        tmp_path, baseline, "screen", mod.SCREEN
    )
    shortlisted = mod.selected_pairs(tmp_path, "advance")
    assert len(shortlisted) == 8
    assert mod.DEFAULT_PAIR in shortlisted
    assert (tmp_path / "screen_scores.csv").is_file()
    assert json.loads(
        (tmp_path / "screen_selection.json").read_text()
    )["n_common_holdouts"] == 12

    # Top 2 for final + original anchor, using 46 common parcels.
    ranked = pd.DataFrame([{
        "k": k, "alpha": alpha, "rmse": i * .01
    } for i, (k, alpha) in enumerate(shortlisted)])
    ranked.to_csv(tmp_path / "advance_scores.csv", index=False)
    (tmp_path / "advance_selection.json").write_text("{}")
    final = mod.selected_pairs(tmp_path, "final")
    assert len(final) == 3 and mod.DEFAULT_PAIR in final


def test_mismatched_provenance_fails_closed(monkeypatch, tmp_path):
    mod = budget(monkeypatch)
    info = {
        "invariant_hashes": {
            "teammate_code_hash": "correct",
        },
    }
    old = tmp_path / "old"
    old.mkdir()
    (old / "run_spec.json").write_text(json.dumps({
        "teammate_code_hash": "different",
    }))
    with pytest.raises(ValueError, match="differs"):
        mod.ensure_source_compatible(old, info)
