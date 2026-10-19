"""Synthetic QA of 138-fold paired DBSCAN and Local04D result alignment."""
import importlib.util
import json
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def runner(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "tools"
    monkeypatch.syspath_prepend(str(path))
    spec = importlib.util.spec_from_file_location(
        "run_peer_dbscan_loo138", path / "run_peer_dbscan_loo138.py"
    )
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def frames(module):
    ids = [f"AGC_{i:03d}" for i in range(1, 139)]
    y = np.linspace(2, 5, 138)
    orig = pd.DataFrame({
        module.ID: ids, "observed": y,
        "Reference26": y + .3,
        "Compact14": y + .4,
    })
    db = pd.DataFrame({
        module.ID: ids[::-1], "observed": y[::-1],
        module.MODEL: y[::-1] + .2,
    })
    local = pd.DataFrame({
        module.ID: ids, "observed": y, "Local04D": y + .1,
    })
    return orig, db, local


def test_tuned_dbscan_fits_only_one_model_and_masks_heldout(monkeypatch, tmp_path):
    module = runner(monkeypatch)
    ids = [f"P{i:03d}" for i in range(138)]
    data = pd.DataFrame({
        module.ID: ids, module.TARGET: np.linspace(2, 5, 138),
    })
    calls = []

    class Expert:
        def __init__(self, masked, groups, distances, geology, protocol, tr, inner, cache):
            assert len(tr) == 137 and 0 not in tr and len(inner) == 4
            assert np.isnan(masked.iloc[0][module.TARGET])
            calls.append("fit")
            self.tr = tr

        def full_fit(self, tr):
            return {"train_positions": tr, "raw": {"shares": {"k": 4}}}

    def predict(pack, positions, frame, distance):
        assert len(positions) == 1 and positions[0] == 0
        return {"Reference26": np.array([3.0])}

    monkeypatch.setattr(module, "configured_gate", lambda _: nullcontext())
    monkeypatch.setattr(
        module, "_DATA",
        (data, pd.DataFrame(index=ids), np.zeros((138, 138)),
         {"geology": {}}, ids, Expert, predict),
    )
    fold, state, elapsed = module.fit_one(0, str(tmp_path))
    assert (fold, state) == (1, "trained")
    assert elapsed >= 0 and calls == ["fit"]
    result = pd.read_csv(tmp_path / "fold_001.csv", dtype={module.ID: str})
    assert set(result.columns) == {module.ID, "observed", module.MODEL}


def test_four_model_comparison_reuses_completed_original_folds(monkeypatch, tmp_path):
    module = runner(monkeypatch)
    original, dbscan, local = frames(module)
    destination = tmp_path / "dbscan"
    destination.mkdir()
    (destination / "run_spec.json").write_text("{}")
    for pos, row in enumerate(dbscan.iloc[::-1].to_dict("records"), start=1):
        # Fold order is irrelevant for comparison, but filenames are exact.
        module.write_csv(destination / f"fold_{pos:03d}.csv", pd.DataFrame([row]))
    orig_file = tmp_path / "original.csv"
    local_file = tmp_path / "local.csv"
    module.write_csv(orig_file, original)
    module.write_csv(local_file, local)
    module.compare(destination, orig_file, local_file)
    scores = pd.read_csv(destination / "four_model_summary.csv")
    assert set(scores.model) == {
        "Reference26", "Reference26_DBSCAN", "Compact14", "Local04D",
    }
    assert scores.n_holdouts.eq(138).all()
    assert abs(scores.loc[scores.model == "Local04D", "rmse"].iloc[0] - .1) < 1e-12
    assert abs(scores.loc[scores.model == "Reference26", "rmse"].iloc[0] - .3) < 1e-12
    assert len(pd.read_csv(destination / "four_model_predictions_private.csv")) == 138


def test_refuse_misaligned_observed_y(monkeypatch, tmp_path):
    module = runner(monkeypatch)
    original, dbscan, local = frames(module)
    local.loc[5, "observed"] += 1
    local_file = tmp_path / "wrong.csv"
    module.write_csv(local_file, local)
    with pytest.raises(ValueError, match="different observed yields"):
        module.local04d_predictions(original, local_file)


def test_no_artificial_jobs_limit(monkeypatch):
    module = runner(monkeypatch)
    ids = [f"P{i:03}" for i in range(138)]
    frame = pd.DataFrame({module.ID: ids, module.TARGET: np.ones(138)})
    monkeypatch.setattr(
        module, "load_inputs",
        lambda *_args: (frame, None, None, None, ids, None, None),
    )
    monkeypatch.setattr(
        sys, "argv", ["run_peer_dbscan_loo138.py", "--check", "--jobs", "100"],
    )
    assert module.main() == 0
