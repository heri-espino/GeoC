"""Synthetic protocol and argument checks for the first-46 geology gate search."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _runner(monkeypatch):
    tools = Path(__file__).resolve().parents[1] / "tools"
    monkeypatch.syspath_prepend(str(tools))
    path = tools / "run_peer_gate_grid_loo46.py"
    spec = importlib.util.spec_from_file_location("peer_gate_loo46_tests", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_grid_choices_and_paired_model_names(monkeypatch):
    module = _runner(monkeypatch)
    from geocebada.evaluation.peer_gate_grid import GATE_GRIDS, VARIANTS

    assert module.COUNT == 46
    assert VARIANTS == (
        "original", "kmeans_expanded", "dbscan", "support", "global"
    )
    assert len(GATE_GRIDS["kmeans_expanded"]) == 5
    assert len(GATE_GRIDS["dbscan"]) == 8
    assert len(GATE_GRIDS["support"]) == 5
    assert len(GATE_GRIDS["global"]) == 1
    assert len(module.model_columns(VARIANTS)) == 6
    assert module.variant_list("support,original,dbscan") == (
        "original", "dbscan", "support",
    )
    with pytest.raises(ValueError, match="Original KMeans"):
        module.variant_list("dbscan,support")
    with pytest.raises(ValueError, match="distinct"):
        module.variant_list("original,original")


def test_46_same_bundle_positions_and_predictions(monkeypatch, tmp_path):
    module = _runner(monkeypatch)
    variants = ("original", "dbscan")
    ids = [f"P{i:03d}" for i in range(138)]
    frame = pd.DataFrame({
        module.ID: ids,
        module.TARGET: np.linspace(2, 5, 138),
    })
    observed = []

    class FakeColdExperts:
        def __init__(self, data, groups, distance, geology, protocol, tr, inner, cache):
            assert len(tr) == 137
            assert len(inner) == 4
            held = set(range(138)) - set(tr)
            assert len(held) == 1
            position = held.pop()
            observed.append(position)
            assert np.isnan(data.iloc[position][module.TARGET])
            assert np.isfinite(data.iloc[tr][module.TARGET].to_numpy(float)).all()
            self.train = tr

        def full_fit(self, tr):
            return {
                "train_positions": tr,
                "raw": {"shares": {"k": 4}},
            }

    def fake_predict(pack, query, masked, distance):
        assert np.isnan(float(masked.iloc[query[0]][module.TARGET]))
        return {"Reference26": [3.2], "Compact14": [3.1]}

    from contextlib import nullcontext

    monkeypatch.setattr(module, "configured_gate", lambda kind: nullcontext())
    monkeypatch.setattr(module, "_DATA", (
        frame, pd.DataFrame(index=ids), np.zeros((138, 138)), {"geology": {}},
        ids, FakeColdExperts, fake_predict,
    ))
    fold, status, seconds = module.fit_one(0, str(tmp_path), variants)
    assert fold == 1 and status == "trained" and seconds >= 0
    assert observed == [0, 0]
    out = module.validate_fold(
        tmp_path / "fold_001.csv", ids[0], float(frame.iloc[0][module.TARGET]),
        variants,
    )
    assert set(out.columns) == {
        module.ID, "observed", *module.model_columns(variants)
    }
    module.summarize(tmp_path, n=1, variants=variants, complete=False)
    assert (tmp_path / "smoke_scores.csv").exists()


def test_no_artificial_worker_limit(monkeypatch):
    module = _runner(monkeypatch)
    frame = pd.DataFrame({
        module.ID: [f"P{i:03d}" for i in range(138)],
        module.TARGET: np.ones(138),
    })
    monkeypatch.setattr(
        module, "load_inputs",
        lambda *_args: (
            frame, None, None, None, frame[module.ID].tolist(), None, None
        ),
    )
    monkeypatch.setattr(
        sys, "argv",
        ["run_peer_gate_grid_loo46.py", "--check", "--jobs", "128"],
    )
    assert module.main() == 0
