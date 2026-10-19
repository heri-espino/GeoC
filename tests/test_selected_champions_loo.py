"""Synthetic QA of the focused, parallel-capable three-model LOOCV runner."""
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _runner():
    root = Path(__file__).resolve().parents[1]
    tools = root / "tools"
    sys.path.insert(0, str(tools))
    source = tools / "run_selected_champions_loo.py"
    spec = importlib.util.spec_from_file_location("run_selected_champions_loo", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_only_selected_models_in_output_and_summary(tmp_path, capsys):
    module = _runner()
    assert module.MODELS == ("Local04D", "Reference26", "Compact14")
    for i in (1, 2, 3):
        row = pd.DataFrame([{
            module.ID: f"P{i:03}",
            "observed": float(i),
            "Local04D": i + .5,
            "Reference26": i + .2,
            "Compact14": i - .1,
        }])
        module.validate_selected(row, f"P{i:03}", float(i))
        module._write_fold(tmp_path / f"fold_{i:03}.csv", row)
    module._report(tmp_path, n=3, complete=False)
    out = pd.read_csv(tmp_path / "smoke_summary.csv")
    assert set(out.model) == set(module.MODELS)
    assert out.n_parcels.eq(3).all()
    assert "NOT 138-fold" not in capsys.readouterr().out
    assert (tmp_path / "smoke_predictions.csv").is_file()


def test_selected_fit_does_not_call_peer_b1_outer_refit(monkeypatch, tmp_path):
    module = _runner()
    calls = []

    class StubColdExperts:
        def __init__(self, frame, groups, distance, geology, protocol, tr, inner, cache):
            calls.append(("init", tuple(tr), len(inner), protocol))
            self.tr = np.asarray(tr)

        def full_fit(self, tr):
            calls.append(("full_fit", tuple(tr)))
            return {"train_positions": np.asarray(tr), "raw": "two champions"}

    fake = types.ModuleType("src.current_champions")
    fake.ColdExperts = StubColdExperts
    monkeypatch.setitem(sys.modules, "src.current_champions", fake)
    record = {
        "train_positions": [0, 1, 2, 3],
        "validation_positions": [4],
        "inner": [
            {"train_relative_positions": [0, 1, 2], "validation_relative_positions": [3]},
        ],
    }
    pack = module.fit_selected(
        pd.DataFrame(), None, None, {}, record, tmp_path / "fit",
    )
    assert pack["B1_weight"] == 0
    assert [c[0] for c in calls] == ["init", "full_fit"]
    assert pack["train_positions"].tolist() == record["train_positions"]


def test_incorrect_cached_fold_fails(tmp_path):
    module = _runner()
    row = pd.DataFrame([{
        module.ID: "P001", "observed": 3.0,
        "Local04D": 3.0, "Reference26": 4.0, "Compact14": 3.5,
    }])
    with pytest.raises(ValueError, match="Wrong held-out"):
        module.validate_selected(row, "P002", 3.0)
    with pytest.raises(ValueError, match="schema"):
        module.validate_selected(row.drop(columns=["Compact14"]), "P001", 3.0)
