"""Synthetic checks for the peer-only Reference26 and Compact14 LOOCV runner."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def runner():
    path = Path(__file__).resolve().parents[1] / "tools/run_selected_champions_loo.py"
    spec = importlib.util.spec_from_file_location("peer_only_loo", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_only_teammate_selected_models_and_native_bundle():
    m = runner()
    assert m.MODELS == ("Reference26", "Compact14")
    assert m.BUNDLE.relative_to(m.ROOT).as_posix() == (
        "data_n/processed/current_champions/v1"
    )


def test_137_train_1_held_out_and_inner_only_train():
    m = runner()
    for held in (0, 22, 137):
        tr, inner = m.fold_record(held)
        assert len(tr) == 137
        assert held not in tr
        assert len(inner) == 4
        seen = []
        for a, b in inner:
            assert set(a).isdisjoint(b)
            assert held not in tr[a] and held not in tr[b]
            seen.extend(b.tolist())
        assert sorted(seen) == list(range(137))


def test_three_smoke_parcels_not_a_full_loo_result(tmp_path):
    m = runner()
    for i in (1, 2, 3):
        row = pd.DataFrame([{
            m.ID: f"P{i:03d}",
            "observed": float(i),
            "Reference26": float(i) + .2,
            "Compact14": float(i) - .1,
        }])
        m.validate_saved(row, f"P{i:03d}", float(i))
        m.write_csv(tmp_path / f"fold_{i:03d}.csv", row)
    m.summarize(tmp_path, count=3, final=False)
    result = pd.read_csv(tmp_path / "smoke_summary.csv")
    assert set(result.model) == {"Reference26", "Compact14"}
    assert result.held_out_parcels.eq(3).all()
    with pytest.raises(ValueError, match="Need 138 completed"):
        m.summarize(tmp_path, count=138, final=True)


def test_peer_cold_fit_masks_y_and_does_not_refit_b1(tmp_path, monkeypatch):
    m = runner()
    ids = [f"P{i:03}" for i in range(138)]
    frame = pd.DataFrame({
        m.ID: ids, m.TARGET: np.linspace(2, 6, 138),
    })
    calls = []

    class NativeColdExperts:
        def __init__(self, masked, groups, distance, geology, protocol, tr, inner, cache):
            assert np.isnan(masked.iloc[0][m.TARGET])
            assert np.isfinite(masked.iloc[tr][m.TARGET].to_numpy(float)).all()
            assert 0 not in tr
            assert protocol == "random"
            calls.append(("init", len(tr), len(inner)))
            self.tr = tr

        def full_fit(self, tr):
            calls.append(("full_fit", len(tr)))
            return {"train_positions": tr}

    def peer_predict(pack, query, masked, distance):
        assert pack["B1_weight"] == 0.
        assert set(pack["train_positions"]) == set(range(1, 138))
        assert np.isnan(masked.iloc[query[0]][m.TARGET])
        return {"Reference26": np.array([2.5]), "Compact14": np.array([2.6])}

    groups = pd.DataFrame(index=ids)
    monkeypatch.setattr(m, "_DATA", (
        frame, groups, np.zeros((138, 138)), {"geology": {}},
        ids, NativeColdExperts, peer_predict,
    ))
    fold, status, seconds = m.fit_one(0, str(tmp_path))
    assert fold == 1 and status == "trained" and seconds >= 0.
    assert calls == [("init", 137, 4), ("full_fit", 137)]
    result = pd.read_csv(tmp_path / "fold_001.csv")
    assert set(result.columns) == {m.ID, "observed", *m.MODELS}
    assert result.iloc[0]["Reference26"] == 2.5
    assert result.iloc[0]["Compact14"] == 2.6
    assert not (tmp_path / "fold_002.csv").exists()


def test_bad_cached_predictions_are_rejected():
    m = runner()
    row = pd.DataFrame([{
        m.ID: "P001", "observed": 3.0,
        "Reference26": 3.1, "Compact14": 3.2,
    }])
    with pytest.raises(ValueError, match="held-out"):
        m.validate_saved(row, "P002", 3.0)
    with pytest.raises(ValueError, match="schema"):
        m.validate_saved(row.drop(columns=["Compact14"]), "P001", 3.0)
