"""Synthetic contracts for Reference26 fixed geological KMeans search."""
import importlib.util
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from geocebada.evaluation.peer_reference26_fixed_kmeans import (
    DEFAULT_K,
    fixed_geological_kmeans,
)


def _load(monkeypatch):
    tools = Path(__file__).resolve().parents[1] / "tools"
    monkeypatch.syspath_prepend(str(tools))
    spec = importlib.util.spec_from_file_location(
        "reference26_fixed_k_runner", tools / "run_reference26_kmeans_grid.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _baseline(module):
    ids = [f"AGC_{i:03d}" for i in range(138)]
    y = np.linspace(2.0, 5.0, 138)
    data = pd.DataFrame({
        module.ID: ids,
        "observed": y,
        "Reference26": y + 0.3,
        "Compact14": y + 0.4,
    })
    return ids, y, data


def test_k_values_have_separate_names(monkeypatch):
    module = _load(monkeypatch)
    assert module.parse_ks("26,14,8,4") == (4, 8, 14, 26)
    assert module.k_name(14) == "Reference26_KMeans_k14"
    assert module.k_name(26) == "Reference26_KMeans_k26"
    assert 14 in DEFAULT_K and 26 in DEFAULT_K
    with pytest.raises(ValueError, match="distinct"):
        module.parse_ks("14,14")
    with pytest.raises(ValueError, match="positive"):
        module.parse_ks("0,14")


def test_original_reference26_order_and_y_checked(monkeypatch, tmp_path):
    module = _load(monkeypatch)
    ids, y, data = _baseline(module)
    file = tmp_path / "original.csv"
    module.write_csv(file, data.iloc[::-1])
    aligned = module.reference_predictions(file, ids, y, folds=46)
    assert aligned[module.ID].tolist() == ids[:46]
    assert aligned["Reference26_original"].tolist() == pytest.approx(
        (y + 0.3)[:46]
    )
    data.loc[0, "observed"] += 1
    module.write_csv(file, data)
    with pytest.raises(ValueError, match="disagree"):
        module.reference_predictions(file, ids, y, folds=46)


def test_one_fold_uses_137_labels_and_each_fixed_k(monkeypatch, tmp_path):
    module = _load(monkeypatch)
    ids, y, _orig = _baseline(module)
    frame = pd.DataFrame({module.ID: ids, module.TARGET: y})
    calls = []

    class Cold:
        def __init__(self, masked, groups, distance, geo, protocol, tr, inner, cache):
            assert len(tr) == 137 and len(inner) == 4 and 0 not in tr
            assert np.isnan(masked.iloc[0][module.TARGET])
            assert np.isfinite(masked.iloc[tr][module.TARGET].to_numpy(float)).all()
            calls.append("fit")

        def full_fit(self, tr):
            return {
                "train_positions": tr,
                "raw": {"shares": None},
            }

    def predict(pack, query, masked, distance):
        assert query.tolist() == [0]
        assert np.isnan(masked.iloc[0][module.TARGET])
        return {"Reference26": [2.2]}

    monkeypatch.setattr(
        module, "_DATA",
        (frame, pd.DataFrame(index=ids), np.zeros((138, 138)),
         {"geology": {}}, ids, Cold, predict),
    )
    monkeypatch.setattr(module, "fixed_geological_kmeans", lambda k: nullcontext())
    ks = (14, 26)
    pos, state, seconds = module.fit_one(0, str(tmp_path), ks)
    assert pos == 1 and state == "trained" and seconds >= 0
    assert calls == ["fit", "fit"]
    file = tmp_path / "fold_001.csv"
    result = pd.read_csv(file)
    assert set(result.columns) == {
        module.ID, "observed",
        "Reference26_KMeans_k14", "Reference26_KMeans_k26",
    }
    baseline = pd.DataFrame({
        module.ID: [ids[0]], "observed": [y[0]],
        module.BASE_NAME: [y[0] + 0.4],
    })
    module.summaries(tmp_path, baseline, ks, 1, complete=False)
    scores = pd.read_csv(tmp_path / "smoke_summary.csv")
    assert set(scores.model) == {
        "Reference26_original",
        "Reference26_KMeans_k14",
        "Reference26_KMeans_k26",
    }


def test_no_fixed_max_jobs(monkeypatch):
    module = _load(monkeypatch)
    ids, y, _orig = _baseline(module)
    frame = pd.DataFrame({module.ID: ids, module.TARGET: y})
    monkeypatch.setattr(
        module, "load_inputs",
        lambda *args: (frame, None, None, None, ids, None, None),
    )
    monkeypatch.setattr(
        module, "reference_predictions",
        lambda *a, **k: pd.DataFrame({
            module.ID: ids[:46], "observed": y[:46],
            module.BASE_NAME: y[:46] + .2,
        }),
    )
    monkeypatch.setattr(
        sys, "argv", [
            "run_reference26_kmeans_grid.py", "--check", "--jobs", "300",
        ],
    )
    assert module.main() == 0


def test_kmeans_patch_checks_positive_k():
    with pytest.raises(ValueError, match="positive"):
        with fixed_geological_kmeans(0):
            pass
