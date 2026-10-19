"""Contracts for Compact14+geological KMeans paired LOOCV."""
import importlib.util
import json
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _runner(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "tools"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        "run_compact14_kmeans_loo", scripts / "run_compact14_kmeans_loo.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _saved(module):
    ids = [f"P{i:03d}" for i in range(138)]
    y = np.linspace(2.1, 5.0, 138)
    original = pd.DataFrame({
        module.ID: ids, "observed": y,
        "Reference26": y + 0.2, "Compact14": y + 0.3,
    })
    new = pd.DataFrame({
        module.ID: ids[::-1], "observed": y[::-1],
        module.MODEL: y[::-1] + 0.1,
    })
    return ids, y, original, new


def test_model_is_distinct_and_138_folds(monkeypatch):
    module = _runner(monkeypatch)
    assert module.N == 138
    assert module.MODEL == "Compact14_KMeans"
    assert module.ORIGINAL == (
        module.ROOT / "reports/selected_champions_loo/loo_predictions.csv"
    )
    assert module.OUTPUT != module.ORIGINAL.parent


def test_comparison_reuses_reference26_no_retraining(monkeypatch, tmp_path):
    module = _runner(monkeypatch)
    ids, y, original, new = _saved(module)
    original_path = tmp_path / "old.csv"
    module.write_csv(original_path, original)
    out = tmp_path / "km"
    out.mkdir()
    (out / "run_spec.json").write_text("{}")
    for i, row in enumerate(new.to_dict("records"), start=1):
        module.write_csv(
            out / f"fold_{i:03d}.csv", pd.DataFrame([row])
        )
        (out / f"kmeans_{i:03d}.json").write_text(
            json.dumps({"k": 4 if i % 2 else 8})
        )

    module.compare(out, original_path, ids, y)
    result = pd.read_csv(out / "paired_summary.csv")
    assert set(result["model"]) == {
        "Reference26", "Compact14", "Compact14_KMeans",
    }
    assert result["n_holdouts"].eq(138).all()
    assert abs(
        result.loc[result["model"] == "Compact14_KMeans", "rmse"].iloc[0] - .1
    ) < 1e-12
    frequencies = pd.read_csv(out / "selected_k_counts.csv")
    assert frequencies["folds"].sum() == 138
    assert sorted(frequencies["selected_k"].tolist()) == [4, 8]


def test_mismatched_original_targets_fail_closed(monkeypatch, tmp_path):
    module = _runner(monkeypatch)
    ids, y, original, _new = _saved(module)
    original.loc[11, "observed"] += 0.1
    file = tmp_path / "old.csv"
    module.write_csv(file, original)
    with pytest.raises(ValueError, match="observed yields differ"):
        module.load_original(file, ids, y)


def test_one_fold_masks_heldout_before_original_factory(monkeypatch, tmp_path):
    module = _runner(monkeypatch)
    ids = [f"P{i:03d}" for i in range(138)]
    y = np.linspace(2, 5, 138)
    frame = pd.DataFrame({module.ID: ids, module.TARGET: y})
    jobs = []

    class OnlyCompact:
        def __init__(self, masked, groups, distance, geology, mode, tr, inner, cache):
            assert 0 not in tr and len(tr) == 137
            assert len(inner) == 4 and mode == "random"
            assert np.isnan(masked.iloc[0][module.TARGET])
            assert np.isfinite(masked.iloc[tr][module.TARGET].to_numpy(float)).all()
            jobs.append("compact-only")
            self.train = tr

    def fake_fit(factory, tr):
        assert 0 not in tr
        return {
            "train_positions": tr,
            "outer_experts": {"raw": {"compact": {"geology_gate": {"k": 8}}}},
        }

    monkeypatch.setattr(module, "compact_kmeans_factory", nullcontext)
    monkeypatch.setattr(module, "fit_compact_only", fake_fit)
    monkeypatch.setattr(module, "predict_compact_kmeans", lambda *a: np.array([3.3]))
    monkeypatch.setattr(module, "selected_k", lambda *a: {"k": 8})
    monkeypatch.setattr(
        module, "_DATA",
        (frame, pd.DataFrame(index=ids), np.zeros((138, 138)),
         {"geology": {}}, ids, OnlyCompact, None),
    )
    index, state, elapsed = module.fit_one(0, str(tmp_path))
    assert (index, state) == (1, "trained")
    assert elapsed >= 0 and jobs == ["compact-only"]
    fold = pd.read_csv(tmp_path / "fold_001.csv")
    assert set(fold.columns) == {module.ID, "observed", module.MODEL}
    assert (tmp_path / "kmeans_001.json").exists()


def test_unlimited_jobs_positive(monkeypatch):
    module = _runner(monkeypatch)
    ids, y, original, _new = _saved(module)
    monkeypatch.setattr(
        module, "load_inputs",
        lambda *_args: (
            pd.DataFrame({module.ID: ids, module.TARGET: y}),
            None, None, None, ids, None, None,
        ),
    )
    monkeypatch.setattr(
        module, "load_original", lambda *_args: original
    )
    monkeypatch.setattr(
        sys, "argv", ["run_compact14_kmeans_loo.py", "--check", "--jobs", "150"],
    )
    assert module.main() == 0
