"""Checkpoint 07C regression tests: nested CV, frozen Local04D, safe resume."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from geocebada.evaluation.checkpoint07c import (
    FoldRepresentations,
    _balanced_columns,
    _predict_once,
    aggregate_07c_results,
    load_development,
    run_nested_search,
)

ID = "ID_POLIGONO"
Y = "RENDIMIENTO_T_HA"


def _synthetic_development(seed: int = 2026):
    rng = np.random.default_rng(seed)
    ids = [f"AGC_{i:03d}" for i in range(1, 198)]
    matrix = rng.normal(size=(197, 32))
    matrix[8:30, 6] = np.nan
    names = [f"hls_all__ndvi_metric_{i}" for i in range(12)] + [
        f"clim_official__thermal_{i}" for i in range(10)
    ] + [f"smap_AM__mean_{i}" for i in range(10)]
    features = pd.DataFrame(matrix, columns=names)
    features.insert(0, ID, ids)
    train_ids = ids[:138]
    y = 3.7 + 0.2 * np.nan_to_num(matrix[:138, 0]) + rng.normal(
        scale=0.08, size=138
    )
    targets = pd.DataFrame({
        ID: ids,
        "CONJUNTO": ["ENTRENAMIENTO"] * 138 + ["PREDICCION"] * 59,
        Y: [*y, *([np.nan] * 59)],
    })
    by_id = dict(zip(train_ids, y, strict=True))
    memberships, local = [], []
    for s in range(16):
        split = f"target_matched_{s + 1:02d}"
        order = np.random.default_rng(s + 300).permutation(train_ids)
        pseudo_test = set(order[:41])
        for pid in train_ids:
            role = "pseudo_target" if pid in pseudo_test else "pseudo_train"
            memberships.append({
                ID: pid, "split_id": split, "family": "target_matched",
                "role": role,
            })
            if role == "pseudo_target":
                local.append({
                    ID: pid, "split_id": split, "family": "target_matched",
                    "method": "Local04D", "observed": by_id[pid],
                    "predicted": by_id[pid] + 0.07,
                })
    return (
        features, targets, pd.DataFrame(memberships), pd.DataFrame(local)
    )


def test_07c_load_aligns_frozen_target_matched_incumbent() -> None:
    x, y, splits, local = _synthetic_development()
    data = load_development(x, y, splits, local)
    assert data.x.shape == (197, 32)
    assert len(data.known_y) == 138
    assert len(data.splits) == 16
    assert len(data.splits["target_matched_01"][0]) == 97
    assert len(data.splits["target_matched_01"][1]) == 41
    broken = local.copy()
    broken.loc[0, "observed"] = 88
    with pytest.raises(ValueError, match="observed target mismatch"):
        load_development(x, y, splits, broken)
    leaky = x.assign(RENDIMIENTO_T_HA=999)
    with pytest.raises(ValueError, match="Prohibited"):
        load_development(leaky, y, splits, local)
    leaky2 = x.assign(siap_2025__yield=999)
    with pytest.raises(ValueError, match="Prohibited"):
        load_development(leaky2, y, splits, local)


def test_07c_representations_are_train_only_and_finite() -> None:
    x, y, splits, local = _synthetic_development()
    data = load_development(x, y, splits, local)
    train, test = data.splits["target_matched_01"]
    selected = _balanced_columns(data.x.loc[train], budget=16)
    assert len(selected) == 16
    assert len(set(selected)) == len(selected)
    original = FoldRepresentations(data.x, train, test)
    a, b = original.get("raw256")
    assert a.shape[0] == 97 and b.shape[0] == 41
    assert np.isfinite(a).all() and np.isfinite(b).all()
    pca_train, pca_test = original.get("pca16")
    assert pca_train.shape == (97, 16)
    assert pca_test.shape == (41, 16)
    # Never fit PCA from the test values.
    altered = data.x.copy()
    altered.loc[test, :] = altered.loc[test, :] * 1e3
    modified = FoldRepresentations(altered, train, test)
    new_train, _ = modified.get("pca16")
    np.testing.assert_allclose(pca_train, new_train, atol=1e-8)
    rng = np.random.default_rng(9)
    label = rng.normal(size=97)
    for family, params in (
        ("krr", {"alpha": 1.0, "gamma": 0.01}),
        ("pls", {"n_components": 4}),
    ):
        pred = _predict_once(
            family, params, (pca_train, pca_test), label,
            device="cpu", threads=1, seed=7,
        )
        assert pred.shape == (41,)
        assert np.isfinite(pred).all()


def test_07c_nested_search_resumes_completed_fit_and_preserves_baseline(
    tmp_path: Path,
) -> None:
    pytest.importorskip("optuna")
    x, y, splits, local = _synthetic_development()
    data = load_development(x, y, splits, local)
    output = tmp_path / "nested"
    pred, meta = run_nested_search(
        data, "target_matched_01", "krr",
        output=output, trials=2, inner_folds=2,
        device="cpu", threads=1, seed=123,
    )
    assert len(pred) == 205
    assert meta["tuning_trials_complete"] == 2
    assert meta["outer_train_count"] == 97
    assert meta["outer_test_count"] == 41
    assert np.isfinite(meta["outer_rmse_model"])
    assert set(pred["model"]) == {
        "Local04D", "krr", "Local04D_plus_krr_w0.10",
        "Local04D_plus_krr_w0.25", "Local04D_plus_krr_w0.50",
    }
    baseline = pred.loc[pred["model"].eq("Local04D")]
    np.testing.assert_allclose(
        baseline["predicted"] - baseline["observed"], 0.07, atol=1e-10
    )
    pred2, meta2 = run_nested_search(
        data, "target_matched_01", "krr",
        output=output, trials=2, inner_folds=2,
        device="cpu", threads=1, seed=123,
    )
    pd.testing.assert_frame_equal(pred, pred2, check_exact=False)
    assert meta2["best_params"] == meta["best_params"]
    # Simulate an interruption after CSV rename but before JSON rename.
    (output / "target_matched_01__krr.json").unlink()
    pred3, meta3 = run_nested_search(
        data, "target_matched_01", "krr",
        output=output, trials=2, inner_folds=2,
        device="cpu", threads=1, seed=123,
    )
    assert len(pred3) == 205
    assert meta3["tuning_trials_complete"] == 2
    assert (output / "target_matched_01__krr.csv.incomplete").is_file()
    full, summary = aggregate_07c_results(output)
    assert len(full) == 205
    assert len(summary) == 5
    assert (summary["n_splits_completed"] == 1).all()


def test_07c_resume_fingerprint_cannot_mix_data_and_parameters(
    tmp_path: Path,
) -> None:
    script = (
        Path(__file__).resolve().parents[1] / "tools" / "run_checkpoint_07c.py"
    )
    spec = importlib.util.spec_from_file_location("checkpoint07_runner_test", script)
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    runner._lock_run_spec(tmp_path, {
        "n_trials_per_split_family": 48,
        "device": "gpu",
        "data_sha256": {"features": "abc"},
    })
    runner._lock_run_spec(tmp_path, {
        "n_trials_per_split_family": 48,
        "device": "gpu",
        "data_sha256": {"features": "abc"},
    })
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        runner._lock_run_spec(tmp_path, {
            "n_trials_per_split_family": 48,
            "device": "gpu",
            "data_sha256": {"features": "changed"},
        })
    assert json.loads((tmp_path / "run_spec.json").read_text())[
        "data_sha256"
    ] == {"features": "abc"}
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "target_matched_01__catboost.sqlite").touch()
    with pytest.raises(ValueError, match="unversioned"):
        runner._lock_run_spec(legacy, {"n_trials_per_split_family": 48})
