"""Synthetic, lightweight tests of three-branch Checkpoint 07 continuation."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from geocebada.evaluation.checkpoint07_local import _predict_local, run_local07
from geocebada.evaluation.checkpoint07_prithvi_head import (
    _ridge_head,
    evaluate_prithvi,
)

ID = "ID_POLIGONO"


def _import_tool(name: str):
    source = Path(__file__).resolve().parents[1] / "tools" / name
    spec = importlib.util.spec_from_file_location(name.replace(".", "_"), source)
    assert spec is not None and spec.loader is not None
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def _development():
    from geocebada.evaluation.checkpoint07c import load_development

    rng = np.random.default_rng(72026)
    ids = [f"AGC_{i:03d}" for i in range(1, 198)]
    frame = pd.DataFrame(
        rng.normal(size=(197, 48)),
        columns=[f"hls_all__f_{i}" for i in range(48)],
    )
    frame["base_centroid_lat"] = np.linspace(19, 20, 197)
    frame["base_centroid_lon"] = np.linspace(-99, -98, 197)
    frame.insert(0, ID, ids)
    y = 4.0 + 0.25 * frame.iloc[:138, 1].to_numpy() + rng.normal(
        0, 0.2, 138
    )
    targets = pd.DataFrame({
        ID: ids,
        "CONJUNTO": ["ENTRENAMIENTO"]*138 + ["PREDICCION"]*59,
        "RENDIMIENTO_T_HA": [*y, *([np.nan]*59)],
    })
    membership, incumbent = [], []
    for k in range(16):
        split = f"target_matched_{k+1:02d}"
        heldout = set(np.random.default_rng(k).choice(
            ids[:138], size=41, replace=False
        ))
        for j, pid in enumerate(ids[:138]):
            role = "pseudo_target" if pid in heldout else "pseudo_train"
            membership.append({
                "split_id": split, "family": "target_matched",
                "role": role, ID: pid,
            })
            if role == "pseudo_target":
                incumbent.append({
                    "split_id": split, "family": "target_matched",
                    ID: pid, "method": "Local04D",
                    "observed": float(y[j]),
                    "predicted": float(y[j]+0.1),
                })
    return load_development(
        frame, targets, pd.DataFrame(membership), pd.DataFrame(incumbent)
    )


def test_local07_predictions_use_train_labels_and_geometry() -> None:
    rng = np.random.default_rng(12)
    a = rng.normal(size=(33, 8))
    b = rng.normal(size=(6, 8))
    xy = rng.normal(size=(33, 2))
    test_xy = rng.normal(size=(6, 2))
    y = rng.normal(size=33)
    output = _predict_local(
        a, b, xy, test_xy, y,
        k=12, alpha=30.0, geo_share=0.5,
    )
    assert output.shape == (6,)
    assert np.isfinite(output).all()
    with pytest.raises(ValueError, match="Invalid"):
        _predict_local(
            a, b, xy, test_xy, y,
            k=1, alpha=30.0, geo_share=0.5,
        )


def test_local07_nested_resume_and_paired_local04d(tmp_path: Path) -> None:
    pytest.importorskip("optuna")
    data = _development()
    report = run_local07(
        data, split_id="target_matched_01", output=tmp_path,
        trials=1, inner_folds=2, seed=77,
    )
    assert report["outer_train_count"] == 97
    assert report["outer_test_count"] == 41
    assert report["n_complete_trials"] == 1
    rows = pd.read_csv(tmp_path / "target_matched_01__local07.csv")
    assert len(rows) == 205
    baseline = rows[rows["model"].eq("Local04D")]
    np.testing.assert_allclose(
        baseline["predicted"] - baseline["observed"], 0.1, atol=1e-10
    )
    again = run_local07(
        data, split_id="target_matched_01", output=tmp_path,
        trials=1, inner_folds=2, seed=77,
    )
    assert again == report


def test_prithvi_head_only_uses_frozen_embeddings_and_inner_y(tmp_path: Path) -> None:
    data = _development()
    rng = np.random.default_rng(8)
    embeddings = pd.DataFrame(
        rng.normal(size=(197, 15)),
        columns=[f"prithvi__f{i}" for i in range(15)],
    )
    embeddings.insert(0, ID, data.x.index.tolist())
    location = tmp_path / "embeddings.csv"
    embeddings.to_csv(location, index=False)
    result = evaluate_prithvi(
        data, embedding_path=location, output=tmp_path / "heads",
        inner_folds=2, max_splits=1,
    )
    assert result["n_completed_splits"] == 1
    assert result["embedding_dim"] == 15
    rows = pd.read_csv(
        tmp_path / "heads/target_matched_01__prithvi.csv"
    )
    assert len(rows) == 205
    assert set(rows["model"]) == {
        "Local04D", "Prithvi07",
        "Local04D_plus_Prithvi07_w0.10",
        "Local04D_plus_Prithvi07_w0.25",
        "Local04D_plus_Prithvi07_w0.50",
    }
    again = evaluate_prithvi(
        data, embedding_path=location, output=tmp_path / "heads",
        inner_folds=2, max_splits=1,
    )
    assert again["n_completed_splits"] == 1
    broken = embeddings.assign(RENDIMIENTO_T_HA=7)
    broken.to_csv(location, index=False)
    with pytest.raises(ValueError, match="may not include y"):
        evaluate_prithvi(
            data, embedding_path=location, output=tmp_path / "else",
            inner_folds=2, max_splits=1,
        )


def test_prithvi_head_predict_is_finite() -> None:
    rng = np.random.default_rng(31)
    a, b = rng.normal(size=(27, 12)), rng.normal(size=(3, 12))
    y = rng.normal(size=27)
    guess = _ridge_head(a, b, y, rank=16, alpha=10.0)
    assert guess.shape == (3,)
    assert np.isfinite(guess).all()


def test_three_model_run_spec_rejects_incompatible_resume(tmp_path: Path) -> None:
    runner = _import_tool("run_checkpoint_07_three.py")
    spec = {"models": ["catboost", "local07", "prithvi"], "trials": 48}
    runner._ensure_spec(tmp_path, spec)
    runner._ensure_spec(tmp_path, spec)
    assert json.loads((tmp_path / "run_spec.json").read_text()) == spec
    with pytest.raises(ValueError, match="Prior run specification differs"):
        runner._ensure_spec(tmp_path, {**spec, "trials": 49})
    stale = tmp_path / "stale"
    (stale / "local07").mkdir(parents=True)
    (stale / "local07/target_matched_01__local07.sqlite").touch()
    with pytest.raises(RuntimeError, match="Existing studies"):
        runner._ensure_spec(stale, spec)


def test_three_model_summary_keeps_paired_baseline(tmp_path: Path) -> None:
    runner = _import_tool("run_checkpoint_07_three.py")
    for name, target in (
        ("catboost", "catboost"),
        ("local07", "local07"),
        ("prithvi/300M/heads", "prithvi"),
    ):
        folder = tmp_path / name
        folder.mkdir(parents=True)
        df = pd.DataFrame({
            "split_id": ["target_matched_01"]*4,
            "family": [target]*4,
            ID: ["AGC_001"]*4,
            "model": ["Local04D", target, "Local04D", target],
            "observed": [2.0, 2.0, 3.0, 3.0],
            "predicted": [2.1, 2.4, 3.1, 3.4],
        })
        df.loc[2:, ID] = "AGC_002"
        df.to_csv(folder / "target_matched_01__out.csv", index=False)
    summary = runner._summary(tmp_path, require_complete=False)
    assert set(summary["branch"]) == {"catboost", "local07", "prithvi"}
    assert "Local04D" in set(summary["model"])
    assert (tmp_path / "three_model_summary.csv").is_file()
