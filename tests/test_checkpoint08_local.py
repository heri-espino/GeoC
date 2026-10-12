"""Synthetic leakage, alignment, and resume tests for Checkpoint 08."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from geocebada.evaluation.checkpoint08_local import (
    FAMILIES,
    ID,
    Inputs,
    label_blind_memberships,
    summarize,
    tune_and_predict,
    validate_memberships,
)


def toy_metadata() -> tuple[pd.DataFrame, pd.DataFrame]:
    ids = [f"AGC_{i:03d}" for i in range(1, 198)]
    r = pd.DataFrame({
        ID: ids,
        "CONJUNTO": ["ENTRENAMIENTO"] * 138 + ["PREDICCION"] * 59,
        "meta_estado": ["Puebla"] * 120 + ["Hidalgo"] * 77,
        "meta_municipio": ["A"] * 120 + ["B"] * 77,
    })
    historical = []
    for j in range(16):
        random = np.random.default_rng(j).choice(ids[:138], 41, replace=False)
        chosen = set(random)
        for pid in ids[:138]:
            historical.append({
                "split_id": f"target_matched_{j+1:02d}",
                ID: pid,
                "role": "pseudo_target" if pid in chosen else "pseudo_train",
            })
    return r, pd.DataFrame(historical)


def test_confirmation_membership_is_label_blind_and_reproducible() -> None:
    meta, historical = toy_metadata()
    one = label_blind_memberships(meta, historical, seed=55)
    two = label_blind_memberships(meta.copy(), historical.copy(), seed=55)
    pd.testing.assert_frame_equal(one, two)
    assert len(one) == 16 * 138
    assert validate_memberships(meta, one).keys() == validate_memberships(meta, two).keys()
    assert one["max_jaccard"].max() < 1
    assert not (one.columns == "RENDIMIENTO_T_HA").any()
    for a, b in validate_memberships(meta, one).values():
        assert len(a) == 97 and len(b) == 41
        assert not set(a) & set(b)


def test_membership_rejects_any_hidden_target_leakage() -> None:
    meta, historical = toy_metadata()
    member = label_blind_memberships(meta, historical, seed=17)
    member.loc[0, ID] = meta.loc[196, ID]  # Actual unknown target
    with pytest.raises(ValueError, match="Invalid parcel coverage"):
        validate_memberships(meta, member)


def test_summary_bootstraps_unique_parcels_and_paired_losses() -> None:
    rng = np.random.default_rng(81)
    rows = []
    for fold in range(16):
        for pid in [f"AGC_{i:03d}" for i in range(1, 42)]:
            y = float(rng.normal(3, 0.5))
            for name, bias in [
                ("Local04D_frozen", 0.3),
                ("Local04D_nested", 0.1),
                ("Local07_nested", 0.5),
            ]:
                rows.append({
                    "split_id": f"confirm_{fold+1:02d}",
                    ID: pid, "model": name, "observed": y,
                    "predicted": y + bias,
                })
    summary, diffs = summarize(pd.DataFrame(rows), bootstraps=20, seed=14)
    assert set(summary["model"]) == {"Local04D_frozen", *FAMILIES}
    assert summary.iloc[0]["model"] == "Local04D_nested"
    assert len(diffs) == 2
    delta = dict(zip(diffs["model"], diffs["delta_mse"], strict=True))
    assert np.isclose(delta["Local04D_nested"], 0.01-0.09)
    assert np.isclose(delta["Local07_nested"], 0.25-0.09)
    assert diffs["bootstrap_unit"].eq("unique_parcel").all()


def test_run_spec_blocks_changes_that_would_corrupt_resume(tmp_path: Path) -> None:
    target = Path(__file__).resolve().parents[1] / "tools/run_checkpoint_08_local.py"
    spec = importlib.util.spec_from_file_location("run_checkpoint08", target)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ensure_spec(tmp_path, {"trials": 96})
    module.ensure_spec(tmp_path, {"trials": 96})
    with pytest.raises(ValueError, match="fingerprint differs"):
        module.ensure_spec(tmp_path, {"trials": 64})


def test_source_code_never_auto_promotes_predictions() -> None:
    source = (Path(__file__).resolve().parents[1] /
              "tools/run_checkpoint_08_local.py").read_text(encoding="utf-8")
    assert "reports/checkpoint_05/final_predictions.csv" not in source
    assert "git push" not in source


def test_both_model_families_nested_train_and_resume(tmp_path: Path) -> None:
    pytest.importorskip("optuna")
    rng = np.random.default_rng(42)
    ids = [f"AGC_{i:03d}" for i in range(1, 198)]
    meta = pd.DataFrame({
        ID: ids,
        "CONJUNTO": ["ENTRENAMIENTO"] * 138 + ["PREDICCION"] * 59,
        "meta_estado": ["Puebla"] * 197,
        "meta_municipio": ["A"] * 197,
        "base_centroid_lat": np.linspace(19.0, 20.0, 197),
        "base_centroid_lon": np.linspace(-99.0, -98.0, 197),
    })
    x = pd.DataFrame(
        rng.normal(size=(197, 65)),
        index=ids,
        columns=[f"hls__f{i}" for i in range(65)],
    ).astype(np.float32)
    geo = rng.uniform(0.1, 5.0, size=(197, 197))
    geo = (geo + geo.T) * 0.5
    np.fill_diagonal(geo, 0)
    data = Inputs(
        x07=x, old=rng.normal(size=(197, 20)), geographic=geo,
        index={pid: i for i, pid in enumerate(ids)},
        y=pd.Series(rng.normal(3.5, 0.5, 138), index=ids[:138]),
        meta=meta,
        folds={"confirm_01": (ids[:97], ids[97:138])},
    )
    for family in FAMILIES:
        first = tune_and_predict(
            data, "confirm_01", family, trials=2, seed=42,
            n_folds=2, folder=tmp_path,
        )
        second = tune_and_predict(
            data, "confirm_01", family, trials=2, seed=42,
            n_folds=2, folder=tmp_path,
        )
        assert len(first) == len(second) == 41
        pd.testing.assert_frame_equal(first, second)
        assert first["predicted"].notna().all()
