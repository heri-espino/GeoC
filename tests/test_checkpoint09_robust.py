"""Synthetic, label-safe regression tests for Checkpoint 09 robust Local04D."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from geocebada.evaluation.checkpoint08_local import Inputs, predict_old, FROZEN
from geocebada.evaluation.checkpoint09_robust import (
    METHODS,
    build_review_queue,
    leave_one_split_out_selection,
    loo_reliability,
    paired_cluster_interval,
    predict_robust_local,
    summarize_bank,
    training_loo_residuals,
)


def _toy(*, with_corruption: bool = True) -> tuple[Inputs, list[str], list[str]]:
    size, n_observed = 46, 40
    ids = [f"AGC_{i:03d}" for i in range(size)]
    rng = np.random.default_rng(71)
    x = rng.normal(0, 0.05, (size, 4))
    y = 3.5 + 0.03 * np.arange(n_observed, dtype=float)
    if with_corruption:
        y[-1] = 35.0
    meta = pd.DataFrame({
        "ID_POLIGONO": ids,
        "CONJUNTO": ["ENTRENAMIENTO"] * n_observed + ["PREDICCION"] * 6,
        "meta_estado": ["Puebla"] * size,
        "meta_municipio": ["A"] * size,
    })
    distance = np.abs(np.arange(size)[:, None] - np.arange(size)).astype(float) + 1
    data = Inputs(
        x07=pd.DataFrame(),
        old=x,
        geographic=distance,
        index={pid: i for i, pid in enumerate(ids)},
        y=pd.Series(y, index=ids[:n_observed]),
        meta=meta,
        folds={},
    )
    return data, ids[:n_observed], ids[n_observed:]


def test_robust_local_does_not_use_unknown_query_yield() -> None:
    data, known, unknown = _toy()
    baseline = predict_old(data, known, unknown[:2], FROZEN)
    huber = predict_robust_local(
        data, known, unknown[:2], method="Huber_c1p5",
    )
    assert baseline.shape == huber.shape == (2,)
    assert np.isfinite(huber).all()
    # The contaminated training parcel has a large residual; reducing its
    # leverage moves the affected predictions toward the uncorrupted scale.
    assert abs(huber[0] - 4.5) < abs(baseline[0] - 4.5)
    assert set(unknown).isdisjoint(data.y.index)


def test_training_only_loo_weights_and_quality_model() -> None:
    data, known, unknown = _toy()
    scores = training_loo_residuals(data, known)
    assert scores.shape == (40,)
    assert abs(scores[-1]) > 15
    w = loo_reliability(scores, multiplier=2.0)
    assert w.shape == (40,)
    assert w.min() >= 0.35
    assert w.max() <= 1.0
    assert w[-1] < np.median(w)
    weighted = predict_robust_local(
        data, known, unknown, method="LOOReliability_c2p0",
        reliability=w,
    )
    assert weighted.shape == (6,)
    assert np.isfinite(weighted).all()


def test_methods_reject_leakage_and_missing_weights() -> None:
    data, known, unknown = _toy()
    with pytest.raises(ValueError, match="Query labels"):
        predict_robust_local(data, known, [known[0]], method="Huber_c2p5")
    with pytest.raises(ValueError, match="TRAINING-ONLY"):
        predict_robust_local(
            data, known, unknown, method="LOOReliability_c2p0"
        )
    with pytest.raises(ValueError):
        loo_reliability(np.ones(4), multiplier=0)


def test_audit_queue_never_declares_error_or_includes_missing_targets() -> None:
    data, known, unknown = _toy()
    queue = build_review_queue(data)
    assert len(queue) == len(known) == 40
    assert set(queue["ID_POLIGONO"]) == set(known)
    assert not (set(queue["ID_POLIGONO"]) & set(unknown))
    assert queue["review_status"].eq("unreviewed").all()
    assert not queue["independent_source_verified"].any()
    assert queue.loc[0, "ID_POLIGONO"] == known[-1]


def _paired_predictions() -> pd.DataFrame:
    rows = []
    biases = {
        "Local04D_frozen": 0.5,
        "Huber_c1p5": 0.2,
        "Huber_c2p5": 0.45,
        "LOOReliability_c2p0": 0.40,
        "LOOReliability_c3p0": 0.43,
    }
    for j in range(16):
        for i in range(41):
            # Repeated IDs mimic dependent target-matched masks.
            pid = f"AGC_{i + ((j % 3) * 10):03d}"
            y = 1.0 if i < 8 else (6.0 if i > 33 else 3.5)
            tail = "low" if i < 8 else ("high" if i > 33 else "center")
            for method, b in biases.items():
                rows.append({
                    "split_id": f"target_matched_{j + 1:02d}",
                    "ID_POLIGONO": pid,
                    "model": method,
                    "observed": y,
                    "predicted": y + b,
                    "regime": tail,
                })
    return pd.DataFrame(rows)


def test_loso_evaluates_complete_composite_rule() -> None:
    predictions = _paired_predictions()
    assert set(predictions["model"]) == set(METHODS)
    selection, chosen, result = leave_one_split_out_selection(predictions)
    assert len(selection) == 16
    assert len(chosen) == 16 * 41
    assert set(selection["selected_on_other_15"]) == {"Huber_c1p5"}
    assert result["LOSO_STABILITY_GATE"] == "PASS"
    assert abs(result["loso_rmse_pooled"] - 0.2) < 1e-10
    assert abs(result["baseline_rmse_pooled"] - 0.5) < 1e-10
    summary = summarize_bank(predictions)
    assert summary.iloc[0]["model"] == "Huber_c1p5"


def test_paired_bootstrap_groups_repeated_parcels() -> None:
    frame = _paired_predictions()
    paired = frame.loc[frame["model"].isin(["Local04D_frozen", "Huber_c1p5"])]
    interval = paired_cluster_interval(paired, "Huber_c1p5", bootstraps=50)
    assert interval["delta_mse"] < 0
    assert interval["ci95_high"] < 0
    assert interval["bootstrap_unit"] == "unique_parcel"
