"""Checkpoints14 focused search: 1,000 seeds, exact Local04D and fold isolation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from geocebada.evaluation.checkpoint10_pca import FROZEN, PcaData, predict
from geocebada.evaluation.checkpoint12_representations import RepresentationData
from geocebada.evaluation.checkpoint14_focused_cv import (
    ANCHOR_NAME,
    BASELINE,
    COUNT,
    FAMILIES,
    FAMILY_QUOTAS,
    JOINT,
    evaluate_outer,
    focused_candidates,
    model_names,
    score_candidates,
    shortlist_candidates,
    summarize,
)

ID = "ID_POLIGONO"


def fixture_data() -> RepresentationData:
    rng = np.random.default_rng(14)
    ids = [f"F{i:03d}" for i in range(138)]
    x = rng.normal(size=(138, 200))
    z = rng.normal(size=(138, 65))
    y = pd.Series(3.0 + rng.normal(size=138) * .45, index=ids)
    geo = np.abs(np.arange(138)[:, None] - np.arange(138)[None, :]).astype(float)
    base = PcaData(
        ids=ids, index={pid: i for i, pid in enumerate(ids)},
        y=y, geographic=geo, x={20: x[:, :20]},
        variance={20: .8},
        folds={"confirm_01": (ids[:97], ids[97:])},
        parity_max_diff=0.,
    )
    representations = {
        ("PCA", n): x[:, :n] for n in
        (5, 8, 12, 16, 20, 24, 32, 40, 60, 80, 120, 160, 192)
    }
    return RepresentationData(base, z, representations, [])


def tiny_grid() -> pd.DataFrame:
    grid = focused_candidates()
    anchor = grid.loc[grid["candidate_id"].eq(ANCHOR_NAME)]
    variants = []
    for family in FAMILIES:
        pool = grid.loc[grid["family"].eq(family)]
        if family == "PCA_Ridge":
            pool = pool.loc[~pool["candidate_id"].eq(ANCHOR_NAME)]
        row = pool.iloc[0].copy()
        # Synthetic: keep regression components small for speed.
        if row["representation"] == "PLS":
            row["components"] = 4
        elif row["representation"] == "PCA":
            row["components"] = 20
        variants.append(row)
    return pd.concat([anchor, pd.DataFrame(variants)], ignore_index=True)


def test_exactly_1000_focused_models_and_permanent_local04d() -> None:
    candidates = focused_candidates()
    assert len(candidates) == COUNT == 1000
    assert candidates["candidate_id"].is_unique
    assert set(candidates["family"]) == set(FAMILIES)
    assert candidates.groupby("family").size().to_dict() == FAMILY_QUOTAS
    old = candidates.loc[candidates["candidate_id"].eq(ANCHOR_NAME)].iloc[0]
    assert old["representation"] == "PCA"
    assert old["regressor"] == "ridge"
    assert (old["components"], old["k"], old["alpha"], old["power"]) == (
        20, FROZEN["k"], FROZEN["alpha"], FROZEN["power"],
    )
    pd.testing.assert_frame_equal(candidates, focused_candidates())
    assert not candidates["family"].str.contains("RP|ICA").any()
    assert not candidates["candidate_id"].eq(ANCHOR_NAME).sum() != 1


def test_full_inner_cv_ignores_41_external_labels() -> None:
    data = fixture_data()
    tr, outer = data.baseline.folds["confirm_01"]
    candidates = tiny_grid()
    first = score_candidates(data, tr, candidates, seed=123, folds=3)
    assert len(first) == len(candidates)
    assert first["inner_predictions"].eq(97).all()
    data.baseline.y.loc[outer] = 9999
    second = score_candidates(data, tr, candidates, seed=123, folds=3)
    pd.testing.assert_frame_equal(first, second)


def test_shortlist_retains_original_even_when_worst_rank() -> None:
    candidates = focused_candidates()
    scores = pd.DataFrame({
        "candidate_id": candidates["candidate_id"],
        "family": candidates["family"],
        "inner_rmse": np.arange(COUNT, dtype=float),
    })
    scores.loc[scores["candidate_id"].eq(ANCHOR_NAME), "inner_rmse"] = 100000.
    selected = shortlist_candidates(candidates, scores)
    assert selected["candidate_id"].eq(ANCHOR_NAME).sum() == 1
    assert selected["family"].nunique() == len(FAMILIES)


def test_selected_outer_predictions_include_paired_control() -> None:
    data = fixture_data()
    tr, held = data.baseline.folds["confirm_01"]
    candidates = tiny_grid()
    rankings = score_candidates(data, tr, candidates, seed=25, folds=2)
    predictions, report = evaluate_outer(
        data, "confirm_01", candidates, rankings
    )
    assert len(predictions) == len(model_names()) * 41
    assert set(predictions["model"]) == model_names()
    assert predictions.groupby(ID)["model"].nunique().eq(len(model_names())).all()
    expected = predict(data.baseline, tr, held, components=20, **FROZEN)
    control = predictions.loc[predictions["model"].eq(BASELINE), "predicted"]
    np.testing.assert_allclose(control, expected, rtol=0, atol=1e-10)
    assert report["incumbent_repeated_cv_rank_all"] >= 1
    assert report["hidden_y_accessed"] is False
    assert report["chosen"][JOINT]["family"] in FAMILIES


def test_summary_requires_all_paired_splits() -> None:
    data = fixture_data()
    tr, _ = data.baseline.folds["confirm_01"]
    candidates = tiny_grid()
    ranks = score_candidates(data, tr, candidates, seed=23, folds=2)
    outer, _ = evaluate_outer(data, "confirm_01", candidates, ranks)
    with pytest.raises(ValueError, match="Incomplete paired"):
        summarize(outer)


def test_bootstrap_clustered_on_repeated_parcels() -> None:
    rng = np.random.default_rng(2)
    records = []
    for n in range(16):
        for j in range(41):
            pid = f"S{(n * 3 + j) % 73:03d}"
            target = float(rng.normal(3, .35))
            for model in model_names():
                records.append({
                    "split_id": f"confirm_{n + 1:02d}", ID: pid,
                    "model": model, "observed": target,
                    "predicted": target + .12,
                })
    scores, intervals = summarize(pd.DataFrame(records), bootstraps=35)
    assert len(scores) == len(model_names())
    assert scores["repeated_predictions"].eq(656).all()
    assert len(intervals) == len(model_names()) - 1
    assert intervals["delta_mse"].abs().max() < 1e-10
