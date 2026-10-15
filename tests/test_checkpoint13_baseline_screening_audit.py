"""Checkpoint13 incumbent 12-parcel screening audit tests."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from geocebada.evaluation.checkpoint10_pca import PcaData
from geocebada.evaluation.checkpoint13_randomized import FAMILIES, SEED

# The tools directory consists of runnable scripts; it is not installed as a package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
_audit = importlib.import_module("audit_checkpoint_13_baseline")
baseline_inner_scores = _audit.baseline_inner_scores
screening_folds = _audit.screening_folds
virtual_ranks = _audit.virtual_ranks

ID = "ID_POLIGONO"


def sample() -> PcaData:
    rng = np.random.default_rng(711)
    ids = [f"X{n:03d}" for n in range(138)]
    x = rng.normal(size=(138, 20))
    geo = np.abs(np.arange(138)[:, None] - np.arange(138)[None, :]).astype(float)
    y = pd.Series(rng.normal(3, .5, 138), index=ids)
    return PcaData(
        ids=ids, index={pid: i for i, pid in enumerate(ids)},
        y=y, geographic=geo, x={20: x}, variance={20: 0.8},
        folds={"confirm_01": (ids[:97], ids[97:])}, parity_max_diff=0,
    )


def test_screen_split_exactly_matches_target_first_permutation() -> None:
    train = sample().folds["confirm_01"][0]
    fit, held = screening_folds(train, seed=SEED + 1)
    permutation = np.random.default_rng(SEED + 1).permutation(97)
    assert held == [train[i] for i in permutation[:12]]
    assert fit == [train[i] for i in permutation[12:]]
    assert len(held) == 12 and len(fit) == 85
    assert not set(fit).intersection(held)


def test_baseline_screen_and_repeated_cv_never_see_outer_heldout_y() -> None:
    data = sample()
    screen, inner, count = baseline_inner_scores(data, "confirm_01")
    assert count == 41
    assert screen > 0 and inner > 0
    data.y.loc[data.folds["confirm_01"][1]] = 100_000
    screen2, inner2, count2 = baseline_inner_scores(data, "confirm_01")
    assert (screen, inner, count) == (screen2, inner2, count2)


def test_virtual_screen_ranks_and_pca_top_cutoff() -> None:
    screened = pd.DataFrame([
        {
            "candidate_id": f"{family}:{j}", "family": family,
            "screen_rmse": float(j) / 10,
        }
        for family in FAMILIES
        for j in (1, 2, 3)
    ])
    reranked = pd.DataFrame([
        {"family": family, "inner_rmse": .15}
        for family in FAMILIES
    ])
    result = virtual_ranks(
        screened, reranked, baseline_screen=.25, baseline_cv=.14,
        finalists_per_family=2,
    )
    assert result["incumbent_virtual_screen_rank_all"] == 11
    assert result["incumbent_virtual_screen_rank_pca_ridge"] == 3
    assert result["would_pass_pca_ridge_top12_screen"] is False
    assert result["incumbent_virtual_inner_cv_rank_among_finalists"] == 1
    assert result["pca_ridge_12th_screen_rmse"] == .2


def test_duplicate_candidate_rejected() -> None:
    screened = pd.DataFrame({
        "candidate_id": ["dup", "dup"], "family": ["PCA_Ridge"] * 2,
        "screen_rmse": [.1, .2],
    })
    reranked = pd.DataFrame({"family": list(FAMILIES), "inner_rmse": [.2] * 5})
    with pytest.raises(ValueError, match="Duplicated"):
        virtual_ranks(
            screened, reranked, baseline_screen=.1,
            baseline_cv=.1, finalists_per_family=1,
        )
