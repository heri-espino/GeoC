"""Synthetic correctness checks for the Checkpoint 10 joint PCA/k/alpha/p grid."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from geocebada.evaluation.checkpoint10_pca import (
    ALPHA_GRID,
    DIMS,
    FROZEN,
    K_GRID,
    POWER_GRID,
    PcaData,
    assemble_split,
    grid_parameters,
    inner_scores,
    summarize,
)

ID = "ID_POLIGONO"


def synthetic_data() -> PcaData:
    generator = np.random.default_rng(510)
    count = 60
    ids = [f"P{i:03d}" for i in range(count)]
    raw = generator.normal(size=(count, 160))
    # Each dimension is genuinely a distinct PCA-style prefix.
    x = {n: raw[:, :n] for n in DIMS}
    meta = np.arange(count)
    distance = np.abs(meta[:, None] - meta[None, :]).astype(float)
    outcomes = pd.Series(
        generator.normal(loc=3.0, scale=0.4, size=count),
        index=ids,
    )
    train, test = ids[:44], ids[44:]
    return PcaData(
        ids=ids, index={pid: i for i, pid in enumerate(ids)},
        y=outcomes, geographic=distance, x=x,
        variance={n: n / 160 for n in DIMS},
        folds={"confirm_01": (train, test)}, parity_max_diff=0.0,
    )


def test_exhaustive_four_dimensional_grid_and_smoke() -> None:
    grid = grid_parameters()
    assert len(grid) * len(DIMS) == 192
    assert len(set((v["k"], v["alpha"], v["power"]) for v in grid)) == len(grid)
    assert set(v["k"] for v in grid) == set(K_GRID)
    assert set(v["alpha"] for v in grid) == set(ALPHA_GRID)
    assert set(v["power"] for v in grid) == set(POWER_GRID)
    assert FROZEN in grid
    assert len(grid_parameters(smoke=True)) == 2


def test_inner_cv_is_independent_of_outer_test_y() -> None:
    data = synthetic_data()
    train, test = data.folds["confirm_01"]
    original = inner_scores(data, train, components=20, seed=4, smoke=True)
    data.y.loc[test] = 99_999
    changed = inner_scores(data, train, components=20, seed=4, smoke=True)
    pd.testing.assert_frame_equal(original, changed)
    assert set(original.columns) == {
        "components", "k", "alpha", "power", "inner_rmse",
    }


def test_nested_outer_outputs_are_paired() -> None:
    data = synthetic_data()
    train, test = data.folds["confirm_01"]
    grids = {n: inner_scores(data, train, components=n, seed=5, smoke=True) for n in DIMS}
    frame, report = assemble_split(data, "confirm_01", grids)
    assert len(frame) == 6 * len(test)
    assert set(frame["model"]) == {
        "PCA20_frozen", "PCA20_nested", "PCA40_nested",
        "PCA80_nested", "PCA160_nested", "PCA_joint_nested",
    }
    assert frame.groupby(ID)["model"].nunique().eq(6).all()
    assert report["joint_best"]["components"] in DIMS
    assert report["joint_best"]["k"] in {20, 24}
    assert report["joint_best"]["power"] == 1.0
    assert report["hidden_y_accessed"] is False


def test_summary_rejects_incomplete_banks() -> None:
    data = synthetic_data()
    train, _ = data.folds["confirm_01"]
    grids = {n: inner_scores(data, train, components=n, seed=2, smoke=True) for n in DIMS}
    frame, _ = assemble_split(data, "confirm_01", grids)
    with pytest.raises(ValueError, match="16 paired splits"):
        summarize(frame)


def test_paired_bootstrap_groups_repeated_parcels() -> None:
    rng = np.random.default_rng(72)
    rows = []
    models = [
        "PCA20_frozen", "PCA20_nested", "PCA40_nested",
        "PCA80_nested", "PCA160_nested", "PCA_joint_nested",
    ]
    for i in range(16):
        for j in range(41):
            pid = f"P{(j + i * 3) % 70:03d}"
            observed = rng.normal(3.0, .2)
            for model in models:
                rows.append({
                    "split_id": f"confirm_{i+1:02d}",
                    ID: pid, "model": model,
                    "observed": observed, "predicted": observed + .1,
                })
    scores, comparison = summarize(pd.DataFrame(rows), bootstraps=30)
    assert len(scores) == 6
    assert scores["repeated_predictions"].eq(656).all()
    assert scores["unique_parcels"].eq(70).all()
    assert len(comparison) == 5
    assert comparison["delta_mse"].abs().max() < 1e-10
    assert comparison["ci95_low"].abs().max() < 1e-10
    assert comparison["ci95_high"].abs().max() < 1e-10
