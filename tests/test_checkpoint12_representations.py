"""Leakage, completeness and candidate tests for Checkpoint12."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from geocebada.evaluation.checkpoint10_pca import PcaData
from geocebada.evaluation.checkpoint12_representations import (
    BASELINE,
    FAMILIES,
    RepresentationData,
    configurations,
    evaluate,
    inner_grid,
    paired_split,
    summarize,
)

ID = "ID_POLIGONO"


def _data() -> RepresentationData:
    rng = np.random.default_rng(202612)
    ids = [f"P{i:03d}" for i in range(138)]
    x = rng.normal(size=(138, 90))
    raw = StandardScaler().fit_transform(x)
    y = pd.Series(3 + rng.normal(size=138) * .4, index=ids)
    distance = np.abs(np.arange(138)[:, None] - np.arange(138)[None, :]).astype(float)
    baseline = PcaData(
        ids=ids,
        index={pid: j for j, pid in enumerate(ids)},
        y=y,
        geographic=distance,
        x={20: raw[:, :20], 40: raw[:, :40]},
        variance={20: .5, 40: .8},
        folds={"confirm_01": (ids[:97], ids[97:])},
        parity_max_diff=0.0,
    )
    rep = {}
    for name, dims in (
        ("PCA", (20, 40)), ("ICA", (8, 16, 32)),
        ("RP", (16, 32, 64)),
    ):
        for n in dims:
            rep[(name, n)] = raw[:, :n]
    return RepresentationData(baseline, raw, rep, [])


def test_candidate_grids_nontrivial_and_lightweight() -> None:
    assert len(FAMILIES) == 5
    for family in FAMILIES:
        candidates = configurations(family)
        assert len(candidates) > 1
        assert len(candidates) <= 12
        assert len(configurations(family, smoke=True)) == 1
    assert all(
        c["representation"] == "PLS"
        for c in configurations("PLS_ElasticNet")
    )


def test_pls_supervised_projection_does_not_use_outer_heldout_y() -> None:
    data = _data()
    train, test = data.baseline.folds["confirm_01"]
    config = configurations("PLS_Ridge", smoke=True)[0]
    control = evaluate(data, train, test, config, cache={})
    data.baseline.y.loc[test] = 100000
    repeated = evaluate(data, train, test, config, cache={})
    np.testing.assert_array_equal(control, repeated)


def test_all_inner_grids_independent_of_outer_targets() -> None:
    data = _data()
    train, held = data.baseline.folds["confirm_01"]
    original = {
        f: inner_grid(data, train, f, seed=43, smoke=True)
        for f in FAMILIES
    }
    data.baseline.y.loc[held] = 99999
    changed = {
        f: inner_grid(data, train, f, seed=43, smoke=True)
        for f in FAMILIES
    }
    for family in FAMILIES:
        pd.testing.assert_frame_equal(original[family], changed[family])


def test_nested_paired_predictions_six_models() -> None:
    data = _data()
    train, held = data.baseline.folds["confirm_01"]
    grids = {
        f: inner_grid(data, train, f, seed=45, smoke=True)
        for f in FAMILIES
    }
    output, selection = paired_split(data, "confirm_01", grids)
    assert len(output) == 6 * len(held)
    assert set(output["model"]) == {BASELINE, *FAMILIES}
    assert output.groupby(ID)["model"].nunique().eq(6).all()
    assert selection["hidden_y_accessed"] is False


def test_summary_rejects_single_split() -> None:
    data = _data()
    train, _ = data.baseline.folds["confirm_01"]
    grids = {
        f: inner_grid(data, train, f, seed=45, smoke=True)
        for f in FAMILIES
    }
    paired, _ = paired_split(data, "confirm_01", grids)
    with pytest.raises(ValueError, match="Incomplete paired"):
        summarize(paired)


def test_bootstrap_groups_parcels_on_complete_synthetic_bank() -> None:
    rng = np.random.default_rng(8)
    records = []
    for i in range(16):
        for j in range(41):
            pid = f"Q{(j + 3 * i) % 72:03d}"
            y = float(rng.normal(3, .3))
            for model in {BASELINE, *FAMILIES}:
                records.append({
                    "split_id": f"confirm_{i + 1:02d}",
                    ID: pid, "model": model,
                    "observed": y, "predicted": y + 0.15,
                })
    scores, bootstrap = summarize(pd.DataFrame(records), bootstraps=30)
    assert len(scores) == 6
    assert scores["repeated_predictions"].eq(656).all()
    assert len(bootstrap) == 5
    assert bootstrap["delta_mse"].abs().max() < 1e-10
    assert bootstrap["ci95_high"].abs().max() < 1e-10
