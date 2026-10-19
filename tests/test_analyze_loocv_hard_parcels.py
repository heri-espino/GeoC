"""Synthetic tests for read-only LOOCV hard-parcel diagnostics."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def load(monkeypatch):
    folder = Path(__file__).resolve().parents[1] / "tools"
    monkeypatch.syspath_prepend(str(folder))
    spec = importlib.util.spec_from_file_location(
        "analyze_loocv_hard_parcels", folder / "analyze_loocv_hard_parcels.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic(mod, *, common=True):
    n = 138
    y = np.full(n, 3.0)
    table = pd.DataFrame({
        mod.ID: [f"P{i:03d}" for i in range(n)],
        mod.TARGET: y,
    })
    for name in ("Reference26_mean", "Reference26_original",
                 "Compact14", "Local04D", "Local04D_temporal_climate"):
        prediction = y.copy()
        high = (11, 97) if common or name.startswith("Reference") else (24, 75)
        prediction[list(high)] += [2.2, -1.8]
        prediction[5] += 0.3
        table[name] = prediction
    return table


def test_shared_top_two_and_cumulative_rmse(monkeypatch):
    mod = load(monkeypatch)
    models, curves, parcels, top10, manifest = mod.analyze(synthetic(mod))
    assert set(manifest["intersection_top2_across_families_private"]) == {
        "P011", "P097"
    }
    assert set(manifest["two_highest_consensus_private"]) == {
        "P011", "P097"
    }
    assert set(manifest["representatives_by_family"]) == {
        "Reference26", "Compact14", "GeoPCA20"
    }
    assert len(models) == 5
    assert len(curves) == 5 * 138
    for row in models.itertuples():
        final = curves.loc[
            (curves["model"] == row.model)
            & (curves["n_easiest_parcels_included"] == 138),
            "cumulative_rmse_t_ha"
        ].iloc[0]
        assert final == pytest.approx(row.rmse_full_t_ha)
        assert row.sse_worst2_pct > 95
    assert set(top10["rank_worst_first"]) == set(range(1, 11))
    assert parcels["family_top2_votes"].max() == 3


def test_no_false_cross_family_intersection(monkeypatch):
    mod = load(monkeypatch)
    _, _, _, _, manifest = mod.analyze(synthetic(mod, common=False))
    assert manifest["intersection_top2_across_families_private"] == []


def test_invalid_geometry_fails_closed(monkeypatch):
    mod = load(monkeypatch)
    table = synthetic(mod)
    with pytest.raises(ValueError, match="138 distinct"):
        mod.analyze(table.iloc[:-1])
    with pytest.raises(ValueError, match="138 distinct"):
        mod.analyze(pd.concat([table.iloc[:-1], table.iloc[[0]]]))
    table.loc[0, "Compact14"] = np.nan
    with pytest.raises(ValueError, match="Nonfinite"):
        mod.analyze(table)


def test_related_methods_are_not_independent_families(monkeypatch):
    mod = load(monkeypatch)
    assert mod.family("Reference26_mean_fresh") == "Reference26"
    assert mod.family("Reference26_DBSCAN") == "Reference26"
    assert mod.family("Compact14_fresh") == "Compact14"
    assert mod.family("Local04D_temporal_climate") == "GeoPCA20"
    assert mod.family("Ensemble_fixed_weight") == "Ensemble"
