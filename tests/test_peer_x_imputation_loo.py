"""Synthetic safety tests for the 138-fold Reference26 X-only imputation comparison."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from geocebada.evaluation.peer_x_imputation import (
    BASIC,
    ID,
    PRO,
    TARGET,
    audit_imputations,
    candidate_neighbors,
    prepare,
    spatial_fill,
)


def build_inputs(tmp_path):
    ids = [f"P{i:03}" for i in range(138)]
    distance = np.abs(np.arange(138)[:, None] - np.arange(138)[None, :])
    distance = distance.astype(float) / 137.0
    data = pd.DataFrame({
        ID: ids,
        TARGET: np.linspace(2.0, 5.0, 138),
        "Estado": ["Puebla"] * 70 + ["Hidalgo"] * 68,
        "Municipio": ["A"] * 70 + ["B"] * 68,
        "climate_profile": ["dry"] * 138,
        "base_centroid_lat": np.linspace(19.0, 19.9, 138),
        "base_centroid_lon": np.linspace(-98, -97, 138),
        "compact_precip_total": np.linspace(100.0, 300.0, 138),
        "S2_ndvi_month_05": np.full(138, .4),
        "agro_pheno__y2025_planet_ndvi__jun_to_aug_delta": np.full(138, .3),
        "geology_physiography::geo250_share__Q(B)": np.full(138, .2),
    })
    data.loc[0:1, "S2_ndvi_month_05"] = np.nan
    data.loc[2, "compact_precip_total"] = np.nan
    data.loc[3, "geology_physiography::geo250_share__Q(B)"] = np.nan
    data.loc[4, "base_centroid_lat"] = np.nan
    raw_basic = pd.DataFrame({
        ID: [ids[0], ids[2], ids[0]],
        "fecha_captura": ["14/05/2025", "15/05/2025", "15/05/2023"],
        "sensor": ["Sentinel-2"] * 3,
        "porcentaje_nubosidad": [0, 5, 3],
        "ndvi_promedio": [.83, .22, .5],
        "ndwi_promedio": [.1, .3, .5],
    })
    raw_pro = pd.DataFrame({
        ID: [ids[0], ids[0]],
        "fecha_captura": ["14/06/2025", "14/08/2025"],
        "sensor": ["Planet", "Planet"],
        "porcentaje_nubosidad": [0., 0.],
        "ndvi_promedio": [.2, .6],
    })
    tab = tmp_path / "data"
    tab.mkdir()
    basic = tab / BASIC
    pro = tab / PRO
    raw_basic.to_csv(basic, index=False)
    raw_pro.to_csv(pro, index=False)
    return data, distance, basic, pro


def test_spatial_neighbors_exclude_heldout_x_donor_and_respect_region(tmp_path):
    frame, distance, _basic, _pro = build_inputs(tmp_path)
    neighbors = candidate_neighbors(0, frame, distance, k=8)
    assert len(neighbors) == 8 and 0 not in neighbors
    assert all(frame.iloc[n]["Municipio"] == "A" for n in neighbors)
    output = spatial_fill(
        frame, distance, ["compact_precip_total", "S2_ndvi_month_05"]
    )
    assert np.isfinite(output.loc[2, "compact_precip_total"])
    assert np.isfinite(output.loc[0, "S2_ndvi_month_05"])
    assert output.loc[17, "S2_ndvi_month_05"] == frame.loc[17, "S2_ndvi_month_05"]


def test_temporal_donor_and_spatiotemporal_are_distinct(tmp_path):
    original, distance, basic, pro = build_inputs(tmp_path)
    result = prepare(
        original, distance,
        [
            "compact_precip_total", "S2_ndvi_month_05",
            "geology_physiography::geo250_share__Q(B)",
            "base_centroid_lat",
            "agro_pheno__y2025_planet_ndvi__jun_to_aug_delta",
        ],
        basic, pro,
    )
    t = result.frames["temporal"]
    sp = result.frames["spatial"]
    joint = result.frames["spatiotemporal"]
    assert t.loc[0, "S2_ndvi_month_05"] == pytest.approx(.83, rel=.01)
    assert np.isnan(t.loc[1, "S2_ndvi_month_05"])
    assert np.isfinite(sp.loc[1, "S2_ndvi_month_05"])
    assert np.isfinite(joint.loc[1, "S2_ndvi_month_05"])
    assert np.isnan(t.loc[2, "compact_precip_total"])
    assert np.isfinite(joint.loc[2, "compact_precip_total"])
    assert np.isnan(joint.loc[3, "geology_physiography::geo250_share__Q(B)"])
    assert np.isnan(joint.loc[4, "base_centroid_lat"])
    for name in ("temporal", "spatial", "spatiotemporal"):
        assert result.frames[name][TARGET].equals(original[TARGET])
        assert result.frames[name].loc[15, "S2_ndvi_month_05"] == .4
    assert result.raw_observations == {"BASIC": 3, "PRO": 2}
    assert result.original_missing == 3
    summary = result.audit.groupby("method")["newly_filled"].sum()
    assert summary["spatial"] >= 3
    assert summary["temporal"] >= 1
    assert summary["spatiotemporal"] >= 3


def test_refuses_changing_any_known_x_or_y(tmp_path):
    original, distance, _basic, _pro = build_inputs(tmp_path)
    features = ["S2_ndvi_month_05", "compact_precip_total"]
    changed = original.copy()
    changed.loc[10, "S2_ndvi_month_05"] = .75
    with pytest.raises(ValueError, match="known feature"):
        audit_imputations(original, {"tampered": changed}, features)
    changed = original.copy()
    changed.loc[3, TARGET] += 1
    with pytest.raises(ValueError, match="target"):
        audit_imputations(original, {"tampered": changed}, features)


def test_temporal_requires_real_raw_dates(tmp_path):
    original, distance, basic, pro = build_inputs(tmp_path)
    with pytest.raises(FileNotFoundError, match="dated satellite"):
        prepare(
            original, distance, ["S2_ndvi_month_05"],
            tmp_path / "nonexistent.csv", pro,
        )
    basic.write_text("wrong;columns\n")
    with pytest.raises(ValueError):
        prepare(
            original, distance, ["S2_ndvi_month_05"], basic, pro
        )


def runner(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "tools"
    monkeypatch.syspath_prepend(str(scripts))
    path = scripts / "run_reference26_imputation_loo.py"
    spec = importlib.util.spec_from_file_location(
        "run_reference26_imputation_loo", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_heldout_y_masked_and_saved_per_strategy(monkeypatch, tmp_path):
    module = runner(monkeypatch)
    frame, distance, basic, pro = build_inputs(tmp_path)
    result = prepare(
        frame, distance, ["compact_precip_total", "S2_ndvi_month_05"],
        basic, pro,
    )
    calls = []

    class Native:
        def __init__(self, modified, groups, dist, geology, mode, tr, inner, cache):
            assert mode == "random" and 0 not in tr and len(tr) == 137
            assert len(inner) == 4
            assert np.isnan(modified.iloc[0][TARGET])
            assert np.isfinite(modified.iloc[tr][TARGET].to_numpy(float)).all()
            calls.append("fit")
            self.tr = tr

        def full_fit(self, tr):
            return {"train_positions": tr}

    def native_predict(pack, queries, modified, distances):
        assert queries.tolist() == [0]
        return {"Reference26": [2.9]}

    ids = frame[ID].tolist()
    monkeypatch.setattr(
        module, "_WORKER",
        (
            frame, pd.DataFrame(index=ids), distance,
            {"geology": {}}, ids, Native, native_predict,
            result.frames,
        ),
    )
    methods = ("spatial", "temporal", "spatiotemporal")
    pos, info, _elapsed = module.do_fold(
        0, methods, str(tmp_path / "outputs"),
        baseline=2.4, unchanged=("spatial",),
    )
    assert pos == 1 and info == {
        "trained": 2, "cached": 0, "unchanged": 1,
    }
    assert len(calls) == 2
    assert module.validated_row(
        tmp_path / "outputs", 0, "spatial",
        ids, frame[TARGET].to_numpy(float),
    ) == 2.4
    row = pd.DataFrame({
        ID: ids,
        "observed": frame[TARGET].to_numpy(float),
        "median_original": np.full(138, 2.4),
    })
    with pytest.raises(FileNotFoundError):
        module.scores(tmp_path / "outputs", row, methods, n=138)


def test_native_median_baseline_alignment(monkeypatch, tmp_path):
    module = runner(monkeypatch)
    ids = [f"P{i:03}" for i in range(138)]
    y = np.linspace(2, 5, 138)
    original = pd.DataFrame({
        ID: ids, "observed": y,
        "Reference26": y + .2, "Compact14": y + .3,
    })
    file = tmp_path / "existing.csv"
    original.iloc[::-1].to_csv(file, index=False)
    joined = module.baseline_predictions(file, ids, y)
    assert joined["median_original"].to_numpy() == pytest.approx(y + .2)
    original.loc[6, "observed"] += 10
    original.to_csv(file, index=False)
    with pytest.raises(ValueError, match="different observations"):
        module.baseline_predictions(file, ids, y)


def test_no_artificial_max_jobs(monkeypatch):
    module = runner(monkeypatch)
    assert module.selected_methods("temporal,spatial") == (
        "spatial", "temporal",
    )
    with pytest.raises(ValueError):
        module.selected_methods("spatial,spatial")
    monkeypatch.setattr(sys, "argv", [
        "run_reference26_imputation_loo.py", "--check", "--jobs", "1000",
    ])
    # Option validation permits jobs=1000. Missing private inputs are
    # the expected next step on CI; do not fabricate user's files.
    with pytest.raises((FileNotFoundError, ValueError, ImportError)):
        module.main()
