"""Synthetic QA of final analog statistical study. No private labels or IDs."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


def reporter(monkeypatch):
    tools = Path(__file__).resolve().parents[1] / "tools"
    monkeypatch.syspath_prepend(str(tools))
    spec = importlib.util.spec_from_file_location(
        "report_reference26_top5_statistics",
        tools/"report_reference26_top5_statistics.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def artificial(m):
    rng = np.random.default_rng(1919)
    z = rng.normal(size=(197, 26))
    y = rng.normal(4., .6, 138)
    original = y + rng.normal(0., .4, 138)
    return z, y, original


def test_selected_rule_matches_native_loo_correction(monkeypatch):
    m = reporter(monkeypatch)
    z, y, base = artificial(m)
    cfg = m.nested.TOP5[0]
    selected = np.empty(138)
    for i in range(138):
        train = np.delete(np.arange(138), i)
        selected[i] = m.nested.apply_candidate(
            cfg, z, y, train, np.array([i]),
            base[[i]], .5
        )[0]
    details = m.correction_details(
        y, base, selected, m.analog.signature(cfg), {"robust26": z}, .5
    )
    assert len(details) == 138
    assert details["support_gate"].between(0, 1).all()
    assert np.max(abs(details["correction_t_ha"])) <= .5 + 1e-12
    with pytest.raises(ValueError, match="parity"):
        m.correction_details(
            y, base, selected + .03, m.analog.signature(cfg),
            {"robust26": z}, .5
        )


def test_reference_fallback_has_zero_correction(monkeypatch):
    m=reporter(monkeypatch)
    z,y,p=artificial(m)
    report=m.correction_details(y,p,p.copy(),m.nested.BASE,{},.5)
    assert np.all(report["correction_t_ha"] == 0)
    with pytest.raises(ValueError, match="mismatches"):
        m.correction_details(y,p,p+.1,m.nested.BASE,{},.5)


def test_calibration_is_not_algorithmic_coefficient_inference(monkeypatch):
    m=reporter(monkeypatch)
    y=np.linspace(1.,6.,138)
    estimate=y+.1
    result=m.calibration_ols_hc3(y,{"chosen":estimate})
    assert result["diagnostic_parameter"].tolist()==[
        "intercept_t_ha","slope"
    ]
    assert result.iloc[1]["estimate"] == pytest.approx(1.)
    assert result.iloc[0]["estimate"] == pytest.approx(-.1)
    assert np.isfinite(result["conditional_HC3_se"]).all()


def test_tail_grouping_is_disjoint_and_conditional(monkeypatch):
    m=reporter(monkeypatch)
    z,y,p=artificial(m)
    rows=m.tails(y,{m.BASE:p,m.FIXED:p+.01,m.POLICY:p-.01})
    assert len(rows)==18
    for (_,model),frame in rows.groupby(["definition","model"]):
        assert frame["n"].sum()==138
    assert rows["warning"].str.contains("Post-hoc").all()


def test_spatial_moran_includes_observed_signal(monkeypatch):
    m=reporter(monkeypatch)
    rng=np.random.default_rng(31)
    y=rng.normal(4,1,138)
    p=y+rng.normal(0,.3,138)
    d=np.abs(np.arange(138)[:,None]-np.arange(138)[None,:]).astype(float)
    table=m.supplemental_spatial(
        y,{m.BASE:p,m.FIXED:p+.01,m.POLICY:p-.01},d,99,11
    )
    assert set(table.statistic)=={"Moran_I","Geary_C"}
    assert "observed_yield" in set(table.spatial_variable)
    assert "correction_delta" in set(table.spatial_variable)
    assert len(table)==4*5*2
