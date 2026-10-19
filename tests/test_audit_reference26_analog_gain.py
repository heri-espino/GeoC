"""Synthetic-only QA for paired Reference26 analog improvement stability."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def loaded(monkeypatch):
    folder=Path(__file__).resolve().parents[1]/"tools"
    monkeypatch.syspath_prepend(str(folder))
    spec=importlib.util.spec_from_file_location(
        "audit_reference26_analog_gain",
        folder/"audit_reference26_analog_gain.py"
    )
    m=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def artificial(m):
    n=138
    y=np.linspace(2.,5.,n)
    # Baseline RMSE must equal the authentic check value.
    base=y+0.4681381271
    candidate=y+0.45
    frame=pd.DataFrame({
        m.analog.ID:[f"TEST_{i:03d}" for i in range(n)],
        "observed":y,
        "Reference26_mean":base,
        m.BASELINE:base,
        m.WINNER:candidate,
    })
    for i in range(29):
        frame[f"candidate_{i:02d}"]=y+(0.49+i*.001)
    return frame,y,base,candidate


def test_exact_baseline_schema_and_sample_size(monkeypatch):
    m=loaded(monkeypatch)
    frame,y,base,pred=artificial(m)
    target,baseline,cols=m.validate(frame)
    assert len(cols)==30
    np.testing.assert_array_equal(target,y)
    np.testing.assert_array_equal(baseline,base)
    assert m.WINNER in cols
    broken=frame.drop(columns=[m.WINNER])
    with pytest.raises(ValueError):
        m.validate(broken)
    changed=frame.copy()
    changed.loc[0,"Reference26_mean"]=0
    with pytest.raises(ValueError, match="differ"):
        m.validate(changed)


def test_known_paired_effect_and_bootstrap(monkeypatch):
    m=loaded(monkeypatch)
    frame,y,base,pred=artificial(m)
    score=m.statistics(y,base,pred)
    assert score["rmse_improvement_t_ha"]==pytest.approx(.0181381271)
    assert score["n_improved_squared_error"]==138
    b=m.iid_bootstrap(
        y,base,pred,draws=200,rng=np.random.default_rng(1)
    )
    np.testing.assert_allclose(
        b["iid_bootstrap_rmse_gain_ci95"],[.0181381271]*2
    )
    assert b["iid_bootstrap_fraction_positive"]==1


def test_delete_one_and_selection_aware_maxT(monkeypatch):
    m=loaded(monkeypatch)
    frame,y,base,pred=artificial(m)
    v=m.leave_one_out_stability(y,base,pred)
    assert v["n_of_138_deletions_still_positive"]==138
    family=frame[[m.WINNER]+[
        f"candidate_{i:02d}" for i in range(29)
    ]].to_numpy(float)
    tests=m.familywise_signflip(
        y,base,family,draws=200,rng=np.random.default_rng(123)
    )
    assert tests["family_size"]==30
    assert 0<tests["maxT_signflip_approx_p"]<=1


def test_geographic_cluster_bootstrap_uses_all_original_138(monkeypatch):
    m=loaded(monkeypatch)
    frame,y,base,pred=artificial(m)
    x=np.arange(len(y))
    matrix=np.abs(x[:,None]-x[None,:]).astype(float)
    report=m.block_bootstrap(
        y,base,pred,matrix,groups=10,draws=200,
        rng=np.random.default_rng(7)
    )
    assert report["geo_blocks"]==10
    assert report["geo_block_size_min"]>=1
    np.testing.assert_allclose(
        report["geo_cluster_bootstrap_rmse_gain_ci95"],
        [.0181381271]*2
    )
    with pytest.raises(ValueError,match="Invalid"):
        m.block_bootstrap(
            y,base,pred,matrix[:50,:50],groups=10,
            draws=100,rng=np.random.default_rng(1)
        )
