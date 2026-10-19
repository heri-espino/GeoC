"""Synthetic QA: analog predictions only use allowed real training labels."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def load(monkeypatch):
    folder=Path(__file__).resolve().parents[1]/"tools"
    monkeypatch.syspath_prepend(str(folder))
    spec=importlib.util.spec_from_file_location(
        "run_reference26_analog_tail_screen",
        folder/"run_reference26_analog_tail_screen.py"
    )
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def example():
    rng=np.random.default_rng(42)
    x=rng.normal(size=(197, 26))
    y=rng.normal(loc=4,scale=0.7,size=138)
    x[0]=np.array([6.]*26)
    x[1]=np.array([5.9]*26)
    return x,y


def test_real_neighbor_y_is_train_only(monkeypatch):
    m=load(monkeypatch)
    z,y=example()
    train=np.arange(1,138)
    p,d,s=m.neighborhood(z,y,train,np.array([0]),k=3)
    altered=y.copy()
    altered[0]=1e9
    p2,d2,s2=m.neighborhood(z,altered,train,np.array([0]),k=3)
    np.testing.assert_array_equal(p,p2)
    np.testing.assert_array_equal(d,d2)
    np.testing.assert_array_equal(s,s2)
    with pytest.raises(ValueError,match="disjoint"):
        m.neighborhood(z,y,np.arange(138),np.array([0]),k=3)


def test_target59_neighbors_are_training_only(monkeypatch):
    m=load(monkeypatch)
    z,y=example()
    queries=np.arange(138,197)
    mean,nearest,std=m.neighborhood(
        z,y,np.arange(138),queries,k=8
    )
    assert mean.shape==(59,)
    assert nearest.shape==(59,)
    assert std.shape==(59,)
    assert np.isfinite(mean).all()


def test_zero_blend_preserves_native_reference26(monkeypatch):
    m=load(monkeypatch)
    base=np.array([3.1, 4.8, 6.1])
    pred,weight=m.analog_correction(
        base,np.array([4., 5., 2.]),np.array([.1,.2,10.]),
        .8,.7,np.array([.1,.1,2.]),blend=0.,gate="supported",
        max_delta=.5
    )
    np.testing.assert_array_equal(base,pred)
    assert np.all((weight>=0)&(weight<=1))
    changed,weight2=m.analog_correction(
        base,np.array([4.,5.,2.]),np.array([.1,.2,10.]),
        .8,.7,np.array([.1,.1,2.]),blend=.25,gate="none",
        max_delta=.5
    )
    np.testing.assert_allclose(changed,[3.325,4.85,5.6])


def test_tail_expansion_follows_train_only_y(monkeypatch):
    m=load(monkeypatch)
    x,y=example()
    actual=m.tail_correction(
        np.array([2.,4.,6.]),y[1:],
        gain=.2,threshold_sd=.5,max_delta=.5,
    )
    y2=y.copy()
    y2[0]=1e9
    revised=m.tail_correction(
        np.array([2.,4.,6.]),y2[1:],
        gain=.2,threshold_sd=.5,max_delta=.5,
    )
    np.testing.assert_array_equal(actual,revised)
    identical=m.tail_correction(
        np.array([2.,4.,6.]),y[1:],
        gain=0.,threshold_sd=.5,max_delta=.5
    )
    np.testing.assert_array_equal(identical,[2.,4.,6.])


def test_full_screen_fold_prediction_ignores_heldout_y(monkeypatch):
    m=load(monkeypatch)
    x,y=example()
    spaces={"robust26":x,"pca8":x[:,:8]}
    baseline=np.full(138,4.,dtype=float)
    first=m.evaluate_folds(y,spaces,baseline,[0],max_delta=.5)
    y2=y.copy()
    y2[0]=-1000.
    replay=m.evaluate_folds(y2,spaces,baseline,[0],max_delta=.5)
    assert first.keys()==replay.keys()
    for key in first:
        np.testing.assert_array_equal(first[key],replay[key])
    assert len(first)==len(m.model_grid(None))


def test_feature_spaces_require_197_x_and_26_names(monkeypatch):
    m=load(monkeypatch)
    rng=np.random.default_rng(0)
    features=[f"x_{i}" for i in range(26)]
    frame=pd.DataFrame(rng.normal(size=(197,26)),columns=features)
    frame.loc[10,"x_0"]=np.nan
    spaces=m.feature_spaces(frame,{"features":features})
    assert spaces["robust26"].shape==(197,26)
    assert spaces["pca8"].shape==(197,8)
    assert np.isfinite(spaces["robust26"]).all()


def test_aggregate_has_genuine_baseline_and_common_holdout_length(monkeypatch):
    m=load(monkeypatch)
    y=np.ones(138)
    ref=y+.25
    folds=list(range(138))
    pred={
        "Reference26_mean_unchanged":ref.copy().tolist(),
        "candidate":(ref-.1).tolist(),
    }
    result=m.aggregate(y,ref,pred,folds)
    assert result.iloc[0]["model"]=="candidate"
    assert result.iloc[0]["rmse_t_ha"]==pytest.approx(.15)
    control=result[result["model"]=="Reference26_mean_unchanged"].iloc[0]
    assert control["delta_mse_vs_ref"]==pytest.approx(0)
    assert control["n_predictions_modified"]==0
