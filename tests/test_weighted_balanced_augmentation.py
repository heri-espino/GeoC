"""Fast synthetic QA: genuine fold-local weighting, uniform X perturbations."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def module(monkeypatch):
    tools = Path(__file__).resolve().parents[1] / "tools"
    monkeypatch.syspath_prepend(str(tools))
    spec = importlib.util.spec_from_file_location(
        "run_weighted_balanced_augmentation",
        tools / "run_weighted_balanced_augmentation.py",
    )
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def data():
    rng = np.random.default_rng(20261019)
    z = rng.normal(size=(197, 5))
    y = rng.normal(2., .6, 138)
    y[0] = 9.
    roots = {i: np.eye(5) for i in range(138)}
    return z,y,roots


def test_real_train_only_difficulty_and_outlier_weights(monkeypatch):
    m=module(monkeypatch)
    z,y,roots=data()
    train = np.arange(1,138)
    first,selected = m.weights_for_training(
        z,y,train,criterion="press",top_k=8,
        factor=5.,alpha=30.,
    )
    assert len(selected)==8 and 0 not in selected
    np.testing.assert_allclose(first.sum(), len(train))
    assert len(np.unique(np.round(first,10)))==2
    changed=y.copy(); changed[0]=-100000.
    again,selected2=m.weights_for_training(
        z,changed,train,criterion="press",top_k=8,
        factor=5.,alpha=30.,
    )
    np.testing.assert_array_equal(first,again)
    assert selected == selected2
    geo,far=m.weights_for_training(
        z,y,train,criterion="sparse",top_k=8,
        factor=5.,alpha=30.,
    )
    np.testing.assert_allclose(geo.sum(),len(train))
    assert 0 not in far


def test_equal_augments_every_real_training_point(monkeypatch):
    m=module(monkeypatch)
    z,y,roots=data()
    train=np.arange(1,138)
    x,ys=m.balanced_gaussian(
        z,y,train,roots,replicas=10,sigma=0.,seed=12,
    )
    assert len(x)==1370 and len(ys)==1370
    np.testing.assert_allclose(x,np.repeat(z[train],10,axis=0))
    np.testing.assert_allclose(ys,np.repeat(y[train],10))
    altered=y.copy();altered[0]=-99999.
    other=m.balanced_gaussian(
        z,altered,train,roots,replicas=10,sigma=0.,seed=12,
    )
    np.testing.assert_array_equal(x,other[0])
    np.testing.assert_array_equal(ys,other[1])


def test_equal_mass_sigma_zero_matches_same_global_ridge_plain(monkeypatch):
    m=module(monkeypatch)
    z,y,roots=data()
    train=np.arange(1,138);query=np.array([0,150])
    kw=dict(model="ridge",alpha=30.,trees=8,leaf=2,seed=12)
    plain=m.fit_predict(z,y,train,query,**kw)
    artificial=m.balanced_gaussian(
        z,y,train,roots,replicas=10,sigma=0.,seed=12,
    )
    aug=m.fit_predict(
        z,y,train,query,synthetic=artificial,
        synthetic_mass="equal",**kw
    )
    np.testing.assert_allclose(plain,aug,rtol=1e-9,atol=1e-9)
    w=m.combined_sample_weights(137,1370,"equal")
    assert len(w)==1507
    np.testing.assert_allclose(w.sum(),137)
    assert w[:137].sum()==pytest.approx(w[137:].sum())
    fw=m.combined_sample_weights(137,1370,"full")
    np.testing.assert_allclose(fw.sum(),137)
    assert fw[137:].sum()>w[137:].sum()


def test_isolation_of_heldout_y_in_complete_fold(monkeypatch):
    m=module(monkeypatch)
    z,y,roots=data()
    ids=[f"P{i:03d}" for i in range(138)]
    args=dict(ridge_alpha=30.,trees=8,leaf=2,seed=13,top_k=8)
    grid=(("ridge",),("press","sparse"),(2.,),
          (2,),(0.1,),("equal",))
    state=(z,y,ids,np.ones(138),roots,args,grid)
    m.init_worker(state)
    _,a=m.one_fold(0)
    changed=y.copy(); changed[0]=1.e12
    m.init_worker((z,changed,ids,np.ones(138),roots,args,grid))
    _,b=m.one_fold(0)
    for key in m.expected_columns(grid,args["top_k"]):
        assert a[key]==pytest.approx(b[key], rel=0, abs=0)
    assert a["observed"]!=b["observed"]


def test_reject_query_inside_supervised_training(monkeypatch):
    m=module(monkeypatch)
    z,y,roots=data()
    with pytest.raises(ValueError,match="Heldout"):
        m.fit_predict(
            z,y,np.arange(138),np.array([0]),
            model="ridge",alpha=30.,trees=8,leaf=2,seed=1
        )


def test_grid_dimensions(monkeypatch):
    m=module(monkeypatch)
    args=SimpleNamespace(
        models="ridge,extra_trees",weight_modes="press,sparse",
        weight_factors="2,5",replicas="2,10",sigmas="0,0.1",
        synth_masses="equal,full",top_k=8,local_neighbors=20,
        ridge_alpha=30.,trees=64,leaf=3,jobs=4
    )
    grid=m.parse_grid(args)
    assert len(m.expected_columns(grid,8))==26
    assert "balanced__ridge__r10__sigma0.1__massequal" in (
        m.expected_columns(grid,8)
    )
    assert "weighted__ridge__press__top8__w5" in (
        m.expected_columns(grid,8)
    )
