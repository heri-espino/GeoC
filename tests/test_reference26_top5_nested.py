"""Leakage and selection QA for preselected-five nested Reference26 analog CV."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def loaded(monkeypatch):
    tools=Path(__file__).resolve().parents[1]/"tools"
    monkeypatch.syspath_prepend(str(tools))
    spec=importlib.util.spec_from_file_location(
        "run_reference26_top5_nested",
        tools/"run_reference26_top5_nested.py"
    )
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def artificial(m):
    rng=np.random.default_rng(71)
    z=rng.normal(size=(197,26))
    y=4+rng.normal(0,.6,138)
    ids=[f"T{i:03d}" for i in range(138)]
    ref=np.full(138,4.,dtype=float)
    spaces={"robust26":z,"pca8":z[:,:8]}
    data=pd.DataFrame({
        m.ID:ids,"observed":y,
        "Reference26_mean":ref
    })
    for cfg in m.TOP5:
        key=m.analog.signature(cfg)
        p=m.apply_candidate(
            cfg,spaces[cfg["geometry"]],y,
            np.arange(1,138),np.array([0]),
            np.array([ref[0]]),.5
        )
        data[key]=np.full(138,p[0])
    frame=pd.DataFrame({
        m.ID:ids,m.TARGET:y,
    })
    return frame,spaces,y,ids,ref,data


def test_top5_names_are_exact_screen_winners(monkeypatch):
    m=loaded(monkeypatch)
    assert len(m.TOP5)==5
    assert len(set(m.NAMES))==5
    assert m.NAMES==(
        "analog__robust26__k3__blend0.25__supported",
        "analog__pca8__k3__blend0.25__supported",
        "analog__pca8__k8__blend0.25__supported",
        "analog__robust26__k3__blend0.1__supported",
        "analog__robust26__k3__blend0.1__none",
    )


def test_choice_includes_native_fallback_and_inner_gain_margin(monkeypatch):
    m=loaded(monkeypatch)
    n=30
    true=np.ones(n)
    preds={name:np.full(n,1.5) for name in m.CHOICES}
    preds[m.NAMES[0]]=np.full(n,1.2)
    chosen,loss=m.choose_on_inner(
        true,preds,min_relative_gain=.01
    )
    assert chosen==m.NAMES[0]
    assert loss[chosen]<loss[m.BASE]
    preds[m.NAMES[0]]=np.full(n,1.498)
    chosen,loss=m.choose_on_inner(
        true,preds,min_relative_gain=.01
    )
    assert chosen==m.BASE
    with pytest.raises(ValueError,match="Missing"):
        m.choose_on_inner(true,{m.BASE:preds[m.BASE]},
                          min_relative_gain=.01)


def test_outer_candidate_y_never_uses_heldout_y(monkeypatch):
    m=loaded(monkeypatch)
    frame,spaces,y,ids,ref,saved=artificial(m)
    for cfg in m.TOP5:
        train=np.arange(1,138)
        a=m.apply_candidate(cfg,spaces[cfg["geometry"]],
                            y,train,np.array([0]),ref[:1],.5)
        modified=y.copy()
        modified[0]=1e9
        b=m.apply_candidate(cfg,spaces[cfg["geometry"]],
                            modified,train,np.array([0]),ref[:1],.5)
        np.testing.assert_array_equal(a,b)
        assert np.isfinite(a).all()


def test_full_nested_fold_uses_only_outer_train_labels(monkeypatch,tmp_path):
    m=loaded(monkeypatch)
    frame,spaces,y,ids,ref,saved=artificial(m)
    model_calls=[]
    def fake_build(frame,groups,d,manifest,train,inner,
                   factory,cache,*,mean):
        train=np.asarray(train,dtype=int)
        assert mean is True
        assert len(train) in (102,103)
        assert 0 not in train
        model_calls.append(set(train.tolist()))
        masked=frame.copy()
        excluded=np.setdiff1d(np.arange(138),train)
        masked.loc[excluded,m.TARGET]=np.nan
        assert masked.loc[excluded,m.TARGET].isna().all()
        assert masked.loc[train,m.TARGET].notna().all()
        return {"train_positions":train},masked

    def fake_predict(fitted,pack,query,masked,d,key):
        assert key=="Reference26"
        assert 0 not in query
        training=pack["train_positions"]
        assert not np.intersect1d(training,query).size
        return np.full(len(query),np.mean(masked.iloc[training][m.TARGET]))

    monkeypatch.setattr(m.native,"build_fresh_fit",fake_build)
    monkeypatch.setattr(m.native,"predict_model",fake_predict)
    m._WORKER=(
        frame,None,np.eye(138),{"features":[f"x{i}" for i in range(26)]},
        ids,None,None,spaces,y,ref,saved,.5,.01
    )
    fold,status,selected=m.one_outer_fold(0,str(tmp_path))
    assert fold==1 and status=="trained" and selected in m.CHOICES
    assert len(model_calls)==4
    assert all(0 not in calls for calls in model_calls)
    generated=pd.read_csv(tmp_path/"fold_001.csv",dtype={m.ID:str})
    m.verify_fold(generated,0,ids,y,ref,saved)
    assert generated.iloc[0]["outer_train_n"]==137
    assert generated.iloc[0]["inner_oof_n"]==137


def test_exact_old_screen_alignment_is_required(monkeypatch,tmp_path):
    m=loaded(monkeypatch)
    frame,spaces,y,ids,ref,saved=artificial(m)
    path=tmp_path/"screen.csv"
    saved.to_csv(path,index=False)
    got=m.read_screen(path,ids,y,ref)
    assert len(got)==138
    tampered=saved.copy()
    tampered.loc[3,m.ID]="WRONG"
    tampered.to_csv(path,index=False)
    with pytest.raises(ValueError,match="IDs/order"):
        m.read_screen(path,ids,y,ref)


def test_five_fixed_metrics_are_not_new_model_retraining(monkeypatch):
    m=loaded(monkeypatch)
    frame,spaces,y,ids,ref,screen=artificial(m)
    report=m.fixed_results(screen)
    assert len(report)==6
    assert set(report["model"])==set(m.CHOICES)
    assert set(report["selection"])=={"fixed_posthoc_top5"}
    assert (report["n_real_heldouts"]==138).all()
