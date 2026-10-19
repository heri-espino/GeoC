"""QA: comprehensive seven-model metrics use real heldout shape and no private exports."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


def script(monkeypatch):
    directory = Path(__file__).resolve().parents[1] / "tools"
    monkeypatch.syspath_prepend(str(directory))
    path = directory / "report_reference26_top5_all_metrics.py"
    spec = importlib.util.spec_from_file_location(
        "report_reference26_top5_all_metrics", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sample(m):
    n=138
    y = np.linspace(1.5, 7, n)
    ref = y+0.4681381271
    selected = y+0.462892
    frame = pd.DataFrame({
        m.ID: [f"PRIV_{j:03d}" for j in range(n)],
        "observed": y,
        "Reference26_mean": ref,
        "chosen_model": [m.TOP5[0]]*n,
        "outer_train_n": [137]*n,
        "inner_oof_n": [137]*n,
        m.NESTED: selected,
    })
    for name in m.TOP5:
        frame[name] = selected if name == m.TOP5[0] else y+.466
    return frame


def test_audited_input_only_138_original_137_train(monkeypatch, tmp_path):
    m = script(monkeypatch)
    frame=sample(m)
    path=tmp_path/"nested.csv"
    frame.to_csv(path, index=False)
    data=m.load_verified(path)
    assert set(data.columns) == {m.ID,"observed",*m.MODELS}
    assert len(data) == 138
    assert not any(x in data.columns for x in (
        "chosen_model", "outer_train_n", "inner_oof_n"
    ))
    bad=frame.copy()
    bad.loc[0,"outer_train_n"]=138
    bad.to_csv(path,index=False)
    with pytest.raises(ValueError, match="137-train"):
        m.load_verified(path)
    bad=frame.copy()
    bad.loc[0,m.NESTED]=500.
    bad.to_csv(path,index=False)
    with pytest.raises(ValueError, match="route mismatch|NestedSelection"):
        m.load_verified(path)


def test_more_than_twenty_statistical_metrics_and_guards(monkeypatch):
    m=script(monkeypatch)
    data=sample(m)
    data=data.rename(columns={"Reference26_mean":m.BASE})
    selected=m.full_metrics(data[[m.ID,"observed",*m.MODELS]])
    assert len(selected)==7
    assert selected["model"].nunique()==7
    row=selected.set_index("model").loc[m.NESTED]
    assert row["mae_t_ha"]==pytest.approx(.462892)
    assert row["l1_t_ha"]==pytest.approx(138*.462892)
    assert row["l2_t_ha"]==pytest.approx(np.sqrt(138)*.462892)
    assert row["linf_t_ha"]==pytest.approx(.462892)
    assert row["p99_ae_t_ha"]==pytest.approx(.462892)
    assert row["mse_t_ha2"]==pytest.approx(.462892**2)
    assert row["median_ae_t_ha"]==pytest.approx(.462892)
    assert row["trimmed_rmse_10pct_t_ha"]==pytest.approx(.462892)
    assert row["huber_mean_delta_0.5"]==pytest.approx(.5*.462892**2)
    assert row["calibration_slope_y_on_pred"]==pytest.approx(1.)
    assert np.isnan(row["adjusted_r2"])
    assert row["mape_excluded_near_zero_count"]==0
    assert row["coverage_abs_error_le_0.5_t_ha_pct"]==100.


def test_relative_metrics_near_zero_are_finite_and_logged(monkeypatch):
    m=script(monkeypatch)
    y=np.array([0.,0.05,1.,2.])
    pred=np.array([0.4,0.5,1.1,1.9])
    s=m.robust_and_relative(y,pred)
    assert s["mape_excluded_near_zero_count"]==2
    assert np.isfinite(s["mape_floor_0p1_pct"])
    assert np.isfinite(s["smape_floor_0p1_pct"])
    assert np.isfinite(s["rmspe_floor_0p1_pct"])
    assert s["p99_ae_t_ha"]>=s["p95_signed_error_t_ha"]-1e-10


def test_paired_and_selection_statistics(monkeypatch):
    m=script(monkeypatch)
    data=sample(m).rename(columns={"Reference26_mean":m.BASE})
    table=data[[m.ID,"observed",*m.MODELS]]
    y=table["observed"].to_numpy()
    values=table[list(m.MODELS)].to_numpy(float)
    boot=m.bootstrap_metrics(y,values,draws=200,seed=7)
    assert len(boot)==7
    assert np.isfinite(boot.filter(regex="ci95").to_numpy(float)).all()
    paired=m.paired_tests(table,draws=200,seed=7)
    assert len(paired)==6
    assert (paired["n_common_heldouts"]==138).all()
    assert (paired["delta_rmse_candidate_minus_baseline_t_ha"]<0).all()
    assert (paired["signflip_mse_maxT_six_one_sided_p"]>=
            paired["signflip_mse_one_sided_p"]).all()
    assert (paired["sign_test_two_sided_p"].between(0,1)).all()
    influence=m.influence(table)
    assert len(influence)==7
    assert influence.set_index("model").loc[m.NESTED][
        "n_deletions_with_candidate_better_than_baseline"
    ]==138
    parts=m.strata(table)
    assert len(parts)==21
    assert (parts["n_heldouts"]==46).all()


def test_spatial_optional_and_strict_distance_shape(monkeypatch,tmp_path):
    m=script(monkeypatch)
    data=sample(m).rename(columns={"Reference26_mean":m.BASE})
    table=data[[m.ID,"observed",*m.MODELS]]
    skip=m.spatial_sensitivity(table,None,draws=200,blocks=10,seed=1)
    assert (skip["status"].str.startswith("SKIPPED")).all()
    wrong=tmp_path/"distance.npy"
    np.save(wrong,np.eye(200))
    with pytest.raises(ValueError,match="distance matrix"):
        m.spatial_sensitivity(table,wrong,draws=200,blocks=10,seed=1)


def test_aggregate_outputs_and_pngs_never_publish_individual_y(
    monkeypatch,tmp_path,
):
    m=script(monkeypatch)
    source=tmp_path/"nested.csv"
    sample(m).to_csv(source,index=False)
    out=tmp_path/"reports"
    args=SimpleNamespace(input=source,output=out,bootstrap=200,
                         seed=7,geo_blocks=10,distance197=None)
    m.run(args)
    for name in m.AGG_FILES:
        file=out/name
        assert file.is_file(), name
        contents=file.read_text(encoding="utf-8")
        assert "PRIV_" not in contents
        assert m.ID not in contents
    for file in (
        "absolute_error_ecdf_7.png", "bootstrap_rmse_7.png",
        "rmse_by_yield_tertile.png", "paired_rmse_delta.png"
    ):
        assert (out/file).is_file()
    spec=json.loads((out/"metric_definitions.json").read_text())
    assert spec["n_target59_with_known_y"]==0
