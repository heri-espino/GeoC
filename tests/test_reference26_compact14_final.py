"""Synthetic, no-private-data QA of full final ensemble statistics and spatial tests."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from geocebada.evaluation.jury_residual_diagnostics import (
    adjust_spatial_tests,
    conditional_error_diagnostics,
    multiplicity,
    normality_and_distribution,
    paired_loss_diagnostics,
)


def load_tool(monkeypatch, name):
    folder = Path(__file__).resolve().parents[1] / "tools"
    monkeypatch.syspath_prepend(str(folder))
    spec = importlib.util.spec_from_file_location(name, folder / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def labels():
    rng = np.random.default_rng(812)
    y = rng.normal(4, .6, 138)
    pred_a = y + rng.normal(0, .4, 138)
    pred_b = y + rng.normal(0, .35, 138)
    return y, {"Reference26_mean": pred_a, "Compact14": pred_b,
               "Ensemble_final": .7049462505599942*pred_a
               + .29505374944000595*pred_b}


def test_frozen_ensemble_config_is_valid(monkeypatch):
    tool = load_tool(monkeypatch, "run_reference26_compact14_final")
    a, b = tool.frozen_weights(tool.CONFIG)
    assert a == pytest.approx(.7049462505599942)
    assert a+b == pytest.approx(1.)
    assert b == pytest.approx(.29505374944000595)


def test_multitest_bh_holm_expected_and_nan_handling():
    p = np.array([.01, .04, .03, np.nan])
    assert np.allclose(
        multiplicity(p, "holm")[:3], [.03, .06, .06]
    )
    assert np.allclose(
        multiplicity(p, "fdr_bh")[:3], [.03, .04, .04]
    )
    assert np.isnan(multiplicity(p, "holm")[3])
    with pytest.raises(ValueError):
        multiplicity(p, "nope")


def test_normality_bias_heteroscedasticity_and_paired_losses():
    y, estimates = labels()
    normal = normality_and_distribution(y, estimates)
    assert set(normal.model) == set(estimates)
    assert normal.n.eq(138).all()
    assert normal["shapiro_p_exploratory"].between(0, 1).all()
    assert normal["anderson_darling_critical_at_5pct"].gt(0).all()
    assert (normal["jarque_bera_p_not_reported"]
            == "chi_square_asymptotic_inaccurate_at_n138").all()
    cond = conditional_error_diagnostics(y, estimates)
    assert set(cond.model) == set(estimates)
    assert cond["absolute_error_vs_predicted_spearman_rho"].between(
        -1, 1
    ).all()
    assert cond["koenker_breusch_pagan_asymptotic_p_exploratory"].between(
        0, 1
    ).all()
    paired = paired_loss_diagnostics(y, estimates)
    assert len(paired) == 4
    assert set(paired.loss) == {"squared_error", "absolute_error"}
    assert paired["wilcoxon_p_exploratory_holm_all"].between(0, 1).all()


def test_genuinely_identical_paired_models_produce_sign_p_one():
    y, estimates = labels()
    estimates["Compact14"] = estimates["Reference26_mean"].copy()
    paired = paired_loss_diagnostics(
        y, {"Reference26_mean": estimates["Reference26_mean"],
            "Compact14": estimates["Compact14"]}
    )
    assert paired.n_non_tied.eq(0).all()
    assert paired["sign_test_p_exploratory"].eq(1.).all()
    assert paired["wilcoxon_p_exploratory"].eq(1.).all()


def test_spatial_knn_permutation_null_and_known_clustering(monkeypatch):
    module = load_tool(monkeypatch, "report_reference26_compact14_final")
    # Small XY grid embedded in 138 valid centroids; no private coordinates.
    lat = 18.5 + np.arange(138)//14 * .05
    lon = -98.5 + np.arange(138)%14 * .05
    d = module.geo_km(lat.astype(float), lon.astype(float))
    assert d.shape == (138, 138)
    assert np.allclose(d, d.T)
    assert np.allclose(np.diag(d), 0)
    knn = module.neighbors(d, 4)
    assert knn.shape == (138, 4)
    rng = np.random.default_rng(10)
    signal = lat + rng.normal(0, .01, 138)
    assert module.moran(signal, knn) > 0.5
    assert module.geary(signal, knn) < .5
    p = module.permutation_test(
        signal, knn, "moran", permutations=199, seed=12
    )
    assert 0 < p["p_two_sided"] <= 1
    assert p["p_right"] <= .05
    result = module.spatial_metrics(
        signal, {"Reference26_mean": signal+.1*rng.normal(size=138)},
        d, permutations=99, seed=32,
    )
    assert set(result.knn) == {4, 8, 12, 24}
    assert len(result) == 4*3*2
    corrected = adjust_spatial_tests(result)
    assert corrected.p_two_sided_holm_all.ge(
        corrected.p_two_sided-1e-12
    ).all()
    assert corrected.p_two_sided_fdr_bh_all.ge(
        corrected.p_two_sided-1e-12
    ).all()


def test_extended_metrics_and_safe_export(monkeypatch, tmp_path):
    module = load_tool(monkeypatch, "report_reference26_compact14_final")
    y, estimates = labels()
    more = module.extended_metrics(y, estimates)
    assert more["rmse_kg_ha"].gt(0).all()
    assert more["concordance_ccc"].between(-1, 1).all()
    base = pd.DataFrame({
        module.ID: [f"row_{i}" for i in range(138)],
        "observed": y,
        **estimates,
    })
    from geocebada.evaluation.loocv_comprehensive_metrics import all_metrics

    standard = all_metrics(
        base, {name: "synthetic_loo" for name in estimates}
    )
    geo = pd.DataFrame([{
        "model": name, "knn": 4, "spatial_variable": "signed_residual",
        "statistic": "Moran_I", "estimate": .05,
        "expected_permutation": -.01,
        "p_two_sided": .3, "p_right": .2, "p_left": .8,
        "permutations": 999,
        "p_two_sided_holm_all": .9,
        "p_two_sided_fdr_bh_all": .3,
    } for name in estimates])
    outfile = tmp_path / "public.json"
    module.export_public(
        standard, more, geo, (.7049462505599942, .29505374944000595),
        outfile,
    )
    payload = json.loads(outfile.read_text())
    assert len(payload["models"]) == 3
    assert payload["not_independent_weight_validation"]
    dumped = outfile.read_text()
    assert "ID_POLIGONO" not in dumped
    assert "RENDIMIENTO_T_HA" not in dumped
    assert "row_1" not in dumped
    assert "base_centroid_lat" not in dumped


def test_loo_record_catches_weight_tampering(monkeypatch, tmp_path):
    module = load_tool(monkeypatch, "run_reference26_compact14_final")
    a, b = module.frozen_weights(module.CONFIG)
    ids = [f"PARCEL_{i:03d}" for i in range(138)]
    y = np.linspace(2, 5, 138)
    target = tmp_path / "fold_001.csv"
    pd.DataFrame([{
        module.ID: ids[0], "observed": y[0],
        "outer_train": 137, "outer_heldout": 1,
        "Reference26_mean": 3., "Compact14": 4.,
        "Ensemble_final": a*3.+b*4.,
    }]).to_csv(target, index=False)
    module.verify_record(target, 0, ids, y, (a, b))
    wrong = pd.read_csv(target)
    wrong["Ensemble_final"] += .01
    wrong.to_csv(target, index=False)
    with pytest.raises(ValueError, match="weighted"):
        module.verify_record(target, 0, ids, y, (a, b))
