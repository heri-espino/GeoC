"""Precision and privacy contracts for full LOOCV model metric reporting."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from geocebada.evaluation.loocv_comprehensive_metrics import (
    MAIN_FIELDS,
    OTHER_FIELDS,
    all_metrics,
    metrics,
    paired_metrics,
)


def sample():
    y = np.linspace(2, 4, 138)
    p = y + 0.2
    other = y + np.where(np.arange(138)%2 == 0, 0.1, -0.1)
    table = pd.DataFrame({
        "ID_POLIGONO": [f"A{i:03}" for i in range(138)],
        "observed": y,
        "Reference26_original": p,
        "Compact14": other,
    })
    return table


def reporter(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "tools"
    monkeypatch.syspath_prepend(str(path))
    spec = importlib.util.spec_from_file_location(
        "report_loocv_metrics", path / "report_loocv_metrics.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_error_norms_have_explicit_sum_not_mean_definitions():
    data = sample()
    y = data["observed"].to_numpy(float)
    p = data["Reference26_original"].to_numpy(float)
    result = metrics(y, p, "Reference26_original", source="test")
    assert result["n_holdouts"] == 138
    assert result["l1_t_ha"] == pytest.approx(138*.2)
    assert result["l2_t_ha"] == pytest.approx(np.sqrt(138)*.2)
    assert result["linf_t_ha"] == pytest.approx(.2)
    assert result["rmse_t_ha"] == pytest.approx(.2)
    assert result["mae_t_ha"] == pytest.approx(.2)
    assert result["median_ae_t_ha"] == pytest.approx(.2)
    assert result["p95_ae_t_ha"] == pytest.approx(.2)
    assert result["bias_t_ha"] == pytest.approx(.2)
    assert result["max_overprediction_t_ha"] == pytest.approx(.2)
    assert result["max_underprediction_t_ha"] == pytest.approx(0)
    assert result["error_sd_t_ha"] == pytest.approx(0, abs=1e-12)
    assert result["explained_variance"] == pytest.approx(1)
    assert result["r2_cv"] < 1
    assert np.isnan(result["adjusted_r2"])
    assert result["adjusted_r2_status"].startswith("not_applicable")


def test_validates_ids_finite_values_and_provenance():
    data = sample()
    report = all_metrics(data, {
        "Reference26_original": "test", "Compact14": "test",
    })
    assert set(report.columns) == set((*MAIN_FIELDS, *OTHER_FIELDS))
    assert len(report) == 2
    with pytest.raises(ValueError, match="provenance"):
        all_metrics(data, {"Compact14": "test"})
    broken = data.copy()
    broken.loc[1, "ID_POLIGONO"] = broken.loc[0, "ID_POLIGONO"]
    with pytest.raises(ValueError, match="distinct"):
        all_metrics(broken, {
            "Reference26_original": "test", "Compact14": "test",
        })
    broken = data.copy()
    broken.loc[0, "observed"] = np.nan
    with pytest.raises(ValueError, match="finite"):
        all_metrics(broken, {
            "Reference26_original": "test", "Compact14": "test",
        })


def test_paired_metrics_and_naive_bootstrap_have_correct_signs():
    data = sample()
    paired = paired_metrics(
        data, baseline="Reference26_original", draws=200, seed=11
    )
    row = paired.iloc[0]
    assert row["model"] == "Compact14"
    assert row["delta_rmse_t_ha"] == pytest.approx(-.1)
    assert row["fraction_lower_ae"] == pytest.approx(1)
    assert row["delta_mse_t_ha2"] == pytest.approx(.01-.04)
    assert row["naive_boot_delta_rmse_ci95_lo"] <= row["naive_boot_delta_rmse_ci95_hi"]
    assert "spatial" in row["bootstrap_caveat"]


def test_strict_138_model_alignment_prevents_mismatched_y(monkeypatch):
    r = reporter(monkeypatch)
    data = sample()
    original = r.loader.audited_model(
        data, "Reference26_original", "Reference26_original",
    )
    compact = r.loader.audited_model(
        data, "Compact14", "Compact14"
    )
    combined = r.loader.join_models(original, [compact])
    assert len(combined) == 138
    wrong = compact.copy()
    wrong.loc[0, "observed"] += .1
    with pytest.raises(ValueError, match="observed yield mismatch"):
        r.loader.join_models(original, [wrong])


def test_publish_prohibits_per_parcel_columns_and_copies_only_aggregate(
    monkeypatch, tmp_path,
):
    r = reporter(monkeypatch)
    source = tmp_path / "private"
    source.mkdir()
    table = sample()
    summary = all_metrics(table, {
        "Reference26_original": "saved_full_loo",
        "Compact14": "saved_full_loo",
    })
    comparison = paired_metrics(table, baseline="Reference26_original",
                                draws=100)
    summary.to_csv(source / "all_models_loocv_metrics.csv", index=False)
    comparison.to_csv(source / "paired_model_comparisons.csv", index=False)
    target = tmp_path / "public"
    args = SimpleNamespace(output=source, publish_dir=target)
    r.publish(args)
    copied = pd.read_csv(target / "LOOCV_FULL_METRICS_138.csv")
    assert set(copied.columns) == set((*MAIN_FIELDS, *OTHER_FIELDS))
    assert not {"observed", "predicted", "ID_POLIGONO"} & set(copied)
    summary["ID_POLIGONO"] = "PRIVATE_ID"
    summary.to_csv(source / "all_models_loocv_metrics.csv", index=False)
    with pytest.raises(ValueError, match="allowlist"):
        r.publish(args)
