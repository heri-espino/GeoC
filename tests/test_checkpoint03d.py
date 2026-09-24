from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from geocebada.evaluation.checkpoint03c import RepresentationSpec
from geocebada.evaluation.checkpoint03d import (
    _eligible_pairs,
    _fit_large_search,
    _make_large_global_search_spec,
)

ROOT = Path(__file__).resolve().parents[1]


def _config() -> dict:
    return yaml.safe_load(
        (ROOT / "configs" / "checkpoint03d.yaml").read_text(encoding="utf-8")
    )


def test_checkpoint03d_contract_is_competition_only() -> None:
    config = _config()
    assert config["scope"]["active_track"] == "competition"
    assert config["scope"]["clean_track_enabled"] is False
    assert Path(config["inputs"]["base_table"]).name == "parcel_features_competition.csv"
    assert config["deployment"]["required"] is True
    assert Path(config["outputs"]["onnx_model"]).suffix == ".onnx"


def test_checkpoint03d_eligible_pairs_respect_model_whitelist() -> None:
    config = _config()
    pairs = _eligible_pairs(
        config=config,
        representation_names=["G0_base", "G1_agronomic"],
        model_names=["HistGBLarge", "RidgeControl"],
    )
    assert ("G1_agronomic", "HistGBLarge") in pairs
    assert ("G0_base", "HistGBLarge") not in pairs
    assert ("G0_base", "RidgeControl") in pairs


def test_checkpoint03d_ridge_search_smoke() -> None:
    rng = np.random.default_rng(42)
    x = pd.DataFrame(
        {
            "x1": rng.normal(size=18),
            "x2": rng.normal(size=18),
        }
    )
    y = np.linspace(2.0, 5.0, 18)
    representation = RepresentationSpec(
        name="synthetic",
        track="competition",
        features=("x1", "x2"),
        layers=("base",),
        discovery=False,
        primitive_columns=(),
        description="synthetic",
    )
    spec = _make_large_global_search_spec(
        model_name="RidgeControl",
        model_config={
            "kind": "ridge",
            "candidates": [{"alpha": 1.0}, {"alpha": 10.0}],
        },
        representation=representation,
        random_state=42,
        compute="CPU",
        catboost_devices="0",
        xgboost_device="cuda:0",
    )
    splits = [
        (np.arange(12), np.arange(12, 18)),
        (
            np.concatenate([np.arange(6), np.arange(12, 18)]),
            np.arange(6, 12),
        ),
    ]
    search = _fit_large_search(
        spec=spec,
        x_train=x,
        y_train=y,
        inner_splits=splits,
        search_n_jobs=1,
    )
    assert np.isfinite(search.best_score_)
    assert search.best_estimator_.predict(x.iloc[:2]).shape == (2,)



def test_balanced_03d_budget() -> None:
    config = _config()

    assert config["validation"]["inner_folds"] == 3
    assert config["runtime"]["budget_profile"] == "balanced"
    for name in [
        "CatBoostLarge",
        "XGBoostLarge",
        "LightGBMLarge",
        "ExtraTreesLarge",
        "HistGBLarge",
    ]:
        assert len(config["models"][name]["candidates"]) == 3
    assert config["models"]["ExtraTreesLarge"]["fixed"]["n_estimators"] == 1500
