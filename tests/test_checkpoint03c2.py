"""Tests for competition-only Checkpoint 03C.2."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from geocebada.evaluation import (
    RepresentationSpec,
    build_competition_representation_specs,
    evaluate_nested_model_families,
    make_inner_cv_splits,
    summarize_nested_results,
)
from geocebada.features import join_agronomic_features, join_empirical_features

ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _config() -> dict:
    return yaml.safe_load(
        (ROOT / "configs" / "checkpoint03c2.yaml").read_text(encoding="utf-8")
    )


def _empirical_config() -> dict:
    return yaml.safe_load(
        (ROOT / "configs" / "empirical_features_v1.yaml").read_text(
            encoding="utf-8"
        )
    )


def test_03c2_is_competition_only_and_representation_sizes() -> None:
    base_dir = ROOT / "data" / "processed" / "features_v1"
    agro_dir = ROOT / "data" / "processed" / "agronomic_features_v1"
    empirical_dir = ROOT / "data" / "processed" / "empirical_features_v1"

    base = pd.read_csv(base_dir / "parcel_features_all.csv")
    agronomic = pd.read_csv(agro_dir / "parcel_agronomic_features.csv")
    empirical = pd.read_csv(empirical_dir / "parcel_empirical_features.csv")
    joined = join_agronomic_features(base, agronomic)
    joined = join_empirical_features(joined, empirical)

    specs = build_competition_representation_specs(
        joined=joined,
        manifests={
            "base": _load_json(base_dir / "feature_manifest.json"),
            "agronomic": _load_json(agro_dir / "feature_manifest.json"),
            "empirical": _load_json(empirical_dir / "feature_manifest.json"),
        },
        config=_config(),
        empirical_config=_empirical_config(),
    )

    assert set(specs) == {
        "C0_base",
        "C1_agronomic",
        "C2_all_plus_discovery",
        "C3_base_agro_plus_discovery",
    }
    assert {spec.track for spec in specs.values()} == {"competition"}

    assert len(specs["C0_base"].features) == 1400
    assert len(specs["C1_agronomic"].features) == 350
    assert len(specs["C2_all_plus_discovery"].features) == 2085
    assert len(specs["C3_base_agro_plus_discovery"].features) == 1754

    assert len(specs["C2_all_plus_discovery"].primitive_columns) == 24
    assert len(specs["C3_base_agro_plus_discovery"].primitive_columns) == 24
    assert not any(
        "chirps" in feature.casefold()
        for spec in specs.values()
        for feature in spec.features
    )


def test_inner_grouped_cv_never_splits_a_municipality() -> None:
    frame = pd.DataFrame(
        {
            "meta_estado": ["A"] * 6 + ["B"] * 6,
            "meta_municipio": [
                "m1",
                "m1",
                "m2",
                "m2",
                "m3",
                "m3",
                "m4",
                "m4",
                "m5",
                "m5",
                "m6",
                "m6",
            ],
        }
    )
    splits = make_inner_cv_splits(
        frame,
        protocol="fold_municipality_grouped",
        n_splits=3,
        random_state=42,
        state_column="meta_estado",
        municipality_column="meta_municipio",
    )

    groups = (
        frame["meta_estado"].astype(str)
        + "|"
        + frame["meta_municipio"].astype(str)
    )
    for train_idx, valid_idx in splits:
        assert set(groups.iloc[train_idx]).isdisjoint(
            set(groups.iloc[valid_idx])
        )


def test_inner_state_cv_preserves_all_states() -> None:
    frame = pd.DataFrame(
        {
            "meta_estado": ["A", "B", "C"] * 6,
            "meta_municipio": [f"m{i}" for i in range(18)],
        }
    )
    splits = make_inner_cv_splits(
        frame,
        protocol="fold_state_stratified",
        n_splits=3,
        random_state=42,
        state_column="meta_estado",
        municipality_column="meta_municipio",
    )

    for _, valid_idx in splits:
        assert set(frame.iloc[valid_idx]["meta_estado"]) == {"A", "B", "C"}


def test_nested_ridge_smoke_has_complete_oof_coverage() -> None:
    rng = np.random.default_rng(42)
    rows = 18
    frame = pd.DataFrame(
        {
            "ID_POLIGONO": [f"P{i:02d}" for i in range(rows)],
            "CONJUNTO": ["ENTRENAMIENTO"] * rows,
            "RENDIMIENTO_T_HA": np.linspace(2.0, 5.0, rows),
            "meta_estado": ["A", "B", "C"] * 6,
            "meta_municipio": [f"m{i // 3}" for i in range(rows)],
            "x1": rng.normal(size=rows),
            "x2": rng.normal(size=rows),
        }
    )
    folds = pd.DataFrame(
        {
            "ID_POLIGONO": frame["ID_POLIGONO"],
            "fold_state_stratified": [1, 2, 3] * 6,
            "fold_municipality_grouped": [
                1,
                1,
                1,
                2,
                2,
                2,
                3,
                3,
                3,
                1,
                1,
                1,
                2,
                2,
                2,
                3,
                3,
                3,
            ],
        }
    )
    representations = {
        "C0_base": RepresentationSpec(
            name="C0_base",
            track="competition",
            features=("x1", "x2"),
            layers=("base",),
            discovery=False,
            primitive_columns=(),
            description="synthetic",
        )
    }
    config = {
        "identity": {
            "id_column": "ID_POLIGONO",
            "target_column": "RENDIMIENTO_T_HA",
            "split_column": "CONJUNTO",
            "train_value": "ENTRENAMIENTO",
            "state_column": "meta_estado",
            "municipality_column": "meta_municipio",
        },
        "validation": {
            "outer_protocols": [
                "fold_state_stratified",
                "fold_municipality_grouped",
            ],
            "inner_folds": 2,
            "random_state": 42,
        },
        "runtime": {
            "search_n_jobs": 1,
            "catboost_task_type": "CPU",
            "catboost_devices": "0",
            "catboost_allow_cpu_fallback": True,
            "progress": False,
        },
        "models": {
            "Ridge": {
                "kind": "ridge",
                "candidates": [{"alpha": 1.0}, {"alpha": 10.0}],
            }
        },
    }
    empirical_config = {
        "supervised_discovery": {
            "top_primitives": 2,
            "top_expressions": 2,
            "operations": ["product"],
            "epsilon": 1.0e-6,
        }
    }

    outer, oof, inner, discovery, task_type = evaluate_nested_model_families(
        joined=frame,
        folds=folds,
        representations=representations,
        config=config,
        empirical_config=empirical_config,
    )
    protocol_summary, robustness = summarize_nested_results(outer, oof)

    assert len(outer) == 6
    assert len(oof) == 36
    assert len(inner) == 12
    assert discovery.empty
    assert task_type == "CPU"
    assert len(protocol_summary) == 2
    assert len(robustness) == 1
    assert np.isfinite(protocol_summary["oof_rmse"]).all()
