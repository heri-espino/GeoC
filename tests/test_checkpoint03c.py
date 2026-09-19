"""Tests for Checkpoint 03C representation benchmarking."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from geocebada.evaluation import (
    RepresentationSpec,
    build_representation_specs,
    evaluate_representation_benchmark,
    summarize_representation_benchmark,
)
from geocebada.features import join_agronomic_features, join_empirical_features

ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _config() -> dict:
    return yaml.safe_load(
        (ROOT / "configs" / "checkpoint03c.yaml").read_text(encoding="utf-8")
    )


def _empirical_config() -> dict:
    return yaml.safe_load(
        (ROOT / "configs" / "empirical_features_v1.yaml").read_text(
            encoding="utf-8"
        )
    )


def test_canonical_representation_counts_and_track_provenance() -> None:
    base_dir = ROOT / "data" / "processed" / "features_v1"
    agro_dir = ROOT / "data" / "processed" / "agronomic_features_v1"
    empirical_dir = ROOT / "data" / "processed" / "empirical_features_v1"

    base = pd.read_csv(base_dir / "parcel_features_all.csv")
    agronomic = pd.read_csv(agro_dir / "parcel_agronomic_features.csv")
    empirical = pd.read_csv(empirical_dir / "parcel_empirical_features.csv")
    joined = join_agronomic_features(base, agronomic)
    joined = join_empirical_features(joined, empirical)

    specs = build_representation_specs(
        joined=joined,
        manifests={
            "base": _load_json(base_dir / "feature_manifest.json"),
            "agronomic": _load_json(agro_dir / "feature_manifest.json"),
            "empirical": _load_json(empirical_dir / "feature_manifest.json"),
        },
        config=_config(),
        empirical_config=_empirical_config(),
    )

    assert len(specs[("clean", "B0_base")].features) == 692
    assert len(specs[("clean", "B1_agronomic")].features) == 123
    assert len(specs[("clean", "B2_empirical")].features) == 91
    assert len(specs[("clean", "B5_all")].features) == 906

    assert len(specs[("competition", "B0_base")].features) == 1400
    assert len(specs[("competition", "B1_agronomic")].features) == 350
    assert len(specs[("competition", "B2_empirical")].features) == 335
    assert len(specs[("competition", "B5_all")].features) == 2085

    clean_discovery = specs[("clean", "B7_all_plus_discovery")]
    competition_discovery = specs[("competition", "B7_all_plus_discovery")]
    assert len(clean_discovery.primitive_columns) == 7
    assert len(competition_discovery.primitive_columns) == 24

    assert not any("chirps" in column.casefold() for column in competition_discovery.features)
    assert "clim_official_2025__prec__season_sum" not in clean_discovery.features
    assert "clim_official_2025__prec__season_sum" in competition_discovery.features


def test_fixed_fold_benchmark_returns_complete_oof_predictions() -> None:
    rng = np.random.default_rng(42)
    rows = 8
    frame = pd.DataFrame(
        {
            "ID_POLIGONO": [f"P{i:02d}" for i in range(rows)],
            "CONJUNTO": ["ENTRENAMIENTO"] * rows,
            "RENDIMIENTO_T_HA": np.linspace(2.0, 5.0, rows),
            "x1": rng.normal(size=rows),
            "x2": rng.normal(size=rows),
        }
    )
    folds = pd.DataFrame(
        {
            "ID_POLIGONO": frame["ID_POLIGONO"],
            "fold_state_stratified": [1, 1, 1, 1, 2, 2, 2, 2],
            "fold_municipality_grouped": [1, 1, 2, 2, 1, 1, 2, 2],
        }
    )
    representations = {
        ("clean", "B0_base"): RepresentationSpec(
            name="B0_base",
            track="clean",
            features=("x1", "x2"),
            layers=("base",),
            discovery=False,
            primitive_columns=(),
            description="synthetic base",
        ),
        ("clean", "B7_all_plus_discovery"): RepresentationSpec(
            name="B7_all_plus_discovery",
            track="clean",
            features=("x1", "x2"),
            layers=("base",),
            discovery=True,
            primitive_columns=("x1", "x2"),
            description="synthetic discovery",
        ),
    }
    empirical_config = {
        "supervised_discovery": {
            "top_primitives": 2,
            "top_expressions": 2,
            "operations": ["product", "sum"],
            "epsilon": 1.0e-6,
        }
    }

    fold_scores, oof, discovery = evaluate_representation_benchmark(
        joined=frame,
        folds=folds,
        representations=representations,
        model_configs={"Ridge10": {"kind": "ridge", "alpha": 10.0}},
        empirical_config=empirical_config,
        protocols=("fold_state_stratified", "fold_municipality_grouped"),
        random_state=42,
    )
    summary = summarize_representation_benchmark(fold_scores, oof)

    assert len(fold_scores) == 8
    assert len(oof) == 32
    assert len(discovery) == 8
    assert len(summary) == 4

    coverage = oof.groupby(
        ["protocol", "track", "representation", "model"]
    )["ID_POLIGONO"].nunique()
    assert set(coverage) == {8}
    assert np.isfinite(summary["oof_rmse"]).all()
    assert np.isfinite(summary["oof_mae"]).all()
