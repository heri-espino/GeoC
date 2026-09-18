import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor

from geocebada.evaluation import (
    build_feature_catalog,
    build_fixed_fold_assignments,
    classify_feature_family,
    evaluate_fixed_folds,
    resolve_ablation_features,
    summarize_feature_inventory,
    summarize_fixed_fold_scores,
    train_prediction_diagnostics,
    validate_fixed_fold_assignments,
)


def _synthetic_frame() -> pd.DataFrame:
    n_train = 30
    n_prediction = 9
    total = n_train + n_prediction
    states = ["Puebla", "Tlaxcala", "Hidalgo"]
    rows = []
    for i in range(total):
        state = states[i % len(states)]
        municipality = f"M{(i // 3) % 10:02d}"
        is_train = i < n_train
        rows.append(
            {
                "ID_POLIGONO": f"P{i:03d}",
                "CONJUNTO": "ENTRENAMIENTO" if is_train else "PREDICCION",
                "RENDIMIENTO_T_HA": float(i % 7 + 1) if is_train else np.nan,
                "meta_estado": state,
                "meta_municipio": municipality,
                "base_area_ha": float(i + 1),
                "soilgrids__phh2o__0_5cm__mean": 6.0 + 0.01 * i,
                "clim_official_hist__prec__m04_mean": 50.0 + i,
                "sat_basic_hist__sentinel_2__ndvi_promedio__mean": 0.2 + 0.01 * i,
                "wapor_2025__aeti__season_total": 100.0 + 4.0 * i,
            }
        )
    return pd.DataFrame(rows)


def _manifest() -> dict:
    return {
        "features": [
            {
                "column": "base_area_ha",
                "source": "official_split+official_parcels+admin",
                "mode": "clean",
                "description": "area",
            },
            {
                "column": "soilgrids__phh2o__0_5cm__mean",
                "source": "external_soilgrids",
                "mode": "clean",
                "description": "soil",
            },
            {
                "column": "clim_official_hist__prec__m04_mean",
                "source": "official_climate",
                "mode": "clean",
                "description": "climate",
            },
            {
                "column": "sat_basic_hist__sentinel_2__ndvi_promedio__mean",
                "source": "sat_basic",
                "mode": "clean",
                "description": "satellite",
            },
            {
                "column": "wapor_2025__aeti__season_total",
                "source": "external_wapor",
                "mode": "competition",
                "description": "wapor",
            },
        ]
    }


def test_feature_family_catalog_and_inventory() -> None:
    frame = _synthetic_frame()
    catalog = build_feature_catalog(_manifest(), frame)

    assert classify_feature_family(_manifest()["features"][0]) == "geometry"
    assert set(catalog["family"]) == {
        "geometry",
        "soilgrids",
        "climate_official",
        "basic",
        "wapor",
    }

    inventory = summarize_feature_inventory(catalog)
    assert int(inventory["feature_count"].sum()) == 5
    assert int(inventory["numeric_features"].sum()) == 5


def test_train_prediction_diagnostics_report_support_shift() -> None:
    frame = _synthetic_frame()
    catalog = build_feature_catalog(_manifest(), frame)
    diagnostics = train_prediction_diagnostics(
        frame,
        features=catalog["feature"],
        smd_threshold=0.1,
        outside_support_threshold=0.01,
    ).set_index("feature")

    row = diagnostics.loc["wapor_2025__aeti__season_total"]
    assert row["prediction_outside_train_range_fraction"] > 0
    assert bool(row["flag_any_shift"])


def test_fixed_folds_are_deterministic_and_group_municipalities() -> None:
    frame = _synthetic_frame()
    folds_a = build_fixed_fold_assignments(frame, n_splits=3, random_state=42)
    folds_b = build_fixed_fold_assignments(frame, n_splits=3, random_state=42)

    pd.testing.assert_frame_equal(folds_a, folds_b)
    validate_fixed_fold_assignments(frame, folds_a)

    grouped = (
        folds_a.assign(
            group=folds_a["meta_estado"].astype(str)
            + "|"
            + folds_a["meta_municipio"].astype(str)
        )
        .groupby("group")["fold_municipality_grouped"]
        .nunique()
    )
    assert int(grouped.max()) == 1


def test_ablation_resolution_and_fixed_fold_evaluation() -> None:
    frame = _synthetic_frame()
    catalog = build_feature_catalog(_manifest(), frame)
    folds = build_fixed_fold_assignments(frame, n_splits=3, random_state=42)

    clean_features = resolve_ablation_features(
        catalog,
        {"include_all_clean": True, "numeric_only": True},
    )
    full_features = resolve_ablation_features(
        catalog,
        {
            "include_all_clean": True,
            "include_all_competition": True,
            "numeric_only": True,
        },
    )
    assert len(clean_features) == 4
    assert len(full_features) == 5

    scores = evaluate_fixed_folds(
        frame,
        folds,
        {"clean": clean_features, "full": full_features},
        fold_column="fold_state_stratified",
        regressors={"DummyMean": DummyRegressor(strategy="mean")},
    )
    summary = summarize_fixed_fold_scores(scores)

    assert len(scores) == 6
    assert set(summary["ablation"]) == {"clean", "full"}
    assert set(summary["model"]) == {"DummyMean"}
