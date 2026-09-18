"""Tests for shared GeoCebada integration-fixture selection/helpers."""

from __future__ import annotations

import runpy
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = runpy.run_path(ROOT / "tools/build_integration_fixtures.py")


def synthetic_metrics() -> pd.DataFrame:
    rows = []
    states = ["Hidalgo", "Puebla", "Tlaxcala"]
    splits = ["ENTRENAMIENTO", "PREDICCION"]
    counter = 1
    for state in states:
        for split_name in splits:
            for local_idx in range(4):
                rows.append(
                    {
                        "ID_POLIGONO": f"AGC_{counter:03d}",
                        "Estado": state,
                        "Municipio": f"M{counter}",
                        "CONJUNTO": split_name,
                        "area_ha": float(local_idx + 1),
                        "basic_cloud_mean": float(local_idx * 10),
                        "basic_missing_mean": float(local_idx) / 10,
                        "basic_sensor_count": 2,
                        "basic_row_count": 100 + local_idx,
                        "pro_cloud_mean": float(30 - local_idx * 5),
                        "pro_missing_mean": float(3 - local_idx) / 10,
                        "pro_row_count": 50 + local_idx,
                    }
                )
                counter += 1
    return pd.DataFrame(rows)


def test_representative_ids_choose_two_per_state_split() -> None:
    metrics = synthetic_metrics()

    selected = FIXTURES["representative_ids"](
        metrics,
        per_stratum=2,
        forced_ids=["AGC_001", "AGC_010"],
    )

    chosen = metrics.loc[metrics["ID_POLIGONO"].isin(selected)]
    counts = chosen.groupby(["Estado", "CONJUNTO"]).size()

    assert len(selected) == 12
    assert len(set(selected)) == 12
    assert set(counts) == {2}
    assert {"AGC_001", "AGC_010"}.issubset(selected)


def test_representative_ids_are_deterministic() -> None:
    metrics = synthetic_metrics()

    first = FIXTURES["representative_ids"](metrics, per_stratum=2)
    second = FIXTURES["representative_ids"](
        metrics.sample(frac=1.0, random_state=42),
        per_stratum=2,
    )

    assert first == second


def test_sample_longitudinal_preserves_selected_ids_and_sensors() -> None:
    rows = []
    for parcel in ["AGC_001", "AGC_002", "AGC_999"]:
        for sensor in ["Landsat", "Sentinel-2"]:
            for day in range(1, 11):
                rows.append(
                    {
                        "ID_POLIGONO": parcel,
                        "fecha_captura": f"{day:02d}/04/2025",
                        "sensor": sensor,
                        "porcentaje_nubosidad": day,
                        "ndvi_promedio": float(day),
                    }
                )
    frame = pd.DataFrame(rows)

    sample = FIXTURES["sample_longitudinal"](
        frame,
        {"AGC_001", "AGC_002"},
        rows_per_sensor=3,
    )

    assert set(sample["ID_POLIGONO"]) == {"AGC_001", "AGC_002"}
    assert set(sample["sensor"]) == {"Landsat", "Sentinel-2"}
    assert len(sample) == 12


def test_canonicalize_siap_frame_accepts_historical_crop_alias() -> None:
    frame = pd.DataFrame(
        {
            "Idestado": [21],
            "Idmunicipio": [19],
            "Nomcultivo Sin Um": ["Cebada grano"],
            "Rendimiento": [4.2],
        }
    )

    result = FIXTURES["canonicalize_siap_frame"](
        frame,
        ["Nomcultivo", "Nomcultivo Sin Um"],
    )

    assert "Nomcultivo" in result.columns
    assert result.loc[0, "Nomcultivo"] == "Cebada grano"
    assert result.loc[0, "cvegeo"] == "21019"


def test_rank_matrix_is_finite_with_missing_values() -> None:
    frame = pd.DataFrame(
        {
            "a": [1.0, np.nan, 3.0],
            "b": [np.nan, np.nan, np.nan],
        }
    )

    matrix = FIXTURES["_rank_matrix"](frame, ["a", "b"])

    assert matrix.shape == (3, 2)
    assert np.isfinite(matrix).all()


def test_integration_fixture_contract_is_declared() -> None:
    contract = FIXTURES["load_contract"](ROOT / "configs/data_contract_v2.yaml")
    settings = contract["integration_fixtures"]

    assert settings["expected_parcels"] == 12
    assert settings["parcels_per_state_split"] == 2
    assert {"AGC_020", "AGC_048", "AGC_129"}.issubset(settings["forced_ids"])
