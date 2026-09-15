import numpy as np
import pandas as pd

from geocebada.features import (
    align_temporal_knn,
    aligned_to_wide,
    make_temporal_grid,
    temporal_backend_available,
    temporal_coverage_summary,
)


def _example_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ID_POLIGONO": ["A", "A", "A", "B", "B"],
            "fecha_captura": [
                "2025-04-01",
                "2025-04-11",
                "2025-04-21",
                "2025-04-01",
                "2025-04-21",
            ],
            "sensor": ["S2", "S2", "S2", "S2", "S2"],
            "ndvi_promedio": [0.2, 0.4, 0.8, 0.1, 0.5],
            "evi_promedio": [0.1, 0.2, 0.4, 0.05, 0.25],
            "porcentaje_nubosidad": [0.0, 50.0, 0.0, 0.0, 0.0],
        }
    )


def test_make_temporal_grid_is_inclusive_from_start() -> None:
    grid = make_temporal_grid("2025-04-01", "2025-04-29", freq="14D")
    assert list(grid) == [
        pd.Timestamp("2025-04-01"),
        pd.Timestamp("2025-04-15"),
        pd.Timestamp("2025-04-29"),
    ]


def test_temporal_coverage_summary_reports_gaps_and_months() -> None:
    result = temporal_coverage_summary(
        _example_frame(),
        start="2025-04-01",
        end="2025-04-30",
    ).set_index(["ID_POLIGONO", "sensor"])

    assert result.loc[("A", "S2"), "n_unique_dates"] == 3
    assert result.loc[("A", "S2"), "median_gap_days"] == 10.0
    assert result.loc[("A", "S2"), "max_gap_days"] == 10.0
    assert result.loc[("A", "S2"), "n_2025_04"] == 3


def test_temporal_knn_exact_capture_matches_when_k_one() -> None:
    aligned = align_temporal_knn(
        _example_frame(),
        ["ndvi_promedio", "evi_promedio"],
        grid=[pd.Timestamp("2025-04-11")],
        k=1,
        bandwidth_days=14.0,
        max_distance_days=20.0,
        backend="cpu",
    )
    row = aligned.loc[aligned["ID_POLIGONO"] == "A"].iloc[0]
    assert np.isclose(row["ndvi_promedio"], 0.4)
    assert np.isclose(row["evi_promedio"], 0.2)
    assert row["nearest_gap_days"] == 0.0
    assert row["backend"] == "cpu"


def test_temporal_knn_selects_nearest_valid_neighbor_per_variable() -> None:
    frame = pd.DataFrame(
        {
            "ID_POLIGONO": ["A", "A", "A"],
            "fecha_captura": ["2025-04-10", "2025-04-11", "2025-04-18"],
            "sensor": ["S2", "S2", "S2"],
            "ndvi_promedio": [np.nan, np.nan, 0.75],
            "evi_promedio": [0.20, 0.30, 0.90],
            "porcentaje_nubosidad": [0.0, 0.0, 0.0],
        }
    )

    aligned = align_temporal_knn(
        frame,
        ["ndvi_promedio", "evi_promedio"],
        grid=[pd.Timestamp("2025-04-11")],
        k=1,
        max_distance_days=10.0,
        cloud_weighting=False,
        backend="cpu",
    )

    assert np.isclose(aligned.loc[0, "ndvi_promedio"], 0.75)
    assert np.isclose(aligned.loc[0, "evi_promedio"], 0.30)


def test_cloud_weighting_downweights_cloudy_neighbor() -> None:
    frame = pd.DataFrame(
        {
            "ID_POLIGONO": ["A", "A"],
            "fecha_captura": ["2025-04-10", "2025-04-20"],
            "sensor": ["S2", "S2"],
            "ndvi_promedio": [0.0, 1.0],
            "porcentaje_nubosidad": [90.0, 0.0],
        }
    )
    weighted = align_temporal_knn(
        frame,
        ["ndvi_promedio"],
        grid=[pd.Timestamp("2025-04-15")],
        k=2,
        cloud_weighting=True,
        backend="cpu",
    )
    unweighted = align_temporal_knn(
        frame,
        ["ndvi_promedio"],
        grid=[pd.Timestamp("2025-04-15")],
        k=2,
        cloud_weighting=False,
        backend="cpu",
    )
    assert weighted.loc[0, "ndvi_promedio"] > unweighted.loc[0, "ndvi_promedio"]


def test_aligned_to_wide_keeps_sensor_in_column_name() -> None:
    aligned = pd.DataFrame(
        {
            "ID_POLIGONO": ["A", "A"],
            "sensor": ["S2", "L8"],
            "grid_date": [pd.Timestamp("2025-04-01"), pd.Timestamp("2025-04-01")],
            "ndvi_promedio": [0.2, 0.3],
        }
    )
    wide = aligned_to_wide(aligned, value_columns=["ndvi_promedio"])
    assert set(wide.columns) == {
        "ID_POLIGONO",
        "S2__ndvi_promedio__2025_04_01",
        "L8__ndvi_promedio__2025_04_01",
    }


def test_auto_backend_always_resolves() -> None:
    assert temporal_backend_available("auto") in {"cpu", "gpu"}
