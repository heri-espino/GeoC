"""Tests for the project-wide data catalog and mini fixture generator."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CATALOG_TOOL = runpy.run_path(ROOT / "tools/build_data_catalog.py")


def test_evenly_spaced_indices_cover_ends() -> None:
    indices = CATALOG_TOOL["evenly_spaced_indices"](100, 5)
    assert indices[0] == 0
    assert indices[-1] == 99
    assert len(indices) == 5


def test_select_tabular_sample_preserves_multiple_parcels() -> None:
    frame = pd.DataFrame(
        {
            "ID_POLIGONO": [f"AGC_{parcel:03d}" for parcel in range(1, 11) for _ in range(10)],
            "fecha_captura": pd.date_range("2025-04-01", periods=100, freq="D"),
            "ndvi_promedio": range(100),
        }
    )
    sample = CATALOG_TOOL["select_tabular_sample"](frame, 24)
    assert len(sample) <= 24
    assert sample["ID_POLIGONO"].nunique() > 1
    assert sample["fecha_captura"].min() < sample["fecha_captura"].max()


def test_catalog_profiles_csv_and_writes_versionable_sample(tmp_path: Path) -> None:
    source = tmp_path / "data/source/tabular/example.csv"
    source.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "ID_POLIGONO": ["AGC_001", "AGC_001", "AGC_002", "AGC_002"],
            "fecha_captura": ["2025-04-01", "2025-04-15", "2025-04-01", "2025-04-15"],
            "value": [1.0, 2.0, None, 4.0],
        }
    ).to_csv(source, index=False)

    catalog = CATALOG_TOOL["build_catalog"](
        tmp_path,
        source_groups={"example": Path("data/source/tabular/example.csv")},
        sample_dir=tmp_path / "data/samples",
        tabular_profile_rows=100,
        sample_rows=3,
    )
    output = tmp_path / "data/data_catalog.json"
    CATALOG_TOOL["write_catalog"](catalog, output)

    saved = json.loads(output.read_text(encoding="utf-8"))
    record = saved["sources"][0]
    assert record["available"] is True
    assert record["file_count"] == 1
    profile = record["representative_profiles"][0]
    assert profile["profile"]["column_count"] == 3
    names = [column["name"] for column in profile["profile"]["columns"]]
    assert names == ["ID_POLIGONO", "fecha_captura", "value"]
    sample_path = tmp_path / profile["sample"]
    assert sample_path.exists()
    assert len(pd.read_csv(sample_path)) <= 3


def test_catalog_tolerates_missing_sources(tmp_path: Path) -> None:
    catalog = CATALOG_TOOL["build_catalog"](
        tmp_path,
        source_groups={"missing": Path("data/raw/external/not_here")},
        sample_dir=None,
    )
    record = catalog["sources"][0]
    assert record["available"] is False
    assert record["file_count"] == 0
    assert record["representative_profiles"] == []


def test_wapor_surrogate_units_are_repaired() -> None:
    """WaPOR NPP metadata must recover the intended UTF-8 superscript."""
    worker = runpy.run_path(ROOT / "tools/_netcdf_catalog_worker.py")

    repaired, changed = worker["repair_text"]("gC/m\udcc2\udcb2/day")

    assert changed is True
    assert repaired == "gC/m\u00b2/day"

