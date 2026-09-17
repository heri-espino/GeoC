"""Regression tests for external-data command-line utilities."""

from __future__ import annotations

import json
import math
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_TOOL = runpy.run_path(ROOT / "tools/build_external_data_manifest.py")
DOWNLOADER = runpy.run_path(ROOT / "tools/download_external_data.py")


def test_manifest_tolerates_missing_sources_and_writes_portable_paths(tmp_path: Path) -> None:
    """The metadata utility must be useful in a checkout with partial data."""
    chirps_dir = tmp_path / "data/raw/external/chirps_v3_daily"
    chirps_dir.mkdir(parents=True)
    (chirps_dir / "chirps_v3_2025-04-01.tif").write_bytes(b"not-a-raster")

    manifest = MANIFEST_TOOL["build_manifest"](tmp_path)
    output = tmp_path / "data/external_manifest.json"
    MANIFEST_TOOL["write_manifest"](manifest, output)

    saved = json.loads(output.read_text(encoding="utf-8"))
    sources = {source["source"]: source for source in saved["sources"]}
    chirps = sources["chirps_diario"]
    assert chirps["available"] is True
    assert chirps["file_count"] == 1
    expected_path = "data/raw/external/chirps_v3_daily/chirps_v3_2025-04-01.tif"
    assert chirps["files"][0]["path"] == expected_path
    assert chirps["date_range_inferred"] == {"min": "2025-04-01", "max": "2025-04-01"}
    assert sources["soilgrids"]["available"] is False


def test_manifest_shrinkage_detects_partial_checkout() -> None:
    """A partial workstation must not silently replace a more complete manifest."""
    existing = {
        "sources": [
            {"source": "inegi_municipios", "available": True, "file_count": 3},
            {"source": "wapor", "available": True, "file_count": 2},
        ]
    }
    candidate = {
        "sources": [
            {"source": "inegi_municipios", "available": True, "file_count": 1},
            {"source": "wapor", "available": True, "file_count": 2},
        ]
    }

    issues = MANIFEST_TOOL["manifest_shrinkage"](existing, candidate)

    assert issues == ["inegi_municipios: file count would shrink (3 -> 1)"]


def test_manifest_shrinkage_allows_equal_or_larger_inventory() -> None:
    """Equal or more complete local inventories should be safe to write."""
    existing = {
        "sources": [
            {"source": "inegi_municipios", "available": True, "file_count": 3},
        ]
    }
    candidate = {
        "sources": [
            {"source": "inegi_municipios", "available": True, "file_count": 3},
        ]
    }

    assert MANIFEST_TOOL["manifest_shrinkage"](existing, candidate) == []


def test_soilgrids_bbox_uses_proj_registered_esri_crs() -> None:
    """SoilGrids' WCS pseudo-EPSG code must not be passed to PyProj."""
    bbox = DOWNLOADER["soilgrids_bbox"]((-98.7, 19.4, -98.1, 20.1))

    assert DOWNLOADER["SOILGRIDS_LOCAL_CRS"] == "ESRI:54052"
    assert DOWNLOADER["SOILGRIDS_WCS_CRS"].endswith("/152160")
    assert all(math.isfinite(value) for value in bbox)
    assert bbox[0] < bbox[2]
    assert bbox[1] < bbox[3]


def test_soilgrids_wcs_request_keeps_the_service_pseudo_epsg(
    tmp_path: Path, monkeypatch
) -> None:
    """The WCS request must retain SoilGrids' documented pseudo-EPSG code."""
    captured: dict[str, object] = {}

    def fake_request_bytes(url: str, **kwargs: object) -> bytes:
        captured["url"] = url
        captured.update(kwargs)
        return b"x" * 1001

    globals_dict = DOWNLOADER["download_soil_layer"].__globals__
    monkeypatch.setitem(globals_dict, "request_bytes", fake_request_bytes)
    output = DOWNLOADER["download_soil_layer"](
        tmp_path, "phh2o", "0-5cm", (-10.0, 1.0, 10.0, 20.0)
    )

    params = dict(captured["params"])
    expected_crs = DOWNLOADER["SOILGRIDS_WCS_CRS"]
    assert output.exists()
    assert params["SUBSETTINGCRS"] == expected_crs
    assert params["OUTPUTCRS"] == expected_crs
