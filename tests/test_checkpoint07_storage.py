"""Metadata-only checks for Checkpoint 07 raw storage inventory."""

from __future__ import annotations

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "checkpoint07_storage",
    Path(__file__).resolve().parents[1] / "tools" / "audit_checkpoint_07_storage.py",
)
assert spec is not None and spec.loader is not None
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_hls_groups_require_correct_sensor_specific_bands(tmp_path: Path) -> None:
    l30 = "HLS.L30.T14QNH.2025093T165916.v2.0"
    s30 = "HLS.S30.T14QNH.2025093T165916.v2.0"
    paths = []
    for band in ("B02", "B03", "B04", "B05", "B06", "B07", "Fmask"):
        path = tmp_path / f"{l30}.{band}.tif"
        path.touch()
        paths.append(path)
    for band in ("B02", "B03", "B04", "B8A", "B11", "Fmask"):
        path = tmp_path / f"{s30}.{band}.tif"
        path.touch()
        paths.append(path)

    result = mod._hls_band_inventory(paths)
    assert result["parsed_granules"] == 2
    assert result["granules_by_sensor"] == {"L30": 1, "S30": 1}
    assert result["granules_missing_prithvi_bands"] == 1
    assert result["missing_examples"][0]["missing"] == ["B12"]


def test_inventory_scans_file_sizes_without_reading_contents(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    (raw / "hls_v2").mkdir(parents=True)
    tif = raw / "hls_v2" / "HLS.L30.T14QNH.2025093T165916.v2.0.B02.tif"
    tif.write_bytes(b"0123456789")
    (raw / "acquisition_manifests").mkdir()
    (raw / "acquisition_manifests" / "source_status.json").write_text(
        '{"hls": {"state": "complete"}}', encoding="utf-8"
    )
    result = mod.build_inventory(raw, tmp_path / "models")
    assert result["raw"]["sources"]["hls_v2"]["files"] == 1
    assert result["raw"]["total_files"] == 2
    assert result["source_status"]["hls"]["state"] == "complete"
    assert result["hls"]["parsed_granules"] == 1
    assert result["hls"]["granules_missing_prithvi_bands"] == 1
    assert result["sample_raster_headers"] == []
