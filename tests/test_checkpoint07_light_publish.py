"""Checks for the explicit Checkpoint 07 lightweight sharing allowlist."""

from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

import pytest

from tools import publish_checkpoint_07_light as share


def _write_x(path: Path, rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as out:
        writer = csv.writer(out, lineterminator="\n")
        writer.writerows(rows)


def _local_inputs(root: Path) -> None:
    _write_x(
        root / "data/processed/checkpoint_07/hls_parcel_observations.csv",
        [
            ["ID_POLIGONO", "date", "sensor", "ndvi_mean"],
            ["AGC_001", "2025-04-01", "L30", "0.20"],
            ["AGC_002", "2025-04-01", "S30", "0.55"],
        ],
    )
    _write_x(
        root / "data/processed/checkpoint_07/smap_parcel_daily.csv",
        [
            ["ID_POLIGONO", "date", "overpass", "soil_moisture_m3_m3"],
            ["AGC_001", "2025-04-01", "AM", "0.25"],
            ["AGC_002", "2025-04-01", "PM", "0.31"],
        ],
    )
    hls_report = {
        "source_scenes": 2,
        "source_parcels": 2,
        "panel": {"panel_rows": 2, "parcels_with_observations": 2},
        "chips": {"parcels_with_4_quality_frames": 2},
        "source_path": "C:\\Users\\somebody\\secret.txt",
    }
    smap_report = {
        "hdf5_files": 2,
        "rows": 4,
        "recommended_quality_rows": 3,
        "geolocation_methods": {
            "Soil_Moisture_Retrieval_Data_AM": "EPSG6933_fixed_global",
        },
        "csv": "C:\\Users\\somebody\\file.csv",
    }
    for name, data in (
        ("processing_report.json", hls_report),
        ("smap_processing_report.json", smap_report),
    ):
        (root / "data/processed/checkpoint_07" / name).write_text(
            json.dumps(data), encoding="utf-8"
        )


def test_check_header_excludes_hidden_y_and_siap_2025(tmp_path: Path) -> None:
    for bad in ("RENDIMIENTO_T_HA", "CONJUNTO", "siap_2025__rendimiento"):
        file = tmp_path / "not_safe.csv"
        _write_x(file, [["ID_POLIGONO", bad], ["AGC_001", "12"]])
        with pytest.raises(ValueError, match="banned columns"):
            share._check_header(file)
    good = tmp_path / "safe.csv"
    _write_x(good, [["ID_POLIGONO", "ndvi_mean"], ["AGC_001", "0.5"]])
    assert share._check_header(good)[0] == 2


def test_portable_build_checksum_and_sanitized_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(share, "ROOT", tmp_path)
    _local_inputs(tmp_path)
    manifest = share.build(
        include_highdim=False, max_file_mib=1, max_total_mib=2
    )
    assert len(manifest["files"]) == 2
    assert share.verify() == 2
    assert not any("expanded_parcel_features" in x["name"] for x in manifest["files"])
    for entry in manifest["files"]:
        result = tmp_path / entry["relative_path"]
        assert result.exists()
        with gzip.open(result, "rt", encoding="utf-8") as stream:
            assert next(csv.reader(stream))[0] == "ID_POLIGONO"
    summary_path = tmp_path / share.PORTABLE_REPORTS / share.QC_NAME
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["smap"]["recommended_quality_fraction"] == 0.75
    assert "C:\\Users" not in summary_path.read_text(encoding="utf-8")
    assert "C:\\Users" not in (
        tmp_path / share.PORTABLE_REPORTS / share.MANIFEST_NAME
    ).read_text(encoding="utf-8")


def test_portable_full_feature_export_is_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(share, "ROOT", tmp_path)
    _local_inputs(tmp_path)
    with pytest.raises(FileNotFoundError, match="Process locally first"):
        share.build(include_highdim=True, max_file_mib=1, max_total_mib=3)

    _write_x(
        tmp_path / share.X_FULL,
        [
            ["ID_POLIGONO", "hls_all__ndvi_mean", "smap_AM__soil"],
            ["AGC_001", "0.35", "0.27"],
            ["AGC_002", "0.41", "0.36"],
        ],
    )
    result = share.build(
        include_highdim=True, max_file_mib=1, max_total_mib=3
    )
    assert len(result["files"]) == 3
    assert share.verify() == 3


def test_file_size_guard_prevents_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(share, "ROOT", tmp_path)
    _local_inputs(tmp_path)
    with pytest.raises(ValueError, match="exceeds cap"):
        share.build(
            include_highdim=False,
            max_file_mib=0.00001,
            max_total_mib=1,
        )
    assert not (tmp_path / share.PORTABLE_REPORTS / share.MANIFEST_NAME).exists()


def test_very_small_smoke_summary_is_not_treated_as_final(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(share, "ROOT", tmp_path)
    _local_inputs(tmp_path)
    smoke = tmp_path / "reports/checkpoint_07/highdim/smoke"
    smoke.mkdir(parents=True)
    (smoke / "run_report.json").write_text(
        json.dumps({
            "evaluation": {
                "n_splits": 1,
                "n_models": 4,
                "best_development_model": "PCA_0.8__MLP_16__a1",
                "best_development_pooled_rmse": 0.615971,
            }
        }), encoding="utf-8",
    )
    _write_x(
        smoke / "benchmark_summary.csv",
        [
            [
                "model", "n_splits", "n_rows_repeated", "rmse_pooled",
                "rmse_split_mean", "rmse_split_worst",
            ],
            ["PCA_0.8__MLP_16__a1", "1", "41", "0.615971", "0.615971", "0.615971"],
        ],
    )
    share.build(include_highdim=False, max_file_mib=1, max_total_mib=2)
    summary = json.loads(
        (tmp_path / share.PORTABLE_REPORTS / share.QC_NAME).read_text()
    )
    assert summary["highdim_smoke"]["n_splits"] == 1
    assert "not final" in summary["highdim_smoke"]["warning"]
    assert (
        tmp_path / share.PORTABLE_REPORTS / share.SMOKE_NAME
    ).is_file()
