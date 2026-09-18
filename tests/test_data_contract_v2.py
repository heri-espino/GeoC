"""Tests for the Data Contract v2 audit."""

from __future__ import annotations

import runpy
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
AUDIT = runpy.run_path(ROOT / "tools/audit_data_contract_v2.py")


def test_contract_v2_loads() -> None:
    contract = AUDIT["load_contract"](ROOT / "configs/data_contract_v2.yaml")

    assert contract["schema_version"] == 2
    assert contract["identity"]["expected_total_parcels"] == 197
    assert contract["external"]["chirps_daily"]["expected_count"] == 214
    assert contract["external"]["soilgrids"]["expected_count"] == 36
    assert contract["external"]["era5_land"]["required"] is False


def test_strict_day_first_date_parser() -> None:
    frame = pd.Series(["05/02/2025", "31/12/2025", "2025-04-01"])

    parsed = AUDIT["parse_strict_dates"](frame, "%d/%m/%Y")

    assert parsed.iloc[0] == pd.Timestamp("2025-02-05")
    assert parsed.iloc[1] == pd.Timestamp("2025-12-31")
    assert pd.isna(parsed.iloc[2])


def test_date_range_is_closed() -> None:
    result = AUDIT["date_range"](date(2025, 4, 1), date(2025, 4, 3))

    assert result == [date(2025, 4, 1), date(2025, 4, 2), date(2025, 4, 3)]


def test_siap_year_files_keeps_duplicate_copies(tmp_path: Path) -> None:
    pattern = __import__("re").compile(
        r"^Cierre_agricola_mun_(\d{4})(?: \(\d+\))?\.csv$"
    )
    for name in [
        "Cierre_agricola_mun_2023.csv",
        "Cierre_agricola_mun_2025.csv",
        "Cierre_agricola_mun_2025 (1).csv",
        "ignore.csv",
    ]:
        (tmp_path / name).write_text("x\n1\n", encoding="utf-8")

    grouped = AUDIT["siap_year_files"](tmp_path, pattern)

    assert sorted(grouped) == [2023, 2025]
    assert len(grouped[2025]) == 2


def test_soilgrids_conversion_contract_matches_official_units() -> None:
    contract = AUDIT["load_contract"](ROOT / "configs/data_contract_v2.yaml")
    properties = contract["external"]["soilgrids"]["properties"]

    assert properties["phh2o"]["divisor"] == 10
    assert properties["clay"]["divisor"] == 10
    assert properties["sand"]["divisor"] == 10
    assert properties["silt"]["divisor"] == 10
    assert properties["soc"]["divisor"] == 10
    assert properties["nitrogen"]["divisor"] == 100
    assert properties["cec"]["divisor"] == 10
    assert properties["bdod"]["divisor"] == 100
    assert properties["cfvo"]["divisor"] == 10


def test_official_sources_have_no_contract_failures() -> None:
    contract = AUDIT["load_contract"](ROOT / "configs/data_contract_v2.yaml")

    report = AUDIT["build_audit"](ROOT, contract, official_only=True)

    failures = [item for item in report["checks"] if item["status"] == "fail"]
    assert failures == []
