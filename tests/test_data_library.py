from pathlib import Path

import pandas as pd

from geocebada.data import (
    build_raster_inventory,
    concatenate_csvs,
    load_yield_split,
    partition_yield_split,
)


def test_official_yield_split_contract() -> None:
    frame = load_yield_split()
    train, prediction = partition_yield_split(frame)

    assert len(frame) == 197
    assert len(train) == 138
    assert len(prediction) == 59
    assert train["RENDIMIENTO_T_HA"].notna().all()
    assert prediction["RENDIMIENTO_T_HA"].isna().all()


def test_concatenate_csvs_tracks_provenance(tmp_path: Path) -> None:
    pd.DataFrame({"value": [1]}).to_csv(tmp_path / "a.csv", index=False)
    pd.DataFrame({"value": [2]}).to_csv(tmp_path / "b.csv", index=False)

    frame = concatenate_csvs(tmp_path, source_column="source_file")

    assert frame["value"].tolist() == [1, 2]
    assert frame["source_file"].tolist() == ["a.csv", "b.csv"]


def test_build_raster_inventory_parses_official_names(tmp_path: Path) -> None:
    for name in ["PREC_2022_01.tif", "Tmin_2023_12.tif", "ignore.tif"]:
        (tmp_path / name).touch()

    inventory = build_raster_inventory(tmp_path)

    assert inventory["name"].tolist() == ["PREC_2022_01.tif", "Tmin_2023_12.tif"]
    assert inventory["variable"].tolist() == ["PREC", "Tmin"]
    assert inventory["year"].tolist() == [2022, 2023]
    assert inventory["month"].tolist() == [1, 12]
