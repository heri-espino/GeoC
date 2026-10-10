"""Synthetic coverage for streaming Checkpoint 07A HLS and SMAP processing."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pytest

from geocebada.data.checkpoint07_hls import (
    band_valid,
    choose_prithvi_scenes,
    clear_land_mask,
    parse_scene_name,
    run_chips,
    run_panel,
    scan_hls,
)
from geocebada.data.checkpoint07_smap import (
    _parcel_nearest_cells,
    _usable_moisture,
    parse_smap_day,
)


def test_hls_date_parser_and_sensor_band_semantics() -> None:
    got = parse_scene_name("HLS.S30.T14QNH.2025093T165916.v2.0.B8A.tif")
    assert got == (
        "HLS.S30.T14QNH.2025093T165916.v2.0",
        "S30",
        date(2025, 4, 3),
        "B8A",
    )
    assert parse_scene_name("random.tif") is None


def test_hls_fmask_and_reflectance_reject_cloud_and_saturation() -> None:
    qa = np.array([[0, 2, 4], [8, 16, 32], [64, 255, 0]], dtype=np.uint8)
    clear = clear_land_mask(qa)
    assert clear.tolist() == [
        [True, False, False],
        [False, False, False],
        [True, False, True],
    ]
    raw = np.ma.masked_array(
        [1000, 12000, -9999, -2000],
        mask=[False, False, True, False],
    )
    good = band_valid(raw, np.ones(4, dtype=bool))
    assert good.tolist() == [True, False, False, False]


def test_prithvi_selection_requires_all_four_windows_and_no_y() -> None:
    observations = [
        {
            "date": day,
            "scene_id": f"s_{i}",
            "valid_fraction": coverage,
            "n_pixels_all_bands_valid": 20,
        }
        for i, (day, coverage) in enumerate(
            [
                ("2025-04-20", 0.6),
                ("2025-05-20", 0.85),
                ("2025-06-10", 0.7),
                ("2025-08-12", 0.8),
                ("2025-09-20", 0.9),
            ]
        )
    ]
    choices = choose_prithvi_scenes(observations)
    assert [item["date"] for item in choices if item] == [
        "2025-05-20", "2025-06-10", "2025-08-12", "2025-09-20"
    ]
    choices = choose_prithvi_scenes(observations[:-1])
    assert choices[-1] is None


def test_smap_quality_and_nearest_grid() -> None:
    assert _usable_moisture(0.28, 0)
    assert _usable_moisture(0.28, 8)
    assert not _usable_moisture(0.28, 1)
    assert not _usable_moisture(-9999, 0)
    assert parse_smap_day("SMAP_L3_SM_P_E_20250401_R19240_002.h5") == date(
        2025, 4, 1
    )
    from shapely.geometry import Point

    lat = np.array([[19.0, 19.0], [20.0, 20.0]])
    lon = np.array([[-99.0, -98.0], [-99.0, -98.0]])
    cells = _parcel_nearest_cells(lat, lon, [("A", Point(-98.02, 19.98))])
    row, col, distance = cells["A"]
    assert (row, col) == (1, 1)
    assert distance < 5


def _make_four_scenes(tmp_path: Path):
    import rasterio
    from pyproj import Transformer
    from rasterio.transform import from_origin
    from shapely.geometry import box
    from shapely.ops import transform

    crs = "EPSG:32614"
    source_transform = from_origin(500_000, 2_200_000, 30, 30)
    dates = ["2025110", "2025165", "2025225", "2025270"]
    for i, doy in enumerate(dates):
        scene_name = f"HLS.L30.T14QNH.{doy}T165916.v2.0"
        for band in ("B02", "B03", "B04", "B05", "B06", "B07", "Fmask"):
            image = np.full(
                (64, 64), 1000 + i * 50, dtype=np.int16
            )
            if band == "B05":
                image += 1400
            if band == "Fmask":
                image = np.zeros((64, 64), dtype=np.uint8)
                image[0:3, :] = 2
            with rasterio.open(
                tmp_path / f"{scene_name}.{band}.tif",
                "w",
                driver="GTiff",
                height=64,
                width=64,
                count=1,
                dtype=image.dtype,
                crs=crs,
                transform=source_transform,
                nodata=255 if band == "Fmask" else -9999,
            ) as dst:
                dst.write(image, 1)

    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    polygon_utm = box(500_500, 2_199_100, 500_850, 2_199_450)
    polygon = transform(transformer.transform, polygon_utm)
    return [("AGC_TEST", polygon)]


def test_hls_panel_and_four_frame_chip_tiny_geotiffs(tmp_path: Path) -> None:
    pytest.importorskip("rasterio")
    parcels = _make_four_scenes(tmp_path)
    scenes, missing = scan_hls(tmp_path)
    assert len(scenes) == 4
    assert not missing

    destination = tmp_path / "processed"
    metrics = run_panel(scenes, parcels, destination)
    assert metrics["panel_rows"] == 4
    assert metrics["parcels_with_observations"] == 1
    resumed = run_panel(scenes, parcels, destination)
    assert resumed["reused_scenes"] == 4

    chip_report = run_chips(scenes, parcels, destination)
    assert chip_report["parcels_with_4_quality_frames"] == 1
    filename = destination / "prithvi_chips" / "AGC_TEST.npz"
    assert filename.is_file()
    with np.load(filename) as data:
        assert data["hls_dn"].shape == (4, 6, 224, 224)
        assert data["valid_mask"].shape == (4, 224, 224)
        assert data["parcel_mask"].shape == (224, 224)
        assert data["valid_mask"].sum() > 0
        assert data["hls_dn"].dtype == np.int16
        assert data["parcel_mask"].sum() > 0


def test_smap_hdf5_small_grid(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    from shapely.geometry import Point

    root = tmp_path / "raw"
    root.mkdir()
    file = root / "SMAP_L3_SM_P_E_20250401_R19240_002.h5"
    with h5py.File(file, "w") as h5:
        for suffix in ("AM", "PM"):
            group = h5.create_group(
                f"Soil_Moisture_Retrieval_Data_{suffix}"
            )
            group.create_dataset(
                "latitude", data=np.array([[19., 19.], [20., 20.]])
            )
            group.create_dataset(
                "longitude", data=np.array([[-99., -98.], [-99., -98.]])
            )
            group.create_dataset(
                "soil_moisture",
                data=np.array([[0.2, 0.3], [0.4, 0.25]]),
            )
            group.create_dataset(
                "retrieval_qual_flag",
                data=np.zeros((2, 2), dtype=np.uint16),
            )

    from geocebada.data.checkpoint07_smap import run_smap

    report = run_smap(
        root, [("AGC_TEST", Point(-98.05, 19.98))], tmp_path / "output"
    )
    assert report["hdf5_files"] == 1
    assert report["rows"] == 2
    assert report["recommended_quality_rows"] == 2



def test_smap_v006_missing_geolocation_uses_projected_global_grid(tmp_path: Path) -> None:
    """Reproduce real 2025 HDF5s lacking latitude/longitude arrays.

    The full 1624x3856 shape is represented by sparse chunked datasets, not
    25MB dense global arrays. AM fields have no suffix, PM fields have _pm.
    """
    h5py = pytest.importorskip("h5py")
    from pyproj import Transformer
    from shapely.geometry import Point

    from geocebada.data.checkpoint07_smap import (
        EASE2_GLOBAL_RES,
        EASE2_GLOBAL_SHAPE,
        EASE2_GLOBAL_UL_X,
        EASE2_GLOBAL_UL_Y,
        run_smap,
    )

    raw = tmp_path / "raw"
    raw.mkdir()
    path = raw / "SMAP_L3_SM_P_E_20250520_R19240_001.h5"
    row, col = 600, 860
    x = EASE2_GLOBAL_UL_X + (col + 0.5) * EASE2_GLOBAL_RES
    y = EASE2_GLOBAL_UL_Y - (row + 0.5) * EASE2_GLOBAL_RES
    lon, lat = Transformer.from_crs(
        "EPSG:6933", "EPSG:4326", always_xy=True
    ).transform(x, y)
    with h5py.File(path, "w") as h5:
        for suffix, moisture in (("AM", 0.23), ("PM", 0.27)):
            group = h5.create_group(f"Soil_Moisture_Retrieval_Data_{suffix}")
            value_name = "soil_moisture_pm" if suffix == "PM" else "soil_moisture"
            flag_name = (
                "retrieval_qual_flag_pm" if suffix == "PM"
                else "retrieval_qual_flag"
            )
            value = group.create_dataset(
                value_name, shape=EASE2_GLOBAL_SHAPE, dtype="float32",
                chunks=(32, 32), fillvalue=-9999,
            )
            flag = group.create_dataset(
                flag_name, shape=EASE2_GLOBAL_SHAPE, dtype="uint16",
                chunks=(32, 32), fillvalue=65534,
            )
            value[row, col] = moisture
            flag[row, col] = 0 if suffix == "AM" else 8

    out = tmp_path / "processed"
    report = run_smap(raw, [("AGC_X", Point(lon, lat))], out)
    assert report["rows"] == 2
    assert report["recommended_quality_rows"] == 2
    assert set(report["geolocation_methods"].values()) == {"EPSG6933_fixed_global"}
    assert report["unique_grid_cells"] == {
        "Soil_Moisture_Retrieval_Data_AM": 1,
        "Soil_Moisture_Retrieval_Data_PM": 1,
    }
    import pandas as pd

    panel = pd.read_csv(out / "smap_parcel_daily.csv")
    assert set(panel["overpass"]) == {"AM", "PM"}
    assert set(panel["smap_grid_row"]) == {row}
    assert set(panel["smap_grid_col"]) == {col}
    assert np.allclose(
        panel.sort_values("overpass")["soil_moisture_m3_m3"], [0.23, 0.27],
    )


def test_smap_unknown_geolocation_shape_rejected(tmp_path: Path) -> None:
    """Never silently treat subregion/polar grids as global EASE2."""
    h5py = pytest.importorskip("h5py")
    from shapely.geometry import Point

    from geocebada.data.checkpoint07_smap import run_smap

    raw = tmp_path / "raw"
    raw.mkdir()
    with h5py.File(raw / "SMAP_L3_SM_P_E_20250520_R19240_001.h5", "w") as h5:
        g = h5.create_group("Soil_Moisture_Retrieval_Data_AM")
        g.create_dataset("soil_moisture", data=np.full((2, 2), 0.3))
        g.create_dataset(
            "retrieval_qual_flag", data=np.zeros((2, 2), dtype=np.uint16),
        )
    with pytest.raises(ValueError, match="Cannot geolocate SMAP"):
        run_smap(raw, [("AGC_X", Point(-99.0, 19.0))], tmp_path / "processed")


def test_smap_fill_before_scaling_and_quality_bits(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    from geocebada.data.checkpoint07_smap import _dataset_scalar

    with h5py.File(tmp_path / "packed.h5", "w") as h5:
        values = h5.create_dataset(
            "soil_moisture", data=np.array([[-9999, 230]], dtype=np.int16),
        )
        values.attrs["_FillValue"] = -9999
        values.attrs["scale_factor"] = 0.001
        assert np.isnan(_dataset_scalar(values, 0, 0))
        assert np.isclose(_dataset_scalar(values, 0, 1), 0.23)
    assert _usable_moisture(0.23, 8)
    assert not _usable_moisture(0.23, 1)
    assert not _usable_moisture(0.23, 2)
    assert not _usable_moisture(0.23, 4)
    assert not _usable_moisture(0.23, 65534)
