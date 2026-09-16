#!/usr/bin/env python
"""Download external GeoCebada covariates for the 2025 barley cycle.

The script derives a WGS84 bounding box from the official parcel layer and downloads only
that area whenever the source supports spatial subsetting.

Automated sources
-----------------
- INEGI municipal GeoJSON: Hidalgo, Puebla and Tlaxcala.
- CHIRPS v3 RNL daily precipitation: 2025-04-01 through 2025-10-31 by default.
- SoilGrids 250 m median predictions: selected properties and depths via WCS.
- ERA5-Land hourly fields: one request per month through the Copernicus CDS API.
- WaPOR v3 Level-1 dekadal NPP and AETI through ``wapordl``.

Manual sources deliberately excluded
-------------------------------------
- SIAP monthly agricultural-progress web interface.
- FIRA Agrocostos interactive web interface.
- INEGI CEM 4.0 15 m interactive area download.

Examples
--------
Public sources that do not require a CDS account::

    python tools/download_external_data.py --sources inegi chirps soilgrids wapor

Everything, including ERA5-Land::

    python tools/download_external_data.py --sources all

ERA5-Land only::

    python tools/download_external_data.py --sources era5

The default output directory is ``data/raw/external`` and is ignored by git.
"""

from __future__ import annotations

import argparse
import calendar
import concurrent.futures as futures
import json
import tempfile
import time
from collections.abc import Iterable
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARCELS = ROOT / "data/source/geospatial/Parcelas_Reto_AGC_CONJUNTO_70_30.zip"
DEFAULT_OUTPUT = ROOT / "data/raw/external"

STATE_CODES = {
    "hidalgo": "13",
    "puebla": "21",
    "tlaxcala": "29",
}

SOIL_PROPERTIES = [
    "phh2o",
    "clay",
    "sand",
    "silt",
    "soc",
    "nitrogen",
    "cec",
    "bdod",
    "cfvo",
]
SOIL_DEPTHS = ["0-5cm", "5-15cm", "15-30cm", "30-60cm"]

ERA5_VARIABLES = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "surface_solar_radiation_downwards",
    "potential_evaporation",
    "total_evaporation",
    "total_precipitation",
    "surface_runoff",
    "volumetric_soil_water_layer_1",
    "volumetric_soil_water_layer_2",
    "volumetric_soil_water_layer_3",
    "volumetric_soil_water_layer_4",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
]

CHIRPS_HTTP_ROOT = "https://data.source.coop/wfp/chirps-rnl-daily/v3.0"
USER_AGENT = "GeoCebada/0.1 external-data-downloader"


def log(message: str) -> None:
    """Print a timestamped progress message."""
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {message}", flush=True)


def daterange(start: date, end: date) -> Iterable[date]:
    """Yield every date in the closed interval ``[start, end]``."""
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def ensure_parent(path: Path) -> None:
    """Create a file's parent directory if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)


def request_bytes(
    url: str,
    *,
    timeout: int = 120,
    params: list[tuple[str, str]] | dict[str, Any] | None = None,
) -> bytes:
    """GET bytes with bounded exponential-backoff retries."""
    last_error: Exception | None = None
    headers = {"User-Agent": USER_AGENT}
    for attempt in range(5):
        try:
            response = requests.get(
                url,
                params=params,
                headers=headers,
                timeout=timeout,
            )
            response.raise_for_status()
            return response.content
        except requests.RequestException as exc:
            last_error = exc
            wait_seconds = 2**attempt
            log(f"Retry {attempt + 1}/5 for {url}: {exc}; waiting {wait_seconds}s")
            time.sleep(wait_seconds)
    raise RuntimeError(f"Could not download {url}") from last_error


def parcel_bbox(
    parcel_path: Path,
    margin_deg: float,
) -> tuple[float, float, float, float]:
    """Read official parcels and return a buffered WGS84 bbox."""
    try:
        import geopandas as gpd
    except ImportError as exc:
        raise RuntimeError(
            "GeoPandas is required. Install GeoCebada with `pip install -e '.[geo]'`."
        ) from exc

    if not parcel_path.exists():
        raise FileNotFoundError(
            f"Parcel file not found: {parcel_path}. Use --parcels or --bbox explicitly."
        )

    uri = f"zip://{parcel_path.resolve()}" if parcel_path.suffix.lower() == ".zip" else parcel_path
    frame = gpd.read_file(uri)
    if frame.crs is None:
        raise RuntimeError(
            "The parcel layer has no CRS. The downloader will not guess one; pass "
            "--bbox WEST SOUTH EAST NORTH after verifying the source CRS."
        )

    frame = frame.to_crs(4326)
    west, south, east, north = map(float, frame.total_bounds)
    return (
        west - margin_deg,
        south - margin_deg,
        east + margin_deg,
        north + margin_deg,
    )


# ---------------------------------------------------------------------------
# INEGI municipal polygons
# ---------------------------------------------------------------------------


def download_inegi_state(output: Path, state: str, code: str) -> Path:
    """Download one state's municipal polygons from the INEGI GeoJSON service."""
    destination = output / "inegi_municipios" / f"{state}_municipios.geojson"
    if destination.exists() and destination.stat().st_size > 100:
        log(f"INEGI skip: {destination.name}")
        return destination

    url = f"https://gaia.inegi.org.mx/wscatgeo/v2/geo/mgem/{code}"
    content = request_bytes(url)
    ensure_parent(destination)

    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"INEGI returned invalid JSON for {state}") from exc
    if not payload:
        raise RuntimeError(f"INEGI returned an empty payload for {state}")

    destination.write_bytes(content)
    log(f"INEGI done: {destination.name}")
    return destination


def download_inegi(output: Path, workers: int) -> list[Path]:
    """Download Hidalgo, Puebla and Tlaxcala municipal polygons in parallel."""
    with futures.ThreadPoolExecutor(max_workers=min(workers, 3)) as executor:
        jobs = [
            executor.submit(download_inegi_state, output, state, code)
            for state, code in STATE_CODES.items()
        ]
        return [job.result() for job in futures.as_completed(jobs)]


# ---------------------------------------------------------------------------
# CHIRPS v3 RNL daily COGs
# ---------------------------------------------------------------------------


def chirps_url(day: date) -> str:
    """Return the public WFP/Source Cooperative CHIRPS v3 RNL COG URL."""
    stamp = day.strftime("%Y.%m.%d")
    filename = f"chirps-v3.0.rnl.{stamp}.tif"
    return (
        f"{CHIRPS_HTTP_ROOT}/{day.year}/{day.month:02d}/{day.day:02d}/{filename}"
    )


def crop_remote_cog(
    url: str,
    destination: Path,
    bbox: tuple[float, float, float, float],
) -> Path:
    """Read just ``bbox`` from a remote COG, with a full-download fallback."""
    if destination.exists() and destination.stat().st_size > 100:
        return destination

    try:
        import rasterio
        from rasterio.windows import from_bounds
    except ImportError as exc:
        raise RuntimeError(
            "Rasterio is required. Install GeoCebada with `pip install -e '.[geo]'`."
        ) from exc

    west, south, east, north = bbox
    ensure_parent(destination)

    def write_crop(source: str | Path) -> None:
        with rasterio.open(source) as src:
            window = from_bounds(west, south, east, north, src.transform)
            window = window.round_offsets().round_lengths()
            array = src.read(window=window, boundless=True)
            profile = src.profile.copy()
            profile.update(
                width=array.shape[2],
                height=array.shape[1],
                transform=src.window_transform(window),
                compress="deflate",
            )
            with rasterio.open(destination, "w", **profile) as dst:
                dst.write(array)

    try:
        write_crop(url)
    except Exception as remote_error:
        log(f"CHIRPS range-read fallback for {Path(url).name}: {remote_error}")
        with tempfile.TemporaryDirectory() as temp_dir:
            temporary = Path(temp_dir) / Path(url).name
            temporary.write_bytes(request_bytes(url, timeout=300))
            write_crop(temporary)

    return destination


def download_chirps(
    output: Path,
    bbox: tuple[float, float, float, float],
    start: date,
    end: date,
    workers: int,
) -> list[Path]:
    """Download daily CHIRPS v3 RNL precipitation clipped to the challenge area."""
    tasks: list[tuple[str, Path]] = []
    for day in daterange(start, end):
        destination = output / "chirps_v3_daily" / f"chirps_v3_{day.isoformat()}.tif"
        tasks.append((chirps_url(day), destination))

    outputs: list[Path] = []
    with futures.ThreadPoolExecutor(max_workers=workers) as executor:
        jobs = {
            executor.submit(crop_remote_cog, url, destination, bbox): destination
            for url, destination in tasks
        }
        for index, job in enumerate(futures.as_completed(jobs), 1):
            outputs.append(job.result())
            if index % 20 == 0 or index == len(jobs):
                log(f"CHIRPS: {index}/{len(jobs)}")
    return outputs


# ---------------------------------------------------------------------------
# SoilGrids 250 m WCS
# ---------------------------------------------------------------------------


def soilgrids_bbox(
    bbox: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Transform a WGS84 bbox to SoilGrids' Interrupted Goode Homolosine CRS."""
    try:
        from pyproj import Transformer
    except ImportError as exc:
        raise RuntimeError(
            "PyProj is required. Install GeoCebada with `pip install -e '.[geo]'`."
        ) from exc

    west, south, east, north = bbox
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:152160", always_xy=True)
    corners = [
        transformer.transform(west, south),
        transformer.transform(west, north),
        transformer.transform(east, south),
        transformer.transform(east, north),
    ]
    xs = [point[0] for point in corners]
    ys = [point[1] for point in corners]
    return min(xs), min(ys), max(xs), max(ys)


def download_soil_layer(
    output: Path,
    property_name: str,
    depth: str,
    bbox_native: tuple[float, float, float, float],
) -> Path:
    """Download one SoilGrids median-prediction layer via WCS 2.0.1."""
    destination = output / "soilgrids_250m" / f"{property_name}_{depth}_Q0.5.tif"
    if destination.exists() and destination.stat().st_size > 100:
        return destination

    xmin, ymin, xmax, ymax = bbox_native
    url = "https://maps.isric.org/mapserv"
    params = [
        ("map", f"/map/{property_name}.map"),
        ("SERVICE", "WCS"),
        ("VERSION", "2.0.1"),
        ("REQUEST", "GetCoverage"),
        ("COVERAGEID", f"{property_name}_{depth}_Q0.5"),
        ("FORMAT", "GEOTIFF_INT16"),
        ("SUBSET", f"X({xmin},{xmax})"),
        ("SUBSET", f"Y({ymin},{ymax})"),
        ("SUBSETTINGCRS", "http://www.opengis.net/def/crs/EPSG/0/152160"),
        ("OUTPUTCRS", "http://www.opengis.net/def/crs/EPSG/0/152160"),
    ]

    content = request_bytes(url, timeout=300, params=params)
    prefix = content[:500].lower()
    if b"exception" in prefix or b"<html" in prefix or len(content) < 1000:
        raise RuntimeError(
            f"SoilGrids WCS failed for {property_name} {depth}: {content[:250]!r}"
        )

    ensure_parent(destination)
    destination.write_bytes(content)
    return destination


def download_soilgrids(
    output: Path,
    bbox: tuple[float, float, float, float],
    workers: int,
) -> list[Path]:
    """Download selected SoilGrids properties and depths in parallel."""
    native_bbox = soilgrids_bbox(bbox)
    tasks = [(prop, depth) for prop in SOIL_PROPERTIES for depth in SOIL_DEPTHS]
    outputs: list[Path] = []

    with futures.ThreadPoolExecutor(max_workers=min(workers, 4)) as executor:
        jobs = {
            executor.submit(
                download_soil_layer,
                output,
                prop,
                depth,
                native_bbox,
            ): (prop, depth)
            for prop, depth in tasks
        }
        for index, job in enumerate(futures.as_completed(jobs), 1):
            outputs.append(job.result())
            if index % 4 == 0 or index == len(jobs):
                log(f"SoilGrids: {index}/{len(jobs)}")
    return outputs


# ---------------------------------------------------------------------------
# ERA5-Land via Copernicus CDS
# ---------------------------------------------------------------------------


def iter_months(start: date, end: date) -> Iterable[tuple[int, int]]:
    """Yield year/month pairs intersecting the closed date interval."""
    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        yield year, month
        if month == 12:
            year, month = year + 1, 1
        else:
            month += 1


def era5_month(
    output: Path,
    year: int,
    month: int,
    bbox: tuple[float, float, float, float],
    start: date,
    end: date,
) -> Path:
    """Submit and download one monthly ERA5-Land request."""
    try:
        import cdsapi
    except ImportError as exc:
        raise RuntimeError(
            "cdsapi is required for ERA5-Land. Install `pip install cdsapi` and configure "
            "your Copernicus CDS credentials first."
        ) from exc

    destination = output / "era5_land" / f"era5_land_{year}_{month:02d}.nc"
    if destination.exists() and destination.stat().st_size > 1000:
        return destination

    first = max(start, date(year, month, 1))
    month_end = date(year, month, calendar.monthrange(year, month)[1])
    last = min(end, month_end)
    days = [f"{value:02d}" for value in range(first.day, last.day + 1)]
    west, south, east, north = bbox

    request = {
        "variable": ERA5_VARIABLES,
        "year": str(year),
        "month": f"{month:02d}",
        "day": days,
        "time": [f"{hour:02d}:00" for hour in range(24)],
        "area": [north, west, south, east],
        "data_format": "netcdf",
        "download_format": "unarchived",
    }

    ensure_parent(destination)
    client = cdsapi.Client()
    result = client.retrieve("reanalysis-era5-land", request)
    result.download(str(destination))
    return destination


def download_era5(
    output: Path,
    bbox: tuple[float, float, float, float],
    start: date,
    end: date,
    workers: int,
) -> list[Path]:
    """Download ERA5-Land monthly chunks with conservative CDS parallelism."""
    months = list(iter_months(start, end))
    outputs: list[Path] = []
    with futures.ThreadPoolExecutor(max_workers=min(workers, 2)) as executor:
        jobs = {
            executor.submit(era5_month, output, year, month, bbox, start, end): (year, month)
            for year, month in months
        }
        for index, job in enumerate(futures.as_completed(jobs), 1):
            outputs.append(job.result())
            log(f"ERA5-Land: {index}/{len(jobs)}")
    return outputs


# ---------------------------------------------------------------------------
# WaPOR v3 through wapordl
# ---------------------------------------------------------------------------


def download_wapor(
    output: Path,
    bbox: tuple[float, float, float, float],
    start: date,
    end: date,
) -> list[Path]:
    """Download global WaPOR v3 Level-1 NPP and actual ET/interception."""
    try:
        from wapordl import wapor_map
    except ImportError as exc:
        raise RuntimeError(
            "wapordl is required for WaPOR. Install `pip install wapordl>=1.2`."
        ) from exc

    folder = output / "wapor_v3"
    folder.mkdir(parents=True, exist_ok=True)
    region = list(bbox)
    period = [start.isoformat(), end.isoformat()]
    outputs: list[Path] = []

    for variable in ("L1-NPP-D", "L1-AETI-D"):
        log(f"WaPOR downloading {variable}")
        result = wapor_map(region, variable, period, str(folder), extension=".nc")
        outputs.append(Path(result))
    return outputs


# ---------------------------------------------------------------------------
# CLI / orchestration
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["inegi", "chirps", "soilgrids", "wapor"],
        choices=["all", "inegi", "chirps", "soilgrids", "era5", "wapor"],
        help="Sources to download. 'all' also includes ERA5-Land.",
    )
    parser.add_argument("--start", default="2025-04-01")
    parser.add_argument("--end", default="2025-10-31")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--margin-deg", type=float, default=0.10)
    parser.add_argument("--parcels", type=Path, default=DEFAULT_PARCELS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        metavar=("WEST", "SOUTH", "EAST", "NORTH"),
        help="Override parcel-derived WGS84 bbox.",
    )
    return parser.parse_args()


def main() -> int:
    """Run selected source downloaders and write a manifest."""
    args = parse_args()
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    if start > end:
        raise ValueError("--start must be earlier than or equal to --end")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")

    selected = set(args.sources)
    if "all" in selected:
        selected = {"inegi", "chirps", "soilgrids", "era5", "wapor"}

    bbox = tuple(args.bbox) if args.bbox else parcel_bbox(args.parcels, args.margin_deg)
    output = args.out.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    log(f"Output: {output}")
    log(f"WGS84 bbox: {bbox}")
    log(f"Period: {start} -> {end}")
    log(f"Sources: {sorted(selected)}")

    manifest: dict[str, Any] = {
        "created_at": datetime.now().isoformat(),
        "bbox_wgs84": list(bbox),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "sources": sorted(selected),
        "files": {},
        "errors": {},
    }

    source_jobs: dict[futures.Future[list[Path]], str] = {}
    with futures.ThreadPoolExecutor(max_workers=min(len(selected), 5)) as executor:
        if "inegi" in selected:
            source_jobs[executor.submit(download_inegi, output, args.workers)] = "inegi"
        if "chirps" in selected:
            source_jobs[
                executor.submit(download_chirps, output, bbox, start, end, args.workers)
            ] = "chirps"
        if "soilgrids" in selected:
            source_jobs[
                executor.submit(download_soilgrids, output, bbox, args.workers)
            ] = "soilgrids"
        if "era5" in selected:
            source_jobs[
                executor.submit(download_era5, output, bbox, start, end, args.workers)
            ] = "era5"
        if "wapor" in selected:
            source_jobs[executor.submit(download_wapor, output, bbox, start, end)] = "wapor"

        for job in futures.as_completed(source_jobs):
            source = source_jobs[job]
            try:
                files = job.result()
                manifest["files"][source] = [str(path) for path in files]
                log(f"{source}: DONE ({len(files)} files)")
            except Exception as exc:  # source failures must not cancel other downloads
                manifest["errors"][source] = repr(exc)
                log(f"{source}: ERROR: {exc}")

    manifest_path = output / "download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log(f"Manifest: {manifest_path}")

    if manifest["errors"]:
        log("Finished with source errors; see download_manifest.json")
        return 1

    log("All requested sources finished successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
