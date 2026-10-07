#!/usr/bin/env python
"""Download Checkpoint 07 new-data inputs.

Sources:
- NASA HLS v2 (HLSL30 + HLSS30) through earthaccess.
- NASA SMAP Enhanced L3 9 km daily soil moisture v6 through earthaccess.
- Copernicus Sentinel-1 GRD as terrain-corrected regional GeoTIFFs.
- Copernicus AgERA5 v2 through the official ARCO Zarr store.
- Optional Prithvi-EO-2.0-TL weights from Hugging Face.

Credentials:
- HLS: NASA Earthdata Login through EARTHDATA_USERNAME/EARTHDATA_PASSWORD or ~/.netrc.
- Sentinel-1: CDSE_CLIENT_ID and CDSE_CLIENT_SECRET.
- AgERA5: CDSAPI_KEY or ~/.cdsapirc.
- Prithvi: public Hugging Face repositories normally need no token.

Examples:
  python tools/download_checkpoint_07_data.py --preflight
  python tools/download_checkpoint_07_data.py --catalog-only --sources hls sentinel1
  python tools/download_checkpoint_07_data.py --sources all
"""

from __future__ import annotations

import argparse
import getpass
import importlib.util
import json
import math
import os
import sys
from collections.abc import Mapping
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARCELS = ROOT / "data/source/geospatial/Parcelas_Reto_AGC_CONJUNTO_70_30.zip"
DEFAULT_OUTPUT = ROOT / "data/raw/checkpoint_07"
DEFAULT_MODELS = ROOT / "models/checkpoint_07"

HLS_SHORT_NAMES = ["HLSL30", "HLSS30"]
SMAP_SHORT_NAME = "SPL3SMP_E"
SMAP_VERSION = "006"
CDSE_STAC_SEARCH = "https://stac.dataspace.copernicus.eu/v1/search"
CDSE_TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
    "protocol/openid-connect/token"
)
CDSE_PROCESS_URL = "https://sh.dataspace.copernicus.eu/process/v1"

AGERA5_ARCO_TIME_URL = (
    "https://arco.datastores.ecmwf.int/cadl-arco-time-001/arco/"
    "sis_agrometeorological_indicators/all/timeChunked.zarr"
)
AGERA5_VARIABLES = [
    "Temperature_Air_2m_Max_24h",
    "Temperature_Air_2m_Mean_24h",
    "Temperature_Air_2m_Min_24h",
    "Dew_Point_Temperature_2m_Mean_24h",
    "Derived_Relative_Humidity_2m_Max_24h",
    "Derived_Relative_Humidity_2m_Min_24h",
    "Precipitation_Flux",
    "Precipitation_Duration_Fraction",
    "ReferenceET_PenmanMonteith_FAO56",
    "Solar_Radiation_Flux",
    "Vapour_Pressure_Deficit_at_Maximum_Temperature",
    "Vapour_Pressure_Mean_24h",
    "Wind_Speed_10m_Mean_24h",
    "Cloud_Cover_Mean_24h",
]
DEFAULT_PRITHVI_MODELS = [
    "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL",
    "ibm-nasa-geospatial/Prithvi-EO-2.0-600M-TL",
]

S1_EVALSCRIPT = r"""
//VERSION=3
function setup() {
  return {
    input: [{
      bands: [
        "VV",
        "VH",
        "localIncidenceAngle",
        "scatteringArea",
        "shadowMask",
        "dataMask"
      ]
    }],
    output: {
      bands: 6,
      sampleType: "FLOAT32"
    }
  };
}

function evaluatePixel(sample) {
  return [
    sample.VV,
    sample.VH,
    sample.localIncidenceAngle,
    sample.scatteringArea,
    sample.shadowMask,
    sample.dataMask
  ];
}
"""


def _log(message: str) -> None:
    stamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _parcel_bbox(
    parcel_path: Path,
    margin_deg: float,
) -> tuple[float, float, float, float]:
    try:
        import geopandas as gpd
    except ImportError as exc:
        raise RuntimeError(
            "GeoPandas is required. Install the geo and checkpoint07 extras."
        ) from exc

    if not parcel_path.is_file():
        raise FileNotFoundError(f"Parcel archive not found: {parcel_path}")

    frame = gpd.read_file(parcel_path)
    if len(frame) != 197:
        raise ValueError(f"Expected 197 parcel polygons, found {len(frame)}.")
    if frame.crs is None:
        raise ValueError("Parcel layer has no CRS; refusing to guess.")

    frame = frame.to_crs(4326)
    west, south, east, north = map(float, frame.total_bounds)
    return (
        west - float(margin_deg),
        south - float(margin_deg),
        east + float(margin_deg),
        north + float(margin_deg),
    )


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _cds_api_key() -> str | None:
    value = os.getenv("CDSAPI_KEY")
    if value:
        return value.strip()

    path = Path.home() / ".cdsapirc"
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("key:"):
            return stripped.split(":", 1)[1].strip()
    return None


def _credential_summary() -> dict[str, bool]:
    netrc = Path.home() / ".netrc"
    return {
        "earthdata_env_or_netrc": bool(
            (os.getenv("EARTHDATA_USERNAME") and os.getenv("EARTHDATA_PASSWORD"))
            or netrc.is_file()
        ),
        "cdse_oauth_env": bool(
            os.getenv("CDSE_CLIENT_ID") and os.getenv("CDSE_CLIENT_SECRET")
        ),
        "cds_api_key": _cds_api_key() is not None,
    }


def _preflight(
    parcel_path: Path,
    bbox: tuple[float, float, float, float],
    sources: set[str],
) -> None:
    print("Checkpoint 07 data preflight")
    print(f"parcels: {parcel_path}")
    print(f"bbox WGS84: {bbox}")
    print(f"sources: {sorted(sources)}")
    print(f"credentials: {_credential_summary()}")

    required_modules = {"geopandas": "geo"}
    if "hls" in sources or "smap" in sources:
        required_modules["earthaccess"] = "checkpoint07"
    if "agera5" in sources:
        required_modules.update(
            {
                "xarray": "checkpoint07",
                "zarr": "checkpoint07",
                "h5netcdf": "checkpoint07",
            }
        )
    if "prithvi" in sources:
        required_modules["huggingface_hub"] = "checkpoint07"

    missing = [name for name in required_modules if not _module_available(name)]
    if missing:
        raise RuntimeError(
            "Missing Python modules: "
            + ", ".join(missing)
            + ". Install the geo and checkpoint07 extras."
        )

    print("Checkpoint 07 data preflight: PASS")


def _safe_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    try:
        return dict(value)
    except Exception:
        return {"repr": str(value)}


def _hls_manifest_row(result: Any) -> dict[str, Any]:
    payload = _safe_mapping(result)
    meta = payload.get("meta", {})
    umm = payload.get("umm", {})
    return {
        "native_id": meta.get("native-id") if isinstance(meta, Mapping) else None,
        "concept_id": meta.get("concept-id") if isinstance(meta, Mapping) else None,
        "size": payload.get("size"),
        "begin": (
            umm.get("TemporalExtent", {})
            .get("RangeDateTime", {})
            .get("BeginningDateTime")
            if isinstance(umm, Mapping)
            else None
        ),
        "collection": (
            umm.get("CollectionReference", {}).get("ShortName")
            if isinstance(umm, Mapping)
            else None
        ),
    }


def _search_hls(
    bbox: tuple[float, float, float, float],
    start: date,
    end: date,
) -> list[Any]:
    try:
        import earthaccess
    except ImportError as exc:
        raise RuntimeError("Install the checkpoint07 extra first.") from exc

    _log("Searching all HLSL30 + HLSS30 v2 granules in the target season")
    results: list[Any] = []
    for short_name in HLS_SHORT_NAMES:
        found = earthaccess.search_data(
            short_name=short_name,
            version="2.0",
            bounding_box=bbox,
            temporal=(start.isoformat(), end.isoformat()),
            count=-1,
        )
        _log(f"HLS {short_name}: {len(found)} granules")
        results.extend(found)
    return results


def _download_hls(
    output: Path,
    bbox: tuple[float, float, float, float],
    start: date,
    end: date,
    *,
    catalog_only: bool,
) -> dict[str, Any]:
    results = _search_hls(bbox, start, end)
    records = [_hls_manifest_row(result) for result in results]
    folder = output / "hls_v2"
    _write_json(
        folder / "search_manifest.json",
        {
            "generated_at": datetime.now(UTC).isoformat(),
            "bbox": bbox,
            "start": start,
            "end": end,
            "short_names": HLS_SHORT_NAMES,
            "granule_count": len(results),
            "granules": records,
        },
    )
    _log(f"HLS catalogue: {len(results)} granules")

    files: list[str] = []
    if not catalog_only:
        import earthaccess

        _log("Earthdata login for HLS download")
        earthaccess.login()
        folder.mkdir(parents=True, exist_ok=True)
        downloaded = earthaccess.download(results, str(folder))
        files = [str(Path(path)) for path in downloaded]
        _log(f"HLS downloaded/reused files: {len(files)}")

    return {
        "granule_count": len(results),
        "downloaded_files": len(files),
        "directory": str(folder),
    }



def _download_smap(
    output: Path,
    bbox: tuple[float, float, float, float],
    start: date,
    end: date,
    *,
    catalog_only: bool,
) -> dict[str, Any]:
    """Search/download daily SMAP Enhanced L3 9 km soil moisture v6."""

    try:
        import earthaccess
    except ImportError as exc:
        raise RuntimeError("Install the checkpoint07 extra first.") from exc

    _log("Searching SMAP SPL3SMP_E v006 daily soil-moisture granules")
    results = earthaccess.search_data(
        short_name=SMAP_SHORT_NAME,
        version=SMAP_VERSION,
        bounding_box=bbox,
        temporal=(start.isoformat(), end.isoformat()),
        count=-1,
    )
    folder = output / "smap_spl3smp_e_v006"
    _write_json(
        folder / "search_manifest.json",
        {
            "generated_at": datetime.now(UTC).isoformat(),
            "bbox": bbox,
            "start": start,
            "end": end,
            "short_name": SMAP_SHORT_NAME,
            "version": SMAP_VERSION,
            "granule_count": len(results),
            "granules": [_hls_manifest_row(result) for result in results],
        },
    )
    _log(f"SMAP catalogue: {len(results)} granules")

    files: list[str] = []
    if not catalog_only:
        _log("Earthdata login for SMAP download")
        earthaccess.login()
        folder.mkdir(parents=True, exist_ok=True)
        downloaded = earthaccess.download(results, str(folder))
        files = [str(Path(path)) for path in downloaded]
        _log(f"SMAP downloaded/reused files: {len(files)}")

    return {
        "granule_count": len(results),
        "downloaded_files": len(files),
        "directory": str(folder),
    }


def _stac_next(payload: Mapping[str, Any]) -> str | None:
    for link in payload.get("links", []):
        if isinstance(link, Mapping) and link.get("rel") == "next":
            href = link.get("href")
            if isinstance(href, str):
                return href
    return None


def _search_sentinel1(
    bbox: tuple[float, float, float, float],
    start: date,
    end: date,
) -> list[dict[str, Any]]:
    params: dict[str, Any] | None = {
        "collections": "sentinel-1-grd",
        "bbox": ",".join(str(value) for value in bbox),
        "datetime": (
            f"{start.isoformat()}T00:00:00Z/"
            f"{end.isoformat()}T23:59:59Z"
        ),
        "limit": 100,
        "sortby": "+datetime",
    }
    url: str | None = CDSE_STAC_SEARCH
    features: list[dict[str, Any]] = []

    while url:
        response = requests.get(url, params=params, timeout=120)
        response.raise_for_status()
        payload = response.json()
        features.extend(payload.get("features", []))
        url = _stac_next(payload)
        params = None

    selected: list[dict[str, Any]] = []
    for item in features:
        props = item.get("properties", {})
        mode = (
            props.get("sar:instrument_mode")
            or props.get("s1:instrument_mode")
            or props.get("instrumentMode")
        )
        if mode and str(mode).upper() != "IW":
            continue

        pols = props.get("sar:polarizations")
        if pols:
            normalized = {str(value).upper() for value in pols}
            if not {"VV", "VH"}.issubset(normalized):
                continue

        timestamp = (
            props.get("datetime")
            or props.get("start_datetime")
            or props.get("startDate")
        )
        if not timestamp:
            continue

        orbit = (
            props.get("sat:orbit_state")
            or props.get("s1:orbit_direction")
            or props.get("orbitDirection")
            or "unknown"
        )
        selected.append(
            {
                "id": item.get("id"),
                "datetime": str(timestamp),
                "date": str(timestamp)[:10],
                "orbit_direction": str(orbit).upper(),
                "platform": props.get("platform"),
            }
        )
    return selected


def _cdse_credentials() -> tuple[str, str]:
    client_id = os.getenv("CDSE_CLIENT_ID")
    client_secret = os.getenv("CDSE_CLIENT_SECRET")
    if not client_id and sys.stdin.isatty():
        client_id = input("CDSE_CLIENT_ID: ").strip()
    if not client_secret and sys.stdin.isatty():
        client_secret = getpass.getpass("CDSE_CLIENT_SECRET: ").strip()
    if not client_id or not client_secret:
        raise RuntimeError(
            "Sentinel-1 download needs CDSE_CLIENT_ID and CDSE_CLIENT_SECRET."
        )
    return client_id, client_secret


def _cdse_token(client_id: str, client_secret: str) -> str:
    response = requests.post(
        CDSE_TOKEN_URL,
        data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        },
        timeout=60,
    )
    response.raise_for_status()
    return str(response.json()["access_token"])


def _output_dimensions(
    bbox: tuple[float, float, float, float],
    resolution_m: float,
) -> tuple[int, int, float]:
    west, south, east, north = bbox
    mid_lat = math.radians((south + north) / 2.0)
    width_m = abs(east - west) * 111_320.0 * math.cos(mid_lat)
    height_m = abs(north - south) * 110_574.0
    effective = float(resolution_m)
    max_dimension = max(width_m / effective, height_m / effective)
    if max_dimension > 2400:
        effective *= max_dimension / 2400.0
    width = max(1, int(math.ceil(width_m / effective)))
    height = max(1, int(math.ceil(height_m / effective)))
    return width, height, effective


def _sentinel1_request(
    bbox: tuple[float, float, float, float],
    scene_date: date,
    orbit_direction: str,
    width: int,
    height: int,
) -> dict[str, Any]:
    next_day = scene_date + timedelta(days=1)
    data_filter: dict[str, Any] = {
        "timeRange": {
            "from": f"{scene_date.isoformat()}T00:00:00Z",
            "to": f"{next_day.isoformat()}T00:00:00Z",
        },
        "mosaickingOrder": "mostRecent",
        "acquisitionMode": "IW",
        "polarization": "DV",
    }
    if orbit_direction in {"ASCENDING", "DESCENDING"}:
        data_filter["orbitDirection"] = orbit_direction

    return {
        "input": {
            "bounds": {
                "bbox": list(bbox),
                "properties": {
                    "crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84"
                },
            },
            "data": [
                {
                    "type": "sentinel-1-grd",
                    "dataFilter": data_filter,
                    "processing": {
                        "backCoeff": "GAMMA0_TERRAIN",
                        "orthorectify": "true",
                        "demInstance": "COPERNICUS_30",
                        "radiometricTerrainOversampling": 2,
                    },
                }
            ],
        },
        "output": {
            "width": width,
            "height": height,
            "responses": [
                {
                    "identifier": "default",
                    "format": {"type": "image/tiff"},
                }
            ],
        },
        "evalscript": S1_EVALSCRIPT,
    }


def _download_sentinel1(
    output: Path,
    bbox: tuple[float, float, float, float],
    start: date,
    end: date,
    *,
    resolution_m: float,
    catalog_only: bool,
) -> dict[str, Any]:
    _log("Searching Copernicus Sentinel-1 GRD catalogue")
    scenes = _search_sentinel1(bbox, start, end)
    groups = sorted({(row["date"], row["orbit_direction"]) for row in scenes})
    folder = output / "sentinel1_grd"
    width, height, effective_resolution = _output_dimensions(
        bbox,
        resolution_m,
    )
    _write_json(
        folder / "search_manifest.json",
        {
            "generated_at": datetime.now(UTC).isoformat(),
            "bbox": bbox,
            "start": start,
            "end": end,
            "catalog_items": scenes,
            "acquisition_groups": [
                {"date": day, "orbit_direction": orbit}
                for day, orbit in groups
            ],
            "requested_resolution_m": resolution_m,
            "effective_resolution_m": effective_resolution,
            "output_width": width,
            "output_height": height,
            "bands": [
                "VV",
                "VH",
                "localIncidenceAngle",
                "scatteringArea",
                "shadowMask",
                "dataMask",
            ],
            "backscatter": "GAMMA0_TERRAIN",
        },
    )
    _log(
        f"Sentinel-1 catalogue: {len(scenes)} items, "
        f"{len(groups)} date/orbit groups"
    )

    if catalog_only:
        return {
            "catalog_items": len(scenes),
            "acquisition_groups": len(groups),
            "directory": str(folder),
        }

    client_id, client_secret = _cdse_credentials()
    token = _cdse_token(client_id, client_secret)
    folder.mkdir(parents=True, exist_ok=True)
    completed = 0

    for index, (day_text, orbit) in enumerate(groups, 1):
        destination = folder / f"s1_{day_text}_{orbit.lower()}.tif"
        if destination.is_file() and destination.stat().st_size > 1000:
            completed += 1
            _log(f"Sentinel-1 skip {index}/{len(groups)}: {destination.name}")
            continue

        payload = _sentinel1_request(
            bbox,
            date.fromisoformat(day_text),
            orbit,
            width,
            height,
        )
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "image/tiff",
            "Content-Type": "application/json",
        }
        response = requests.post(
            CDSE_PROCESS_URL,
            json=payload,
            headers=headers,
            timeout=600,
        )
        if response.status_code == 401:
            token = _cdse_token(client_id, client_secret)
            headers["Authorization"] = f"Bearer {token}"
            response = requests.post(
                CDSE_PROCESS_URL,
                json=payload,
                headers=headers,
                timeout=600,
            )
        if response.status_code >= 400:
            raise RuntimeError(
                f"Sentinel-1 Process API failed for {day_text} {orbit}: "
                f"{response.status_code} {response.text[:1000]}"
            )

        destination.write_bytes(response.content)
        completed += 1
        _log(f"Sentinel-1 {index}/{len(groups)}: {destination.name}")

    return {
        "catalog_items": len(scenes),
        "acquisition_groups": len(groups),
        "downloaded_or_reused": completed,
        "directory": str(folder),
    }


def _coord_slice(values: Any, low: float, high: float) -> slice:
    first = float(values[0])
    last = float(values[-1])
    if first <= last:
        return slice(low, high)
    return slice(high, low)


def _download_agera5(
    output: Path,
    bbox: tuple[float, float, float, float],
    start: date,
    end: date,
    *,
    catalog_only: bool,
) -> dict[str, Any]:
    key = _cds_api_key()
    if key is None:
        raise RuntimeError(
            "AgERA5 ARCO requires CDSAPI_KEY or a key entry in ~/.cdsapirc."
        )

    folder = output / "agera5_v2"
    destination = folder / (
        f"agera5_v2_{start.isoformat()}_{end.isoformat()}_region.nc"
    )
    if catalog_only:
        return {
            "variables": AGERA5_VARIABLES,
            "directory": str(folder),
            "status": "catalog-only; ARCO not opened",
        }
    if destination.is_file() and destination.stat().st_size > 1000:
        _log(f"AgERA5 skip: {destination.name}")
        return {
            "variables": len(AGERA5_VARIABLES),
            "file": str(destination),
            "status": "reused",
        }

    try:
        import xarray as xr
    except ImportError as exc:
        raise RuntimeError("Install the checkpoint07 extra first.") from exc

    _log("Opening official AgERA5 v2 ARCO time-chunked Zarr")
    ds = xr.open_zarr(
        AGERA5_ARCO_TIME_URL,
        consolidated=True,
        storage_options={
            "headers": {"Authorization": f"Bearer {key}"}
        },
        chunks={},
    )
    missing = sorted(set(AGERA5_VARIABLES) - set(ds.data_vars))
    if missing:
        raise RuntimeError(
            "AgERA5 ARCO schema changed; missing variables: "
            + ", ".join(missing)
        )

    west, south, east, north = bbox
    latitude = _coord_slice(ds["latitude"].values, south, north)
    longitude = _coord_slice(ds["longitude"].values, west, east)
    subset = ds[AGERA5_VARIABLES].sel(
        time=slice(start.isoformat(), end.isoformat()),
        latitude=latitude,
        longitude=longitude,
    )
    if int(subset.sizes.get("time", 0)) == 0:
        raise RuntimeError("AgERA5 subset returned zero time steps.")

    _log(
        "Loading AgERA5 subset "
        f"{dict(subset.sizes)} with {len(AGERA5_VARIABLES)} variables"
    )
    subset = subset.load()
    folder.mkdir(parents=True, exist_ok=True)
    subset.to_netcdf(destination, engine="h5netcdf")
    _write_json(
        folder / "manifest.json",
        {
            "generated_at": datetime.now(UTC).isoformat(),
            "source": AGERA5_ARCO_TIME_URL,
            "bbox": bbox,
            "start": start,
            "end": end,
            "variables": AGERA5_VARIABLES,
            "sizes": dict(subset.sizes),
            "file": str(destination),
        },
    )
    _log(f"AgERA5 done: {destination}")
    return {
        "variables": len(AGERA5_VARIABLES),
        "file": str(destination),
        "sizes": dict(subset.sizes),
    }


def _download_prithvi(
    models_output: Path,
    model_ids: list[str],
    *,
    catalog_only: bool,
) -> dict[str, Any]:
    if catalog_only:
        return {"models": model_ids, "status": "catalog-only"}

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("Install the checkpoint07 extra first.") from exc

    downloaded: list[str] = []
    for model_id in model_ids:
        destination = models_output / model_id.rsplit("/", 1)[-1]
        _log(f"Downloading/reusing {model_id}")
        snapshot_download(
            repo_id=model_id,
            local_dir=destination,
            allow_patterns=[
                "*.pt",
                "*.json",
                "*.yaml",
                "*.yml",
                "*.py",
                "README.md",
                "requirements.txt",
            ],
        )
        downloaded.append(str(destination))
    return {"models": model_ids, "directories": downloaded}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["all"],
        choices=["all", "hls", "smap", "sentinel1", "agera5", "prithvi"],
    )
    parser.add_argument("--start", default="2025-04-01")
    parser.add_argument("--end", default="2025-10-31")
    parser.add_argument("--margin-deg", type=float, default=0.02)
    parser.add_argument("--s1-resolution-m", type=float, default=30.0)
    parser.add_argument("--parcels", type=Path, default=DEFAULT_PARCELS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--models-out", type=Path, default=DEFAULT_MODELS)
    parser.add_argument(
        "--prithvi-models",
        nargs="+",
        default=DEFAULT_PRITHVI_MODELS,
    )
    parser.add_argument("--catalog-only", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    return parser.parse_args()


def _main() -> int:
    args = _parse_args()
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    if start > end:
        raise ValueError("--start must be on or before --end.")
    if args.s1_resolution_m <= 0:
        raise ValueError("--s1-resolution-m must be positive.")

    sources = set(args.sources)
    if "all" in sources:
        sources = {"hls", "smap", "sentinel1", "agera5", "prithvi"}

    parcel_path = args.parcels.expanduser().resolve()
    output = args.out.expanduser().resolve()
    models_output = args.models_out.expanduser().resolve()
    bbox = _parcel_bbox(parcel_path, args.margin_deg)

    if args.preflight:
        _preflight(parcel_path, bbox, sources)
        return 0

    output.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "checkpoint": "07",
        "bbox_wgs84": bbox,
        "start": start,
        "end": end,
        "catalog_only": bool(args.catalog_only),
        "sources": {},
    }

    if "hls" in sources:
        manifest["sources"]["hls"] = _download_hls(
            output,
            bbox,
            start,
            end,
            catalog_only=args.catalog_only,
        )
    if "smap" in sources:
        manifest["sources"]["smap"] = _download_smap(
            output,
            bbox,
            start,
            end,
            catalog_only=args.catalog_only,
        )
    if "sentinel1" in sources:
        manifest["sources"]["sentinel1"] = _download_sentinel1(
            output,
            bbox,
            start,
            end,
            resolution_m=args.s1_resolution_m,
            catalog_only=args.catalog_only,
        )
    if "agera5" in sources:
        manifest["sources"]["agera5"] = _download_agera5(
            output,
            bbox,
            start,
            end,
            catalog_only=args.catalog_only,
        )
    if "prithvi" in sources:
        manifest["sources"]["prithvi"] = _download_prithvi(
            models_output,
            list(args.prithvi_models),
            catalog_only=args.catalog_only,
        )

    _write_json(output / "download_manifest.json", manifest)
    _log(f"Checkpoint 07 manifest: {output / 'download_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
