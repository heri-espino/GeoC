#!/usr/bin/env python3
"""Checkpoint 07 credential-safe runner for GeoC's existing scientific downloader.

No credentials are embedded, printed, committed or sent in --simulate mode.
Requires Python 3.11+; production downloads use the repository's
`tools/download_checkpoint_07_data.py` and its project extras.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SOURCES = ("hls", "smap", "sentinel1", "agera5", "prithvi")
# Stable publicly catalogued data-file URLs used only for an authenticated check.
# These are example 2025 scenes, not credentials or generated download links.
NASA_PROBE_URLS = {
    "hls": (
        "https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/"
        "HLSL30.020/HLS.L30.T14QNH.2025093T165916.v2.0/"
        "HLS.L30.T14QNH.2025093T165916.v2.0.B06.tif"
    ),
    "smap": (
        "https://data.nsidc.earthdatacloud.nasa.gov/nsidc-cumulus-prod-protected/"
        "SMAP/SPL3SMP_E/006/2025/04/01/SMAP_L3_SM_P_E_20250401_R19240_002.h5"
    ),
}
CREDENTIAL_KEYS = ("EARTHDATA_TOKEN", "EARTHDATA_USERNAME", "EARTHDATA_PASSWORD",
                   "CDSE_CLIENT_ID", "CDSE_CLIENT_SECRET", "CDSAPI_KEY")
TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
    "protocol/openid-connect/token"
)
ARCO_META_URL = (
    "https://arco.datastores.ecmwf.int/cadl-arco-time-001/arco/"
    "sis_agrometeorological_indicators/all/timeChunked.zarr/.zmetadata"
)
HF_CONFIG_URL = (
    "https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL/"
    "resolve/main/config.json"
)
TEMPLATE = """# PRIVATE local credentials. NEVER commit this file.\n# Replace EXAMPLE values with real values locally; do not share them in chat.\n# For NASA, an EARTHDATA_TOKEN is preferred over a password.\nEARTHDATA_TOKEN=REPLACE_WITH_NASA_EARTHDATA_TOKEN\nCDSE_CLIENT_ID=sh-REPLACE_WITH_CDSE_CLIENT_ID\nCDSE_CLIENT_SECRET=REPLACE_WITH_CDSE_CLIENT_SECRET\nCDSAPI_KEY=REPLACE_WITH_CDS_API_KEY\n"""


def locate_root(override: str | None) -> Path:
    return Path(override).expanduser().resolve() if override else Path(__file__).resolve().parents[1]


def config_file(root: Path, override: str | None) -> Path:
    return Path(override).expanduser().resolve() if override else root / ".env.checkpoint07"


def parse_env(path: Path) -> dict[str, str]:
    """Read simple KEY=value secrets without interpolating or executing them."""
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for i, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        item = line.strip()
        if not item or item.startswith("#"):
            continue
        if item.startswith("export "):
            item = item[7:].lstrip()
        if "=" not in item:
            raise ValueError(f"Malformed credential file at line {i} (expected KEY=value)")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key.isidentifier() or not key.isupper():
            raise ValueError(f"Invalid variable name on line {i}")
        value = value.strip()
        if value.startswith(("'", '"')) and value.endswith(value[0]) and len(value) > 1:
            value = value[1:-1]
        values[key] = value
    return values


def available(value: str | None) -> bool:
    if not value:
        return False
    uppercase = value.upper()
    return not any(marker in uppercase for marker in ("REPLACE_WITH", "YOUR_", "<TOKEN>", "EXAMPLE"))


def requirements(config: dict[str, str], source: str) -> bool:
    if source in {"hls", "smap"}:
        return available(config.get("EARTHDATA_TOKEN")) or (
            available(config.get("EARTHDATA_USERNAME"))
            and available(config.get("EARTHDATA_PASSWORD"))
        )
    if source == "sentinel1":
        return available(config.get("CDSE_CLIENT_ID")) and available(config.get("CDSE_CLIENT_SECRET"))
    if source == "agera5":
        return available(config.get("CDSAPI_KEY"))
    return True


def safe_print(msg: str, secrets: dict[str, str]) -> None:
    for value in secrets.values():
        if available(value) and len(value) >= 8:
            msg = msg.replace(value, "[REDACTED]")
    print(msg, flush=True)


def atomic_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=".manifest-", suffix=".tmp", delete=False, encoding="utf-8") as f:
        tmp = Path(f.name)
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        f.write("\n")
    os.replace(tmp, path)


def initialize(path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as f:
            f.write(TEMPLATE)
    except FileExistsError:
        print(f"Credential file already exists (unchanged): {path}")
        return 0
    try:
        path.chmod(0o600)
    except OSError:
        pass
    print(f"Created private template: {path}")
    print("Replace placeholders locally. No values are sent over the network.")
    return 0


def simulate(root: Path, selected: list[str]) -> int:
    """Purely offline example, safe even when the user supplies sample keys."""
    example = {
        "EARTHDATA_TOKEN": "EXAMPLE_NASA_TOKEN",
        "CDSE_CLIENT_ID": "sh-EXAMPLE-CDSE-ID",
        "CDSE_CLIENT_SECRET": "EXAMPLE_CDSE_SECRET",
        "CDSAPI_KEY": "EXAMPLE_CDS_KEY",
    }
    request = {
        "grant_type": "client_credentials",
        "client_id": "<CDSE_CLIENT_ID>",
        "client_secret": "<CDSE_CLIENT_SECRET>",
    }
    plan = {
        "mode": "offline-simulation",
        "network_requests": 0,
        "credentials_example_only": True,
        "sources": selected,
        "oauth_endpoint": TOKEN_URL,
        "oauth_form_fields": sorted(request),
        "uses_earthaccess_token": "EARTHDATA_TOKEN" in example,
        "agera5_authorization": "Bearer <CDSAPI_KEY>",
        "source_script": str(root / "tools" / "download_checkpoint_07_data.py"),
        "raw_folder": str(root / "data" / "raw" / "checkpoint_07"),
        "model_folder": str(root / "models" / "checkpoint_07"),
        "period": ["2025-04-01", "2025-10-31"],
        "note": "No download was performed; use 'check-auth --execute' then 'download --execute'.",
    }
    print(json.dumps(plan, indent=2, ensure_ascii=False))
    return 0


def _http_session():
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    session = requests.Session()
    retry = Retry(total=3, connect=3, read=2, backoff_factor=1.5,
                  status_forcelist=[429, 500, 502, 503, 504],
                  allowed_methods=frozenset({"GET", "HEAD", "POST"}),
                  respect_retry_after_header=True)
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


@contextmanager
def _earthdata_environment(config: dict[str, str]):
    """Expose NASA credentials to earthaccess temporarily, then restore the process environment.

    In contrast to run_download(), check_auth runs inside this Python process,
    so a private .env file must be passed to os.environ explicitly.
    """
    keys = ("EARTHDATA_TOKEN", "EARTHDATA_USERNAME", "EARTHDATA_PASSWORD")
    previous = {key: os.environ.get(key) for key in keys}
    try:
        for key in keys:
            value = config.get(key)
            if available(value):
                os.environ[key] = value
            else:
                os.environ.pop(key, None)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def doctor(root: Path, env_path: Path, config: dict[str, str]) -> int:
    """Report the discovered credential-file path and key presence, never values."""
    print(f"Repository root: {root}")
    print(f"Credentials file: {env_path}")
    print(f"Credentials file exists: {env_path.is_file()}")
    for key in CREDENTIAL_KEYS:
        print(f"{key}: {'configured' if available(config.get(key)) else 'missing'}")
    return 0


def probe_nasa(config: dict[str, str], selected: list[str]) -> bool:
    """Verify actual protected HLS/SMAP file access without saving data to disk.

    A local earthaccess.login() with EARTHDATA_TOKEN is not proof of validity:
    earthaccess marks the supplied token as authenticated without server check.
    Probe one byte using earthaccess's authenticated session.
    """
    try:
        import earthaccess

        with _earthdata_environment(config):
            auth = earthaccess.login(strategy="environment")
        if not getattr(auth, "authenticated", False):
            print("NASA: Earthdata login did not load any credentials")
            return False
        with auth.get_session() as session:
            for source in selected:
                url = NASA_PROBE_URLS[source]
                with session.get(
                    url,
                    headers={"Range": "bytes=0-0"},
                    stream=True,
                    allow_redirects=True,
                    timeout=(15, 75),
                ) as response:
                    status = int(response.status_code)
                    media_type = response.headers.get("Content-Type", "").lower()
                    if status not in (200, 206) or "text/html" in media_type:
                        if status in (401, 403):
                            print(
                                f"NASA {source}: HTTP {status} on protected file. "
                                "Replace/reissue EARTHDATA_TOKEN or verify DAAC "
                                "application/EULA authorization."
                            )
                        else:
                            print(f"NASA {source}: protected-file check failed (HTTP {status})")
                        return False
                    print(f"NASA {source}: protected-file access verified (HTTP {status})")
        return True
    except Exception as exc:
        # Provider error bodies/redirect URLs can contain signed query strings.
        print(f"NASA protected-file check: {type(exc).__name__} (no file downloaded)")
        return False


def check_auth(config: dict[str, str], selected: list[str], *, execute: bool) -> int:
    """Test credentials without downloading datasets or printing secrets."""
    if not execute:
        print("Real authentication checks require --execute. No network request made.")
        return 2
    needed = [s for s in selected if not requirements(config, s)]
    if needed:
        print("Missing credentials for: " + ", ".join(needed))
        return 2
    ok = True
    session = _http_session()
    nasa = [source for source in ("hls", "smap") if source in selected]
    if nasa and not probe_nasa(config, nasa):
        ok = False
    if "sentinel1" in selected:
        try:
            response = session.post(TOKEN_URL, data={
                "grant_type": "client_credentials",
                "client_id": config["CDSE_CLIENT_ID"],
                "client_secret": config["CDSE_CLIENT_SECRET"],
            }, timeout=(15, 50))
            response.raise_for_status()
            if not response.json().get("access_token"):
                raise RuntimeError("OAuth response missing access_token")
            print("Copernicus OAuth2: OK")
        except Exception as e:
            safe_print(f"Copernicus OAuth2: FAIL ({type(e).__name__}: {e})", config)
            ok = False
    if "agera5" in selected:
        try:
            with session.get(ARCO_META_URL,
                             headers={"Authorization": f"Bearer {config['CDSAPI_KEY']}"},
                             stream=True, timeout=(15, 70)) as response:
                response.raise_for_status()
                # Reads only response headers; never the huge Zarr metadata body.
            print("CDS AgERA5 ARCO metadata: HTTP OK (data access not fully verified)")
        except Exception as e:
            safe_print(f"CDS AgERA5 ARCO: FAIL ({type(e).__name__}: {e})", config)
            ok = False
    if "prithvi" in selected:
        try:
            with session.get(HF_CONFIG_URL, stream=True, timeout=(15, 50)) as response:
                response.raise_for_status()
            print("Hugging Face public Prithvi config: HTTP OK")
        except Exception as e:
            safe_print(f"Hugging Face: FAIL ({type(e).__name__}: {e})", config)
            ok = False
    return 0 if ok else 1


def run_download(root: Path, config: dict[str, str], selected: list[str], *,
                 execute: bool, catalog_only: bool, start: str, end: str,
                 parcels: str | None, margin: float, resolution: float) -> int:
    if not execute:
        print("Execution requires --execute; no API request was made.")
        return 2
    script = root / "tools" / "download_checkpoint_07_data.py"
    if not script.is_file():
        print(f"Missing original downloader: {script}")
        return 2
    incomplete = [s for s in selected if not requirements(config, s)
                  and not (catalog_only and s in {"hls", "smap", "sentinel1", "prithvi"})]
    if incomplete:
        print("Missing credentials: " + ", ".join(incomplete))
        return 2
    # Enforce env override for all configured values, without logging them.
    child_env = os.environ.copy()
    child_env.update({k: v for k, v in config.items() if available(v)})
    output = root / "data" / "raw" / "checkpoint_07"
    per_source_dir = output / "acquisition_manifests"
    status_file = per_source_dir / "source_status.json"
    status: dict[str, Any] = {}
    if status_file.exists():
        try:
            status = json.loads(status_file.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            pass
    for s in selected:
        if s in NASA_PROBE_URLS and not catalog_only:
            print(f"Checking NASA {s} protected-file access before bulk transfer")
            if not probe_nasa(config, [s]):
                print(f"STOP {s}: NASA access not verified; bulk transfer was not started")
                return 2
        cmd = [sys.executable, str(script), "--sources", s,
               "--start", start, "--end", end,
               "--margin-deg", str(margin),
               "--s1-resolution-m", str(resolution)]
        if parcels:
            cmd.extend(["--parcels", parcels])
        if catalog_only:
            cmd.append("--catalog-only")
        print(f"START {s} ({'catalog' if catalog_only else 'download'})", flush=True)
        status[s] = {"state": "running", "started_at": datetime.now(UTC).isoformat(),
                     "catalog_only": catalog_only}
        atomic_json(status_file, status)
        # Avoid passwords in process argv; inherited environment is private to child.
        process = subprocess.Popen(cmd, cwd=root, env=child_env,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace")
        assert process.stdout is not None
        for line in process.stdout:
            safe_print(line.rstrip("\n"), config)
        process.stdout.close()
        return_code = process.wait()
        manifest = output / "download_manifest.json"
        if return_code == 0 and manifest.is_file():
            per_source_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(manifest, per_source_dir /
                         f"{s}_{'catalog' if catalog_only else 'download'}.json")
        status[s] = {"state": "complete" if return_code == 0 else "failed",
                     "exit_code": return_code,
                     "finished_at": datetime.now(UTC).isoformat(),
                     "catalog_only": catalog_only}
        atomic_json(status_file, status)
        if return_code:
            print(f"STOP {s}: exit {return_code}. Other sources remain resumable.")
            return return_code
        print(f"OK {s}")
    print(f"Manifest statuses: {status_file}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["init", "simulate", "doctor", "preflight", "check-auth", "catalog", "download"])
    parser.add_argument("--root", help="GeoC repository root (inferred from script path by default)")
    parser.add_argument("--env-file", help="Local, gitignored credentials file")
    parser.add_argument("--sources", nargs="+", choices=["all", *SOURCES], default=["all"])
    parser.add_argument("--execute", action="store_true", help="Explicitly permit authenticated network requests")
    parser.add_argument("--start", default="2025-04-01")
    parser.add_argument("--end", default="2025-10-31")
    parser.add_argument("--parcels", default=None)
    parser.add_argument("--margin-deg", type=float, default=0.02)
    parser.add_argument("--s1-resolution-m", type=float, default=30.0)
    args = parser.parse_args(argv)
    from datetime import date
    if date.fromisoformat(args.start) > date.fromisoformat(args.end):
        parser.error("--start must be earlier than or equal to --end")
    if args.margin_deg < 0 or args.s1_resolution_m <= 0:
        parser.error("Margin must be nonnegative and S1 resolution must be positive")
    selected = list(SOURCES) if "all" in args.sources else list(dict.fromkeys(args.sources))
    root = locate_root(args.root)
    env_path = config_file(root, args.env_file)
    if args.action == "init":
        return initialize(env_path)
    if args.action == "simulate":
        return simulate(root, selected)
    # Local credentials take precedence; placeholders never override real OS variables.
    config = {k: v for k, v in os.environ.items() if k in CREDENTIAL_KEYS and available(v)}
    config.update({k: v for k, v in parse_env(env_path).items()
                   if k in CREDENTIAL_KEYS and available(v)})
    if args.action == "doctor":
        return doctor(root, env_path, config)
    if args.action == "check-auth":
        return check_auth(config, selected, execute=args.execute)
    if args.action == "preflight":
        source_script = root / "tools" / "download_checkpoint_07_data.py"
        if not source_script.is_file():
            print(f"Missing original downloader: {source_script}")
            return 2
        child_env = os.environ.copy()
        child_env.update({k: v for k, v in config.items() if available(v)})
        cmd = [sys.executable, str(source_script), "--preflight", "--sources", *selected]
        if args.parcels:
            cmd.extend(["--parcels", args.parcels])
        return subprocess.run(cmd, cwd=root, env=child_env, check=False).returncode
    return run_download(root, config, selected, execute=args.execute,
                        catalog_only=args.action == "catalog", start=args.start,
                        end=args.end, parcels=args.parcels,
                        margin=args.margin_deg, resolution=args.s1_resolution_m)


if __name__ == "__main__":
    raise SystemExit(main())