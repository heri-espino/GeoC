# GeoC Checkpoint 07 — credential-safe acquisition runner

## Install

The runner is tracked in the GeoC repository; use `git pull` to obtain it.
From the repository root (PowerShell on Windows):

```powershell
git pull --ff-only origin main
conda activate geocebada
python -m pip install -e ".[geo,checkpoint07]"
python tools/run_checkpoint_07_acquisition.py init
```

Edit **`.env.checkpoint07`** locally, replacing the placeholders. Never paste live keys
into chat or commit the file. This credentials file is already ignored by the
existing repository pattern `.env.*`.

## Confirm credentials are being discovered (no secrets printed)

```powershell
python tools/run_checkpoint_07_acquisition.py doctor
```

`doctor` reports the expected `.env.checkpoint07` location and which key names
are configured, but **never prints their values**. It does not access the internet.

## Test safely without sending any credentials

```powershell
python tools/run_checkpoint_07_acquisition.py simulate
```

The simulation constructs a plan and proves zero network requests in this mode.
It does **not** prove access to the remote scientific datasets.

## Test actual access (explicit opt-in)

```powershell
python tools/run_checkpoint_07_acquisition.py check-auth --execute
```

This loads the NASA token locally and verifies protected-file HTTP access on
one published HLS and one SMAP file by streaming a one-byte Range request.
HTTP 200/206 (non-HTML) means protected-file access works; HTTP 401/403 means
the token or data-provider authorization must be corrected before bulk transfer.
It also calls CDSE OAuth, CDS ARCO metadata and the public Prithvi HTTP endpoint.
The NASA probe transfers only response headers/a minimal payload and saves no files.
It does **not** download any large datasets. The original downloader now runs
this NASA probe before starting each HLS/SMAP bulk download. The CDS check validates
HTTP access to metadata only, not NetCDF writing or full temporal coverage.

## Inventory before downloading

```powershell
python tools/download_checkpoint_07_data.py --preflight
python tools/run_checkpoint_07_acquisition.py catalog --sources hls smap sentinel1 --execute
```

This uses the official catalog endpoints through the existing downloader.

## Actual downloads

```powershell
python tools/run_checkpoint_07_acquisition.py download --sources prithvi --execute
python tools/run_checkpoint_07_acquisition.py download --sources agera5 --execute
python tools/run_checkpoint_07_acquisition.py download --sources hls smap --execute
python tools/run_checkpoint_07_acquisition.py download --sources sentinel1 --execute
```

Or all sources, in order:

```powershell
python tools/run_checkpoint_07_acquisition.py download --sources all --execute
```

The script invokes `tools/download_checkpoint_07_data.py` one source at a time
and preserves a per-source completion manifest under
`data/raw/checkpoint_07/acquisition_manifests/`. These files are gitignored.

**Limits of existing downloader:** HLS downloads full intersecting granules;
SMAP downloads daily full granules; the existing Sentinel-1 raster request may
coarsen its effective resolution if the regional bounding box exceeds its
2400-pixel limit. Validate *effective_resolution_m* in the Sentinel-1 manifest
before treating any raster as 30m. The original downloader's `>1000` byte
resume check is not a full integrity or CRC check. Do not claim data are
scientifically complete until the checkpoint-07 quality audit finishes.

**Security:** no credentials in script, arguments, outputs, Git commits, or
manifest. If any sample credentials provided in chat are actually valid,
revoke/rotate before using them. Test credentials must not be submitted to live
services just to demonstrate the example.

## macOS / Linux

The same Python commands work using `python3` in place of `python` and your
preferred virtual environment.

## Tests

```powershell
python -m unittest discover -s tests -p 'test_checkpoint07_runner.py'
```
## NASA 401 troubleshooting (HLS or SMAP)

If `check-auth --sources hls smap --execute` reports HTTP 401, visit
https://urs.earthdata.nasa.gov/ and generate a fresh, unmodified **user token**.
Replace only `EARTHDATA_TOKEN` in the private `.env.checkpoint07` file.
Earthaccess accepts an environment token locally without checking its signature,
so the protected-file probe is necessary. If a fresh token still returns 401,
open the corresponding HLS/SMAP granule link in a browser while signed in to
Earthdata, and check for pending LP DAAC / NSIDC application authorization or
EULA approval; contact the appropriate DAAC if the error persists.

You may alternatively configure `EARTHDATA_USERNAME` and
`EARTHDATA_PASSWORD` in the private env file, but remove or comment out
`EARTHDATA_TOKEN` first: tokens take precedence in earthaccess. Never store
these secrets in the repository or print them in terminal output.

Run the protected-file check again before retrying the download:

```powershell
python tools/run_checkpoint_07_acquisition.py check-auth --sources hls smap --execute
python tools/run_checkpoint_07_acquisition.py download --sources hls --execute
python tools/run_checkpoint_07_acquisition.py download --sources smap --execute
```
