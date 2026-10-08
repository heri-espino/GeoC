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

This calls NASA login, CDSE OAuth, CDS ARCO metadata and public Prithvi HTTP
endpoint. It does **not** download any large datasets. The CDS check validates
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