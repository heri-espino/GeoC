#!/usr/bin/env python
"""Isolated NetCDF inspector used by build_data_catalog.py.

This process intentionally imports xarray/h5py without importing GDAL/rasterio
first. On Windows this avoids mixing HDF5 runtimes loaded by different binary
packages.

It also repairs surrogate-escaped UTF-8 text in metadata when producing tiny
development fixtures. Source NetCDF files are never modified.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def repair_text(value: str) -> tuple[str, bool]:
    """Repair surrogate-escaped UTF-8 text without changing valid strings."""
    try:
        value.encode("utf-8")
        return value, False
    except UnicodeEncodeError:
        pass

    try:
        repaired = value.encode(
            "utf-8",
            errors="surrogateescape",
        ).decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        repaired = value.encode(
            "utf-8",
            errors="replace",
        ).decode("utf-8")

    return repaired, repaired != value


def sanitize_attr(
    value: Any,
    *,
    location: str,
    repairs: list[dict[str, str]],
) -> Any:
    """Return an HDF5-safe attribute value, recording textual repairs."""
    if isinstance(value, str):
        repaired, changed = repair_text(value)

        if changed:
            repairs.append(
                {
                    "location": location,
                    "repaired_value": repaired,
                }
            )

        return repaired

    if isinstance(value, bytes):
        try:
            repaired = value.decode("utf-8")
            changed = False
        except UnicodeDecodeError:
            repaired = value.decode("utf-8", errors="replace")
            changed = True

        if changed:
            repairs.append(
                {
                    "location": location,
                    "repaired_value": repaired,
                }
            )

        return repaired

    if isinstance(value, list):
        return [
            sanitize_attr(
                item,
                location=f"{location}[{index}]",
                repairs=repairs,
            )
            for index, item in enumerate(value)
        ]

    if isinstance(value, tuple):
        return tuple(
            sanitize_attr(
                item,
                location=f"{location}[{index}]",
                repairs=repairs,
            )
            for index, item in enumerate(value)
        )

    return value


def json_value(value: Any) -> Any:
    """Convert NetCDF metadata values to JSON-compatible objects."""
    if value is None:
        return None

    if isinstance(value, np.generic):
        return json_value(value.item())

    if isinstance(value, np.ndarray):
        return [json_value(item) for item in value.tolist()]

    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]

    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return value.decode("utf-8", errors="replace")

    if isinstance(value, str):
        repaired, _ = repair_text(value)
        return repaired

    if isinstance(value, (bool, int, float)):
        return value

    return str(value)


def sanitize_dataset(dataset: Any) -> tuple[Any, list[dict[str, str]]]:
    """Return a shallow dataset copy with writable textual metadata."""
    clean = dataset.copy(deep=False)
    repairs: list[dict[str, str]] = []

    clean.attrs = {
        str(key): sanitize_attr(
            value,
            location=f"dataset.attrs[{key!r}]",
            repairs=repairs,
        )
        for key, value in dataset.attrs.items()
    }

    for variable_name in clean.variables:
        clean[variable_name].attrs = {
            str(key): sanitize_attr(
                value,
                location=f"{variable_name}.attrs[{key!r}]",
                repairs=repairs,
            )
            for key, value in dataset[variable_name].attrs.items()
        }

    return clean, repairs


def build_profile(dataset: Any) -> dict[str, Any]:
    """Describe NetCDF dimensions, variables, dtypes and sanitized attributes."""
    return {
        "dimensions": {
            name: int(size)
            for name, size in dataset.sizes.items()
        },
        "variables": {
            name: {
                "dims": list(variable.dims),
                "shape": list(variable.shape),
                "dtype": str(variable.dtype),
                "attributes": {
                    str(key): json_value(value)
                    for key, value in variable.attrs.items()
                },
            }
            for name, variable in dataset.variables.items()
        },
        "attributes": {
            str(key): json_value(value)
            for key, value in dataset.attrs.items()
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-dim", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Import only inside this isolated worker.
    import xarray as xr

    dataset = xr.open_dataset(
        args.input,
        decode_times=False,
        engine="h5netcdf",
    )

    try:
        clean, repairs = sanitize_dataset(dataset)

        profile = build_profile(clean)

        if args.output is not None:
            selection = {
                dim: slice(0, min(size, args.max_dim))
                for dim, size in clean.sizes.items()
                if size > args.max_dim
            }

            subset = clean.isel(selection) if selection else clean

            args.output.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            subset.to_netcdf(
                args.output,
                engine="h5netcdf",
            )

        payload = {
            "profile": profile,
            "metadata_repairs": repairs,
        }

        # ASCII JSON avoids console-codepage problems on Windows.
        print(
            json.dumps(
                payload,
                ensure_ascii=True,
                sort_keys=True,
            )
        )

        return 0

    finally:
        dataset.close()


if __name__ == "__main__":
    raise SystemExit(main())
