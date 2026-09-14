"""Coordinate-reference-system safeguards for geospatial processing."""

from __future__ import annotations

from typing import Any


def assert_crs(frame: Any, expected: str | None = None, *, label: str = "GeoDataFrame") -> None:
    """Require a geospatial frame to have a CRS and optionally match one.

    Parameters
    ----------
    frame:
        GeoPandas-like object exposing a ``crs`` attribute.
    expected:
        Optional CRS accepted by :class:`pyproj.CRS`, e.g. ``"EPSG:4326"``.
    label:
        Human-readable name used in error messages.
    """

    crs = getattr(frame, "crs", None)
    if crs is None:
        raise ValueError(f"{label} has no CRS. Assign/verify it before geospatial operations.")

    if expected is None:
        return

    try:
        from pyproj import CRS
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "CRS validation requires pyproj. Install GeoCebada with `pip install -e '.[geo]'`."
        ) from exc

    actual_crs = CRS.from_user_input(crs)
    expected_crs = CRS.from_user_input(expected)
    if actual_crs != expected_crs:
        raise ValueError(f"{label} CRS is {actual_crs.to_string()}, expected {expected_crs.to_string()}.")


def reproject_frame(frame: Any, target_crs: str, *, label: str = "GeoDataFrame") -> Any:
    """Return ``frame`` reprojected to ``target_crs`` after validating source CRS.

    The input object is not modified by this helper; GeoPandas ``to_crs`` returns
    a new frame by default.
    """

    assert_crs(frame, label=label)
    if not hasattr(frame, "to_crs"):
        raise TypeError(f"{label} does not expose a to_crs() method.")
    return frame.to_crs(target_crs)
