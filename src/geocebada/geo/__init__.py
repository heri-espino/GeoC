"""Geospatial validation, CRS and parcel utilities for GeoCebada."""

from geocebada.geo.crs import assert_crs, reproject_frame
from geocebada.geo.parcels import aggregate_to_parcels, load_parcels

__all__ = [
    "aggregate_to_parcels",
    "assert_crs",
    "load_parcels",
    "reproject_frame",
]
