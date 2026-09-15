"""
geo.py — the one place coordinate conventions are enforced and converted.

D020 fixes three conventions for every file this project stores:

    longitude   0-360          (as both ERA5 and HYCOM serve it)
    latitude    ascending
    axis names  lat / lon      (not latitude / longitude)

The reason for 0-360 is not aesthetic. Both source stores serve 0-360, so storing
it unchanged means the number on disk is the number the server gave us, and any
discrepancy later is a real discrepancy rather than possibly our own arithmetic.
A conversion that never happens cannot be got wrong.

Conversion to -180..180 happens at the PRESENTATION boundary only -- figures, the
KML export (R8c) and the frontend -- via `to_display_longitude` below.

Why this file exists at all: on 2026-09-15 the wind fetcher wrote longitude as
-70.0 and the current fetcher wrote 290.0 for the same Gulf Stream cell, and they
named their axes differently on top of that. Neither script was wrong on its own
and both passed their own checks, which is exactly why it survived. See
`concepts/Grids, NetCDF and the convention traps` in the vault -- longitude
convention is trap #1 and latitude order is trap #2.
"""

import numpy as np
import xarray as xr

STORE_LON_RANGE = (0.0, 360.0)
CANONICAL_DIMS = ("lat", "lon")

# ERA5 and HYCOM disagree on what to call the same axis.
_RENAMES = {"latitude": "lat", "longitude": "lon", "Latitude": "lat", "Longitude": "lon"}


def to_store_longitude(lon):
    """-180..180 (or anything) -> 0-360, the convention we store in."""
    return np.asarray(lon) % 360.0


def to_display_longitude(lon):
    """0-360 -> -180..180, for axis labels, KML and the frontend. Display only."""
    return ((np.asarray(lon) + 180.0) % 360.0) - 180.0


def normalise_grid(ds: xr.Dataset) -> xr.Dataset:
    """Put any forcing dataset into the D020 convention.

    Renames the axes, converts longitude to 0-360, and sorts both axes ascending.
    Sorting matters as much as the rename: ERA5 is served descending in latitude,
    and `sel(lat=slice(17, 36))` on a descending axis returns an EMPTY selection
    without raising -- the silent failure that trap #2 is about.
    """
    ds = ds.rename({k: v for k, v in _RENAMES.items() if k in ds.dims or k in ds.coords})
    missing = [d for d in CANONICAL_DIMS if d not in ds.coords]
    if missing:
        raise KeyError(f"no {missing} coordinate after renaming; got {list(ds.coords)}")

    ds = ds.assign_coords(lon=to_store_longitude(ds["lon"]))
    return ds.sortby("lat").sortby("lon")


def assert_conventions(ds: xr.Dataset) -> None:
    """Fail loudly if a dataset is not in the stored convention.

    Call this on every READ as well as every write. D020's own consequence:
    a convention that is only documented is a convention that will be violated.
    """
    for d in CANONICAL_DIMS:
        if d not in ds.coords:
            raise AssertionError(f"missing coordinate {d!r}; got {list(ds.coords)}")

    lon = np.asarray(ds["lon"].values)
    lo, hi = STORE_LON_RANGE
    if not ((lon >= lo).all() and (lon < hi).all()):
        raise AssertionError(
            f"longitude outside {STORE_LON_RANGE} (got {lon.min()}..{lon.max()}) -- "
            "looks like -180..180. D020 stores 0-360."
        )
    for d in CANONICAL_DIMS:
        v = np.asarray(ds[d].values)
        if v.size > 1 and not (np.diff(v) > 0).all():
            raise AssertionError(f"{d} is not strictly ascending -- D020 stores ascending")
