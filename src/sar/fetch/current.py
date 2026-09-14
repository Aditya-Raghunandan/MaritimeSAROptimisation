"""
current.py: surface current lookup against HYCOM GLBy0.08/expt_93.0.

The `_sur` catalogue does NOT contain water_u/water_v: it holds barotropic
(whole-water-column-averaged) velocity, not surface velocity (D019). This
module uses the 3-D dataset instead and takes depth level 0.

    OPeNDAP: https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0
    Variables used: water_u, water_v (eastward/northward velocity, m/s)
    Depth: level 0 = 0 m (measured against the server, not the docs page)
    Cadence: 3-hourly, not hourly
    Longitude: store convention is 0-360, not -180-180

CLI:
    python -m sar.fetch.current --date 2019-01-01 --time 12:00 --lat 25.5 --lon -70.0
"""

import argparse
from datetime import datetime

import numpy as np
import xarray as xr

HYCOM_URL = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0"

VARS = ["water_u", "water_v"]

# D014 study box (same box as code/fetch_wind_arco.py in the vault).
# CHANGE THESE ONCE, HERE, AND NOWHERE ELSE.
LAT_S, LAT_N = 17.0, 36.0
LON_W, LON_E = -82.0, -63.0


def open_current_dataset() -> xr.Dataset:
    """Open the HYCOM surface-current dataset (lazy: nothing downloaded yet).

    Asserts depth level 0 really is 0 m before any caller selects it, since
    that assumption (not "the shallowest level", but "0 m exactly") is what
    D019 rests on.
    """
    # drop_variables=["tau"] is load-bearing: tau's units are "hours since analysis", which is not a parseable reference date, and xarray's CF time decoder chokes on it before we ever touch water_u/water_v.
    ds = xr.open_dataset(HYCOM_URL, drop_variables=["tau"])
    assert ds["depth"].values[0] == 0.0, (
        f"expected depth level 0 to be 0.0 m, got {ds['depth'].values[0]}"
    )
    return ds


def to_store_longitude(lon: float) -> float:
    """Convert a -180..180 longitude to the store's 0..360 convention."""
    return lon % 360


def fetch_current(date: str, time: str, lat: float, lon: float) -> dict:
    """Nearest-neighbour surface current at one point in time and space.

    Parameters
    ----------
    date : "YYYY-MM-DD"
    time : "HH:MM" (UTC)
    lat  : degrees north
    lon  : degrees east, -180..180 (converted internally to 0..360)

    Returns
    -------
    dict with keys water_u, water_v (m/s) and time (the timestamp selected, which is the nearest available 3-hourly step, not necessarily the one requested).
    """
    requested = datetime.fromisoformat(f"{date}T{time}")
    store_lon = to_store_longitude(lon)

    ds = open_current_dataset()
    point = ds[VARS].sel(
        depth=0.0,
        time=np.datetime64(requested),
        lat=lat,
        lon=store_lon,
        method="nearest",
    )
    point = point.load()

    return {
        "water_u": float(point["water_u"].values),
        "water_v": float(point["water_v"].values),
        "time": point["time"].values,
    }


def fetch_current_box(start: str, end: str, lat_bounds: tuple, lon_bounds: tuple) -> xr.Dataset:
    """All surface-current records for a lat/lon box across a date range.

    `start`/`end` are "YYYY-MM-DD", start inclusive and end exclusive, same
    convention as fetch_wind_arco.py. Returns the loaded (in-memory) subset
    at depth level 0 for every 3-hourly timestep in [start, end).
    """
    lat_s, lat_n = lat_bounds
    lon_w, lon_e = (to_store_longitude(lon_bounds[0]), to_store_longitude(lon_bounds[1]))

    ds = open_current_dataset()
    sub = ds[VARS].sel(
        depth=0.0,
        time=slice(start, end),
        lat=slice(lat_s, lat_n),
        lon=slice(lon_w, lon_e),
    )
    assert sub.sizes["lat"] > 0, "empty latitude: check the bounds"
    assert sub.sizes["lon"] > 0, "empty longitude: check the 0-360 convention"
    assert sub.sizes["time"] > 0, f"no timesteps in [{start}, {end})"

    return sub.load()


def main() -> None:
    p = argparse.ArgumentParser(description="Nearest-neighbour HYCOM surface current lookup")
    p.add_argument("--date", required=True, help="UTC date, YYYY-MM-DD")
    p.add_argument("--time", required=True, help="UTC time, HH:MM")
    p.add_argument("--lat", required=True, type=float, help="degrees north")
    p.add_argument("--lon", required=True, type=float, help="degrees east, -180..180")
    args = p.parse_args()

    result = fetch_current(args.date, args.time, args.lat, args.lon)
    print(result)


if __name__ == "__main__":
    main()
