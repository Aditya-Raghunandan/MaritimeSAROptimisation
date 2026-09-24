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

# Metres per degree of latitude on a sphere of the WGS84 mean radius. Constant with
# latitude to within 0.6 % over this project's box, which is far below every other error
# in the system; the longitude equivalent is NOT constant and is the function below.
M_PER_DEG_LAT = 111_320.0

# ERA5 and HYCOM disagree on what to call the same axis.
_RENAMES = {"latitude": "lat", "longitude": "lon", "Latitude": "lat", "Longitude": "lon"}


def to_store_longitude(lon):
    """-180..180 (or anything) -> 0-360, the convention we store in."""
    return np.asarray(lon) % 360.0


def to_display_longitude(lon):
    """0-360 -> -180..180, for axis labels, KML and the frontend. Display only."""
    return ((np.asarray(lon) + 180.0) % 360.0) - 180.0


def metres_per_degree_lon(lat):
    """Metres in one degree of longitude AT THIS LATITUDE. Scalar or array.

    A degree of longitude is 111.32 km at the equator and shrinks as cos(latitude):
    99.6 km at 26.5 N, 90.1 km at 36 N. It exists as a function, and takes latitude as
    an argument, for one reason: SO THAT IT CANNOT BE FROZEN.

    Converting an eastward velocity in m/s into a longitude rate means dividing by this
    number. Evaluate it once -- at the datum, say -- and reuse it while a particle drifts
    north, and every eastward step is wrong: 3.3 % by 30 N, 10.6 % by 36 N, 18.2 % across
    the full 17-36 N domain. Nothing raises and the trajectory still looks like physics.
    See ADR002 section 7; it is the third defect of this family in one month, after the
    raster painted 44 km north (latitude assumed linear on a Mercator screen) and a cell
    width quoted at the domain edge rather than the box centre.

    Callers therefore pass the particle's CURRENT latitude on every step, never a stored
    value. `sar.model.grid` calls it once per grid because a grid's cell size is fixed by
    definition; an integrator must call it per particle per step.
    """
    return M_PER_DEG_LAT * np.cos(np.radians(np.asarray(lat, dtype=float)))


def east_north(bearing_deg, distance_m):
    """A distance along a compass bearing as (east, north) metres. Scalar or array.

    Bearing is degrees clockwise from true north, the direction of travel -- the same
    convention `sar.model.interpolate.speed_direction` returns, so a heading read off the
    drift field can be flown without a conversion.
    """
    b = np.radians(np.asarray(bearing_deg, dtype=float))
    d = np.asarray(distance_m, dtype=float)
    return d * np.sin(b), d * np.cos(b)


def offset_position(lat, lon, east_m, north_m):
    """The point (east, north) metres from (lat, lon), on a local flat earth.

    cos(lat) is taken at the reference point, which is what a search pattern needs: its
    legs are laid out in metres about a datum a few kilometres across, where the change of
    cos(lat) across the pattern is under 0.1 %. Longitude is returned in whatever
    convention it was given in; nothing here wraps it.
    """
    lat = np.asarray(lat, dtype=float)
    dlat = np.asarray(north_m, dtype=float) / M_PER_DEG_LAT
    dlon = np.asarray(east_m, dtype=float) / metres_per_degree_lon(lat)
    return lat + dlat, np.asarray(lon, dtype=float) + dlon


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


def regular_axis_step(axis, name: str = "axis") -> float:
    """The single step of a regular coordinate axis, or raise if it is not regular.

    Every grid this project reads is regular, and three separate places rely on that:
    the bilinear weights in `sar.model.interpolate`, the `lat0 + k * dlat` the browser
    reconstructs from `sar.viz.export`'s manifest, and the pivot in
    `sar.utils.data_io.open_forcing_table`. They must agree on what "regular" means, so
    they all call this.

    JUDGED AGAINST A FITTED LINE, NOT AGAINST NEIGHBOURING GAPS. Comparing consecutive
    differences to each other looks equivalent and is not, because HYCOM stores its
    coordinates as float32 and this repo's real archive fails that test: over the D014
    box the longitude gaps run 0.079956 to 0.080018, which `np.allclose(rtol=1e-4)` and
    `atol=1e-9` both reject, while the axis is in fact a perfectly regular 0.08 grid
    written in a type that cannot represent it exactly. Measured 2026-09-19 on
    `current_2019-01-01_2019-01-03.txt`: the largest departure from the fitted line is
    5.5e-05 degrees, which is 0.07 % of a step and about 5 m on the ground.

    The tolerance is therefore derived from what float32 can represent at the axis's own
    magnitude, which is the actual cause: eight units in the last place, capped at a
    twentieth of a step so a coarse axis can never hide a genuinely irregular one. A
    stretched or curvilinear grid departs by a sizeable fraction of a step and still
    fails by orders of magnitude.
    """
    axis = np.asarray(axis, dtype=float)
    if axis.size < 2:
        raise ValueError(f"{name} needs at least 2 points to have a step, got {axis.size}")

    step = float((axis[-1] - axis[0]) / (axis.size - 1))
    if step == 0.0:
        raise ValueError(f"{name} has a zero step: first and last value are both {axis[0]}")

    ulp32 = float(np.spacing(np.float32(np.abs(axis).max())))
    tolerance = min(8.0 * ulp32, 0.05 * abs(step))
    departure = float(np.abs(axis - (axis[0] + step * np.arange(axis.size))).max())
    if departure > tolerance:
        raise ValueError(
            f"{name} is not regularly spaced: it departs from a {step:.6g} step by "
            f"{departure:.3g} ({departure / abs(step):.2%} of a step), and only "
            f"{tolerance:.3g} is allowed. Coordinates reconstructed as "
            f"{name}0 + k * d{name} would be wrong."
        )
    return step


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
