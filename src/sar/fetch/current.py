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

CLI, two modes:
    # one point, nearest neighbour
    python -m sar.fetch.current --date 2019-01-01 --time 12:00 --lat 25.5 --lon -70.0
    # the raw archive tier: gridded NetCDF over the D014 box
    python -m sar.fetch.current --start 2021-01-05 --end 2021-01-08 --out /home/26p67/data

--out HAS NO DEFAULT, deliberately. On the cluster it is /home/26p67/data --
NOT /data1/26p67 or /data/26p67, which are empty root-owned mount points with
nothing mounted on them (D017, measured 2026-09-14).
"""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr

from sar.utils.geo import assert_conventions, normalise_grid

HYCOM_URL = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0"

VARS = ["water_u", "water_v"]

# D014 study box (same box as code/fetch_wind_arco.py in the vault).
# CHANGE THESE ONCE, HERE, AND NOWHERE ELSE.
LAT_S, LAT_N = 17.0, 36.0
LON_W, LON_E = -82.0, -63.0

# D019: the store is 3-hourly, not hourly. The docs page said hourly.
CADENCE_HOURS = 3

# MEASURED over the D014 box, 2026-09-17, by writing the file: 476 LAT x 238 LON
# x 2 vars x float32 = 885 KB per timestep, nineteen times a wind timestep.
#
# THIS IS THE IN-MEMORY SIZE AND IT IS ROUGHLY TWICE WHAT LANDS ON DISK.
# Measured again 2026-09-18 on a real month: 112 MB for 247 timesteps, so
# **443 KB per timestep**, exactly half. HYCOM serves water_u/water_v packed as
# int16 with `scale_factor = 0.001`, xarray decodes to float32 on read and
# re-applies that encoding on write, so the archive is int16 on disk at 1 mm/s
# resolution -- far finer than the data is accurate to, so nothing is lost.
#
# Consequence: the five-year archive is about **6.6 GB, not the 12.9 GB quoted
# in D019, D021 and issue #12.** The guard below therefore over-estimates by 2x,
# which is the safe direction and is left alone deliberately -- it bounds what
# is HELD IN MEMORY during the pull, which really is the float32 figure.
# The axes are that way round because latitude is the FINER axis here (0.04 deg
# against 0.08), which is the opposite of the intuition and is written as
# "476 x 238" in D019 and issue #12 without saying which is which.
# Everything in the size guard below comes off this one number.
BYTES_PER_TIMESTEP = 885 * 1024

# One year is 2.6 GB and is the unit the wind archive was pulled in; five years
# is 12.9 GB and should never be one file or one job. The threshold sits between.
MAX_BYTES = 3_000_000_000

# The land mask, as a fraction of box cells that are NaN. The same bound
# scripts/check_forcing_pair.py asserts against the live server.
NAN_FRACTION_MIN, NAN_FRACTION_MAX = 0.02, 0.40

# How much of the time axis to ask for in ONE OPeNDAP request.
#
# MEASURED THE HARD WAY, 2026-09-18. A half-year asked for in a single call is
# 1,448 timesteps and the server gives up on it:
#
#     oc_open: server error ... code = 500;
#     message = "java.net.SocketTimeoutException: Read timed out;
#                water_u -- 8981:10428,0:0,2425:2900,3475:3712";
#
# Six array tasks died that way after ~36 minutes each, having written nothing.
# The earlier throughput measurement (4.0 s/timestep over a 4-day request) was
# correct and useless on its own: it established the RATE and said nothing
# about the largest request the server will serve, and those are different
# limits.
#
# REVISED THE SAME EVENING: 8 days (64 steps, 29 MB per variable) ALSO times out
#
#     Read timed out; water_v -- 220:283,0:0,2425:2900,3475:3712
#
# while 4 days (32 steps, 14.5 MB) went through in 128 s. xarray retries, so a
# too-big piece does eventually land, but each retry costs a full server
# timeout and the task runs out of wall clock instead of failing cleanly.
#
# 2 days = 16 steps = 7.3 MB per variable, half the largest size observed to
# work. A deliberate margin, not timidity: throughput here varies by almost an
# order of magnitude between runs (4.0 s/timestep measured clean, 35 s on the
# first 3-day pull), so a size that only just works on a good day will not
# survive a bad one.
REQUEST_DAYS = 2


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


def fetch_current_range(start: str, end: str, lat_bounds: tuple, lon_bounds: tuple,
                        *, chunk_days: int = REQUEST_DAYS, verbose: bool = True) -> xr.Dataset:
    """`fetch_current_box` over a long window, in requests the server will serve.

    Splits [start, end) into `chunk_days` pieces, fetches each, and concatenates.
    See REQUEST_DAYS: a single request for a half-year times the server out, and
    the per-timestep rate does not warn you, because the rate is fine right up
    until the request is too big to answer at all.

    Progress is printed per piece so a long pull shows it is alive -- the failed
    array printed nothing for 36 minutes and then died.
    """
    t0 = np.datetime64(start)
    t1 = np.datetime64(end)
    step = np.timedelta64(chunk_days, "D")

    pieces = []
    at = t0
    n = int(np.ceil((t1 - t0) / step))
    k = 0
    while at < t1:
        stop = min(at + step, t1)
        k += 1
        sub = fetch_current_box(str(at), str(stop), lat_bounds, lon_bounds)
        pieces.append(sub)
        if verbose:
            got = sum(p.sizes["time"] for p in pieces)
            print(f"  [{k}/{n}] {str(at)[:10]} -> {str(stop)[:10]}  "
                  f"{sub.sizes['time']:3d} steps  ({got} so far)", flush=True)
        at = stop

    out = xr.concat(pieces, dim="time").sortby("time")

    # Adjacent pieces share no endpoint because end is exclusive throughout, so
    # a duplicate here means a piece boundary went wrong rather than the server
    # serving something twice.
    times = out["time"].values
    if times.size != np.unique(times).size:
        raise AssertionError(
            f"{times.size - np.unique(times).size} duplicate timestamps after "
            "concatenating -- the piece boundaries overlap"
        )
    return out


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
    # Same trap as src/sar/fetch/wind.py: xarray reads a bare date string as the
    # whole day, so slice(start, end) includes all of `end`. Made exclusive here
    # so the two fetchers cover identical windows -- otherwise a paired pull
    # silently gets one more day of current than of wind.
    sub = sub.sel(time=sub.time < np.datetime64(end))

    assert sub.sizes["lat"] > 0, "empty latitude: check the bounds"
    assert sub.sizes["lon"] > 0, "empty longitude: check the 0-360 convention"
    assert sub.sizes["time"] > 0, f"no timesteps in [{start}, {end})"

    return sub.load()


def box_tag() -> str:
    """The box half of a raw filename, identical in form to the wind fetch's."""
    return f"{int(LAT_S)}-{int(LAT_N)}N_{abs(int(LON_W))}-{abs(int(LON_E))}W"


def estimate_bytes(start: str, end: str) -> int:
    """Estimated size on disk of a gridded pull over [start, end).

    MEASURED, not assumed: one 3-hourly timestep over the D014 box is 476 lat x 238 lon
    cells x 2 variables x float32 = 885 KB. Nineteen times a wind timestep,
    which is the whole reason this guard exists -- the numbers that look
    harmless for wind are not harmless here.

        3 days  ~  21 MB        1 year ~ 2.6 GB        5 years ~ 12.9 GB
    """
    steps = int(
        (np.datetime64(end) - np.datetime64(start)) / np.timedelta64(CADENCE_HOURS, "h")
    )
    return max(steps, 0) * BYTES_PER_TIMESTEP


def write_current_netcdf(start: str, end: str, out, *, force: bool = False):
    """Pull [start, end) over the D014 box and write it as gridded NetCDF.

    This is the RAW tier of D020: the archive the visualiser and the engine read.
    `scripts/fetch_current_range.py` stays the long-format analytics export --
    two days of the full box is ~160 MB as text against ~14 MB here, and the
    five-year archive would be hundreds of GB in that form.

    Returns the path written. `out` has no default anywhere up the call chain:
    the fetch scripts that defaulted to a relative "data" have already written
    raw NetCDF into the synced vault once.
    """
    tag = f"{start.replace('-', '')}-{end.replace('-', '')}"
    raw_dir = Path(out) / "raw"
    existing = raw_dir / f"hycom_{box_tag()}_{tag}.nc"
    if existing.exists() and not force:
        # Resume. This endpoint is slow and erratic enough that a bulk pull WILL
        # be resubmitted, and a resubmit that re-downloads finished windows can
        # cost more than the original attempt. --force re-fetches.
        print(f"skip      {existing.name} already exists "
              f"({existing.stat().st_size / 1e6:.1f} MB) -- pass --force to re-fetch")
        return existing

    estimate = estimate_bytes(start, end)
    if estimate > MAX_BYTES and not force:
        raise SystemExit(
            f"{start}..{end} is an estimated {estimate / 1e9:.1f} GB "
            f"({estimate // BYTES_PER_TIMESTEP} timesteps x 885 KB), over the "
            f"{MAX_BYTES / 1e9:.1f} GB single-file threshold. Pull it a year at a "
            "time -- that is what the wind archive did -- or pass --force."
        )

    sub = fetch_current_range(start, end, (LAT_S, LAT_N), (LON_W, LON_E))

    # depth was selected, not sliced, so it must not have survived as an axis:
    # a length-1 depth dimension propagates into every downstream sel() and is
    # the kind of thing that only shows up three layers away.
    assert "depth" not in sub.dims, f"depth survived as a dimension: {dict(sub.sizes)}"

    for v in VARS:
        units = sub[v].attrs.get("units")
        if units is None:
            # The synthetic test datasets carry no attributes; the real store
            # does. Stamp it so the round trip has something to preserve.
            sub[v].attrs["units"] = "m/s"
        elif units != "m/s":
            raise ValueError(f"{v} is served in {units!r}, not m/s -- D002 assumes m/s")

    sub = normalise_grid(sub)
    assert_conventions(sub)

    # The land mask is the cheapest proof the pull is real. All-finite means the
    # box missed the coast entirely (wrong longitude convention, most likely);
    # mostly-NaN means the depth level or the window is wrong. Same bound
    # scripts/check_forcing_pair.py already asserts against the live server.
    nan_fraction = float(np.isnan(sub["water_u"].values).mean())
    if not (NAN_FRACTION_MIN <= nan_fraction <= NAN_FRACTION_MAX):
        raise SystemExit(
            f"NaN fraction {nan_fraction:.3f} outside [{NAN_FRACTION_MIN}, "
            f"{NAN_FRACTION_MAX}] -- the land mask did not survive. All-finite "
            "means the box is not where you think it is."
        )

    raw_dir.mkdir(parents=True, exist_ok=True)
    path = existing

    sub.to_netcdf(path)

    size = path.stat().st_size
    print(f"wrote {path}  ({size / 1e6:.1f} MB, {sub.sizes['time']} timesteps, "
          f"{nan_fraction:.1%} land)")
    return path


def main() -> None:
    p = argparse.ArgumentParser(
        description="HYCOM surface current: a point lookup, or a gridded NetCDF pull",
        epilog=(
            "point:  --date 2019-01-01 --time 12:00 --lat 25.5 --lon -70.0\n"
            "box:    --start 2021-01-05 --end 2021-01-08 --out /home/26p67/data"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--date", help="point mode: UTC date, YYYY-MM-DD")
    p.add_argument("--time", help="point mode: UTC time, HH:MM")
    p.add_argument("--lat", type=float, help="point mode: degrees north")
    p.add_argument("--lon", type=float, help="point mode: degrees east, -180..180")
    p.add_argument("--start", help="box mode: UTC date, YYYY-MM-DD, inclusive")
    p.add_argument("--end", help="box mode: UTC date, YYYY-MM-DD, exclusive")
    # No default. Every fetch script in this repo that defaulted --out to a
    # relative "data" has written raw NetCDF into the synced vault at least once.
    p.add_argument("--out", help="box mode: archive root; writes <out>/raw/. No default.")
    p.add_argument("--force", action="store_true",
                   help="box mode: write even if the estimate exceeds the threshold")
    args = p.parse_args()

    point_mode = args.date is not None
    box_mode = args.start is not None
    if point_mode == box_mode:
        p.error("give either --date/--time/--lat/--lon (point) or --start/--end/--out (box)")

    if point_mode:
        missing = [f for f, v in (("--time", args.time), ("--lat", args.lat),
                                  ("--lon", args.lon)) if v is None]
        if missing:
            p.error(f"point mode also needs {', '.join(missing)}")
        print(fetch_current(args.date, args.time, args.lat, args.lon))
        return

    if args.end is None or args.out is None:
        p.error("box mode needs --start, --end and --out")
    write_current_netcdf(args.start, args.end, args.out, force=args.force)


if __name__ == "__main__":
    main()
