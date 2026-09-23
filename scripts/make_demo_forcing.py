"""make_demo_forcing.py: writes a tiny pair of forcing files for trying the CLI (issue #10).

Writes two small NetCDF files under <out>/raw/:

    hycom_demo_<start>-<end>.nc     water_u, water_v   3-hourly, 0.08 x 0.04 deg
    era5_demo_<start>-<end>.nc      u10, v10           hourly,   0.25 x 0.25 deg

THESE ARE NOT DATA. Every value is invented by the formula below, and nothing here has
been near HYCOM or ERA5. The point is that `sar.model.interpolate` cannot be run at all
without a pair of files in the stored convention, a real pull takes a long time and a
network, and "I cannot see it work" is a bad reason to trust code less than it deserves
or more. So this writes files with the right SHAPE: D020 conventions (0 to 360 longitude,
ascending axes, lat/lon axis names), the two grids' real steps, the real variable names
and units, and a patch of NaN where land would be, so the land path can be exercised too.

Use it to try the CLI, to see the JSON, and to draw a real cell diagram. Do not use it to
judge a current, a wind or a drift: the numbers mean nothing.

The files land in the gitignored data tree and must never be committed.

Run:
    python scripts/make_demo_forcing.py --out data
    python -m sar.model.interpolate \\
        --current data/raw/hycom_demo_2021-01-05-2021-01-06.nc \\
        --wind data/raw/era5_demo_2021-01-05-2021-01-06.nc \\
        --lat 26.53 --lon 281.4 --time 2021-01-05T07:30 --with-wind --plot cell.png
"""

import argparse
from pathlib import Path

import numpy as np
import xarray as xr

START = "2021-01-05"
END = "2021-01-06"   # the default one-day span; --days moves it


def _end(days: int) -> str:
    """The exclusive end date for a run of `days` days from START."""
    return str(np.datetime64(START) + np.timedelta64(days, "D"))

# A small corner of the study box (D014 is 17-36 N, 82-63 W, stored as 278-297), big
# enough to hold several cells of both grids and small enough to write in a moment.
LAT_RANGE = (26.0, 27.0)
LON_RANGE = (281.0, 282.0)

# Land, as a fraction of the box: a wedge in the south west corner, so a query there
# raises MissingCornerError and a query in the middle does not.
LAND_LAT = (26.0, 26.2)
LAND_LON = (281.0, 281.2)


def _axes(dlat: float, dlon: float, hours: int, step_hours: int):
    lat = np.arange(LAT_RANGE[0], LAT_RANGE[1] + 0.5 * dlat, dlat)
    lon = np.arange(LON_RANGE[0], LON_RANGE[1] + 0.5 * dlon, dlon)
    time = np.datetime64(f"{START}T00:00") + np.arange(0, hours, step_hours) * np.timedelta64(1, "h")
    return lat, lon, time


def _field(lat, lon, time, scale: float, turn: float):
    """A smooth, turning, invented field. Not physics: a shape that is easy to check.

    It varies in both axes and in time, so an interpolated value at a point genuinely
    differs from its four corners, which a constant or linear field would hide.
    """
    t = np.arange(time.size)[:, None, None]
    la = (lat[None, :, None] - LAT_RANGE[0]) * 10.0
    lo = (lon[None, None, :] - LON_RANGE[0]) * 10.0
    u = scale * (np.sin(turn * lo + 0.3 * la) + 0.15 * np.cos(0.4 * t))
    v = scale * (np.cos(0.5 * lo - turn * la) - 0.10 * np.sin(0.3 * t))
    return u, v


def _mask_land(u, v, lat, lon):
    """NaN out a wedge, the way HYCOM stores land, so the land path can be tried."""
    j = (lat >= LAND_LAT[0]) & (lat <= LAND_LAT[1])
    i = (lon >= LAND_LON[0]) & (lon <= LAND_LON[1])
    u[:, np.ix_(j, i)[0][:, None], np.ix_(j, i)[1]] = np.nan
    v[:, np.ix_(j, i)[0][:, None], np.ix_(j, i)[1]] = np.nan
    return u, v


def write_current(out: Path, days: int = 1) -> Path:
    """The HYCOM-shaped file: 3-hourly, 0.08 deg longitude by 0.04 deg latitude."""
    lat, lon, time = _axes(0.04, 0.08, 24 * days, 3)
    u, v = _field(lat, lon, time, scale=0.9, turn=0.6)
    u, v = _mask_land(u, v, lat, lon)
    ds = xr.Dataset(
        {
            "water_u": (("time", "lat", "lon"), u, {"units": "m/s", "long_name": "eastward velocity"}),
            "water_v": (("time", "lat", "lon"), v, {"units": "m/s", "long_name": "northward velocity"}),
        },
        coords={"time": time, "lat": lat, "lon": lon},
        attrs={"title": "INVENTED demo current, not HYCOM", "source": "scripts/make_demo_forcing.py"},
    )
    path = out / "raw" / f"hycom_demo_{START}-{_end(days)}.nc"
    ds.to_netcdf(path)
    return path


def write_wind(out: Path, days: int = 1) -> Path:
    """The ERA5-shaped file: hourly, 0.25 deg in both axes, no land mask (wind has none)."""
    lat, lon, time = _axes(0.25, 0.25, 24 * days, 1)
    u, v = _field(lat, lon, time, scale=7.0, turn=0.4)
    ds = xr.Dataset(
        {
            "u10": (("time", "lat", "lon"), u, {"units": "m s**-1", "long_name": "10 metre U wind"}),
            "v10": (("time", "lat", "lon"), v, {"units": "m s**-1", "long_name": "10 metre V wind"}),
        },
        coords={"time": time, "lat": lat, "lon": lon},
        attrs={"title": "INVENTED demo wind, not ERA5", "source": "scripts/make_demo_forcing.py"},
    )
    path = out / "raw" / f"era5_demo_{START}-{_end(days)}.nc"
    ds.to_netcdf(path)
    return path


def main() -> None:
    p = argparse.ArgumentParser(
        description="Write small INVENTED forcing files so the interpolate CLI can be run"
    )
    p.add_argument("--out", default="data", help="archive root; writes <out>/raw/")
    # One day by default, which is what the CLI examples and the golden fixture use.
    # The browser suite asks for more: past 14 days the site opens on a coarser tier and
    # switches, which is the path that froze the drift arrows on 23 Sep.
    p.add_argument("--days", type=int, default=1, help="length of the run in days (default 1)")
    args = p.parse_args()
    if args.days < 1:
        p.error("--days must be at least 1")

    out = Path(args.out)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    current, wind = write_current(out, args.days), write_wind(out, args.days)

    for path in (current, wind):
        print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KB)")
    print(
        "\nThese values are invented and mean nothing. Try:\n"
        f"  python -m sar.model.interpolate --current {current} --wind {wind} \\\n"
        "      --lat 26.53 --lon 281.4 --time 2021-01-05T07:30 --with-wind\n"
        "  (--lat 26.1 --lon 281.1 is inside the land wedge and will raise)"
    )


if __name__ == "__main__":
    main()
