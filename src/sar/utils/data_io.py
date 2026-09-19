"""
data_io.py -- the read half. Opens what the fetchers wrote.

    python -m sar.utils.data_io --forcing <data>/raw/era5_17-36N_82-63W_20210101-20210108.nc
    python -m sar.utils.data_io --drifters <data>/raw/gdp_hourly_17-36N_82-63W_20190101-20240101.csv
    python -m sar.utils.data_io --box-means <data>/derived

where <data> is C:/maritime-data locally or /home/26p67/data on the cluster.

WHY THIS FILE EXISTS. Three fetchers write NetCDF, CSV and Parquet, and until
now nothing read any of it back. `code/MAP.md` has planned a `data_io.py` since
3 Sep and calls `geo.py` "the first piece" of it; this is the rest. Everything
downstream -- figures, the export to the frontend, the regime clustering, and
eventually D009's ForcingProvider -- comes through here, so the conventions get
enforced once rather than in each consumer.

THE ONE TRAP, AND IT IS WHY THIS NORMALISES BEFORE IT ASSERTS
-------------------------------------------------------------
D020 landed on 2026-09-15. Every file written before that date is in the OLD
convention, and several are still on disk. Measured 2026-09-16 on
`raw/era5_17-36N_82-63W_20210101-20210108.nc` (written 10 Sep):

    axes        latitude / longitude        NOT lat / lon
    latitude    36.00 -> 17.00  DESCENDING  NOT ascending
    longitude   -82.00 -> -63.00            NOT 0-360

while the five-year archive the cluster wrote on 2026-09-16 is fully compliant.
So the archive is MIXED, and a loader that only calls `assert_conventions`
rejects every local file we have. It has to `normalise_grid` first.

Doing that silently would be worse than the bug, though: a stale file would be
repaired on every read and nobody would ever learn it was stale. So the arrival
convention is recorded in `ds.attrs["sar_arrival_convention"]` and a pre-D020
file warns once, loudly, naming itself.

Closes nothing on its own. Feeds R3d (single tested ingestion module), R3e and
D021's export layer.
"""

import argparse
import warnings
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import xarray as xr

from sar.utils.geo import assert_conventions, normalise_grid, to_store_longitude

# D014 study box. Imported by figures and the exporter so the number lives in
# one place; `sar.fetch.wind` remains the source of truth for the fetch itself.
LAT_S, LAT_N = 17.0, 36.0
LON_W, LON_E = -82.0, -63.0

# A gap longer than this starts a new in-box drifter segment. Same value
# `sar.fetch.drifters.report()` uses, and it must stay the same: the 642
# segments quoted in D018 were counted with it.
SEGMENT_GAP = pd.Timedelta("3h")

# sst arrives in Kelvin carrying unmasked fill values -- the raw range observed
# on 2026-09-10 was -76.9 to 1000 C. ~99.9 % of values are plausible, so a naive
# mean looks ALMOST right, which is the worst kind of wrong. D013 needs Celsius
# and this filter.
SST_PLAUSIBLE_C = (-2.0, 40.0)

ARRIVAL_D020 = "d020"
ARRIVAL_PRE_D020 = "pre-d020"


def _arrival_convention(ds: xr.Dataset) -> str:
    """Which convention a dataset was stored in, judged before anything is changed.

    A file is pre-D020 if it fails on ANY of the three counts D020 fixes: axis
    names, latitude order, longitude range. Judged rather than assumed, because
    the archive currently holds both kinds.
    """
    if "lat" not in ds.coords or "lon" not in ds.coords:
        return ARRIVAL_PRE_D020

    lat = np.asarray(ds["lat"].values)
    if lat.size > 1 and not (np.diff(lat) > 0).all():
        return ARRIVAL_PRE_D020

    lon = np.asarray(ds["lon"].values)
    if lon.size and not ((lon >= 0.0).all() and (lon < 360.0).all()):
        return ARRIVAL_PRE_D020

    return ARRIVAL_D020


def open_forcing(path: str | Path) -> xr.Dataset:
    """Open a stored forcing file and return it in the D020 convention.

    Works on both wind (`era5_*.nc`) and current (`hycom_*.nc`) files -- it
    normalises the grid and does not care which variables are present.

    The returned dataset carries `attrs["sar_arrival_convention"]`, either
    `"d020"` or `"pre-d020"`. A pre-D020 file is repaired and warned about; it
    is not an error, because most of the local archive is still that way.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no forcing file at {path}")

    ds = xr.open_dataset(path)
    arrival = _arrival_convention(ds)

    ds = normalise_grid(ds)
    assert_conventions(ds)

    if arrival == ARRIVAL_PRE_D020:
        warnings.warn(
            f"{path.name} was written before D020 (2026-09-15) and has been normalised on "
            "read: check axis names, latitude order and longitude range at the source. "
            "Re-fetching it would produce a compliant file.",
            stacklevel=2,
        )

    ds.attrs["sar_arrival_convention"] = arrival
    ds.attrs["sar_source_path"] = str(path)
    return ds


def select_window(ds: xr.Dataset, start: str, end: str) -> xr.Dataset:
    """Slice a time window, `start` INCLUSIVE and `end` EXCLUSIVE.

    The project-wide convention, and the explicit fix for the off-by-one that
    both fetchers carried until 2026-09-15: xarray reads a bare date string as
    the WHOLE day, so `slice(start, end)` swallows all 24 hours of `end`. The
    local eight-day test slice has 192 timesteps for exactly this reason.

    Raises rather than returning an empty dataset -- an empty selection that
    propagates is the silent failure this whole module exists to prevent.
    """
    sub = ds.sel(time=slice(start, end))
    sub = sub.sel(time=sub.time < np.datetime64(end))

    if sub.sizes.get("time", 0) == 0:
        have = (f"{np.datetime_as_string(ds.time.values[0], unit='h')} to "
                f"{np.datetime_as_string(ds.time.values[-1], unit='h')}"
                if ds.sizes.get("time", 0) else "no time axis")
        raise ValueError(f"no timesteps in [{start}, {end}); file covers {have}")

    return sub


def _erddap_units_row(path: Path) -> list[int]:
    """`[1]` if the file carries ERDDAP's units row, `[]` if it does not.

    ERDDAP's tabledap CSV puts a units line immediately under the header:

        ID,time,latitude,longitude,ve,vn,sst,drogue_lost_date
        ,UTC,degrees_north,degrees_east,m/s,m/s,Kelvin,UTC
        145813,2019-06-04T05:00:00Z,30.82191,-63.00191,...

    `sar.fetch.drifters` drops it with `skiprows=[1]` when reading the live URL,
    but `raw/gdp_hourly_17-36N_82-63W_20190101-20240101.csv` as written on
    2026-09-10 still HAS it -- verified by reading the file on 2026-09-16. Left
    in, every numeric column loads as object dtype and the first comparison
    raises `'>=' not supported between instances of 'str' and 'float'`.

    Detected rather than assumed, so a correctly written file is not truncated
    by one observation: read the second line and ask whether its latitude is a
    number.
    """
    with path.open(encoding="utf-8") as fh:
        header = fh.readline().rstrip("\n").split(",")
        second = fh.readline().rstrip("\n").split(",")

    if "latitude" not in header or len(second) != len(header):
        return []
    try:
        float(second[header.index("latitude")])
    except ValueError:
        return [1]
    return []


def load_drifters(path: str | Path, *, in_box: bool = True) -> pd.DataFrame:
    """Read a GDP drifter CSV into the project's conventions.

    The raw file keeps ERDDAP's own names and units. This applies, in one place,
    every rule `sar.fetch.drifters.report()` established on 2026-09-10:

        latitude/longitude -> lat/lon, and lon to 0-360   (D020, as for grids)
        sst Kelvin -> sst_c Celsius, masked to [-2, 40]   (fill values, D013)
        undrogued  -> bool, per OBSERVATION not per buoy  (D018)
        segment_id -> contiguous in-box run, split on >3h gaps

    `undrogued` is per observation because `drogue_lost_date` is a single date
    per buoy: the same buoy is drogued before it and undrogued after. Treating
    it as a buoy-level flag would mislabel every pre-loss hour of 209 buoys.

    Returns ~926,533 rows for the full five-year pull, about 110 MB on disk.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no drifter CSV at {path}")

    df = pd.read_csv(path, skiprows=_erddap_units_row(path),
                     parse_dates=["time", "drogue_lost_date"])
    df = df.rename(columns={"latitude": "lat", "longitude": "lon"})

    missing = {"ID", "time", "lat", "lon"} - set(df.columns)
    if missing:
        raise ValueError(f"drifter CSV missing {sorted(missing)}; got {list(df.columns)}")

    df["ID"] = df["ID"].astype(str)

    if in_box:
        # Applied on DISPLAY longitude, before the 0-360 conversion, because the
        # file stores -82..-63 and the box constants are in the same frame.
        df = df[df.lat.between(LAT_S, LAT_N) & df.lon.between(LON_W, LON_E)]
        if df.empty:
            raise ValueError(f"no observations inside the study box in {path.name}")

    df["lon"] = to_store_longitude(df["lon"].to_numpy())

    if "sst" in df.columns:
        sst_c = df["sst"] - 273.15
        df["sst_c"] = sst_c.where(sst_c.between(*SST_PLAUSIBLE_C))

    if "drogue_lost_date" in df.columns:
        df["undrogued"] = df.drogue_lost_date.notna() & (df.time >= df.drogue_lost_date)
    else:
        df["undrogued"] = False

    df = df.sort_values(["ID", "time"]).reset_index(drop=True)
    gap = df.groupby("ID")["time"].diff() > SEGMENT_GAP
    new_buoy = df["ID"] != df["ID"].shift()
    df["segment_id"] = (gap | new_buoy).cumsum() - 1

    return df


def open_box_means(paths: str | Path | Sequence[str | Path]) -> pd.DataFrame:
    """Read one or more `wind_box_mean_*.parquet` into a single time-indexed frame.

    Accepts a directory, a single file, or a sequence of either. The five yearly
    files are ~385 KB each, so the whole five-year series is under 2 MB and is
    read whole -- this is the analytics tier D020 describes, and the input to
    the weather-regime clustering.

    Yearly files are written with `end` exclusive, so they abut without
    overlapping. Duplicate timestamps are therefore an error rather than
    something to drop quietly -- but the error names the files that overlap,
    because in practice the cause is usually mundane. `C:/maritime-data/derived`
    on 2026-09-16 held the five yearly files AND three short test slices left
    over from 10 Sep, all inside 2021: 312 duplicate hours, and nothing wrong
    with any individual file.
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]

    files: list[Path] = []
    for p in paths:
        p = Path(p)
        files.extend(sorted(p.glob("wind_box_mean_*.parquet")) if p.is_dir() else [p])

    if not files:
        raise FileNotFoundError(f"no wind_box_mean_*.parquet found under {paths}")

    frames = {f: pd.read_parquet(f) for f in files}
    df = pd.concat(frames.values()).sort_index()

    dupes = int(df.index.duplicated().sum())
    if dupes:
        # Name the files that share timestamps, smallest first -- a short slice
        # sitting inside a yearly file is the common case, and the short one is
        # nearly always the one to remove.
        clashing = df.index[df.index.duplicated()]
        culprits = sorted(
            ((f, len(d)) for f, d in frames.items() if d.index.isin(clashing).any()),
            key=lambda kv: kv[1],
        )
        listed = "\n".join(f"    {f.name}  ({n:,} rows)" for f, n in culprits)
        raise ValueError(
            f"{dupes:,} duplicate timestamps across {len(files)} box-mean files. These "
            f"overlap:\n{listed}\n"
            "  Yearly files are written end-exclusive and should abut. A short slice "
            "inside a yearly file is a leftover -- pass the files you want explicitly, or "
            "move the slice out of the directory."
        )

    return df


def main() -> None:
    p = argparse.ArgumentParser(description="Open what the fetchers wrote, and describe it.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--forcing", help="an era5_*.nc or hycom_*.nc file")
    g.add_argument("--drifters", help="a gdp_hourly_*.csv file")
    g.add_argument("--box-means", help="a derived/ directory or a single parquet")
    p.add_argument("--start", help="with --forcing: window start, inclusive")
    p.add_argument("--end", help="with --forcing: window end, EXCLUSIVE")
    args = p.parse_args()

    if args.forcing:
        ds = open_forcing(args.forcing)
        if args.start and args.end:
            ds = select_window(ds, args.start, args.end)
        lat, lon = ds.lat.values, ds.lon.values
        print(f"arrived as   : {ds.attrs['sar_arrival_convention']}")
        print(f"variables    : {list(ds.data_vars)}")
        print(f"grid         : {lat.size} lat x {lon.size} lon")
        print(f"lat          : {lat[0]:.2f} -> {lat[-1]:.2f}  step {np.diff(lat)[0]:+.3f}")
        print(f"lon          : {lon[0]:.2f} -> {lon[-1]:.2f}  step {np.diff(lon)[0]:+.3f}  (0-360)")
        print(f"time         : {np.datetime_as_string(ds.time.values[0], unit='h')} -> "
              f"{np.datetime_as_string(ds.time.values[-1], unit='h')}   n = {ds.sizes['time']}")
        cells = lat.size * lon.size
        print(f"one timestep : {cells * 2 * 4 / 1024:.1f} KB as u/v float32")

    elif args.drifters:
        df = load_drifters(args.drifters)
        print(f"observations : {len(df):,}")
        print(f"buoys        : {df.ID.nunique()}")
        print(f"time         : {df.time.min()} -> {df.time.max()}")
        print(f"undrogued    : {100 * df.undrogued.mean():.1f} %  <- the leeway-capable set")
        print(f"segments     : {df.segment_id.nunique():,} contiguous in-box runs")
        # The `+ 1` counts OBSERVATIONS, not elapsed span: a run from 00:00 to
        # 47:00 is 48 hourly fixes. `sar.fetch.drifters.report()` defines it that
        # way and D018's cited "642 segments >= 48 h" was counted with it, so
        # dropping the +1 silently reports 640 and makes the vault look wrong.
        hrs = df.groupby("segment_id").time.agg(
            lambda t: (t.max() - t.min()).total_seconds() / 3600 + 1
        )
        for h in (24, 48, 72):
            print(f"  >= {h:3d} h   : {int((hrs >= h).sum()):,} segments")

    else:
        df = open_box_means(args.box_means)
        print(f"rows         : {len(df):,} hourly steps")
        print(f"time         : {df.index.min()} -> {df.index.max()}")
        print(f"columns      : {list(df.columns)}")
        print(f"mean speed   : {df.speed.mean():.2f} m/s   max {df.speed.max():.2f} m/s")
        print(f"mean msl     : {df.msl_hpa.mean():.1f} hPa   "
              f"mean spread {df.msl_spread_hpa.mean():.1f} hPa")


if __name__ == "__main__":
    main()
