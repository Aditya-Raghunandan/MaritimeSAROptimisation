"""
wind.py — pull a region+period subset of ERA5 from ARCO-ERA5
(Google Cloud, Zarr, anonymous) and write four things:

  1. data/raw/era5_<box>_<window>.nc            10u, 10v, msl for the box+window
  2. data/derived/wind_box_mean_<window>.parquet per-hour scalars for analytics
  3. figures/wind_rose_<window>.png              wind direction distribution
  4. figures/pressure_wind_check_<window>.png    THE CORRECTNESS CHECK — see below

No CDS account, no queue. Same numbers as the CDS product.

Run it from a shell that has internet access. Install once, from the repo root:
    python -m venv .venv && pip install -e ".[dev]"   # pins live in pyproject.toml

Then, ONE WEEK first:
    python -m sar.fetch.wind --start 2019-01-01 --end 2019-01-08 --out C:/maritime-data
    # on the cluster:  --out /home/26p67/data   (NOT /data1 -- it is not writable)

  --out IS NOT OPTIONAL in practice: it defaults to "data" RELATIVE TO THE
  WORKING DIRECTORY, so running this from the vault root writes raw NetCDF into
  OneDrive, which HOW WE WORK and data/README both forbid.  One week is 13.7 MB;
  one year is 623 MB.

Then a year at a time. R3 mitigation, and it makes the debug loop seconds
instead of minutes.

--------------------------------------------------------------------------
WHY msl IS IN HERE (decisions/D014, concepts/Temperature, pressure and wind)
--------------------------------------------------------------------------
Mean sea level pressure is NOT a forcing term. It never enters the drift
equation — the 10 m wind already IS the atmosphere's response to the pressure
field, so using both as forces would double-count. It is here for three
separate reasons:

  1. CORRECTNESS CHECK.  Wind must run roughly ALONG the isobars, clockwise
     around a high in the northern hemisphere, tilted 10-20 degrees outward by
     surface friction.  If the plotted vectors run ACROSS the contours, or
     circle the wrong way, we have a sign error or u and v are swapped —
     the exact silent bug that produces a plausible-looking, wrong drift model.
     The quiver plot alone does not catch this; this figure does.
  2. REGIME NAMING.  The pressure pattern is a compact fingerprint of the whole
     synoptic situation (Bermuda High dominant / frontal passage / tropical
     system).  Clustering on it gives scenarios with names and causes rather
     than "cluster 3", which is what D006's held-out split needs.
  3. CLIMATOLOGY.  Tells us in advance what fraction of the record is the
     boring default state, so scenario sampling is planned rather than assumed.

It costs one extra string in VARS and roughly 1 GB over the full five years.
"""

import argparse
import os
from pathlib import Path

import dask
import numpy as np
import pandas as pd
import xarray as xr

from sar.utils.geo import assert_conventions, normalise_grid, to_display_longitude

ARCO = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"

# MEASURED 2026-09-10, not assumed.  This store uses FULL CF-STYLE NAMES.
# "10u"/"10v"/"msl" are the CDS/GRIB short codes and they DO NOT EXIST here --
# selecting them raises KeyError: '10u'.  That was bug #1.
VARS = {
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "mean_sea_level_pressure": "msl",
}

# The store is chunked [1, 721, 1440] -- ONE CHUNK IS A FULL GLOBAL TIMESTEP.
# Subsetting our 77x77 box therefore does NOT reduce what crosses the wire, and
# the cost is pure per-chunk latency.  Measured: 0.72 s/chunk serial -> 26 h for
# five years; 0.246 s/chunk at 32 threads -> 9 h.  Concurrency is the whole fix.
# Concurrency for the ARCO pull. Overridable because 32 threads is right for a
# dedicated machine and antisocial on a shared login node -- on 2026-09-15 this
# ran at 483 % CPU on jaguar1 and made the box unusable for its actual users.
# scripts/run_on_cluster.sh caps it. See D017.
THREADS = int(os.environ.get("SAR_FETCH_THREADS", "32"))

# D014 study box — full Bermuda Triangle, northern edge pushed to 36 N so the
# Gulf Stream's separation and meander field at Cape Hatteras (~35.2 N) is inside
# the domain rather than clipped by it.
# CHANGE THESE ONCE, HERE, AND NOWHERE ELSE.
LAT_N, LAT_S = 36.0, 17.0
LON_W, LON_E = -82.0, -63.0

# D014 study period, CORRECTED 2026-09-10 and confirmed 2026-09-11.
# Was 2021-01-01 -> 2026-01-01.  Three independent product end dates are earlier
# than the docs pages claim, and the old window fell off two of them:
#   HYCOM GLBy0.08 expt_93.0 currents   end 2024-09-05 09:00 UTC
#   NOAA GDP drifter_hourly_qc          end 2022-10-31
#   ERA5 final (this store)             end 2026-05-31   <- the only one that was fine
# 2019-01-01 -> 2024-01-01 keeps all five complete calendar years (five hurricane
# seasons, five winter storm seasons) and sits wholly inside HYCOM.
# CHANGE THESE ONCE, HERE, AND NOWHERE ELSE.
STUDY_START = "2019-01-01"
STUDY_END   = "2024-01-01"

# Pull it ONE YEAR AT A TIME.  The store is chunked at one timestep per chunk, so
# a five-year request is ~44,000 chunk reads in a single call.  Measured: 0.246
# s/chunk at 32 threads -> ~1.8 h per year, ~9 h for the full window.

# Allen (2000), "The Leeway of Persons-In-Water and Three Small Craft",
# USCG R&D Center, DTIC ADA376479: downwind slope 1.93 % (sd 0.083 m/s) for a
# PIW, 2.7 % (SE 0.133 kn, 5-12 kn winds) for a PIW in a survival suit.  The
# 1-4 % RANGE in R1c is right; the 3 % midpoint this file used was not.
ALPHA_MID = 0.02      # midpoint leeway coefficient, 1-4 % range (D002, D018)
CURRENT_TYPICAL = 1.8  # m/s, Gulf Stream surface average (D001)


def open_wind_store() -> xr.Dataset:
    """Open ARCO-ERA5 lazily. Nothing is downloaded until .load().

    Mirrors `sar.fetch.current.open_current_dataset` deliberately: the two
    fetchers are the same shape so a caller can pair them without special-casing
    either one (D020).

    consolidated=True IS LOAD-BEARING. Despite the ".zarr-v3" in the path the
    store is Zarr FORMAT 2 with a .zmetadata sidecar (there is no zarr.json).
    Without this flag, zarr 3.x tries to discover the layout across 277 arrays
    and open_zarr never returns -- observed hanging past 30 minutes with no
    error and no output. That was bug #2 of 2026-09-10.
    """
    return xr.open_zarr(ARCO, chunks={"time": 1},
                        storage_options={"token": "anon"}, consolidated=True)


def fetch_wind_box(start: str, end: str,
                   lat_bounds: tuple = (LAT_S, LAT_N),
                   lon_bounds: tuple = (LON_W, LON_E)) -> xr.Dataset:
    """Every hourly wind record in a lat/lon box between two dates.

    `start` inclusive, `end` exclusive, same convention as
    `sar.fetch.current.fetch_current_box`. Returns a loaded, D020-normalised
    dataset: axes named lat/lon, both ascending, longitude 0-360.
    """
    lat_s, lat_n = lat_bounds
    ds = open_wind_store()

    # The two traps, handled explicitly:
    #   1. ERA5 latitude runs 90 -> -90, so slice(17, 36) returns EMPTY, silently.
    #   2. ERA5 longitude runs 0 -> 360. -82 W is 278.
    sub = ds[list(VARS)].sel(
        time=slice(start, end),
        latitude=slice(lat_n, lat_s),          # descending, on purpose
        longitude=slice(lon_bounds[0] % 360, lon_bounds[1] % 360),
    )
    # xarray reads a bare date string as the WHOLE day, so slice(start, end)
    # includes all of `end` -- 24 extra hours. Pulling year by year, that
    # duplicates 1 January at every boundary and leaves the concatenated
    # archive with repeated timestamps. Make `end` genuinely exclusive.
    sub = sub.sel(time=sub.time < np.datetime64(end))

    assert sub.sizes["latitude"] > 0, "empty latitude -- check the descending slice"
    assert sub.sizes["longitude"] > 0, "empty longitude -- check the 0-360 convention"
    assert sub.sizes["time"] > 0, f"no timesteps in [{start}, {end})"

    sub = sub.rename(VARS)
    with dask.config.set(scheduler="threads", num_workers=THREADS):
        sub = sub.load()

    # The padded-axis trap: ARCO's time axis runs 1900-2050 but valid data stops
    # at 2026-05-31, and a slice past it returns ALL-NaN rather than raising.
    if bool(np.isnan(sub["u10"].values).all()):
        raise SystemExit("all-NaN wind -- window is outside the store's VALID range")

    sub = normalise_grid(sub)
    assert_conventions(sub)
    return sub


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--start", required=True, help="UTC date, YYYY-MM-DD, inclusive")
    p.add_argument("--end", required=True, help="UTC date, YYYY-MM-DD, exclusive")
    p.add_argument("--out", default="data")
    args = p.parse_args()

    # Guard the study window.  Asking for data outside it is not an error the
    # store will report -- ERA5 covers 1940-2026, so an out-of-window request
    # succeeds and quietly produces forcing that no scenario, drifter track or
    # current field can be paired with.  D014 / D019.
    if not (STUDY_START <= args.start < args.end <= STUDY_END):
        print(f"WARNING: {args.start}..{args.end} lies outside the D014 study "
              f"window {STUDY_START}..{STUDY_END}. Continuing, but nothing "
              f"downstream will match it.")

    tag = f"{args.start.replace('-','')}-{args.end.replace('-','')}"
    box = f"{int(LAT_S)}-{int(LAT_N)}N_{abs(int(LON_W))}-{abs(int(LON_E))}W"

    raw_dir = Path(args.out) / "raw"
    der_dir = Path(args.out) / "derived"
    fig_dir = Path("figures")
    for d in (raw_dir, der_dir, fig_dir):
        d.mkdir(parents=True, exist_ok=True)

    print("opening ARCO-ERA5 (lazy, nothing downloaded yet) ...")
    # consolidated=True IS LOAD-BEARING.  Despite the ".zarr-v3" in the path the
    # store is Zarr FORMAT 2 with a .zmetadata sidecar (there is no zarr.json).
    # Without this flag, zarr 3.x tries to discover the layout across 277 arrays
    # and open_zarr never returns -- observed hanging past 30 minutes with no
    # error and no output.  That was bug #2.
    ds = xr.open_zarr(ARCO, chunks={"time": 1},
                      storage_options={"token": "anon"}, consolidated=True)

    # The advertised range is not the real one.  Record what the store actually
    # holds — this is the number that goes in the notebook, not the docs page.
    # The axis is PADDED 1900-01-01 -> 2050-12-31 (1,323,648 steps).  The real
    # data range lives in the attributes, and a slice outside it returns
    # ALL-NaN rather than raising.  Measured 2026-09-10:
    #   valid_time_start 1940-01-01, valid_time_stop 2026-05-31 (final ERA5),
    #   valid_time_stop_era5t 2026-09-04 (preliminary, gets revised).
    print("  padded axis :", str(ds.time.values[0]), "->", str(ds.time.values[-1]))
    print("  VALID final :", ds.attrs.get("valid_time_start"), "->",
          ds.attrs.get("valid_time_stop"))
    print("  VALID era5t :", ds.attrs.get("valid_time_stop_era5t"), "(preliminary)")

    # ---- the two traps, handled explicitly -------------------------------
    # 1. ERA5 latitude runs 90 -> -90. slice(17, 36) returns EMPTY, silently.
    # 2. ERA5 longitude runs 0 -> 360. -82 W is 278.
    lon_w = LON_W % 360
    lon_e = LON_E % 360
    sub = ds[list(VARS)].sel(
        time=slice(args.start, args.end),
        latitude=slice(LAT_N, LAT_S),      # descending, on purpose
        longitude=slice(lon_w, lon_e),
    )
    assert sub.sizes["latitude"] > 0, "empty latitude — check the descending slice"
    assert sub.sizes["longitude"] > 0, "empty longitude — check the 0-360 convention"
    print("  grid:", dict(sub.sizes))

    sub = sub.rename(VARS)
    n_chunks = sub.sizes["time"] * len(VARS)
    print(f"downloading {n_chunks} chunks on {THREADS} threads ...")
    with dask.config.set(scheduler="threads", num_workers=THREADS):
        sub = sub.load()

    # Guard the padded-axis trap: an out-of-range window is silently all-NaN.
    if bool(np.isnan(sub["u10"].values).all()):
        raise SystemExit("all-NaN wind -- window is outside the store's VALID range")

    # D020: store what the server served. Longitude stays 0-360, the axes are
    # renamed latitude/longitude -> lat/lon to match src/sar/fetch/current.py,
    # and both are sorted ascending. This file used to convert back to
    # -180..180 here, which meant the same Gulf Stream cell was -70.0 in the
    # wind file and 290.0 in the current file. Display conversion now happens
    # in the figure functions only.
    sub = normalise_grid(sub)
    assert_conventions(sub)

    raw_path = raw_dir / f"era5_{box}_{tag}.nc"
    sub.to_netcdf(raw_path)
    print("wrote", raw_path)

    # ---- derived scalars: this is what the analytics actually read -------
    u = sub["u10"].mean(dim=("lat", "lon"))
    v = sub["v10"].mean(dim=("lat", "lon"))
    speed = np.hypot(u, v)
    # meteorological convention: direction the wind comes FROM, degrees
    direction = (270.0 - np.degrees(np.arctan2(v, u))) % 360.0
    msl_mean = sub["msl"].mean(dim=("lat", "lon")) / 100.0   # Pa -> hPa
    msl_range = (sub["msl"].max(dim=("lat", "lon"))
                 - sub["msl"].min(dim=("lat", "lon"))) / 100.0

    df = pd.DataFrame(
        {
            "time": sub.time.values,
            "u10": u.values,
            "v10": v.values,
            "speed": speed.values,
            "dir_from_deg": direction.values,
            "msl_hpa": msl_mean.values,
            "msl_spread_hpa": msl_range.values,   # proxy for gradient strength
        }
    ).set_index("time")
    df["hour_utc"] = df.index.hour

    der_path = der_dir / f"wind_box_mean_{tag}.parquet"
    df.to_parquet(der_path)
    print("wrote", der_path)

    report(df)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    wind_rose(df, args, fig_dir, tag, plt)
    pressure_wind_check(sub, fig_dir, tag, plt)


def report(df: pd.DataFrame) -> None:
    """Use 3 — what does 'normal' actually look like in this box."""
    print("\n--- wind speed, box mean, m/s ---")
    print(df["speed"].describe().round(2).to_string())

    bands = [(0, 5, "calm"), (5, 10, "moderate"), (10, 15, "fresh"), (15, 99, "strong")]
    print("\n--- how much of the record is the boring default state ---")
    for lo, hi, name in bands:
        frac = ((df["speed"] >= lo) & (df["speed"] < hi)).mean() * 100
        print(f"  {name:9s} {lo:2d}-{hi:2d} m/s : {frac:5.1f} %")

    print(f"\n--- leeway implied at alpha = {ALPHA_MID:.0%}, against a "
          f"~{CURRENT_TYPICAL} m/s current ---")
    for q in (0.50, 0.90, 0.99):
        w = df["speed"].quantile(q)
        lee = ALPHA_MID * w
        print(f"  p{int(q*100):02d} wind {w:5.2f} m/s -> leeway {lee:5.3f} m/s "
              f"({100*lee/CURRENT_TYPICAL:4.1f} % of current)")
    print("  (the regime where leeway stops being negligible is the one that")
    print("   justifies the three-term model empirically — D002, R1)")

    # Day/night check.  Beware: the inertial period is 23.9 h at 30 N, so a ~24 h
    # signal in the CURRENT field is near-inertial, not solar.  In the WIND field a
    # diurnal cycle is genuinely solar.  See the 2026-09-08 notebook entry.
    print("\n--- mean wind speed by hour of day (UTC) ---")
    print(df.groupby("hour_utc")["speed"].mean().round(3).to_string())

    print("\n--- mean sea level pressure, box mean, hPa ---")
    print(df["msl_hpa"].describe().round(2).to_string())
    print("  (a high box mean with a SMALL msl_spread_hpa is the Bermuda High")
    print("   sitting on us: weak gradient, light wind, the default state)")


def wind_rose(df, args, fig_dir, tag, plt) -> None:
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(6, 6))
    theta = np.radians(df["dir_from_deg"].values)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    counts, edges = np.histogram(theta, bins=36, range=(0, 2 * np.pi))
    ax.bar(edges[:-1], counts, width=np.diff(edges), align="edge", alpha=0.8)
    ax.set_title(f"ERA5 10 m wind direction (from)\n{args.start} to {args.end}")
    path = fig_dir / f"wind_rose_{tag}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def pressure_wind_check(sub, fig_dir, tag, plt) -> None:
    """Use 2 — the correctness check.

    Contour mean sea level pressure, overlay the 10 m wind vectors, one snapshot.

    WHAT PASSING LOOKS LIKE
      - arrows run roughly ALONG the contours, not across them
      - around a high (a closed contour with the largest values) they go CLOCKWISE
      - they are tilted slightly OUTWARD from the high — surface friction, 10-20 deg
      - arrows are longest where contours are closest together

    WHAT FAILING LOOKS LIKE, AND WHAT IT MEANS
      - arrows point straight down the gradient, high to low  -> Coriolis missing;
        almost certainly you are looking at a derived/geostrophic field, not 10u/10v
      - circulation is ANTICLOCKWISE around the high          -> v sign flipped
      - arrows perpendicular to where they should be          -> u and v swapped
      - arrows fine but the map is mirrored                   -> latitude slice
                                                                 ascending/descending
    """
    t = sub.time.values[len(sub.time) // 2]      # a mid-window snapshot
    snap = sub.sel(time=t)

    # 0-360 on disk (D020), -180..180 on the axis so it reads as "70 W".
    # This is the single presentation-boundary conversion that decision buys.
    lon = to_display_longitude(snap.lon.values)
    lat = snap.lat.values
    order = np.argsort(lon)
    lon = lon[order]
    p_hpa = snap["msl"].values[:, order] / 100.0
    u = snap["u10"].values[:, order]
    v = snap["v10"].values[:, order]

    step = max(1, len(lon) // 28)                # thin the arrows so they read

    fig, ax = plt.subplots(figsize=(9, 8))
    cs = ax.contour(lon, lat, p_hpa, levels=14, linewidths=1.0, colors="k", alpha=0.55)
    ax.clabel(cs, inline=True, fontsize=7, fmt="%d")
    ax.quiver(lon[::step], lat[::step],
              u[::step, ::step], v[::step, ::step],
              np.hypot(u, v)[::step, ::step],
              cmap="viridis", scale=250, width=0.0028)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title(
        f"ERA5 msl contours + 10 m wind — {str(t)[:16]} UTC\n"
        "PASS: arrows along the contours, clockwise around a high, tilted slightly outward",
        fontsize=10)
    ax.set_aspect("equal", adjustable="box")
    path = fig_dir / f"pressure_wind_check_{tag}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)
    print("  ^ LOOK AT THIS ONE before trusting anything downstream.")


if __name__ == "__main__":
    main()
