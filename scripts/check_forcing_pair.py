"""
check_forcing_pair.py -- prove wind and current describe the same place and time,
BEFORE nine hours are spent pulling five years of either.

Every failure mode here is silent. A longitude convention error, a flipped
latitude axis, a cm/s-vs-m/s mix-up or a three-hour time offset all produce a
simulation that runs perfectly and puts particles in the wrong ocean. Nothing
crashes; the maps look plausible. See `concepts/Grids, NetCDF and the convention
traps` in the vault -- those are traps 1 to 6, and this script is the test for them.

It is also R3f, the last open Week 1 exit criterion: the currents quiver plot.

    python scripts/check_forcing_pair.py --date 2021-01-05

Pulls ONE day of each field (24 hourly wind steps, 8 three-hourly current steps,
a few MB) and asserts the pairing rather than printing it. Exit code 0 means the
two fields can be handed to the same ForcingProvider (D009).

WHAT IT DOES NOT DO: regrid. The fields are never merged onto a common grid, and
they do not have to be -- D009's provider samples each independently at the
particle's own (x, y, t). What must agree is the CONVENTIONS, not the grids.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr

from sar.fetch.current import LAT_N, LAT_S, LON_E, LON_W, fetch_current_box
from sar.fetch.wind import fetch_wind_box
from sar.utils.geo import assert_conventions, to_display_longitude

# Gulf Stream surface speed, D001 / concepts/Why the Gulf Stream. This is the
# cm/s-vs-m/s check from trap #3: a core reading ~180 instead of ~1.8 is a
# factor-of-100 error that looks entirely plausible on a colour map.
STREAM_CORE_MIN, STREAM_CORE_MAX = 0.8, 3.0
HATTERAS_LAT = 35.2

_fails = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        _fails.append(name)


def main():
    p = argparse.ArgumentParser(description="Verify wind and current pair correctly")
    p.add_argument("--date", required=True, help="UTC date, YYYY-MM-DD. One day.")
    p.add_argument("--figures", default="figures", help="where to write the quiver plot")
    args = p.parse_args()

    day = args.date
    nxt = str(np.datetime64(day) + np.timedelta64(1, "D"))

    print(f"fetching one day of each field for {day} ...")
    wind = fetch_wind_box(day, nxt)
    print(f"  wind    {dict(wind.sizes)}")
    cur = fetch_current_box(day, nxt, (LAT_S, LAT_N), (LON_W, LON_E))
    print(f"  current {dict(cur.sizes)}")

    print("\n--- window size (end must be EXCLUSIVE) ---")
    # A one-day window is 24 hourly wind steps and 8 three-hourly current steps.
    # Until 2026-09-15 both fetchers returned 48 and 16: xarray reads a bare date
    # string as the whole day, so slice(start, end) swallowed all of `end` too.
    # Pulled year by year that duplicates 1 January at every boundary -- and the
    # "one verified week" of 2026-09-10 was in fact eight days, 192 steps.
    check("wind returns 24 hourly steps for a one-day window",
          wind.sizes["time"] == 24, f"got {wind.sizes['time']}")
    check("current returns 8 three-hourly steps for a one-day window",
          cur.sizes["time"] == 8, f"got {cur.sizes['time']}")

    print("\n--- conventions (D020) ---")
    for label, ds in (("wind", wind), ("current", cur)):
        try:
            assert_conventions(ds)
            check(f"{label}: axes lat/lon, ascending, longitude 0-360", True)
        except AssertionError as exc:
            check(f"{label}: axes lat/lon, ascending, longitude 0-360", False, str(exc))

    print("\n--- co-location ---")
    wl, cl = wind.lat.values, cur.lat.values
    wo, co = wind.lon.values, cur.lon.values
    check("latitude ranges overlap",
          max(wl.min(), cl.min()) < min(wl.max(), cl.max()),
          f"wind {wl.min():.2f}..{wl.max():.2f}  current {cl.min():.2f}..{cl.max():.2f}")
    check("longitude ranges overlap",
          max(wo.min(), co.min()) < min(wo.max(), co.max()),
          f"wind {wo.min():.2f}..{wo.max():.2f}  current {co.min():.2f}..{co.max():.2f}")
    print(f"         wind    grid {wl.size:4d} x {wo.size:4d} at "
          f"{abs(np.diff(wl)[0]):.3f} lat x {abs(np.diff(wo)[0]):.3f} lon deg")
    print(f"         current grid {cl.size:4d} x {co.size:4d} at "
          f"{abs(np.diff(cl)[0]):.3f} lat x {abs(np.diff(co)[0]):.3f} lon deg"
          "   <- NOT 1/12 deg and NOT square (measured 2026-09-15)")

    print("\n--- time axes ---")
    wt = set(wind.time.values.astype("datetime64[s]").tolist())
    ct = cur.time.values.astype("datetime64[s]").tolist()
    orphans = [t for t in ct if t not in wt]
    check("every current timestamp exists in the hourly wind axis",
          not orphans,
          "so no time interpolation is needed at coincident stamps" if not orphans
          else f"{len(orphans)} orphan stamps, e.g. {orphans[:2]}")
    hours = sorted({t.hour for t in ct})
    check("current stamps sit on whole 3-hour boundaries",
          all(h % 3 == 0 for h in hours) and all(t.minute == 0 for t in ct),
          f"hours {hours}")

    print("\n--- units and sign (traps 3 and 4) ---")
    spd = np.sqrt(cur.water_u ** 2 + cur.water_v ** 2)
    core = float(spd.quantile(0.999))
    check("Gulf Stream core speed is m/s, not cm/s",
          STREAM_CORE_MIN <= core <= STREAM_CORE_MAX,
          f"99.9th pct = {core:.2f} m/s (expect {STREAM_CORE_MIN}-{STREAM_CORE_MAX}; "
          "~180 would mean cm/s)")
    check("current units attribute says m/s",
          cur.water_u.attrs.get("units", "").replace(" ", "") in ("m/s", "meters/second"),
          f"units={cur.water_u.attrs.get('units')!r}")

    # The Stream is a NORTHEASTWARD jet. If v is negative in the core then the
    # sign convention is inverted and every particle is sent down-coast -- which
    # looks perfectly plausible on a map. That is trap #4.
    fast = spd.values > np.nanquantile(spd.values, 0.99)
    with np.errstate(invalid="ignore"):
        u_core = float(np.nanmean(np.where(fast, cur.water_u.values, np.nan)))
        v_core = float(np.nanmean(np.where(fast, cur.water_v.values, np.nan)))
    check("fastest water flows NORTHEASTWARD (u>0, v>0)",
          u_core > 0 and v_core > 0,
          f"core mean u={u_core:+.2f}, v={v_core:+.2f} m/s")

    print("\n--- land mask (trap 6) ---")
    land = float(np.isnan(cur.water_u.isel(time=0).values).mean())
    check("current carries a land mask, plausible for this box",
          0.02 < land < 0.40, f"{land:.1%} of cells are NaN")
    check("wind carries NO land mask (ERA5 10 m wind is defined over land too)",
          not bool(np.isnan(wind.u10.values).any()),
          "a particle over land still gets wind; D016's beaching rule is what stops it")

    print("\n--- joint sampling at the same physical points ---")
    rng = np.random.default_rng(0)
    pts = list(zip(rng.uniform(cl.min() + 1, cl.max() - 1, 20),
                   rng.uniform(co.min() + 1, co.max() - 1, 20)))
    t0 = cur.time.values[0]
    both = 0
    for la, lo in pts:
        w = wind.sel(lat=la, lon=lo, time=t0, method="nearest")
        c = cur.sel(lat=la, lon=lo, time=t0, method="nearest")
        if not np.isnan(float(w.u10)) and not np.isnan(float(c.water_u)):
            both += 1
    check("both fields answer at the same arbitrary point",
          both >= 12, f"{both}/20 sample points returned both (the rest are land)")

    print("\n--- R3f: the quiver plot ---")
    quiver(cur, args.figures, day)

    print()
    if _fails:
        print(f"FAILED {len(_fails)} check(s): " + "; ".join(_fails))
        sys.exit(1)
    print("All checks passed. The two fields can be paired.")


def quiver(cur: xr.Dataset, fig_dir: str, day: str) -> None:
    """The Week 1 exit criterion, as a NUMBER and not only a picture.

    The eyeball version of this test is "does it look like the Gulf Stream",
    which is not re-runnable and is not evidence. So this also emits the jet
    separation longitudes, which a later run can regress on -- the same trick
    that turned the wind check into a median 103.6 degrees on 2026-09-10.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    snap = cur.isel(time=0)
    s = np.sqrt(snap.water_u ** 2 + snap.water_v ** 2).values
    lat = snap.lat.values
    lon = to_display_longitude(snap.lon.values)
    order = np.argsort(lon)
    lon, s = lon[order], s[:, order]
    u = snap.water_u.values[:, order]
    v = snap.water_v.values[:, order]

    # Where is the jet at each latitude? South of Cape Hatteras it hugs the
    # Florida and Carolina shelf; north of it the Stream separates and runs
    # offshore. That separation is the single most recognisable feature here,
    # so it is what gets measured.
    with np.errstate(invalid="ignore"):
        jet_lon = np.array([lon[np.nanargmax(row)] if np.isfinite(row).any() else np.nan
                            for row in s])
    south = np.nanmean(jet_lon[lat < HATTERAS_LAT])
    north = np.nanmean(jet_lon[lat >= HATTERAS_LAT])
    print(f"  mean jet longitude south of {HATTERAS_LAT} N : {south:7.2f}")
    print(f"  mean jet longitude north of {HATTERAS_LAT} N : {north:7.2f}")
    check(f"jet runs OFFSHORE (further east) north of {HATTERAS_LAT} N",
          bool(np.isfinite(south) and np.isfinite(north) and north > south),
          f"delta = {north - south:+.2f} deg east")

    step = max(1, len(lat) // 45)
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.pcolormesh(lon, lat, s, shading="auto", cmap="viridis", vmin=0, vmax=2.0)
    ax.quiver(lon[::step], lat[::step], u[::step, ::step], v[::step, ::step],
              scale=25, width=0.0018, color="white", alpha=0.85)
    ax.axhline(HATTERAS_LAT, color="red", lw=0.8, ls="--", alpha=0.7)
    ax.text(lon.min() + 0.3, HATTERAS_LAT + 0.2, "Cape Hatteras 35.2 N",
            color="red", fontsize=8)
    fig.colorbar(im, ax=ax, label="surface current speed (m/s)")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title(
        f"HYCOM surface current -- {str(snap.time.values)[:16]} UTC\n"
        "PASS: coherent NE jet hugging Florida, separating offshore near 35.2 N"
    )
    Path(fig_dir).mkdir(parents=True, exist_ok=True)
    out = Path(fig_dir) / f"current_quiver_{day.replace('-', '')}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
