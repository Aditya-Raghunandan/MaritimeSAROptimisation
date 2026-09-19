"""
fields.py -- matplotlib renderers for ANY u/v field, and the numbers they measure.

    python -m sar.viz.fields --forcing <data>/raw/era5_17-36N_82-63W_20210101-20210108.nc
    python -m sar.viz.fields --box-means <data>/derived/wind_box_mean_20210101-20220101.parquet

`code/MAP.md` has planned this module since 3 Sep with one constraint, and it is
the constraint that shapes the whole file: *"one plotting module for ANY u/v
field -- quiver, streamplot, wind rose. Wind, current and ensemble all go
through it; IT MUST NOT KNOW WHICH PRODUCT IT WAS HANDED."* So every function
here takes arrays and labels, never a "wind dataset" or a "current dataset". The
drift ensemble will render through the same code.

EVERY FUNCTION RETURNS A DICT OF NUMBERS
----------------------------------------
`scripts/check_forcing_pair.py` established the pattern on 2026-09-15: it does
not just draw the Gulf Stream, it measures the jet as sitting 1.35 deg further
east north of Cape Hatteras than south of it, and asserts that. The reasoning
there applies to every figure in this project:

    "The eyeball version of this test is 'does it look like the Gulf Stream',
     which is not re-runnable and is not evidence."

A figure is a claim. The dict is what makes the claim checkable next month, and
what `figures/FIGURES.md` rows are written from.

D009 forbids the engine importing a plotting library. Nothing here is imported
by the engine; this is a consumer, and matplotlib is imported inside functions
so that merely importing this module costs nothing.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sar.utils.data_io import open_box_means, open_forcing, select_window
from sar.utils.geo import to_display_longitude

# Above this, leeway stops being a rounding error on the drift budget, and it is
# also where L1 bites: ERA5 underestimates winds over ~10 m/s by about 10 %, so
# the fraction of time spent above it is the size of that exposure.
LEEWAY_MATTERS_MS = 10.0

# Allen 2000, frozen by D002. Reported beside the wind so the leeway a given
# wind implies is readable off the figure rather than computed in someone's head.
ALPHA = 0.02


def _plt():
    """Import matplotlib configured for headless use.

    Imported inside the call, immediately after setting the Agg backend, exactly
    as `wind.py` and `check_forcing_pair.py` do -- a module-level import picks up
    whatever backend is already active and fails differently on the cluster than
    on a laptop.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save(fig, path: Path, plt, dpi: int = 150) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)
    return path


def _display_axes(lat, lon, *arrays):
    """Convert to display longitude and re-sort every array to match.

    The `argsort` is not optional and is the reason this is a helper. Wrapping
    0-360 to -180..180 leaves longitude UNSORTED (278..297 maps to -82..-63
    fine, but any box spanning the meridian does not), and pcolormesh on an
    unsorted axis draws a mirrored smear rather than raising. Every plot in this
    repository does this dance; doing it in one place means it can be got wrong
    only once.
    """
    lon_d = to_display_longitude(np.asarray(lon))
    order = np.argsort(lon_d)
    return (np.asarray(lat), lon_d[order]) + tuple(np.asarray(a)[..., order] for a in arrays)


def speed_map(
    lat,
    lon,
    u,
    v,
    *,
    path: Path,
    title: str,
    vmax: float | None = None,
    arrows: int = 30,
    cmap: str = "viridis",
) -> dict:
    """Speed as colour with direction arrows over it. Wind, current or ensemble.

    `arrows` is the approximate number of arrows across the shorter axis; the
    field is thinned to roughly that, because a 77 x 77 grid drawn in full is a
    black rectangle and a 476 x 238 current grid is worse.

    Returns the speed distribution and where the maximum sits, so the figure is
    checkable without being looked at.
    """
    plt = _plt()
    lat, lon_d, u, v = _display_axes(lat, lon, u, v)
    speed = np.hypot(u, v)

    fig, ax = plt.subplots(figsize=(11, 8))
    im = ax.pcolormesh(lon_d, lat, speed, shading="auto", cmap=cmap,
                       vmin=0, vmax=vmax if vmax else float(np.nanmax(speed)))
    step = max(1, min(lat.size, lon_d.size) // arrows)
    ax.quiver(lon_d[::step], lat[::step], u[::step, ::step], v[::step, ::step],
              scale=None, width=0.0022, color="white", alpha=0.85)

    ax.set_xlabel("longitude (deg E, negative west)")
    ax.set_ylabel("latitude (deg N)")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="speed (m/s)")
    _save(fig, path, plt)

    j, i = np.unravel_index(int(np.nanargmax(speed)), speed.shape)
    return {
        "figure": str(path),
        "cells": int(speed.size),
        "speed_mean_ms": float(np.nanmean(speed)),
        "speed_p99_ms": float(np.nanquantile(speed, 0.99)),
        "speed_max_ms": float(np.nanmax(speed)),
        "max_at_lat": float(lat[j]),
        "max_at_lon": float(lon_d[i]),
        "nan_fraction": float(np.isnan(speed).mean()),
    }


def wind_rose(direction_from_deg, speed=None, *, path: Path, title: str,
              sectors: int = 36) -> dict:
    """Direction histogram on polar axes, meteorological convention.

    `direction_from_deg` is the direction the wind comes FROM, which is what
    `wind_box_mean_*.parquet` stores and what every mariner means. North at the
    top, clockwise, because that is a compass.

    The returned numbers are the point of this figure. **Steadiness** -- the
    magnitude of the vector mean over the mean of the magnitudes -- is the one
    worth reading: 1.0 is a wind that never changes direction, near 0 is one
    that blows equally in all directions and averages to nothing. It separates a
    steady trade flow from a variable one when the mean speeds are identical,
    which no rose can be eyeballed for.
    """
    plt = _plt()
    d = np.asarray(direction_from_deg, dtype=float)
    d = d[np.isfinite(d)]
    if d.size == 0:
        raise ValueError("no finite directions to plot")

    theta = np.radians(d)
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(6.5, 6.5))
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    counts, edges = np.histogram(theta, bins=sectors, range=(0, 2 * np.pi))
    ax.bar(edges[:-1], counts, width=np.diff(edges), align="edge", alpha=0.85)
    ax.set_title(title)
    _save(fig, path, plt)

    # Vector mean in the "from" frame: average the unit vectors, not the angles.
    # Averaging 350 deg and 10 deg arithmetically gives 180, the exact opposite.
    mean_x, mean_y = np.cos(theta).mean(), np.sin(theta).mean()
    resultant = float(np.hypot(mean_x, mean_y))
    top = int(np.argmax(counts))

    out = {
        "figure": str(path),
        "n": int(d.size),
        "mean_direction_from_deg": _bearing(np.degrees(np.arctan2(mean_y, mean_x))),
        "directional_constancy": resultant,
        "dominant_sector_deg": _bearing(np.degrees(edges[top])),
        "dominant_sector_share": float(counts[top] / counts.sum()),
    }
    if speed is not None:
        s = np.asarray(speed, dtype=float)
        s = s[np.isfinite(s)]
        out["speed_mean_ms"] = float(s.mean())
        out["steadiness"] = float(np.hypot(*_vector_mean(d, s)) / s.mean())
    return out


def _bearing(degrees: float) -> float:
    """Wrap an angle into [0, 360), closed at the bottom and OPEN at the top.

    `x % 360.0` is not enough. For a tiny negative x -- which is exactly what
    `arctan2` returns for a mean direction of due north -- the true result is
    just under 360 and floating point rounds it to 360.0 itself. A bearing of
    360.0 then breaks any consumer that bins into 36 sectors as `int(d // 10)`,
    because that yields index 36 in a length-36 array.
    """
    wrapped = float(degrees) % 360.0
    return 0.0 if wrapped >= 360.0 else wrapped


def _vector_mean(direction_from_deg, speed):
    """Vector-mean wind components from direction-from and speed.

    The `270 - angle` form is the inverse of the `dir_from_deg` that
    `sar.fetch.wind` writes, and it has to stay exactly that: getting it wrong
    mirrors the field, which is trap #1's cousin and just as silent.
    """
    ang = np.radians(270.0 - np.asarray(direction_from_deg, dtype=float))
    s = np.asarray(speed, dtype=float)
    return float(np.mean(s * np.cos(ang))), float(np.mean(s * np.sin(ang)))


def speed_histogram(speed, *, path: Path, title: str, bins: int = 60,
                    threshold: float = LEEWAY_MATTERS_MS,
                    ylabel: str = "count") -> dict:
    """Speed distribution with the leeway threshold marked.

    The number that matters is `fraction_above_threshold`: how much of the
    record sits in the band where leeway stops being negligible AND where L1
    says ERA5 is about 10 % too gentle.

    THE ANNOTATION STATES THE MEASURED FRACTION, ALWAYS. A dashed line captioned
    "leeway matters above here" invites the reader to assume something crosses
    it, and on box-mean data nothing ever does: measured 2026-09-16, the
    box-mean speed never reaches 10 m/s in the whole five years, while over one
    ordinary week 15.5 % of individual CELLS do and the per-cell maximum is
    3.13x the box-mean maximum. Printing the fraction on the figure makes a
    box-mean histogram state its own limitation instead of implying the
    opposite. See L13.
    """
    plt = _plt()
    s = np.asarray(speed, dtype=float)
    s = s[np.isfinite(s)]
    if s.size == 0:
        raise ValueError("no finite speeds to plot")

    above = float((s > threshold).mean())

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.hist(s, bins=bins, alpha=0.85)
    ax.axvline(threshold, color="red", ls="--", lw=1.2)
    caption = (f"  {threshold:.0f} m/s leeway threshold\n"
               f"  {100 * above:.1f} % of this record above it"
               + ("\n  (never reached -- see L13)" if above == 0.0 else ""))
    ax.text(threshold, ax.get_ylim()[1] * 0.95, caption, color="red", fontsize=9, va="top")
    ax.set_xlabel("wind speed (m/s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    _save(fig, path, plt)

    return {
        "figure": str(path),
        "n": int(s.size),
        "mean_ms": float(s.mean()),
        "median_ms": float(np.median(s)),
        "p90_ms": float(np.quantile(s, 0.90)),
        "p99_ms": float(np.quantile(s, 0.99)),
        "max_ms": float(s.max()),
        "fraction_above_threshold": float((s > threshold).mean()),
        "leeway_at_mean_ms": float(s.mean() * ALPHA),
    }


def seasonal_cycle(index: pd.DatetimeIndex, speed, *, path: Path, title: str) -> dict:
    """Monthly mean speed across the record, with the spread around it.

    Answers whether a held-out split can be drawn on season at all: if January
    and July are indistinguishable there is no seasonal structure to stratify on,
    and the split has to come from the regime clustering instead.
    """
    plt = _plt()
    s = pd.Series(np.asarray(speed, dtype=float), index=pd.DatetimeIndex(index))
    by_month = s.groupby(s.index.month)
    mean, std = by_month.mean(), by_month.std()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.errorbar(mean.index, mean.values, yerr=std.values, marker="o", capsize=4)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
    ax.set_xlabel("month")
    ax.set_ylabel("mean wind speed (m/s)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    _save(fig, path, plt)

    # A seasonal signal is only usable for stratifying if it is larger than the
    # within-month scatter it sits inside. Guarded because a synthetic or
    # single-valued record has zero scatter, and an unguarded divide returns inf
    # -- which reads as "an overwhelming seasonal signal", the exact opposite of
    # what no variance at all means.
    scatter = float(std.mean())
    ratio = float((mean.max() - mean.min()) / scatter) if scatter > 0 else float("nan")

    return {
        "figure": str(path),
        "windiest_month": int(mean.idxmax()),
        "windiest_mean_ms": float(mean.max()),
        "calmest_month": int(mean.idxmin()),
        "calmest_mean_ms": float(mean.min()),
        "seasonal_range_ms": float(mean.max() - mean.min()),
        "within_month_sd_ms": scatter,
        "range_over_within_month_sd": ratio,
    }


def _report(name: str, numbers: dict) -> None:
    print(f"\n{name}")
    for k, v in numbers.items():
        print(f"  {k:28s} {v:.4g}" if isinstance(v, float) else f"  {k:28s} {v}")


def main() -> None:
    p = argparse.ArgumentParser(description="Render a u/v field or a box-mean series.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--forcing", help="an era5_*.nc or hycom_*.nc file")
    g.add_argument("--box-means", nargs="+", help="wind_box_mean_*.parquet, or a directory")
    p.add_argument("--start", help="with --forcing: window start, inclusive")
    p.add_argument("--end", help="with --forcing: window end, EXCLUSIVE")
    p.add_argument("--figures", default="figures", help="output directory")
    args = p.parse_args()

    fig_dir = Path(args.figures)

    if args.forcing:
        ds = open_forcing(args.forcing)
        if args.start and args.end:
            ds = select_window(ds, args.start, args.end)
        snap = ds.isel(time=0)
        uname, vname = ("u10", "v10") if "u10" in ds.data_vars else ("water_u", "water_v")
        stamp = str(snap.time.values)[:13]
        label = "10 m wind" if uname == "u10" else "surface current"
        numbers = speed_map(
            snap.lat.values, snap.lon.values, snap[uname].values, snap[vname].values,
            path=fig_dir / f"speed_map_{stamp.replace(':', '').replace('-', '')}.png",
            title=f"{label} speed, {stamp} UTC",
        )
        _report("speed_map", numbers)

    else:
        df = open_box_means(args.box_means)
        span = f"{df.index[0]:%Y-%m-%d} to {df.index[-1]:%Y-%m-%d}"
        _report("wind_rose", wind_rose(
            df.dir_from_deg, df.speed, path=fig_dir / "wind_rose_study_period.png",
            title=f"ERA5 10 m wind direction (from)\nbox mean, {span}"))
        _report("speed_histogram", speed_histogram(
            df.speed, path=fig_dir / "speed_histogram_study_period.png",
            title=f"ERA5 10 m wind speed, box mean\n{span}"))
        _report("seasonal_cycle", seasonal_cycle(
            df.index, df.speed, path=fig_dir / "seasonal_cycle_study_period.png",
            title=f"Seasonal cycle of box-mean wind speed\n{span}"))


if __name__ == "__main__":
    main()
