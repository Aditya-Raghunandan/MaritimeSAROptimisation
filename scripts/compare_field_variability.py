"""
compare_field_variability.py -- does the wind really change faster than the
current, or does the map only make it look that way?

    python scripts/compare_field_variability.py
    python scripts/compare_field_variability.py --start 2019-07-01 --days 30

Writes nothing. Prints a table and one ratio, read off the PUBLISHED archive on
Hugging Face rather than the cluster, because the question is about what the
site shows and the site reads the published copy.

WHY THIS EXISTS
---------------
Watching the visualiser at hourly resolution, the wind field visibly churns
while the current looks nearly frozen. That is either physics worth knowing or
a bug in the frame mapping, and the two look identical on screen. This measures
it instead of arguing about it.

THE COMPARISON HAS TO BE LIKE FOR LIKE, and two things make that easy to get
wrong. Wind is hourly and current 3-hourly, so "change per frame" means
different elapsed times for each -- the wind is therefore SUBSAMPLED to the
current's cadence here rather than the current being interpolated up, because
interpolating would invent the very smoothness under test. And the two stores
carry different time epochs (hours since 1900 for ERA5, hours since 2000 for
HYCOM), so the index of a given moment differs between them; both are resolved
from each store's own first stamp and then asserted equal.

Change is reported NORMALISED by each field's own standard deviation. The
absolute numbers differ by a factor of twenty for the obvious reason that one
is measured in tens of m/s and the other in tenths, and that ratio says nothing
about which field is more variable.

MEASURED 2026-09-19, 20 days from 2019-02-01, four points:

    wind     normalised 3-hourly change 0.335   lag-1 autocorrelation 0.890
    current  normalised 3-hourly change 0.386   lag-1 autocorrelation 0.873

So relative to its own spread the wind changes 0.9x as much as the current --
slightly LESS, not more -- and their decorrelation timescales overlap (12-36 h
against 9-57 h). The current is not a static field, and on the Bahamas shelf it
is the more variable of the two.

What differs is absolute magnitude: 0.6-1.2 m/s per 3 h against 0.04-0.06.
Three things then combine on screen, and none of them is a bug: that magnitude
difference; the nearest-frame mapping holding the current for three frames at
hourly playback; and the magma ramp's near-black low end rendering the weak
background -- which is where most of the relative variability lives -- as
almost nothing, so the only visible part of the current is the jet, and the jet
is the most persistent structure in the domain.

NOTE ON READING THE PUBLISHED STORE. This uses `zarr` directly rather than
`xr.open_zarr`, because the published stores carry no consolidated metadata:
every array opens by name but `array_keys()` returns empty, so xarray sees a
dataset with no variables. See the note in the vault.
"""
import argparse

import sys

import numpy as np
import zarr
sys.stdout.reconfigure(encoding="utf-8")
B = "https://huggingface.co/datasets/AdityaRugs/MaritimeSARoperations/resolve/main"

p = argparse.ArgumentParser(description="wind vs current variability, like for like")
p.add_argument("--start", default="2019-02-01", help="UTC date, YYYY-MM-DD")
p.add_argument("--days", type=int, default=20, help="window length")
args = p.parse_args()
START, DAYS = f"{args.start}T00", args.days

def hours_since(epoch, when):
    return (np.datetime64(when) - np.datetime64(epoch)) / np.timedelta64(1, "h")

def index_of(g, epoch, when):
    """hours-since-epoch is a VALUE on the axis, not an index into it."""
    t0 = float(g["time"][0]); step = float(g["time"][1]) - t0
    return int(round((hours_since(epoch, when) - t0) / step)), step

gw = zarr.open_group(f"{B}/wind_hourly.zarr", mode="r")
gc = zarr.open_group(f"{B}/current_3-hourly.zarr", mode="r")
w0, wstep = index_of(gw, "1900-01-01T00", START); wn = DAYS * 24
c0, cstep = index_of(gc, "2000-01-01T00", START); cn = DAYS * 8
wt = gw["time"][w0:w0 + wn]; ct = gc["time"][c0:c0 + cn]
print(f"window {START} + {DAYS} d")
print(f"  wind  idx {w0:6d}  step {wstep:g} h  frames {len(wt)}")
print(f"  curr  idx {c0:6d}  step {cstep:g} h  frames {len(ct)}")
wreal = np.datetime64("1900-01-01T00") + np.timedelta64(int(wt[0] * 60), "m")
creal = np.datetime64("2000-01-01T00") + np.timedelta64(int(ct[0] * 60), "m")
print("  wind starts    ", wreal, "\n  current starts ", creal)
assert wreal == creal == np.datetime64(START), "axes do not start at the same moment"
print("  -> same UTC moment, so this is like for like.\n")

wlat, wlon = gw["lat"][:], gw["lon"][:]
clat, clon = gc["lat"][:], gc["lon"][:]
POINTS = [("Gulf Stream jet", 31.0, -79.0), ("open ocean", 25.0, -70.0),
          ("Bahamas shelf edge", 26.5, -76.0), ("east of Bermuda", 33.0, -64.5)]

wu = gw["u10"][w0:w0 + wn]; wv = gw["v10"][w0:w0 + wn]
cu = gc["water_u"][c0:c0 + cn]; cv = gc["water_v"][c0:c0 + cn]
print("read: wind", wu.shape, " current", cu.shape, "\n")

def stats(sp):
    sp = np.asarray(sp, float)
    if not np.isfinite(sp).all():
        return None
    d = np.abs(np.diff(sp))
    x = sp - sp.mean()
    r1 = float(np.sum(x[:-1] * x[1:]) / np.sum(x * x))
    tau = None
    for k in range(1, len(x) // 3):
        if float(np.sum(x[:-k] * x[k:]) / np.sum(x * x)) < 1 / np.e:
            tau = k * 3; break
    return dict(mean=sp.mean(), sd=sp.std(), dstep=d.mean(),
                norm=d.mean() / sp.std(), r1=r1, tau=tau)

hdr = f"{'point':<20} {'field':<8} {'mean':>6} {'sd':>6} {'|d|/3h':>8} {'/sd':>6} {'r(3h)':>7} {'tau':>7}"
print(hdr); print("-" * len(hdr))
rows = {}
for name, lat, lon in POINTS:
    wj = int(np.argmin(np.abs(wlat - lat))); wi = int(np.argmin(np.abs(wlon - lon)))
    cj = int(np.argmin(np.abs(clat - lat))); ci = int(np.argmin(np.abs(clon - lon)))
    ws = np.hypot(wu[::3, wj, wi], wv[::3, wj, wi])   # wind at the CURRENT's cadence
    cs = np.hypot(cu[:, cj, ci], cv[:, cj, ci])
    n = min(len(ws), len(cs)); ws, cs = ws[:n], cs[:n]
    for field, sp in (("wind", ws), ("current", cs)):
        st = stats(sp)
        if st is None:
            print(f"{name:<20} {field:<8}   land / no data"); continue
        rows.setdefault(field, []).append(st)
        tau = f"{st['tau']} h" if st["tau"] else ">2 d"
        print(f"{name:<20} {field:<8} {st['mean']:6.2f} {st['sd']:6.2f} "
              f"{st['dstep']:8.3f} {st['norm']:6.3f} {st['r1']:7.3f} {tau:>7}")
print("-" * len(hdr))
for f in ("wind", "current"):
    r = rows[f]
    print(f"{f:<8} normalised 3-hourly change {np.mean([x['norm'] for x in r]):.3f}"
          f"   r(3h) {np.mean([x['r1'] for x in r]):.3f}")
w, c = rows["wind"], rows["current"]
print(f"\nWIND CHANGES {np.mean([x['norm'] for x in w]) / np.mean([x['norm'] for x in c]):.1f}x "
      f"AS MUCH per 3 h, relative to its own spread.")
