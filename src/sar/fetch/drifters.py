"""
fetch_drifters_gdp.py -- pull NOAA Global Drifter Program tracks for the study
box and window, and report the numbers that decide whether R2 is achievable.

Closes the risk carried in state/Aditya.md since 8 Sep ("GDP drifter coverage
2021-2026 in-box -- unverified, and a real risk to R2") and D014's consequence
"five years of drifter coverage inside one box may be thin".

    python code/fetch_drifters_gdp.py --start 2019-01-01 --end 2024-01-01 --out C:/maritime-data

MEASURED 2026-09-10, so these are facts and not assumptions:

  Server            https://erddap.aoml.noaa.gov/gdp/erddap/
  drifter_hourly_qc 1987-10-02 13:00 -> 2022-10-31        <- the QC product LAGS ~3 y
  drifter_6hour_qc  1979-02-15 00:00 -> 2025-06-18        <- current, but 6-hourly

THE HOURLY PRODUCT ENDS 2022-10-31.  That is one of the three independent
constraints that moved the study window (see D014's correction block).  Do not
assume "present" for any of these archives -- ask the server, as below.

WHY UNDROGUED SEGMENTS ARE THE POINT (D018, and the concept note on leeway)
--------------------------------------------------------------------------
A drogued GDP buoy hangs a drogue at 15 m specifically to follow water and
ignore wind, so alpha ~ 0 and it cannot validate the leeway term at all.  When
the drogue snaps off the buoy floats at the surface and DOES feel wind --
downwind slip ~1 % of wind speed (Pazan & Niiler 2001; Poulain et al. 2009),
revised ~50 % higher by Lumpkin et al. 2013, so ~1-1.5 %.  A person in water is
1.9-2.7 % (Allen 2000).  So an undrogued buoy is a LOWER BOUND on human leeway
-- weaker than a person, but not zero, which is what makes the paired A/B test
in D018 possible.

`drogue_lost_date` is the field that makes this separable.  It is per buoy, and
NaN means the drogue was never lost within the record.
"""

import argparse
import urllib.parse
from pathlib import Path

import numpy as np
import pandas as pd

ERDDAP = "https://erddap.aoml.noaa.gov/gdp/erddap/tabledap"
DATASET = "drifter_hourly_qc"

# D014 study box.  MUST be identical to the wind and current box -- a track
# outside it has no forcing field, so the comparison is meaningless there.
LAT_S, LAT_N = 17.0, 36.0
LON_W, LON_E = -82.0, -63.0

# Real column names, read off the dataset's info endpoint. Do not guess these.
COLS = ["ID", "time", "latitude", "longitude", "ve", "vn", "sst", "drogue_lost_date"]


def coverage_url(dataset: str, start: str, end: str, cols: list[str]) -> str:
    """ERDDAP tabledap query. Constraints are &-joined and must be url-encoded."""
    q = ",".join(cols) + "&" + "&".join([
        f"time>={start}T00:00:00Z", f"time<={end}T00:00:00Z",
        f"latitude>={LAT_S}", f"latitude<={LAT_N}",
        f"longitude>={LON_W}", f"longitude<={LON_E}",
    ])
    return f"{ERDDAP}/{dataset}.csv?{urllib.parse.quote(q, safe=',&=:.-')}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--start", required=True, help="UTC date, YYYY-MM-DD")
    p.add_argument("--end", required=True, help="UTC date, YYYY-MM-DD")
    p.add_argument("--out", default="data")
    args = p.parse_args()

    raw = Path(args.out) / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    tag = f"{args.start.replace('-','')}-{args.end.replace('-','')}"
    box = f"{int(LAT_S)}-{int(LAT_N)}N_{abs(int(LON_W))}-{abs(int(LON_E))}W"
    dest = raw / f"gdp_hourly_{box}_{tag}.csv"

    url = coverage_url(DATASET, args.start, args.end, COLS)
    print("GET", url)
    # skiprows=[1] drops ERDDAP's units row, which would otherwise poison dtypes.
    df = pd.read_csv(url, skiprows=[1], parse_dates=["time", "drogue_lost_date"])
    df.to_csv(dest, index=False)
    print(f"wrote {dest} ({dest.stat().st_size/1e6:.1f} MB)")

    report(df)


def report(df: pd.DataFrame) -> None:
    df = df.copy()
    df["ID"] = df["ID"].astype(str)
    # sst arrives in KELVIN and carries unmasked fill values (observed -76.9 to
    # 1000 C).  Suné's D013 survivable window needs Celsius AND a QC filter.
    df["sst_c"] = df["sst"] - 273.15
    plausible = df.sst_c.between(-2, 40)

    print(f"\nrows (hourly obs) : {len(df):,}")
    print(f"distinct buoys    : {df.ID.nunique()}")
    print(f"actual time span  : {df.time.min()} -> {df.time.max()}")
    print(f"  ^ compare with what you asked for. The hourly product ends 2022-10-31.")
    print(f"sst plausible     : {100*plausible.mean():.1f} % in [-2, 40] C "
          f"-- the rest are fill values, MASK THEM (D013)")

    und = df.drogue_lost_date.notna() & (df.time >= df.drogue_lost_date)
    print(f"\nundrogued obs     : {und.sum():,} ({100*und.mean():.1f} %) "
          f"across {df.loc[und,'ID'].nunique()} buoys   <- the leeway-capable set")
    print(f"drogued obs       : {(~und).sum():,} ({100*(~und).mean():.1f} %) "
          f"across {df.loc[~und,'ID'].nunique()} buoys   <- advection only")

    # A validation track must stay INSIDE the box for the whole horizon it is
    # scored over, because outside it there is no wind and no current. D014's
    # truncation rule: cut the comparison where the buoy leaves, and log the
    # horizon actually achieved. A gap > 3 h starts a new segment.
    rows = []
    for bid, g in df.groupby("ID"):
        t = g.time.sort_values()
        seg = t.groupby((t.diff() > pd.Timedelta("3h")).cumsum()).agg(["min", "max"])
        frac_und = float(und[g.index].mean())
        for _, r in seg.iterrows():
            rows.append({"ID": bid,
                         "hours": (r["max"] - r["min"]).total_seconds() / 3600 + 1,
                         "undrogued_frac": frac_und})
    seg = pd.DataFrame(rows)
    print(f"\ncontiguous in-box segments : {len(seg)}  "
          f"(median {seg.hours.median():.0f} h, max {seg.hours.max():.0f} h)")
    print("  usable validation tracks by horizon -- R6d budgets only 10:")
    for h in (24, 48, 72, 120):
        m = seg.hours >= h
        mu = m & (seg.undrogued_frac > 0.5)
        print(f"    >= {h:3d} h : {m.sum():4d} segments / {seg.loc[m,'ID'].nunique():3d} buoys"
              f"   | mostly-undrogued {mu.sum():4d} / {seg.loc[mu,'ID'].nunique():3d} buoys")


if __name__ == "__main__":
    main()
