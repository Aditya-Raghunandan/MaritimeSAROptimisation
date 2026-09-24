"""Where drifter tracks enter and leave the study box, and which way they are moving.

The site draws a hollow ring where each buoy was first seen, and most rings sit on the
box edge. This measures why: `load_drifters` crops every fix to 17-36 N, 82-63 W, so a
buoy that drifts in from outside is first seen on the edge it crossed.

    python scripts/drifter_edge_entries.py \
        --csv C:/maritime-data/raw/gdp_hourly_17-36N_82-63W_20190101-20240101.csv \
        --buoys C:/maritime-data/raw/gdp_hourly_buoys_17-36N_82-63W_20190101-20240101.csv

WHAT IT MEASURES. For each buoy's first fix, each segment's first fix and each buoy's
last fix: the distance to the nearest box edge (flat earth, cos lat), which edge, and
whether the velocity normal to that edge points into the box. Interior first fixes are
compared with the buoy's deploy date.

WHAT IT FOUND, 2026-09-24, hourly product, all 268 buoys:
    189 / 268 buoys (71 %) are first seen within 10 km of an edge, 99 % of them moving
    inward -- E 134, S 43, W 7, N 5. 49 of the 79 interior starts are deployments.
    526 / 825 segments start near an edge; 107 / 268 buoys end near one, 55 of them on
    the north edge (the Gulf Stream leaving past Cape Hatteras).
Vault: notebook/aditya/2026-09-24 Why the drifters sit on the border.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from sar.fetch.wind import LAT_N, LAT_S, LON_E, LON_W
from sar.utils.data_io import load_drifters
from sar.utils.geo import M_PER_DEG_LAT, to_display_longitude

NEAR_EDGE_KM = 10.0
DEPLOYED_HERE = pd.Timedelta("2D")

# The velocity component that points INTO the box across each edge.
INWARD = {"S": lambda ve, vn: vn, "N": lambda ve, vn: -vn,
          "W": lambda ve, vn: ve, "E": lambda ve, vn: -ve}


def nearest_edge(lat: float, lon: float) -> tuple[str, float]:
    """The closest box edge to a display-longitude point, and its distance in km."""
    km = M_PER_DEG_LAT / 1000.0
    coslat = np.cos(np.radians(lat))
    d = {"S": (lat - LAT_S) * km, "N": (LAT_N - lat) * km,
         "W": (lon - LON_W) * km * coslat, "E": (LON_E - lon) * km * coslat}
    edge = min(d, key=d.get)
    return edge, d[edge]


def edge_table(first: pd.DataFrame, meta: pd.DataFrame | None) -> pd.DataFrame:
    rows = [(*nearest_edge(r.lat, r.lon), r.ve, r.vn, r.ID, r.time) for r in first.itertuples()]
    t = pd.DataFrame(rows, columns=["edge", "km", "ve", "vn", "ID", "time"])
    t["inward"] = [INWARD[e](a, b) > 0 for e, a, b in zip(t.edge, t.ve, t.vn)]
    if meta is not None:
        t = t.merge(meta[["ID", "deploy_date"]], on="ID", how="left")
        t["deployed_here"] = (t.time - t.deploy_date).abs() < DEPLOYED_HERE
    return t


def report(label: str, t: pd.DataFrame) -> None:
    near = t.km < NEAR_EDGE_KM
    print(f"\n== {label}: {len(t)}")
    print(f"  within {NEAR_EDGE_KM:g} km of an edge: {near.sum()} ({near.mean():.0%})")
    print("  by edge:", t[near].edge.value_counts().to_dict())
    print(f"  of those, moving inward at that fix: {t[near].inward.mean():.0%}")
    if "deployed_here" in t:
        print(f"  interior: {(~near).sum()}, deployed within 2 d of the fix: "
              f"{int(t[~near].deployed_here.sum())}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--csv", required=True, help="a gdp_hourly_*.csv drifter pull")
    ap.add_argument("--buoys", help="the matching gdp_hourly_buoys_*.csv, for deploy dates")
    args = ap.parse_args(argv)

    df = load_drifters(args.csv)
    df["lon"] = to_display_longitude(df["lon"].to_numpy())
    meta = None
    if args.buoys:
        meta = pd.read_csv(args.buoys, dtype={"ID": str})
        meta["deploy_date"] = pd.to_datetime(meta.deploy_date, utc=True)

    ordered = df.sort_values("time")
    first_fix = ordered.groupby("ID").first().reset_index()
    segment_start = ordered.groupby("segment_id").first().reset_index()
    report("buoy first fix (the hollow ring)", edge_table(first_fix, meta))
    report("segment first fix", edge_table(segment_start, meta))
    last = edge_table(ordered.groupby("ID").last().reset_index(), None)
    near = last.km < NEAR_EDGE_KM
    print(f"\n== buoy last fix: within {NEAR_EDGE_KM:g} km of an edge {near.sum()}/{len(last)}",
          last[near].edge.value_counts().to_dict())


if __name__ == "__main__":
    main()
