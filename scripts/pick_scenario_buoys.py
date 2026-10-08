"""pick_scenario_buoys.py: dev buoys to start search scenarios from, one row per scenario.

Each scenario is a datum (the buoy's real GPS fix at the start) and a truth (where the buoy
really went in the next hours). The Monte Carlo starts its cloud at the datum; the search is
scored against the truth. The rules, each for a reason:

  * DEV, UNDROGUED ONLY. Sealed and holdout tracks are kept away from anything built on the
    engine before its frozen run (D025). An undrogued buoy floats at the surface like a
    person; a drogued one does not feel a person's leeway and loses to "stays put" (D028).
  * A REAL FIX AT THE START, TRUTH EVERY HOUR. The start position is a GPS fix (fix_gap_h <= 1)
    and every hour to HORIZON_H counts as truth under D025's rule (fix_gap_h <= 3).
  * ON THE MAP. The datum is >= 20 km from HYCOM land and >= 50 km inside the box, and the
    buoy's whole path to HORIZON_H stays >= 10 km from land and >= 30 km inside the box, so
    the cloud neither beaches wholesale (D016) nor leaves the forcing.
  * ONE BUOY PER SHARED-WATER GROUP. Buoys within 10 km at the same time ride the same water
    (D025), so two of them would be one scenario counted twice.
  * ANY HOUR, ANY DAY. The start is a random valid fix, not a fixed weekday or hour, so the
    scenarios sample the diurnal cycle and the seasons instead of one phase of them.
  * STRATIFIED BY HOW FAST THE BUOY MOVED over the horizon, so strong currents, the
    project's subject, are not left to chance.

    python scripts/pick_scenario_buoys.py --data C:/maritime-data \
        --land C:/maritime-data/raw/hycom_<box>_<dates>.nc --out <dir>/scenarios.csv

THE BENCHMARK TABLE, --every 48h. Instead of one start per group, every buoy is walked from
its first valid start to its last, taking a new start every 48 h: every buoy, until its
track ends, under the same rules as above. Aditya, 7 Oct: the benchmark should use the
buoys we have, not 55 of them. Why 48 h and not back to back: the drift model's error
persists (D030, T_L = 25.7 h), so two starts 48 h apart share e^(-48/25.7) ~ 15 % of it,
and two starts 4 h apart 86 %. Rows of one buoy are still not independent, which is why
the scorer's summary averages each shared-water group first and resamples whole groups.
Rows are named D0001, D0002, ... in time order, with their own seeds.

    python scripts/pick_scenario_buoys.py --data C:/maritime-data \
        --land C:/maritime-data/raw/hycom_<box>_<dates>.nc --every 48h --out <dir>/bench.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree

from sar.fetch.drifters import LAT_N, LAT_S, LON_E, LON_W
from sar.utils.geo import M_PER_DEG_LAT, to_display_longitude
from sar.validate.drift_windows import load_tracks, read_split

HORIZON_H = 6                      # truth to 6 h: the 4 h search window plus margin
MAX_START_GAP_H = 1.0
MAX_TRUTH_GAP_H = 3.0              # D025
DATUM_LAND_KM, PATH_LAND_KM = 20.0, 10.0
DATUM_EDGE_KM, PATH_EDGE_KM = 50.0, 30.0
STRATA = ((0.0, 0.3, "quiet"), (0.3, 1.0, "moderate"), (1.0, np.inf, "jet"))
TARGETS = {"jet": 15, "moderate": 20, "quiet": 15}
SPARES = 5
VALIDATION = {"jet": 3, "moderate": 4, "quiet": 3}
SEED = 20261004
TILE_SEED = 20261007


def flat_xy(lat, lon):
    """km on one flat sheet about the box centre: good enough for 'how far from land'."""
    mid = 0.5 * (LAT_S + LAT_N)
    x = ((np.asarray(lon) + 180.0) % 360.0 - 180.0) * M_PER_DEG_LAT * np.cos(np.radians(mid))
    return np.column_stack([x / 1e3, np.asarray(lat) * M_PER_DEG_LAT / 1e3])


def land_tree(path) -> cKDTree:
    """HYCOM's land cells (NaN current at the first time) as a KD tree."""
    with xr.open_dataset(path) as ds:
        u = ds["water_u"].isel(time=0).values
        lat, lon = np.meshgrid(ds["lat"].values, ds["lon"].values, indexing="ij")
    land = ~np.isfinite(u)
    return cKDTree(flat_xy(lat[land], lon[land]))


def edge_km(lat, lon):
    lon = to_display_longitude(lon)
    dlat = np.minimum(np.asarray(lat) - LAT_S, LAT_N - np.asarray(lat)) * M_PER_DEG_LAT / 1e3
    dlon = (np.minimum(lon - LON_W, LON_E - lon) * M_PER_DEG_LAT / 1e3
            * np.cos(np.radians(np.asarray(lat))))
    return np.minimum(dlat, dlon)


def stratum(speed):
    for lo, hi, name in STRATA:
        if lo <= speed < hi:
            return name
    return "quiet"


def candidates(data, land: cKDTree) -> pd.DataFrame:
    """Every valid start in every dev undrogued unit, with its truth path."""
    split = read_split(Path(data) / "derived" / "validation_split.csv")
    units = split[(split["split"] == "dev") & (split["tier"] == "undrogued")]
    tracks = load_tracks(data)
    by_id = {k: g.sort_values("time") for k, g in tracks.groupby("ID", sort=False)}
    rows = []
    for u in units.itertuples(index=False):
        g = by_id.get(str(u.ID))
        if g is None:
            continue
        g = g[(g["time"] >= u.start) & (g["time"] <= u.end)].set_index("time")
        times = g.index
        lat, lon, gap = g["lat"].to_numpy(), g["lon"].to_numpy(), g["fix_gap_h"].to_numpy()
        pos = {t: i for i, t in enumerate(times)}
        d_land = land.query(flat_xy(lat, lon))[0]
        d_edge = edge_km(lat, lon)
        for i, t0 in enumerate(times):
            if (not gap[i] <= MAX_START_GAP_H or d_land[i] < DATUM_LAND_KM
                    or d_edge[i] < DATUM_EDGE_KM):
                continue
            path = [pos.get(t0 + pd.Timedelta(hours=h), -1) for h in range(HORIZON_H + 1)]
            if min(path) < 0:
                continue
            path = np.array(path)
            if (not np.all(gap[path] <= MAX_TRUTH_GAP_H) or d_land[path].min() < PATH_LAND_KM
                    or d_edge[path].min() < PATH_EDGE_KM):
                continue
            k4 = path[4]
            east = (((lon[k4] - lon[i] + 180) % 360 - 180)
                    * M_PER_DEG_LAT * np.cos(np.radians(lat[i])))
            north = (lat[k4] - lat[i]) * M_PER_DEG_LAT
            speed = float(np.hypot(east, north) / (4 * 3600.0))
            rows.append({"ID": str(u.ID), "unit": u.unit, "group": int(u.group), "start": t0,
                         "lat": float(lat[i]), "lon": float(to_display_longitude(lon[i])),
                         "speed_4h_ms": speed, "stratum": stratum(speed),
                         "land_km": float(d_land[i]), "edge_km": float(d_edge[i]),
                         **{f"lat_{h}h": float(lat[path[h]]) for h in range(1, HORIZON_H + 1)},
                         **{f"lon_{h}h": float(to_display_longitude(lon[path[h]]))
                            for h in range(1, HORIZON_H + 1)}})
    return pd.DataFrame(rows)


def pick(cands: pd.DataFrame, seed=SEED) -> pd.DataFrame:
    """One start per buoy, one buoy per group, filling the scarcest stratum first."""
    rng = np.random.default_rng(seed)
    used_groups, used_ids, out = set(), set(), []
    need = dict(TARGETS)
    need_spare = SPARES
    order = sorted(need, key=lambda s: cands["stratum"].eq(s).sum())
    for name in order + ["any"]:
        pool = cands if name == "any" else cands[cands["stratum"] == name]
        groups = pool["group"].unique()
        rng.shuffle(groups)
        for grp in groups:
            if name != "any" and need[name] == 0:
                break
            if name == "any" and need_spare == 0:
                break
            if grp in used_groups:
                continue
            opts = pool[(pool["group"] == grp) & ~pool["ID"].isin(used_ids)]
            if opts.empty:
                continue
            row = opts.iloc[rng.integers(len(opts))].to_dict()
            row["set"] = "spare" if name == "any" else None
            out.append(row)
            used_groups.add(grp)
            used_ids.add(row["ID"])
            if name == "any":
                need_spare -= 1
            else:
                need[name] -= 1
    picked = pd.DataFrame(out)
    main = picked["set"].isna()
    for name, k in VALIDATION.items():
        idx = picked.index[main & (picked["stratum"] == name)].to_numpy()
        val = rng.choice(idx, min(k, len(idx)), replace=False)
        picked.loc[val, "set"] = "validation"
    picked.loc[picked["set"].isna(), "set"] = "train"
    picked = picked.sort_values(["set", "stratum", "start"],
                                key=lambda c: c.map({"train": 0, "validation": 1, "spare": 2})
                                if c.name == "set" else c).reset_index(drop=True)
    picked.insert(0, "scenario", [f"S{k + 1:02d}" for k in range(len(picked))])
    picked["seed"] = SEED + np.arange(len(picked))
    return picked


def tile(cands: pd.DataFrame, every_h: float, seed=TILE_SEED, prefix="D") -> pd.DataFrame:
    """Every buoy, from its first valid start to its last, a new start every `every_h` hours.

    Within a unit the first valid start is taken, then the first valid start at least
    `every_h` later, and so on, so a gap in valid starts (near land, a missing fix) is
    skipped rather than ending the buoy. Rows are named in time order.
    """
    if not every_h > 0:
        raise ValueError(f"every_h must be positive, got {every_h}")
    step = pd.Timedelta(hours=float(every_h))
    taken = []
    for _, g in cands.sort_values(["unit", "start"]).groupby("unit", sort=True):
        due = None
        for k, start in zip(g.index, g["start"]):
            if due is None or start >= due:
                taken.append(k)
                due = start + step
    out = cands.loc[taken].sort_values(["start", "ID"]).reset_index(drop=True)
    width = max(4, len(str(len(out))))
    out.insert(0, "scenario", [f"{prefix}{k + 1:0{width}d}" for k in range(len(out))])
    out["set"] = "dev"
    out["seed"] = seed + np.arange(len(out))
    return out


def hours(text: str) -> float:
    """'48h', '2d' or '48' as hours."""
    text = str(text).strip().lower()
    if text.endswith("d"):
        return float(text[:-1]) * 24.0
    return float(text[:-1] if text.endswith("h") else text)


def main(argv=None) -> pd.DataFrame:
    a = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    a.add_argument("--data", required=True, help="archive root holding raw/ and derived/")
    a.add_argument("--land", required=True, help="any HYCOM NetCDF over the box, for its land mask")
    a.add_argument("--every", default=None,
                   help="the benchmark table: every buoy, a start this often (48h, 2d); "
                        "default the 55-row scenario table")
    a.add_argument("--out", required=True)
    args = a.parse_args(argv)
    cands = candidates(args.data, land_tree(args.land))
    if args.every is not None:
        tiled = tile(cands, hours(args.every))
        tiled["start"] = pd.to_datetime(tiled["start"]).dt.strftime("%Y-%m-%dT%H:%M")
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        tiled.to_csv(args.out, index=False, float_format="%.5f", lineterminator="\n")
        print(json.dumps({"candidates": len(cands), "rows": len(tiled),
                          "buoys": int(tiled["ID"].nunique()),
                          "groups": int(tiled["group"].nunique()),
                          "rows_by_water": tiled["stratum"].value_counts().to_dict(),
                          "years": tiled["start"].str[:4].value_counts().sort_index().to_dict()},
                         indent=1, default=str))
        return tiled
    picked = pick(cands)
    picked["start"] = pd.to_datetime(picked["start"]).dt.strftime("%Y-%m-%dT%H:%M")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    # Unix line ends on every OS: the cluster's shell reads this file.
    picked.to_csv(args.out, index=False, float_format="%.5f", lineterminator="\n")
    print(json.dumps({"candidates": len(cands), "buoys": int(cands["ID"].nunique()),
                      "groups": int(cands["group"].nunique()),
                      "by_stratum": cands.groupby("stratum")["group"].nunique().to_dict(),
                      "picked": {f"{s}, {t}": int(n) for (s, t), n in
                                 picked.groupby(["set", "stratum"]).size().items()}},
                     indent=1, default=str))
    return picked


if __name__ == "__main__":
    main()
