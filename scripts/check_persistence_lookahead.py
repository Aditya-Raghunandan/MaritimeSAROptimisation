"""check_persistence_lookahead.py: does the buoy's starting velocity already know the future?

Persistence ("the buoy keeps its first velocity") uses `ve`, `vn` from the GDP hourly
product at the window's start. If those velocities are the derivative of a curve fitted
through fixes on BOTH sides of t0, a centred estimate, then persistence uses positions
after the start, and it is flattered at short leads (D025's 4 Oct amendment).

Two measurements on the dev windows, using only data the windows already hold. Windows of
one unit are cut back to back (sar.validate.drift_windows), so the previous window's last
hours are the hours before this window's t0.

  1. Regress GDP's velocity on the backward difference (t0 - 1 h to t0) and the forward
     difference (t0 to t0 + 1 h). A centred estimate weights them about equally; a
     velocity built from the past alone puts its weight on the backward one.
  2. Persistence from GDP's velocity against persistence from the backward difference,
     which a forecaster at t0 could really have computed, at each lead, beside the model.

    python scripts/check_persistence_lookahead.py --windows <dir>/windows \
        --stage1 <dir>/stage1 --out <dir>/lookahead.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sar.model.drift import LEEWAY_COEFFICIENT
from sar.utils.geo import M_PER_DEG_LAT, metres_per_degree_lon
from sar.validate.calibrate_sigma import gap_metres, load_stage1
from sar.validate.drift_windows import Windows

LEADS_H = (1, 2, 3, 4, 6, 12, 24)


def velocity(lat_a, lon_a, lat_b, lon_b, seconds: float) -> np.ndarray:
    """[east, north] m/s from a to b."""
    dlon = (np.asarray(lon_b) - lon_a + 180.0) % 360.0 - 180.0
    mid = 0.5 * (np.asarray(lat_a) + lat_b)
    return np.stack([dlon * metres_per_degree_lon(mid),
                     (np.asarray(lat_b) - lat_a) * M_PER_DEG_LAT], axis=-1) / seconds


def previous_window(w: Windows) -> np.ndarray:
    """Index of the window of the same unit that ends at this one's t0, or -1."""
    tab = w.table.reset_index(drop=True)
    key = {(u, t): i for i, (u, t) in enumerate(zip(tab["unit"], tab["t0"]))}
    span = np.timedelta64(w.hours, "h")
    return np.array([key.get((u, t - span), -1) for u, t in zip(tab["unit"], tab["t0"])])


def persistence_miss(w: Windows, rows, v, lead: int) -> np.ndarray:
    """Metres from the buoy at `lead` to its start carried at velocity v (rows x 2)."""
    tab = w.table
    t = lead * 3600.0
    lat0, lon0 = tab["lat0"].to_numpy()[rows], tab["lon0"].to_numpy()[rows]
    p_lat = lat0 + v[:, 1] * t / M_PER_DEG_LAT
    p_lon = lon0 + v[:, 0] * t / metres_per_degree_lon(lat0)
    g = gap_metres(w.lat[rows, lead], w.lon[rows, lead], p_lat, p_lon)
    return np.hypot(g[:, 0], g[:, 1])


def check(w: Windows, stage1_dir) -> dict:
    tab = w.table.reset_index(drop=True)
    prev = previous_window(w)
    h = w.hours
    # Usable: a predecessor whose last hour is a real-fix position, a real start, a real
    # t0 + 1 h, and GDP's velocity.
    have = prev >= 0
    rows = np.flatnonzero(have)
    p = prev[rows]
    ok = (w.ok[p, h - 1] & w.ok[rows, 0] & w.ok[rows, 1]
          & np.isfinite(tab["ve0"].to_numpy()[rows]) & np.isfinite(tab["vn0"].to_numpy()[rows]))
    rows, p = rows[ok], p[ok]
    v_back = velocity(w.lat[p, h - 1], w.lon[p, h - 1], w.lat[rows, 0], w.lon[rows, 0], 3600.0)
    v_fwd = velocity(w.lat[rows, 0], w.lon[rows, 0], w.lat[rows, 1], w.lon[rows, 1], 3600.0)
    v_gdp = tab[["ve0", "vn0"]].to_numpy(float)[rows]

    # 1. GDP's velocity as a weighted sum of the two differences, east and north pooled.
    x = np.column_stack([v_back.ravel(), v_fwd.ravel()])
    coef, *_ = np.linalg.lstsq(x, v_gdp.ravel(), rcond=None)
    resid = v_gdp.ravel() - x @ coef

    def rms(a):
        return float(np.sqrt(np.mean(np.sum(a ** 2, axis=-1))))

    out = {"windows_with_a_past": int(len(rows)), "of_windows": int(len(tab)),
           "weights": {"backward": float(coef[0]), "forward": float(coef[1])},
           "residual_rms_ms": float(np.sqrt(np.mean(resid ** 2))),
           "rms_gap_ms": {"gdp_minus_backward": rms(v_gdp - v_back),
                          "gdp_minus_forward": rms(v_gdp - v_fwd),
                          "gdp_minus_centred": rms(v_gdp - 0.5 * (v_back + v_fwd))}}

    # 2. Persistence both ways, beside the model, per tier, on the same windows.
    s1, _ = load_stage1(w, stage1_dir)
    s1 = s1[s1["valid"] & np.isclose(s1["alpha"], LEEWAY_COEFFICIENT)]
    model = s1.set_index(["w", "hour"])["sep"]
    out["median_miss_km"] = {}
    for tier in ("undrogued", "drogued"):
        is_tier = tab["tier"].to_numpy()[rows] == tier
        out["median_miss_km"][tier] = {}
        for lead in LEADS_H:
            if lead > h:
                continue
            r = rows[is_tier]
            keep = w.ok[r, lead]
            idx = pd.MultiIndex.from_arrays([r, np.full(r.size, lead)])
            m = model.reindex(idx).to_numpy(float)
            keep &= np.isfinite(m)
            gdp = persistence_miss(w, r[keep], v_gdp[is_tier][keep], lead)
            back = persistence_miss(w, r[keep], v_back[is_tier][keep], lead)
            out["median_miss_km"][tier][lead] = {
                "windows": int(keep.sum()), "model": float(np.median(m[keep]) / 1e3),
                "persistence_gdp": float(np.median(gdp) / 1e3),
                "persistence_backward": float(np.median(back) / 1e3)}
    return out


def main(argv=None) -> dict:
    a = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    a.add_argument("--windows", required=True, help="stem written by sar.validate.drift_windows")
    a.add_argument("--stage1", required=True, help="calibrate_sigma stage1 output directory")
    a.add_argument("--out", required=True)
    args = a.parse_args(argv)
    out = check(Windows.load(args.windows), args.stage1)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    main()
