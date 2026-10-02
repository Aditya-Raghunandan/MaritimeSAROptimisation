"""check_current_lag.py: is HYCOM's current in step with the buoys? (#89)

At each dev window's start, compare the buoy's own velocity (ve0, vn0) with HYCOM's
current at the same place, sampled at the same time and at times shifted by a lag. If
the drifter clock and the forcing clock disagreed, the two would agree best at a lag
other than zero. A swapped or flipped axis would show as a rotated correlation, a unit
error as a ratio of speeds far from 1.

The vector correlation is the complex one: r = <conj(h') b'> / sqrt(<|h'|^2><|b'|^2>),
with h' and b' the anomalies from their means. |r| is how well the two agree, and
angle(r) is the mean rotation from HYCOM to the buoy.

    python scripts/check_current_lag.py --data /home/26p67/data --windows <stem> \
        --days 400 --out <json>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sar.pipeline.gridded import GriddedForcing
from sar.validate.calibrate_sigma import FORCING_ERRORS, task_batches
from sar.validate.drift_windows import Windows


def vector_stats(buoy: np.ndarray, model: np.ndarray) -> dict:
    b, h = buoy - buoy.mean(), model - model.mean()
    r = np.mean(np.conj(h) * b) / np.sqrt(np.mean(np.abs(h) ** 2) * np.mean(np.abs(b) ** 2))
    return {"n": int(buoy.size), "abs_r": float(np.abs(r)),
            "angle_deg": float(np.degrees(np.angle(r))),
            "rms_diff_ms": float(np.sqrt(np.mean(np.abs(buoy - model) ** 2))),
            "speed_ratio": float(np.mean(np.abs(model)) / np.mean(np.abs(buoy)))}


def main(argv=None) -> dict:
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--data", required=True)
    p.add_argument("--windows", required=True)
    p.add_argument("--days", type=int, default=400)
    p.add_argument("--lags", type=float, nargs="+",
                   default=[-24, -12, -6, -3, -1, 0, 1, 3, 6, 12, 24], help="hours")
    p.add_argument("--seed", type=int, default=20261002)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    w = Windows.load(args.windows)
    tab = w.table
    reach = np.timedelta64(int(max(abs(x) for x in args.lags) * 3600), "s")
    got = {lag: [] for lag in args.lags}
    rows_used = []
    for _, rows in task_batches(w, days=args.days, seed=args.seed):
        t0 = tab["t0"].iloc[rows[0]].to_datetime64()
        try:
            with GriddedForcing.from_dir(args.data, t0 - reach, t0 + reach) as f:
                for lag in args.lags:
                    at = t0 + np.timedelta64(int(lag * 3600), "s")
                    current, _ = f.sample(tab["lat0"].to_numpy()[rows],
                                          tab["lon0"].to_numpy()[rows], at)
                    got[lag].append(current[:, 0] + 1j * current[:, 1])
        except FORCING_ERRORS:
            continue
        rows_used.append(rows)
    rows_used = np.concatenate(rows_used)
    buoy = tab["ve0"].to_numpy()[rows_used] + 1j * tab["vn0"].to_numpy()[rows_used]
    tier = tab["tier"].to_numpy()[rows_used]
    out = {"windows": int(rows_used.size), "days": args.days, "by_tier": {}}
    for name in ("drogued", "undrogued"):
        out["by_tier"][name] = {}
        for lag in args.lags:
            model = np.concatenate(got[lag])
            keep = (tier == name) & np.isfinite(buoy) & np.isfinite(model)
            out["by_tier"][name][str(lag)] = vector_stats(buoy[keep], model[keep])
    Path(args.out).write_text(json.dumps(out, indent=2))
    for name, lags in out["by_tier"].items():
        best = max(lags, key=lambda k: lags[k]["abs_r"])
        print(name, "best lag", best, "h:", json.dumps(lags[best]))
        print("  |r| by lag:", {k: round(v["abs_r"], 3) for k, v in lags.items()})
    return out


if __name__ == "__main__":
    main()
