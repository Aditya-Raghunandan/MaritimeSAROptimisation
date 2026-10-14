"""How far each real buoy ends up from its cloud, for every game window (55 buoys x 6 arrivals).

Reads the published bundles (scenarios/v4 on Hugging Face) the game itself loads: the window's
JSON (marker track, real buoy track) and, by HTTP range request, only the first and last frames
of its cloud (float32 [frame][particle][east_m, north_m] from the marker at that frame). At the
end of the window (45 min after arrival, the moment the game's reveal shows the real buoy):

  d      the real buoy's distance from the cloud's centre
  r90    the radius about the centre holding 90 % of the particles
  ratio  d / r90 (under 1: inside the cloud's 90 % circle)
  depth  the share of particles nearer the centre than the buoy
  near   the distance from the buoy to the nearest particle

Used 15 Oct to choose which windows the open-day game leaves out (frontend/src/farBuoys.js).
Needs only the network: python scripts/buoy_vs_cloud.py --out buoy_vs_cloud.csv
"""

import argparse
import csv
import json
import math
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

BASE = "https://huggingface.co/datasets/AdityaRugs/MaritimeSARoperations/resolve/main/scenarios/v4"
R_EARTH_M = 6371008.8
M_PER_DEG_LAT = R_EARTH_M * math.pi / 180.0
WINDOW_S = 45 * 60
ARRIVALS = ["0.25", "0.5", "0.75", "1", "2", "3"]


def get(url, rng=None):
    req = urllib.request.Request(url, headers={"Range": f"bytes={rng[0]}-{rng[1]}"} if rng else {})
    for attempt in range(4):
        try:
            return urllib.request.urlopen(req, timeout=60).read()
        except Exception:
            if attempt == 3:
                raise
    return None


def at(track, t):
    return float(np.interp(t, track["t_s"], track["lat"])), float(np.interp(t, track["t_s"], track["lon"]))


def measure(scenario, arrival):
    name = f"rvc_{arrival}h"
    meta = json.loads(get(f"{BASE}/{scenario}/{name}.json"))
    frames, n = meta["frames"], meta["particles"]
    frame_bytes = n * 2 * 4
    last = frames - 1
    raw = get(f"{BASE}/{scenario}/{name}.f32", (last * frame_bytes, (last + 1) * frame_bytes - 1))
    xy = np.frombuffer(raw, dtype="<f4").reshape(n, 2).astype(float)       # east, north from the marker
    t_end = last * meta["step_s"]
    mlat, mlon = at(meta["marker"], t_end)
    blat, blon = at(meta["target"], t_end)
    buoy = np.array([(blon - mlon) * M_PER_DEG_LAT * math.cos(math.radians(mlat)), (blat - mlat) * M_PER_DEG_LAT])
    centre = xy.mean(axis=0)
    r = np.hypot(*(xy - centre).T)
    d = float(np.hypot(*(buoy - centre)))
    r90 = float(np.quantile(r, 0.9))
    return {
        "scenario": scenario, "arrival_h": arrival, "t_end_s": t_end,
        "d_km": round(d / 1000, 3), "r90_km": round(r90 / 1000, 3), "ratio": round(d / r90, 3),
        "depth": round(float(np.mean(r < d)), 4),
        "near_km": round(float(np.min(np.hypot(*(xy - buoy).T))) / 1000, 3),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="buoy_vs_cloud.csv", help="the CSV to write")
    args = parser.parse_args()
    index = json.loads(get(f"{BASE}/index.json"))
    jobs = [(s["scenario"], a) for s in index["scenarios"] for a in ARRIVALS
            if f"rvc_{a}h" in s["windows"]]
    with ThreadPoolExecutor(8) as pool:
        rows = list(pool.map(lambda j: measure(*j), jobs))
    out = Path(args.out)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} windows -> {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
