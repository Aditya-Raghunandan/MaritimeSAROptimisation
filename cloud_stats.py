"""cloud_stats.py: Table D.clouds, distance from the LKP to the cloud centre and R90 at 4 h, for 10 days at the 5 sites.

Run on the cluster from the repository root. Reads the run CSVs in <montes>/derived; a run whose CSV
is gone is listed, and can be made again with the same seed (it gives the same particles):

    /home/26p67/envs/sar/bin/python cloud_stats.py --montes /home/26p67/data/derived/rl/montes \
        --out cloud_stats.csv

Writes one row per (site, day): distance_km, r90_km, the frame's time, the beached share and the
random term recorded in the sidecar JSON, then prints mean and s.d. per site.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

R_KM = 6371.0088
SITES = {"2516N08006W": ("Florida Current", 25.1629, -80.0647),
         "3016N07982W": ("Off Jacksonville", 30.1599, -79.8213),
         "2546N07560W": ("East of Eleuthera", 25.4645, -75.5972),
         "3486N07170W": ("Ring zone", 34.8649, -71.6963),
         "2907N06888W": ("Sargasso eddy field", 29.0705, -68.8835)}
# Every sixth test day: the last Tuesday of January and of July, 2019 to 2023.
DAYS = ["20190129", "20190730", "20200128", "20200728", "20210126",
        "20210727", "20220125", "20220726", "20230131", "20230725"]


def km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * R_KM * np.arcsin(np.sqrt(a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--montes", required=True)
    ap.add_argument("--at-s", type=float, default=4 * 3600)
    ap.add_argument("--out", default="cloud_stats.csv")
    ap.add_argument("--windows", help="the study's windows_*.npz folder, to check each re-run against the original")
    a = ap.parse_args()
    rows, missing = [], []
    for code, (site, lat0, lon0) in SITES.items():
        for day in DAYS:
            csv = Path(a.montes) / "derived" / f"ensemble_{code}_{day}T1700_N10000_dt60s_T4h_seed20261008.csv"
            if not csv.exists():
                missing.append((site, day, lat0, lon0))
                continue
            full = pd.read_csv(csv, usecols=["seconds", "particle", "lat", "lon", "beached"])
            match_m = None
            if a.windows:
                # The re-run reproduces the original only if every particle sits where the windows file has it.
                npz = Path(a.windows) / f"windows_{code}_{day}T1700_N10000_dt60s_T4h_seed20261008_arr30-60-120-180m.npz"
                if npz.exists():
                    with np.load(npz) as w:
                        diffs = []
                        for k, t in enumerate(w["t_s"]):
                            f = full[full.seconds == t].sort_values("particle")
                            if len(f) == w["lat"].shape[1]:
                                diffs.append(km(f.lat.to_numpy(), f.lon.to_numpy(), w["lat"][k], w["lon"][k]).max())
                        match_m = 1000 * max(diffs) if diffs else None
            d = full[full.seconds == full.seconds[(full.seconds - a.at_s).abs().idxmin()]]
            lon = (d.lon.to_numpy() + 180) % 360 - 180
            clat, clon = d.lat.mean(), np.degrees(np.arctan2(np.sin(np.radians(lon)).mean(), np.cos(np.radians(lon)).mean()))
            side = csv.with_suffix(".json")
            noise = json.loads(side.read_text()).get("pipeline", {}) if side.exists() else {}
            rows.append({"site": site, "day": day, "t_s": float(d.seconds.iloc[0]),
                         "distance_km": float(km(lat0, lon0, clat, clon)),
                         "r90_km": float(np.percentile(km(clat, clon, d.lat.to_numpy(), lon), 90)),
                         "beached_share": float(d.beached.mean()),
                         "sigma_u": noise.get("sigma_u"), "sigma": noise.get("sigma"),
                         "max_diff_from_original_m": match_m})
    t = pd.DataFrame(rows)
    t.to_csv(a.out, index=False)
    if len(t):
        print(t.groupby("site")[["distance_km", "r90_km"]].agg(["mean", "std"]).round(2).to_string())
    for site, day, lat0, lon0 in missing:
        print(f"MISSING {site} {day}: PYTHONPATH=src python -m sar.pipeline.ensemble --forcing-dir /home/26p67/data "
              f"--lat {lat0} --lon {lon0} --datum-sigma-km 0 --start {day[:4]}-{day[4:6]}-{day[6:]}T17:00:00 "
              f"--timestep 60 --duration 4h --particles 10000 --seed 20261008 --save-every 5m --out {a.montes}")


if __name__ == "__main__":
    main()
