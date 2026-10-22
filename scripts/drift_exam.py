"""drift_exam.py: the drift model's exam on a benchmark table (D025's R2, as amended 4 Oct).

Is the real buoy inside the cloud's 90 % region, and is the cloud's centre closer to it than
"the buoy stayed where it was reported"? Built for the sealed set's one opening (22 Oct), and
run first on the dev validation rows, so the sealed numbers have a dev twin made by the same
code.

PER ROW. The cloud is the scorer's own (`sar.search.scenario.scenario_search`, the row's seed,
the named noise model), run once to 4 h, the end of the longest search, and read at each whole
hour in --hours. The buoy is the table's fix at that hour (lat_<h>h, lon_<h>h, D025's truth).

  * R2c, CONTAINMENT. The buoy's rank among the cloud's highest-density regions
    (`sar.validate.region.score_cloud`, Hyndman 1996): inside the 90 % region when the rank is
    at most 0.9. The densities are taken on SUBSAMPLE particles drawn at a fixed seed, since
    the kernel is N x N (`region.py`: ~10^3 particles are enough).
  * R2b, THE CENTRE. The distance from the cloud's weighted centre to the buoy, beside the
    distance from the last known position: "it stayed put", the only centre test since 4 Oct.
  * THE SHAPE (D035). The buoy's offset from the centre along the drift (the unit vector from
    the last known position to the centre) and across it, and whether it sits inside the
    cloud's own central 90 % on each axis (the particles' weighted 5th and 95th percentiles).

`summary` reports, per hour, the share inside pooled over rows (D025's test), and the same
averaged within each shared-water group first, with a 95 % interval resampling groups; the
share inside on each axis; the median distances; and the sign test of R2b: per group the median
of (centre - stayed put), the share of groups where the centre is closer, and the one-sided
binomial p. R2 passes at 4 h when the pooled share inside is at least 0.80 and p < 0.05.

    python scripts/drift_exam.py run --csv TABLE --forcing-dir DATA --noise rvsc --out DIR \\
        [--part K --parts P]
    python scripts/drift_exam.py summary DIR
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

HOURS = (1, 2, 3, 4)
SUBSAMPLE = 2000
LEVEL = 0.9
R2C_SHARE = 0.80               # D025, amended 4 Oct: inside the 90 % region in >= 80 %
R2B_P = 0.05                   # D025: the centre beats "stayed put", one-sided sign test
MIN_DRIFT_M = 50.0             # cloud_shape.py's: below this the drift has no direction
BOOTSTRAP_SEED, RESAMPLES = 20261022, 2000


def weighted_quantile(x, w, q) -> np.ndarray:
    order = np.argsort(x)
    x, w = x[order], w[order]
    cum = (np.cumsum(w) - 0.5 * w) / np.sum(w)
    return np.interp(q, cum, x)


def score_frame(lat, lon, weight, lkp, truth, rng) -> dict:
    """One cloud against one fix, in metres about the last known position."""
    from sar.validate.region import local_metres, score_cloud

    ok = np.isfinite(lat) & np.isfinite(lon) & (weight > 0)
    p = local_metres(lat[ok], lon[ok], *lkp)
    w = weight[ok] / np.sum(weight[ok])
    b = local_metres(np.array([truth[0]]), np.array([truth[1]]), *lkp)[0]
    centre = w @ p
    pick = rng.choice(p.shape[0], size=min(SUBSAMPLE, p.shape[0]), replace=False, p=w)
    hdr = score_cloud(p[pick], b, levels=(LEVEL,))
    out = {"rank": hdr.rank, "inside90": bool(hdr.rank <= LEVEL),
           "area90_hdr_km2": hdr.area_m2[LEVEL] / 1e6,
           "d_centre_m": float(np.hypot(*(b - centre))), "d_stayed_m": float(np.hypot(*b)),
           "drift_m": float(np.hypot(*centre))}
    if out["drift_m"] >= MIN_DRIFT_M:
        u = centre / out["drift_m"]
        v = np.array([-u[1], u[0]])                    # left of the drift
        along, across = (p - centre) @ u, (p - centre) @ v
        b_along, b_across = float((b - centre) @ u), float((b - centre) @ v)
        lo_a, hi_a = weighted_quantile(along, w, [0.05, 0.95])
        lo_c, hi_c = weighted_quantile(across, w, [0.05, 0.95])
        out.update({"along_m": b_along, "across_m": b_across,
                    "inside_along": bool(lo_a <= b_along <= hi_a),
                    "inside_across": bool(lo_c <= b_across <= hi_c),
                    "spread_along_m": float(np.sqrt(w @ along ** 2)),
                    "spread_across_m": float(np.sqrt(w @ across ** 2))})
    return out


def exam_row(row, forcing_dir, noise: str, particles: int, hours=HOURS) -> list[dict]:
    from sar.pipeline.gridded import GriddedForcing
    from sar.search.benchmark import NOISE
    from sar.search.platform import STEP_S
    from sar.search.scenario import scenario_search

    first, last = min(hours), max(hours)
    start = np.datetime64(str(row["start"]), "us")
    forcing = GriddedForcing.from_dir(forcing_dir, start,
                                      start + np.timedelta64(int(last * 3600), "s"))
    steps = int(round((last - first) * 3600 / STEP_S))
    setup = scenario_search(row, forcing, first * 3600.0, particles, steps=steps, **NOISE[noise])
    weight = np.asarray(setup.window.weight, dtype=float)
    lkp = (float(row["lat"]), float(row["lon"]))
    rng = np.random.default_rng(int(row["seed"]))
    group = row["group"] if "group" in row.index else None
    lines = []
    for h in hours:
        truth = (float(row[f"lat_{h}h"]), float(row[f"lon_{h}h"]))
        if not np.all(np.isfinite(truth)):
            continue
        k = int(round((h - first) * 3600 / STEP_S))
        lines.append({"scenario": row["scenario"], "set": row.get("set"), "ID": str(row["ID"]),
                      "unit": row.get("unit"), "group": None if group is None else int(group),
                      "water": row.get("stratum"), "speed_4h_ms": float(row["speed_4h_ms"]),
                      "noise": noise, "particles": particles, "hour": h,
                      "start_current_ms": setup.start_current_ms, "sigma_u": setup.sigma_u,
                      **score_frame(setup.window.lat[k], setup.window.lon[k], weight, lkp,
                                    truth, rng)})
    return lines


def run(csv, forcing_dir, noise: str, particles: int, out: Path, part: int, parts: int) -> None:
    from sar.pipeline.gridded import ForcingGapError

    table = pd.read_csv(csv)
    folder = Path(out) / "rows"
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(part, len(table), parts):
        row = table.iloc[i]
        try:
            lines = exam_row(row, forcing_dir, noise, particles)
        except ForcingGapError as gap:
            lines = [{"scenario": row["scenario"], "skipped": str(gap)}]
        (folder / f"{row['scenario']}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
        print(row["scenario"], "done", flush=True)


def _group_first(df: pd.DataFrame, column: str, rng) -> tuple[float, float, float]:
    by = df.groupby("group")[column].mean()
    boot = [by.sample(len(by), replace=True, random_state=rng).mean() for _ in range(RESAMPLES)]
    return float(by.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def sign_test(df: pd.DataFrame) -> dict:
    """R2b: per shared-water group, is the centre closer than "stayed put"? One-sided binomial."""
    from scipy.stats import binomtest

    gap = (df["d_centre_m"] - df["d_stayed_m"]).groupby(df["group"]).median()
    gap = gap[gap != 0]
    wins = int((gap < 0).sum())
    p = binomtest(wins, len(gap), 0.5, alternative="greater").pvalue if len(gap) else float("nan")
    return {"groups": int(len(gap)), "centre_wins": wins, "sign_p": float(p)}


def summary(folder: Path) -> pd.DataFrame:
    rows = [json.loads(line) for p in sorted((Path(folder) / "rows").glob("*.jsonl"))
            for line in p.read_text().splitlines() if line.strip()]
    skipped = [r for r in rows if "skipped" in r]
    df = pd.DataFrame([r for r in rows if "skipped" not in r])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    table = []
    for h, d in df.groupby("hour"):
        g, lo, hi = _group_first(d, "inside90", rng)
        shaped = d.dropna(subset=["inside_along"])
        line = {"hour": h, "rows": len(d), "buoys": d["ID"].nunique(), "groups": d["group"].nunique(),
                "inside90": float(d["inside90"].mean()),
                "inside90_group_first": g, "inside90_lo": lo, "inside90_hi": hi,
                "inside_along": float(shaped["inside_along"].astype(float).mean()),
                "inside_across": float(shaped["inside_across"].astype(float).mean()),
                "median_d_centre_km": float(d["d_centre_m"].median() / 1e3),
                "median_d_stayed_km": float(d["d_stayed_m"].median() / 1e3),
                "median_lag_km": float(-shaped["along_m"].median() / 1e3),
                "median_area90_km2": float(d["area90_hdr_km2"].median()),
                **sign_test(d)}
        line["r2c_pass"] = bool(line["inside90"] >= R2C_SHARE)
        line["r2b_pass"] = bool(line["sign_p"] < R2B_P)
        table.append(line)
    out = pd.DataFrame(table)
    df.to_csv(Path(folder) / "exam.csv", index=False)
    out.to_csv(Path(folder) / "summary.csv", index=False)
    print(f"{len(df)} fixes from {df['scenario'].nunique()} rows; {len(skipped)} rows skipped "
          f"(a gap in the forcing)")
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--csv", required=True, help="a benchmark table (pick_scenario_buoys.py --every)")
    r.add_argument("--forcing-dir", required=True)
    r.add_argument("--noise", default="rvsc", help="the drift model's noise, as the scorer's")
    r.add_argument("--particles", type=int, default=10_000)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--part", type=int, default=0)
    r.add_argument("--parts", type=int, default=1)
    s = sub.add_parser("summary")
    s.add_argument("folder", type=Path)
    args = parser.parse_args(argv)
    if args.cmd == "run":
        run(args.csv, args.forcing_dir, args.noise, args.particles, args.out, args.part, args.parts)
    else:
        with pd.option_context("display.width", 200, "display.max_columns", 40):
            print(summary(args.folder).round(3).T.to_string())


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
