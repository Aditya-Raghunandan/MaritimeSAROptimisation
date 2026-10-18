"""truth_windows.py: real buoys' starts as training windows with the buoy to find (13 Oct, Telemachus 3).

Telemachus 2 was paid for probability (POS) on practice clouds with no real buoy. On real
buoys the Coast Guard's patterns find more than POS says and the agent fewer (13 Oct probe,
290 dev buoys: Parallel Track +10 points at 2 h, +17 at 3 h), so the next agent is paid for
finding the buoy itself. That needs clouds that come with the buoy's real track.

`select` picks the rows once, from the dev table (bench_dev_48h.csv: dev buoys only, D025; the
sealed 55 are never touched): every --val-every-th shared-water group is VALIDATION, the rest
TRAIN, so the agent is picked on buoys in water it never trained on. At most --per-group rows
from each training group and --val-per-group from each validation group, shuffled at a fixed
seed, so one long-lived buoy cannot fill the set. Writes rows_train.csv and rows_validation.csv.

`build` makes one windows file per row (`sar.rl.windows.build_truth`: the benchmark's rvc cloud
at the row's seed, saved every 5 min, the four arrivals, the buoy's hourly fixes) into
OUT/train or OUT/validation, rows K, K + P, ... so P processes share the work. A row whose
forcing has a gap (D011) is skipped and said so.

    python scripts/truth_windows.py select --csv bench_dev_48h.csv --out DIR
    python scripts/truth_windows.py build --forcing-dir DATA --out DIR --part K --parts P
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sar.rl.windows import WINDOW_S, build_truth, save, windows_name
from sar.search.platform import STEP_S

SEED = 20261013


def select(csv, out: Path, per_group: int, val_per_group: int, val_every: int) -> dict:
    rows = pd.read_csv(csv).sample(frac=1.0, random_state=SEED)
    validation = rows["group"] % val_every == 0
    parts = {"train": rows[~validation].groupby("group").head(per_group),
             "validation": rows[validation].groupby("group").head(val_per_group)}
    out.mkdir(parents=True, exist_ok=True)
    for role, part in parts.items():
        part.sort_values("scenario").to_csv(out / f"rows_{role}.csv", index=False)
        print(f"{role}: {len(part)} rows from {part['group'].nunique()} groups")
    return parts


def build(forcing_dir, out: Path, part: int, parts: int, arrivals_min, particles: int,
          shape: dict | None = None, roles=("train", "validation")) -> None:
    from sar.pipeline.gridded import ForcingGapError, GriddedForcing

    arrivals = [60.0 * m for m in arrivals_min]
    for role in roles:
        rows = pd.read_csv(out / f"rows_{role}.csv").iloc[part::parts]
        for _, row in rows.iterrows():
            run = {"lat": float(row["lat"]), "lon": float(row["lon"]), "start": str(row["start"]),
                   "particles": particles, "timestep": STEP_S,
                   "duration": max(arrivals) + WINDOW_S, "seed": int(row["seed"])}
            if (out / role / f"{windows_name(run, arrivals)}.npz").exists():
                continue
            start = np.datetime64(str(row["start"]), "us")
            end = start + np.timedelta64(int(max(arrivals) + WINDOW_S), "s")
            try:
                with GriddedForcing.from_dir(forcing_dir, start, end) as forcing:
                    windows = build_truth(row, forcing, arrivals, particles, shape=shape)
            except (ForcingGapError, ValueError, FileNotFoundError) as error:
                print(f"{row['scenario']}: skipped, {error}", flush=True)
                continue
            save(windows, out / role)
            print(f"{row['scenario']} -> {role}", flush=True)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--csv", required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--per-group", type=int, default=24)
    s.add_argument("--val-per-group", type=int, default=8)
    s.add_argument("--val-every", type=int, default=5,
                   help="groups whose number is a multiple of this are validation")
    b = sub.add_parser("build")
    b.add_argument("--forcing-dir", required=True)
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--part", type=int, default=0)
    b.add_argument("--parts", type=int, default=1)
    b.add_argument("--arrival-min", type=int, nargs="+", default=[30, 60, 120, 180])
    b.add_argument("--particles", type=int, default=5000)
    b.add_argument("--shape-json", help="the cloud-shape correction (cloud_shape.py fit, L41): "
                   "current_scale, wind_scale, along, across")
    b.add_argument("--roles", nargs="+", default=["train", "validation"])
    args = parser.parse_args(argv)
    if args.cmd == "select":
        select(args.csv, args.out, args.per_group, args.val_per_group, args.val_every)
    else:
        shape = json.loads(Path(args.shape_json).read_text()) if args.shape_json else None
        build(args.forcing_dir, args.out, args.part, args.parts, args.arrival_min, args.particles,
              shape, args.roles)


if __name__ == "__main__":
    main()
