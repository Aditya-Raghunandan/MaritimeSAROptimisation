"""fleet.py: more sweep from one helicopter, or more helicopters? (22 Oct, Aditya)

One MH-60T sortie sweeps Z = W x V x T = 23.15 km² (D027). Two ways to sweep k times that in the
same 45 minutes, both flown by the benchmark's referee over the benchmark's own cloud (the row's
seed, the named drift model) and laid out by the same doctrine (the Parallel Track on the
drifting marker, along the datum line or the drift: `sar.search.scenario.doctrinal_searcher`):

  * WIDE, one helicopter with k x the sweep width: track spacing and sweep width kW, over the
    square of side sqrt(k) x 4.81 km on the marker. As if one aircraft could see k times as far.
  * FLEET, k ordinary helicopters: the same square cut into k strips along the axis, one each
    (IAMSAR's sub-areas, one per unit), each flown as a Parallel Track with spacing W. All arrive
    at the marker together and transit to their own strip inside the window. A particle is found
    if any of them sweeps it (pod 1, so the union of what each swept is exact); the real buoy is
    found if any of them finds it, at the first time.

Both cover the same area with the same total track, so what differs is how it is flown: the
wide helicopter turns between legs kW apart (a plain U once kW reaches the 758 m turning circle),
each helicopter of the fleet between legs W apart (a loop), and the fleet spends time getting to
its strips. k = 1 is the benchmark's Parallel Track, flown again here so every comparison is paired.

ONE RULE FOR WHERE EACH UNIT STARTS (k >= 2): on the leg of its own area nearest the datum, then
outwards. A strip beside the axis starts on its inner edge; an area centred on the axis (the wide
helicopter's square, the middle strip of three) starts on the middle leg, sweeps out to one
edge, crosses back and sweeps out to the other. Started on an outer edge instead, a unit reaches
the dense middle of the cloud last (two dev rows, 22 Oct: three helicopters then cleared less
than one at 1 h). k = 1 keeps the benchmark's start, its area's left edge.

NOT MODELLED, and to be said wherever this is shown: cost, fuel, crews, aircraft availability,
keeping k aircraft apart, and that W = 0.1 NM is what a person in the water allows by eye, so a
k-fold sweep width stands for a better sensor, not a better pair of eyes.

    python scripts/fleet.py run --csv TABLE --forcing-dir DATA --noise rvsc --out DIR [--part K --parts P]
    python scripts/fleet.py summary DIR
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

KS = (2, 3, 4)
ARRIVALS_H = (0.5, 1.0, 2.0, 3.0)


def strip_track(length_m: float, width_m: float, spacing_m: float, bearing_deg: float,
                centre_across_m: float, middle_first: bool = True):
    """A Parallel Track over a length x width strip whose centre is `centre_across_m` to the right
    of the marker's axis: legs along the axis, `spacing_m` apart, the first nearest the axis (see
    the module's rule), drawn twice over so it never ends before the window does (the
    autopilot's turns make it slower than the drawing). With `middle_first` False a strip on the
    axis starts on its left edge, as `parallel_track` does: k = 1, the benchmark's own."""
    from sar.search import patterns as P
    from sar.search.platform import ON_SCENE_WINDOW_S, SEARCH_SPEED_MS

    across, near, far = P._parallel_area(length_m, width_m, spacing_m, SEARCH_SPEED_MS,
                                         ON_SCENE_WINDOW_S)
    u, v = P._axes(bearing_deg)
    if centre_across_m < 0:
        across = across[::-1]                         # left of the axis: start on its right edge
    elif centre_across_m == 0 and middle_first:
        right = sorted(a for a in across if a >= 0)
        across = right + sorted((a for a in across if a < 0), reverse=True)

    def points():
        for sweep in range(2):
            order = across if sweep == 0 else across[::-1]
            for k, off in enumerate(order):
                start, end = (near, far) if k % 2 == 0 else (far, near)
                yield P._point(u, v, start, off + centre_across_m)
                yield P._point(u, v, end, off + centre_across_m)

    return P._fly("parallel_track", P._legs_through(points()), SEARCH_SPEED_MS,
                  2 * ON_SCENE_WINDOW_S)


def fly(setup, pattern, sweep_width_m: float):
    """One helicopter flown by the referee: (metrics, which particles it swept)."""
    from sar.search.episode import pattern_policy, run
    from sar.search.scenario import search_episode

    episode = search_episode(setup, sweep_width_m=sweep_width_m)
    before = np.array(episode._weight, copy=True)
    m = run(pattern_policy(pattern), episode)
    return m, episode._weight < before - 1e-15        # pod 1: a swept particle is emptied


def window_fleet(setup, bearing_deg: float) -> dict:
    """POS and the real buoy for k = 1 and every WIDE and FLEET k, on one window."""
    from sar.search.platform import SWEEP_WIDTH_M, search_effort_m2

    weight = np.asarray(setup.window.weight, dtype=float)
    total = float(np.sum(weight))
    side = float(np.sqrt(search_effort_m2()))
    out = {}

    def record(name, metrics_list, swept_list):
        swept = np.logical_or.reduce(swept_list)
        finds = [m.get("target") or {} for m in metrics_list]
        times = [f["found_s"] for f in finds if f.get("found")]
        out[f"{name}_pos"] = float(np.sum(weight[swept]) / total)
        out[f"{name}_found"] = bool(times)
        out[f"{name}_found_s"] = float(min(times)) if times else None
        out[f"{name}_distance_m"] = float(sum(m["distance_m"] for m in metrics_list))

    m, s = fly(setup, strip_track(side, side, SWEEP_WIDTH_M, bearing_deg, 0.0, middle_first=False),
               SWEEP_WIDTH_M)
    record("k1", [m], [s])
    for k in KS:
        big = side * np.sqrt(k)
        m, s = fly(setup, strip_track(big, big, k * SWEEP_WIDTH_M, bearing_deg, 0.0), k * SWEEP_WIDTH_M)
        record(f"wide{k}", [m], [s])
        width = big / k
        flown = [fly(setup, strip_track(big, width, SWEEP_WIDTH_M, bearing_deg,
                                        (i - (k - 1) / 2.0) * width), SWEEP_WIDTH_M)
                 for i in range(k)]
        record(f"fleet{k}", [f[0] for f in flown], [f[1] for f in flown])
    return out


def run_rows(csv, forcing_dir, noise: str, particles: int, out: Path, part: int, parts: int,
             arrivals_h=ARRIVALS_H) -> None:
    from sar.pipeline.gridded import ForcingGapError, GriddedForcing
    from sar.search.benchmark import NOISE
    from sar.search.episode import STEPS
    from sar.search.platform import STEP_S
    from sar.search.scenario import doctrinal_searcher, scenario_search

    table = pd.read_csv(csv)
    folder = Path(out) / "rows"
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(part, len(table), parts):
        row = table.iloc[i]
        start = np.datetime64(str(row["start"]), "us")
        lines = []
        try:
            forcing = GriddedForcing.from_dir(
                forcing_dir, start, start + np.timedelta64(int(max(arrivals_h) * 3600 + STEPS * STEP_S), "s"))
            for hours in arrivals_h:
                setup = scenario_search(row, forcing, round(hours * 3600.0), particles, **NOISE[noise])
                _, bearing, _ = doctrinal_searcher("parallel", setup, (float(row["lat"]), float(row["lon"])))
                lines.append({"scenario": row["scenario"], "group": int(row["group"]),
                              "noise": noise, "arrival_h": hours, "particles": particles,
                              "bearing_deg": bearing, **window_fleet(setup, bearing)})
        except ForcingGapError as gap:
            lines = [{"scenario": row["scenario"], "skipped": str(gap)}]
        (folder / f"{row['scenario']}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
        print(row["scenario"], "done", flush=True)


def summary(folder: Path) -> pd.DataFrame:
    rows = [json.loads(line) for p in sorted((Path(folder) / "rows").glob("*.jsonl"))
            for line in p.read_text().splitlines() if line.strip()]
    df = pd.DataFrame([r for r in rows if "skipped" not in r])
    df.to_csv(Path(folder) / "fleet.csv", index=False)
    names = ["k1"] + [f"{kind}{k}" for k in KS for kind in ("wide", "fleet")]
    table = []
    for h, d in df.groupby("arrival_h"):
        line = {"arrival_h": h, "windows": len(d), "groups": d["group"].nunique()}
        for n in names:
            line[f"{n}_pos"] = float(d.groupby("group")[f"{n}_pos"].mean().mean())
            line[f"{n}_found"] = float(d.groupby("group")[f"{n}_found"].mean().mean())
        for k in KS:
            gap = (d[f"wide{k}_pos"] - d[f"fleet{k}_pos"]) * 100
            line[f"wide{k}_minus_fleet{k}_pts"] = float(gap.groupby(d["group"]).mean().mean())
            line[f"wide{k}_better_groups"] = int((gap.groupby(d["group"]).mean() > 0).sum())
        table.append(line)
    out = pd.DataFrame(table)
    out.to_csv(Path(folder) / "summary.csv", index=False)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--csv", required=True)
    r.add_argument("--forcing-dir", required=True)
    r.add_argument("--noise", default="rvsc")
    r.add_argument("--particles", type=int, default=10_000)
    r.add_argument("--arrival-h", type=float, nargs="+", default=list(ARRIVALS_H))
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--part", type=int, default=0)
    r.add_argument("--parts", type=int, default=1)
    s = sub.add_parser("summary")
    s.add_argument("folder", type=Path)
    args = parser.parse_args(argv)
    if args.cmd == "run":
        run_rows(args.csv, args.forcing_dir, args.noise, args.particles, args.out, args.part,
                 args.parts, tuple(args.arrival_h))
    else:
        with pd.option_context("display.width", 200, "display.max_columns", 60):
            print(summary(args.folder).round(3).T.to_string())


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
