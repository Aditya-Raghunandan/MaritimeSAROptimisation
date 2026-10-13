"""ceiling.py: the most any 45-minute search could clear on each scenario's cloud, beside what each searcher did.

The question (13 Oct): nothing trained beats the Coast Guard's patterns by much, so is there
room above them at all? A searcher can only be beaten by as much as it leaves below the
ceiling, so this measures the ceiling.

THE HELICOPTER'S BUDGET. In one window it sweeps at most Z = W x V x T = 185.2 m x 46.3 m/s
x 2,700 s = 23.15 km² about the marker (`sar.search.platform.search_effort_m2`): a strip W
wide along 125 km of track, never overlapping itself, with no time lost turning. Detection is
certain inside the strip (pod 1, the referee's), so no searcher clears more than the
probability inside the 23.15 km² it paints.

THREE NUMBERS PER WINDOW, each on the cloud frozen at one frame, in the marker's frame
(everyone flies about the marker, ADR003 row 7):

  * ANY SHAPE. The fullest 23.15 km² of the cloud in 250 m cells (D007's episode cell, the
    cells area90_km2 is counted on): the cells sorted by probability and added up until the
    budget runs out. The arithmetic of area90 read the other way. It ignores that a
    helicopter must fly one connected path, so it is a ceiling, not a plan.
  * BEST RECTANGLE. The 23.15 km² rectangle, any centre, any angle (5 degree steps) and
    length over width 1 to 8, holding the most: the most a Parallel Track fitted to this
    cloud could clear.
  * TEXTBOOK SQUARE. The Parallel Track's own 4.81 km square, centred on the marker along
    its bearing (`sar.search.scenario.doctrinal_searcher`), frozen: what doctrine's area
    holds before turns and the cloud's motion take their share.

WHICH FRAME. The cloud moves against the marker over the window (leeway, the random
velocity) and spreads, so its fullest 23.15 km² changes. Any shape is taken at every frame:
`any_max` is the cloud at its most concentrated, the ceiling reported; `any_mid` is minute
22. A searcher could in principle beat a frozen ceiling by catching particles that move
into its strip, so this is an estimate, and the summary counts the flights that exceed it.
On the 55 the cloud thins fast inside the window (at 1 h the fullest 23.15 km² holds 0.96
at arrival, 0.79 at minute 45), and no helicopter paints everything at arrival, so
`any_max` is generous. Hence a fourth number:

  * TELEPORT. A helicopter that may jump anywhere: each minute it paints the fullest 1/45
    of the budget of what is still unfound, on that minute's frame (`teleport`). No path,
    no turns, but the clock is real. The fair ceiling to read a searcher against.

LUCK IN THE CELLS. The fullest cells of 10^4 particles are partly full by chance, which
flatters the ceiling. `any_max_split` chooses the cells with half the particles and counts
them with the other half, both ways round, averaged: it cannot be flattered, and runs a
little low. The truth lies between the two.

THE SAME CLOUDS AS THE BENCHMARK. Each row's cloud is rebuilt at its own seed
(`sar.search.scenario.scenario_search`, noise rvc, as score_scenarios.py does) and its
area90_km2 recomputed; the summary checks it against the benchmark's flights.parquet, which
proves the cloud is the one that was flown.

    python scripts/ceiling.py run --csv scenarios.csv --forcing-dir DATA --rows 1-14 --out DIR
    python scripts/ceiling.py summary DIR --flights RESULTS/flights.parquet
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from sar.search.episode import STEPS
from sar.search.platform import STEP_S, search_effort_m2
from sar.search.sweep import relative_m

BUDGET_M2 = search_effort_m2()
CELL_M = 250.0
AREA_SHARE = 0.9
# The rectangle search: a 100 m grid, coarsened if a cloud would need over MAX_CELLS a side.
RECT_CELL_M, MAX_CELLS = 100.0, 600
ANGLES_DEG = np.arange(0.0, 180.0, 5.0)
ASPECTS = (1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0)
MID = STEPS // 2
SEARCHERS = ("expanding-square", "sector", "parallel", "trackline", "greedy", "random", "ppo",
             "telemachus")


# -- the three numbers, on one frozen frame -----------------------------------------------

def _cells(east, north, cell_m: float):
    keys = np.floor(east / cell_m).astype(np.int64) * 1_000_003 \
        + np.floor(north / cell_m).astype(np.int64)
    return np.unique(keys, return_inverse=True)[1]


def _top(mass_sorted, budget_m2: float, cell_m: float) -> float:
    """The fullest budget's worth of cells, the last one in part."""
    n = budget_m2 / (cell_m * cell_m)
    k = int(n)
    return float(np.sum(mass_sorted[:k]) + (n - k) * (mass_sorted[k] if k < mass_sorted.size else 0.0))


def any_shape(east, north, weight, budget_m2: float = BUDGET_M2, cell_m: float = CELL_M) -> float:
    """The probability in the fullest `budget_m2` of the cloud, on `cell_m` cells."""
    mass = np.bincount(_cells(east, north, cell_m), weights=weight)
    return _top(np.sort(mass)[::-1], budget_m2, cell_m)


def any_shape_split(east, north, weight, budget_m2: float = BUDGET_M2,
                    cell_m: float = CELL_M) -> float:
    """As `any_shape`, the cells chosen on one half of the particles and counted on the other."""
    inverse = _cells(east, north, cell_m)
    halves = np.arange(weight.size) % 2 == 0
    n = budget_m2 / (cell_m * cell_m)
    k = int(n)
    out = []
    for choose, count in ((halves, ~halves), (~halves, halves)):
        chosen = np.bincount(inverse[choose], weights=weight[choose], minlength=inverse.max() + 1)
        counted = np.bincount(inverse[count], weights=weight[count], minlength=inverse.max() + 1)
        order = np.argsort(-chosen, kind="stable")
        got = np.sum(counted[order[:k]]) + (n - k) * (counted[order[k]] if k < order.size else 0.0)
        out.append(got * np.sum(weight) / np.sum(weight[count]))
    return float(np.mean(out))


def area_km2(east, north, weight, share: float = AREA_SHARE, cell_m: float = CELL_M) -> float:
    """The smallest area holding `share` of the cloud: score_scenarios.cloud_area_km2's rule."""
    held = np.cumsum(np.sort(np.bincount(_cells(east, north, cell_m), weights=weight))[::-1])
    return (int(np.searchsorted(held, share * held[-1] - 1e-12)) + 1) * cell_m * cell_m / 1e6


def _axes(east, north, bearing_deg: float):
    """Along and across a bearing, degrees true: the along axis points down the bearing."""
    s, c = np.sin(np.radians(bearing_deg)), np.cos(np.radians(bearing_deg))
    return east * s + north * c, east * c - north * s


def square(east, north, weight, bearing_deg: float, side_m: float = float(np.sqrt(BUDGET_M2))) -> float:
    """The probability inside a square centred on the marker along `bearing_deg`."""
    along, across = _axes(east, north, bearing_deg)
    inside = (np.abs(along) <= side_m / 2) & (np.abs(across) <= side_m / 2)
    return float(np.sum(weight[inside]))


def teleport(clouds, budget_m2: float = BUDGET_M2, cell_m: float = CELL_M,
             split: bool = False) -> float:
    """A helicopter that may jump anywhere: each minute it paints the fullest 1/STEPS of the
    budget of what is still unfound, on that minute's frame. `clouds` is every frame's
    (east, north, weight, ok) over all particles. With `split`, the cells are chosen on one
    half of the particles and the find counted on the other, both ways round, averaged."""
    # Whole cells only, the budget's fraction carried to the next minute: a cell taken in part
    # and chosen again would be charged a whole cell's area for what is left of it.
    n = budget_m2 / STEPS / (cell_m * cell_m)
    weight = clouds[0][2]
    halves = np.arange(weight.size) % 2 == 0
    rounds = ((halves, ~halves), (~halves, halves)) if split else ((None, None),)
    out = []
    for choose, count in rounds:
        left = weight.copy()
        found = 0.0
        for minute, (east, north, _, ok) in enumerate(clouds[:STEPS]):
            k = int((minute + 1) * n) - int(minute * n)
            inverse = np.full(weight.size, -1)
            inverse[ok] = _cells(east[ok], north[ok], cell_m)
            seen = inverse >= 0
            pick = seen if choose is None else seen & choose
            mass = np.bincount(inverse[pick], weights=left[pick], minlength=inverse.max() + 1)
            chosen = np.zeros(mass.size, dtype=bool)
            chosen[np.argsort(-mass, kind="stable")[:k]] = True
            taken = np.where(seen & chosen[np.maximum(inverse, 0)], left, 0.0)
            found += np.sum(taken if count is None else taken[count])
            left -= taken
        out.append(found if count is None else found * np.sum(weight) / np.sum(weight[count]))
    return float(np.mean(out))


def best_rectangle(east, north, weight, budget_m2: float = BUDGET_M2) -> dict:
    """The `budget_m2` rectangle, any centre, angle and length over width, holding the most."""
    best = {"mass": -1.0}
    for bearing in ANGLES_DEG:
        along, across = _axes(east, north, bearing)
        lo_a, lo_c = along.min(), across.min()
        g = max(RECT_CELL_M, (along.max() - lo_a) / MAX_CELLS, (across.max() - lo_c) / MAX_CELLS)
        i = ((along - lo_a) / g).astype(np.int64)
        j = ((across - lo_c) / g).astype(np.int64)
        # Whole cells, never more area than the budget: n_a along, n_c across.
        sizes = []
        for aspect in ASPECTS:
            n_a = max(1, int(round(np.sqrt(budget_m2 * aspect) / g)))
            sizes.append((aspect, n_a, max(1, int(budget_m2 / (n_a * g * g)))))
        shape = (max(i.max() + 1, max(n for _, n, _ in sizes)),
                 max(j.max() + 1, max(n for _, _, n in sizes)))
        grid = np.bincount(i * shape[1] + j, weights=weight,
                           minlength=shape[0] * shape[1]).reshape(shape)
        s = np.zeros((shape[0] + 1, shape[1] + 1))
        s[1:, 1:] = grid.cumsum(0).cumsum(1)
        for aspect, n_a, n_c in sizes:
            sums = s[n_a:, n_c:] - s[:-n_a, n_c:] - s[n_a:, :-n_c] + s[:-n_a, :-n_c]
            a, c = np.unravel_index(int(np.argmax(sums)), sums.shape)
            if sums[a, c] > best["mass"]:
                mid_a, mid_c = lo_a + (a + n_a / 2) * g, lo_c + (c + n_c / 2) * g
                sb, cb = np.sin(np.radians(bearing)), np.cos(np.radians(bearing))
                best = {"mass": float(sums[a, c]), "bearing_deg": float(bearing),
                        "aspect": float(aspect), "length_m": n_a * g, "width_m": n_c * g,
                        "centre_east_m": float(mid_a * sb + mid_c * cb),
                        "centre_north_m": float(mid_a * cb - mid_c * sb)}
    return best


# -- one window ---------------------------------------------------------------------------

def frames(setup):
    """Every frame's particles as (east, north) metres from the marker, their weights, and
    which are usable (finite, with weight)."""
    marker_lat, marker_lon = setup.marker.at(np.arange(STEPS + 1) * STEP_S)
    weight = np.asarray(setup.window.weight, dtype=float)
    for k in range(STEPS + 1):
        east, north = relative_m(setup.window.lat[k], setup.window.lon[k],
                                 marker_lat[k], marker_lon[k])
        yield east, north, weight, np.isfinite(east) & np.isfinite(north) & (weight > 0)


def window_ceiling(setup, parallel_bearing_deg: float) -> dict:
    full = list(frames(setup))
    clouds = [(e[ok], n[ok], w[ok]) for e, n, w, ok in full]
    by_frame = [any_shape(*f) for f in clouds]
    k = int(np.argmax(by_frame))
    rect_max, rect_mid = best_rectangle(*clouds[k]), best_rectangle(*clouds[MID])
    return {"total_mass": float(np.sum(setup.window.weight)),
            "area90_km2": area_km2(*clouds[0]),
            "any_by_frame": by_frame, "any_max": by_frame[k], "any_max_frame": k,
            "any_mid": by_frame[MID], "any_end": by_frame[-1],
            "any_max_split": any_shape_split(*clouds[k]),
            "any_mid_split": any_shape_split(*clouds[MID]),
            "teleport": teleport(full), "teleport_split": teleport(full, split=True),
            "rect_max": rect_max["mass"], "rect_mid": rect_mid["mass"],
            "rect_max_shape": rect_max, "rect_mid_shape": rect_mid,
            "parallel_bearing_deg": parallel_bearing_deg,
            "square_start": square(*clouds[0], parallel_bearing_deg),
            "square_mid": square(*clouds[MID], parallel_bearing_deg),
            "square_end": square(*clouds[-1], parallel_bearing_deg),
            "square_max": square(*clouds[k], parallel_bearing_deg)}


def run_rows(csv, forcing_dir, rows, arrivals_h, particles: int, out: Path) -> None:
    from sar.pipeline.gridded import ForcingGapError, GriddedForcing
    from sar.search.benchmark import NOISE
    from sar.search.scenario import doctrinal_searcher, read_scenarios, scenario_search

    table = read_scenarios(csv)
    a, b = (int(x) for x in rows.split("-")) if "-" in rows else (int(rows),) * 2
    out.mkdir(parents=True, exist_ok=True)
    for _, row in table.iloc[a - 1:min(b, len(table))].iterrows():
        start = np.datetime64(str(row["start"]), "us")
        longest = max(arrivals_h) * 3600.0 + STEPS * STEP_S
        lines = []
        try:
            forcing = GriddedForcing.from_dir(forcing_dir, start,
                                              start + np.timedelta64(int(round(longest)), "s"))
            for hours in arrivals_h:
                setup = scenario_search(row, forcing, round(hours * 3600.0), particles,
                                        **NOISE["rvc"])
                _, bearing, _ = doctrinal_searcher("parallel", setup,
                                                   (float(row["lat"]), float(row["lon"])))
                lines.append({"scenario": row["scenario"], "noise": "rvc", "arrival_h": hours,
                              "particles": particles, **window_ceiling(setup, bearing)})
        except ForcingGapError as gap:
            lines = [{"scenario": row["scenario"], "skipped": str(gap)}]
        (out / f"{row['scenario']}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
        print(row["scenario"], "done", flush=True)


# -- the summary --------------------------------------------------------------------------

def _group_mean(df: pd.DataFrame, column: str) -> float:
    """Averaged within each shared-water group first, then across groups (D025)."""
    return float(df.groupby("group")[column].mean().mean())


def summary(folder: Path, flights_path: Path) -> pd.DataFrame:
    rows = [json.loads(line) for p in sorted(folder.glob("*.jsonl"))
            for line in p.read_text().splitlines() if line.strip()]
    ceil = pd.DataFrame([r for r in rows if "skipped" not in r])
    flights = pd.read_parquet(flights_path)
    flights = flights[flights["noise"] == "rvc"]
    pos = flights.pivot_table(index=["scenario", "arrival_h"], columns="searcher", values="pos")
    about = flights.groupby(["scenario", "arrival_h"])[["group", "area90_km2"]].first()
    df = ceil.merge(about.rename(columns={"area90_km2": "bench_area90_km2"}).reset_index(),
                    on=["scenario", "arrival_h"]).merge(pos.reset_index(), on=["scenario", "arrival_h"])
    mismatch = df[np.abs(df["area90_km2"] - df["bench_area90_km2"]) > 1e-9]
    print(f"{len(df)} windows; area90 matches the benchmark's in {len(df) - len(mismatch)}"
          + ("" if mismatch.empty else f"; NOT in {sorted(mismatch['scenario'].unique())}"))
    searchers = [s for s in SEARCHERS if s in df.columns]
    table = []
    for hours, d in df.groupby("arrival_h"):
        line = {"arrival_h": hours, "windows": len(d), "groups": d["group"].nunique(),
                "ceiling": _group_mean(d, "any_max"),
                "ceiling_split": _group_mean(d, "any_max_split"),
                "rect_share": _group_mean(d.assign(r=d["rect_max"] / d["any_max"]), "r"),
                "square_share": _group_mean(d.assign(r=d["square_max"] / d["any_max"]), "r")}
        line["teleport"] = _group_mean(d, "teleport")
        line["teleport_split"] = _group_mean(d, "teleport_split")
        # The best rectangle's shape, and what choosing the pattern case by case, in hindsight,
        # would have been worth: the most a searcher that picks or fits a pattern could add.
        shapes = d["rect_max_shape"]
        line["rect_aspect_median"] = float(np.median([s["aspect"] for s in shapes]))
        line["rect_offset_median_m"] = float(np.median(
            [np.hypot(s["centre_east_m"], s["centre_north_m"]) for s in shapes]))
        patterns = [s for s in ("expanding-square", "sector", "parallel", "trackline") if s in d]
        line["best_pattern_pos"] = _group_mean(d.assign(r=d[patterns].max(axis=1)), "r")
        line["best_searcher_pos"] = _group_mean(d.assign(r=d[searchers].max(axis=1)), "r")
        for s in searchers:
            line[f"{s}_pos"] = _group_mean(d, s)
            line[f"{s}_share"] = _group_mean(d.assign(r=d[s] / d["any_max"]), "r")
            line[f"{s}_share_split"] = _group_mean(d.assign(r=d[s] / d["any_max_split"]), "r")
            line[f"{s}_of_teleport"] = _group_mean(d.assign(r=d[s] / d["teleport"]), "r")
            line[f"{s}_of_teleport_split"] = _group_mean(d.assign(r=d[s] / d["teleport_split"]), "r")
            line[f"{s}_over_ceiling"] = int(np.sum(d[s] > d["any_max"] + 1e-12))
            line[f"{s}_over_teleport"] = int(np.sum(d[s] > d["teleport"] + 1e-12))
        table.append(line)
    out = pd.DataFrame(table)
    df.drop(columns=["any_by_frame"]).to_csv(folder / "windows.csv", index=False)
    out.to_csv(folder / "summary.csv", index=False)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--csv", required=True)
    r.add_argument("--forcing-dir", required=True)
    r.add_argument("--rows", required=True, help="1-based rows, such as 1-14")
    r.add_argument("--arrival-h", type=float, nargs="+", default=[1.0, 2.0, 3.0])
    r.add_argument("--particles", type=int, default=10_000)
    r.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("summary")
    s.add_argument("folder", type=Path)
    s.add_argument("--flights", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.cmd == "run":
        run_rows(args.csv, args.forcing_dir, args.rows, args.arrival_h, args.particles, args.out)
    else:
        with pd.option_context("display.width", 200, "display.max_columns", 60):
            print(summary(args.folder, args.flights).round(3).T.to_string())


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
