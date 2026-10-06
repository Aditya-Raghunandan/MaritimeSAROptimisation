"""score_scenarios.py: the doctrinal patterns flown over the scenario table, scored by the referee.

Every row of the scenario table (scripts/pick_scenario_buoys.py) is a real buoy's start. For
each row, noise model and arrival this rebuilds the row's cloud at its own seed
(`sar.search.scenario`), flies each searcher through one `SearchEpisode` (ADR004), and writes
what the referee measured, one JSON line per flight:

  * HOW MUCH. pos, the probability the flight cleared: the RL return with no discount.
  * HOW FAST. removed_per_step, the probability cleared in each minute, which is the RL
    reward and the search's rate; its running total at 15 and 30 min; and expected_ttd_s,
    when on average the probability was found.
  * THE REAL BUOY. found, found_s and closest_m, against the buoy's truth every hour, by the
    site's closest-approach rule. One buoy is one coin toss; pos is the measurement.

TWO NOISE MODELS ON THE SAME SEEDS. `rv` is the engine's random velocity (D030, sigma_u =
0.226 m/s). `rw` is the old random walk (D028, sigma = 26.3), which the first scenario runs
and the 4 Oct training clouds used. Same row, same seed, so each flight has a pair.

    python scripts/score_scenarios.py score --csv scenarios.csv --forcing-dir DATA \\
        --rows 1-55 --noise rv rw --arrival-h 1 2 3 --out DIR
    python scripts/score_scenarios.py summary DIR

One file per row, DIR/S01.jsonl, rewritten on a rerun. `summary` reads them all.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sar.model.position import CALIBRATED_SIGMA, CALIBRATED_SIGMA_U
from sar.pipeline.gridded import GriddedForcing
from sar.search.episode import STEPS, SearchEpisode, pattern_policy, run
from sar.search.patterns import PATTERNS
from sar.search.platform import STEP_S, expanding_square_spacing_m
from sar.search.scenario import read_scenarios, scenario_search
from sar.utils.geo import to_display_longitude

SEARCHERS = ("expanding-square", "sector")
NOISE = {"rv": {"sigma_u": CALIBRATED_SIGMA_U, "sigma": 0.0},
         "rw": {"sigma_u": 0.0, "sigma": CALIBRATED_SIGMA}}


def row_names(spec, table: pd.DataFrame) -> list[str]:
    """S01, 7 and 1-55 style row specs as scenario names, checked against the table."""
    names = []
    for item in spec:
        item = str(item).strip().upper()
        if item.startswith("S"):
            names.append(item)
        elif "-" in item:
            a, b = (int(x) for x in item.split("-"))
            names += [f"S{k:02d}" for k in range(a, b + 1)]
        else:
            names.append(f"S{int(item):02d}")
    unknown = sorted(set(names) - set(table["scenario"]))
    if unknown:
        raise ValueError(f"not in the scenario table: {', '.join(unknown)}")
    return names


def searcher(name: str, bearing_deg):
    """A pattern as a searcher, its first leg along the target's drift at the datum."""
    first = 0.0 if bearing_deg is None else float(bearing_deg)
    if name == "expanding-square":
        pattern = PATTERNS[name](expanding_square_spacing_m(), first)
    else:
        pattern = PATTERNS[name](first_bearing_deg=first)
    return pattern_policy(pattern), first


def score_row(row, forcing, noises, arrivals_h, searchers, particles: int) -> list[dict]:
    """Every (noise, arrival, searcher) flight over one row, as flat dicts.

    `forcing` is a backend, or a data root, opened once for the latest arrival's window.
    """
    if isinstance(forcing, (str, Path)):
        start = np.datetime64(str(row["start"]), "us")
        longest = max(arrivals_h) * 3600.0 + STEPS * STEP_S
        forcing = GriddedForcing.from_dir(forcing, start,
                                          start + np.timedelta64(int(round(longest)), "s"))
    flights = []
    for noise in noises:
        for hours in arrivals_h:
            setup = scenario_search(row, forcing, round(hours * 3600.0), particles,
                                    **NOISE[noise])
            beached = float(np.mean(setup.window.beached[-1]))
            for name in searchers:
                policy, first = searcher(name, setup.drift_bearing_deg)
                m = run(policy, SearchEpisode(setup.window, setup.marker, target=setup.target))
                removed = m["removed_per_step"]
                target = m.get("target") or {}
                flights.append({
                    "scenario": row["scenario"], "set": row["set"], "water": row["stratum"],
                    "noise": noise, **NOISE[noise], "arrival_h": hours, "searcher": name,
                    "first_bearing_deg": first,
                    "datum_lat": setup.datum[0],
                    "datum_lon": float(to_display_longitude(setup.datum[1])),
                    "pos": m["pos"],
                    "pos_15m": float(np.sum(removed[:15])),
                    "pos_30m": float(np.sum(removed[:30])),
                    "expected_ttd_s": m["expected_ttd_s"],
                    "found": target.get("found"), "found_s": target.get("found_s"),
                    "closest_m": target.get("closest_m"),
                    "closest_s": target.get("closest_s"), "gaps": target.get("gaps"),
                    "beached_share": beached, "distance_m": m["distance_m"],
                    "particles": particles, "removed_per_step": removed})
    return flights


def read_flights(folder) -> pd.DataFrame:
    paths = sorted(Path(folder).glob("S*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"no S*.jsonl in {folder}")
    return pd.DataFrame([json.loads(line) for p in paths for line in p.read_text().splitlines()
                         if line.strip()])


def summarise(flights: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Means per (noise, searcher, arrival), and the paired rv - rw difference in pos."""
    f = flights.assign(found=flights["found"].astype(float),
                       ttd_min=flights["expected_ttd_s"].astype(float) / 60.0)
    keys = ["noise", "searcher", "arrival_h"]
    means = f.groupby(keys).agg(n=("pos", "size"), pos=("pos", "mean"), pos_sd=("pos", "std"),
                                pos_15m=("pos_15m", "mean"), pos_30m=("pos_30m", "mean"),
                                ttd_min=("ttd_min", "mean"), found=("found", "mean"),
                                beached=("beached_share", "mean")).reset_index()
    pairs = pd.DataFrame()
    if set(f["noise"]) >= {"rv", "rw"}:
        wide = f.pivot_table(index=["scenario", "searcher", "arrival_h"], columns="noise",
                             values="pos").dropna()
        d = (wide["rv"] - wide["rw"]).rename("diff").reset_index()
        pairs = d.groupby(["searcher", "arrival_h"])["diff"].agg(
            n="size", mean="mean", sd="std",
            rv_better=lambda x: int((x > 0).sum())).reset_index()
        pairs["se"] = pairs["sd"] / np.sqrt(pairs["n"])
    return means, pairs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("score", help="fly and score rows of the scenario table")
    s.add_argument("--csv", required=True, help="the scenario table")
    s.add_argument("--forcing-dir", required=True, help="data root holding raw/hycom_*, raw/era5_*")
    s.add_argument("--rows", nargs="+", required=True, help="S01, 7 or 1-55; several allowed")
    s.add_argument("--noise", nargs="+", choices=sorted(NOISE), default=["rv"])
    s.add_argument("--arrival-h", nargs="+", type=float, default=[1.0, 2.0, 3.0],
                   help="hours from the call to arrival, whole minutes; default 1 2 3")
    s.add_argument("--searchers", nargs="+", choices=SEARCHERS, default=list(SEARCHERS))
    s.add_argument("--particles", type=int, default=10_000)
    s.add_argument("--out", required=True, help="folder for one S##.jsonl per row")
    m = sub.add_parser("summary", help="tabulate a folder of scored rows")
    m.add_argument("folder")
    args = parser.parse_args(argv)

    if args.command == "summary":
        means, pairs = summarise(read_flights(args.folder))
        with pd.option_context("display.width", 140, "display.float_format", "{:.4f}".format):
            print(means.to_string(index=False))
            if not pairs.empty:
                print("\npaired pos, random velocity minus random walk, same seeds:")
                print(pairs.to_string(index=False))
        return means, pairs

    table = read_scenarios(args.csv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in row_names(args.rows, table):
        row = table[table["scenario"] == name].iloc[0]
        t0 = time.perf_counter()
        flights = score_row(row, args.forcing_dir, args.noise, args.arrival_h,
                            args.searchers, args.particles)
        (out / f"{name}.jsonl").write_text("".join(json.dumps(f) + "\n" for f in flights))
        best = max(flights, key=lambda f: f["pos"])
        print(f"{name}: {len(flights)} flights in {time.perf_counter() - t0:.1f} s; best pos "
              f"{best['pos']:.3f} ({best['searcher']}, {best['noise']}, {best['arrival_h']:g} h)",
              flush=True)


if __name__ == "__main__":
    main()
