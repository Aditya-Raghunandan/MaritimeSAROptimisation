"""score_scenarios.py: the non-ML benchmark, every searcher flown over the scenario table and scored.

Every row of the scenario table (scripts/pick_scenario_buoys.py) is a real buoy's start. For
each row, noise model and arrival this rebuilds the row's cloud at its own seed
(`sar.search.scenario`), flies each searcher through one `SearchEpisode` (ADR004), and writes
what the referee measured, one JSON line per flight:

  * HOW MUCH. pos, the probability the flight cleared: the RL return with no discount.
  * HOW FAST. drain_rate, the share of the probability cleared in each minute (the coined
    name for the referee's removed_per_step; the RL reward). POS is its sum. Also its
    running total at 15 and 30 min, and expected_ttd_s, when on average it was found.
  * THE REAL BUOY. found, found_s and closest_m, against the buoy's truth every hour, by the
    site's closest-approach rule. One buoy is one coin toss; pos is the measurement.
  * THE SCENARIO. Its shared-water group (D025), water type, speed and straightness (how far
    the buoy got over 4 h divided by how far it travelled), for the summary's slices.

THE SEARCHERS. The four Coast Guard patterns, laid out by `sar.search.scenario.
doctrinal_searcher` exactly as the site lays them; greedy and the random floor
(`sar.search.greedy`, #49). The random floor's seed is the row's seed and the arrival, so
both noise models are searched with the same random headings.

THE HELICOPTER TURNS AT 7 DEG/S (D032). Every flight is flown by a helicopter that turns
at most `--turn-rate` degrees a second (default `TURN_RATE_DEG_S`, a 30 degree bank at
90 kt), arriving along the drift (`sar.search.scenario.search_episode`). `--turn-rate inf`
is the referee before D032, whose helicopter turns at once: the "as the manual draws it"
comparison. The rate is part of the experiment, so a folder never mixes the two.

TWO NOISE MODELS ON THE SAME SEEDS. `rv` is the engine's random velocity (D030, sigma_u =
0.226 m/s). `rw` is the old random walk (D028, sigma = 26.3). Same row, same seed, so each
flight has a pair.

ONE FOLDER, ONE EXPERIMENT. `score` writes DIR/manifest.json the first time (code commit,
the table's sha256, N, noise, arrivals, searchers) and refuses to add flights made with
anything different, so a folder can never mix two experiments. One file per row,
DIR/flights/<scenario>.jsonl, rewritten on a rerun. A row whose window crosses a gap in the
forcing (D011) is skipped and leaves DIR/flights/<scenario>.skipped.json saying why.

THE SUMMARY AVERAGES GROUPS, NOT ROWS. Buoys that drifted within 10 km of each other ride the
same water (D025), and a long-lived buoy gives many rows. So every number is averaged within
each group first and then across groups, and its 95 % interval comes from resampling whole
groups. `summary` writes flights.parquet and summary.csv beside the rows.

    python scripts/score_scenarios.py score --csv scenarios.csv --forcing-dir DATA \\
        --rows 1-55 --noise rv rw --arrival-h 1 2 3 --out DIR
    python scripts/score_scenarios.py summary DIR [--straight-at 0.90]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from sar.pipeline.gridded import ForcingGapError, GriddedForcing
from sar.search.benchmark import BASELINE, GREEDY, NOISE, SEARCHERS, searcher
from sar.search.episode import STEPS, run
from sar.search.platform import STEP_S, TURN_RATE_DEG_S
from sar.search.scenario import (
    arrival_heading,
    read_scenarios,
    scenario_search,
    search_episode,
    straightness,
)
from sar.utils.geo import to_display_longitude

RESAMPLES = 1000
# Straight against turning: 0.95 is a curve turning through about 60 degrees over the 4 h.
# The dev median is 0.995, a straight line to half a percent, so it would split straight
# buoys from straight buoys (7 Oct: 12 % of the 9,538 dev rows are below 0.95).
STRAIGHT_AT = 0.95
BOOTSTRAP_SEED = 20261007
REPO = Path(__file__).resolve().parents[1]


def row_names(spec, table: pd.DataFrame) -> list[str]:
    """Row specs as scenario names: a name (S01, D0007), or 1-based positions such as 7 or 1-55."""
    names = list(table["scenario"])
    known = set(names)
    out = []
    for item in spec:
        item = str(item).strip()
        if item.upper() in known:
            out.append(item.upper())
        elif item.isdigit() or ("-" in item and item.replace("-", "").isdigit()):
            a, b = (int(x) for x in item.split("-")) if "-" in item else (int(item),) * 2
            bad = [k for k in range(a, b + 1) if not 1 <= k <= len(names)]
            if bad:
                raise ValueError(f"row {bad[0]} is not in a table of {len(names)} rows")
            out += names[a - 1:b]
        else:
            raise ValueError(f"not in the scenario table: {item}")
    return out


def _scenario_fields(row) -> dict:
    """What the summary slices by: group, water type, speed and straightness."""
    get = lambda k: row[k] if k in row.index else None          # noqa: E731
    group = get("group")
    return {"ID": None if get("ID") is None else str(get("ID")),
            "group": None if group is None or pd.isna(group) else int(group),
            "speed_4h_ms": None if get("speed_4h_ms") is None else float(get("speed_4h_ms")),
            "straightness_4h": straightness(row) if "lat_4h" in row.index else None}


def score_row(row, forcing, noises, arrivals_h, searchers, particles: int,
              greedy: dict | None = None,
              turn_rate_deg_s: float = TURN_RATE_DEG_S) -> list[dict]:
    """Every (noise, arrival, searcher) flight over one row, as flat dicts.

    `forcing` is a backend, or a data root, opened once for the latest arrival's window.
    """
    if isinstance(forcing, (str, Path)):
        start = np.datetime64(str(row["start"]), "us")
        longest = max(arrivals_h) * 3600.0 + STEPS * STEP_S
        forcing = GriddedForcing.from_dir(forcing, start,
                                          start + np.timedelta64(int(round(longest)), "s"))
    about = _scenario_fields(row)
    flights = []
    for noise in noises:
        for hours in arrivals_h:
            setup = scenario_search(row, forcing, round(hours * 3600.0), particles,
                                    **NOISE[noise])
            beached = float(np.mean(setup.window.beached[-1]))
            for name in searchers:
                policy, first, layout = searcher(name, setup, row, hours, greedy)
                m = run(policy, search_episode(setup, turn_rate_deg_s))
                drain = m["removed_per_step"]
                target = m.get("target") or {}
                flights.append({
                    "scenario": row["scenario"], "set": row.get("set"),
                    "water": row.get("stratum"), **about,
                    "noise": noise, **NOISE[noise], "arrival_h": hours, "searcher": name,
                    "turn_rate_deg_s": m["turn_rate_deg_s"],
                    "arrival_heading_deg": arrival_heading(setup),
                    "first_bearing_deg": first, "layout": layout,
                    "datum_lat": setup.datum[0],
                    "datum_lon": float(to_display_longitude(setup.datum[1])),
                    "pos": m["pos"],
                    "pos_15m": float(np.sum(drain[:15])),
                    "pos_30m": float(np.sum(drain[:30])),
                    "peak_drain_rate": float(np.max(drain)),
                    "expected_ttd_s": m["expected_ttd_s"],
                    "found": target.get("found"), "found_s": target.get("found_s"),
                    "closest_m": target.get("closest_m"),
                    "closest_s": target.get("closest_s"), "gaps": target.get("gaps"),
                    "beached_share": beached, "distance_m": m["distance_m"],
                    "particles": particles, "drain_rate": drain})
    return flights


# -- provenance ---------------------------------------------------------------------------

def git_commit() -> str:
    """The checkout's commit, with -dirty if tracked files differ from it.

    core.checkStat=minimal: on the cluster's compute nodes the shared disk reports file
    metadata the head node's index does not match, and git 2.25 there called files "M"
    whose contents were the commit's byte for byte (7 Oct, jobs 59091-59092).
    """
    held = os.environ.get("SAR_COMMIT")
    if held:
        # Computed once by the batch job (score_scenarios.sbatch). Forty-eight processes each
        # asking git at the same moment on a shared disk did not all get the same answer
        # (7 Oct, job 59217: six tasks refused by their own manifest).
        return held
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                             text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-c", "core.checkStat=minimal", "status", "--porcelain",
                                "--untracked-files=no"],
                               cwd=REPO, capture_output=True, text=True, check=True).stdout
        return sha + ("-dirty" if dirty.strip() else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def experiment(csv, noises, arrivals_h, searchers, particles: int,
               greedy: dict | None = None, turn_rate_deg_s: float = TURN_RATE_DEG_S) -> dict:
    """What makes two flights comparable. A folder holds flights from one of these only."""
    return {"commit": git_commit(), "table": Path(csv).name, "table_sha256": sha256(csv),
            "particles": int(particles), "noise": {n: NOISE[n] for n in noises},
            "arrival_h": [float(a) for a in arrivals_h], "searchers": list(searchers),
            "greedy": dict(GREEDY if greedy is None else greedy),
            "turn_rate_deg_s": None if math.isinf(turn_rate_deg_s) else float(turn_rate_deg_s),
            "steps": STEPS, "step_s": STEP_S}


def claim(out: Path, settings: dict) -> dict:
    """Write DIR/manifest.json if it is new; otherwise refuse settings that differ from it."""
    out.mkdir(parents=True, exist_ok=True)
    path = out / "manifest.json"
    record = {**settings, "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "slurm_job": os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get("SLURM_JOB_ID")}
    # Written whole to a private file, then published by a hard link, which fails if the
    # manifest exists. Four processes start at once on a node; opening the manifest itself
    # with "x" let another read it half-written (7 Oct, jobs 59002-59007).
    tmp = out / f".manifest.{os.getpid()}.{time.time_ns()}.tmp"
    tmp.write_text(json.dumps(record, indent=1))
    try:
        os.link(tmp, path)
        return record
    except FileExistsError:
        held = json.loads(path.read_text())
    finally:
        tmp.unlink(missing_ok=True)
    differ = sorted(k for k in settings if held.get(k) != settings[k])
    if differ:
        raise ValueError(f"{out} already holds an experiment with different {', '.join(differ)}"
                         f" ({'; '.join(f'{k}: {held.get(k)!r} there, {settings[k]!r} here' for k in differ if k == 'commit')})"
                         f"; use a new folder")
    return held


# -- the summary --------------------------------------------------------------------------

def read_flights(folder) -> pd.DataFrame:
    folder = Path(folder)
    paths = sorted((folder / "flights").glob("*.jsonl")) or sorted(folder.glob("S*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"no flights in {folder}")
    rows = [json.loads(line) for p in paths for line in p.read_text().splitlines()
            if line.strip()]
    f = pd.DataFrame(rows)
    if "drain_rate" not in f and "removed_per_step" in f:        # flights from before 7 Oct
        f = f.rename(columns={"removed_per_step": "drain_rate"})
    if "group" not in f or f["group"].isna().all():
        f["group"] = f["scenario"]                               # no groups: each row is its own
    f["group"] = f["group"].astype(str)
    return f


def group_means(f: pd.DataFrame, value: str, keys: list[str]) -> pd.DataFrame:
    """One number per group per key: the mean of `value` over that group's rows."""
    return f.groupby(keys + ["group"], dropna=False)[value].mean().reset_index()


def bootstrap(values: np.ndarray, rng, resamples: int = RESAMPLES) -> tuple[float, float]:
    """95 % interval of the mean, resampling the given per-group values with replacement."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return float("nan"), float("nan")
    draws = values[rng.integers(0, values.size, size=(resamples, values.size))].mean(axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def _tabulate(f: pd.DataFrame, measure: str, value: str, keys: list[str], rng,
              resamples: int, better=None) -> list[dict]:
    rows = []
    per = group_means(f, value, keys)
    windows = f.groupby(keys, dropna=False).size()
    for key, g in per.groupby(keys, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        lo, hi = bootstrap(g[value].to_numpy(), rng, resamples)
        row = {"measure": measure, **dict(zip(keys, key)), "n_rows": int(windows[key]),
               "n_groups": int(g[value].notna().sum()), "mean": float(g[value].mean()),
               "lo": lo, "hi": hi}
        if better is not None:
            row["groups_better"] = int((g[value] > 0).sum())
        rows.append(row)
    return rows


def slices(f: pd.DataFrame, straight_at: float | None) -> list[tuple[str, pd.DataFrame]]:
    """The whole table, then by water type, then straight against turning buoys."""
    out = [("all", f)]
    if "water" in f and f["water"].notna().any():
        out += [(f"water: {w}", f[f["water"] == w]) for w in ("quiet", "moderate", "jet")
                if (f["water"] == w).any()]
    if straight_at is not None and "straightness_4h" in f:
        s = f["straightness_4h"].astype(float)
        out += [(f"path: straight (>= {straight_at:.3f})", f[s >= straight_at]),
                (f"path: turning (< {straight_at:.3f})", f[s < straight_at])]
    return [(name, part) for name, part in out if len(part)]


def summarise(flights: pd.DataFrame, straight_at: float | None = STRAIGHT_AT,
              resamples: int = RESAMPLES) -> pd.DataFrame:
    """Every searcher's numbers, and the paired differences, averaged by group, with 95 % CIs.

    Measures: pos, found (the real buoy's found rate), ttd_min (expected time to detection),
    pos_15m. Paired: pos with memory minus without (rv - rw, same row and seed), and each
    searcher's pos minus the Expanding Square's on the same cloud. `straight_at` splits
    straight from turning buoys (STRAIGHT_AT); None takes the table's own median.
    """
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    f = flights.assign(found=flights["found"].astype(float),
                       ttd_min=flights["expected_ttd_s"].astype(float) / 60.0)
    if straight_at is None and "straightness_4h" in f and f["straightness_4h"].notna().any():
        straight_at = float(f.drop_duplicates("scenario")["straightness_4h"].median())
    keys = ["noise", "searcher", "arrival_h"]
    rows = []
    for name, part in slices(f, straight_at):
        for measure in ("pos", "found", "ttd_min", "pos_15m"):
            rows += [{**r, "slice": name}
                     for r in _tabulate(part, measure, measure, keys, rng, resamples)]
        if set(part["noise"]) >= {"rv", "rw"}:
            wide = part.pivot_table(index=["scenario", "group", "searcher", "arrival_h"],
                                    columns="noise", values="pos").dropna().reset_index()
            wide["diff"] = wide["rv"] - wide["rw"]
            wide["noise"] = "rv - rw"
            rows += [{**r, "slice": name} for r in _tabulate(
                wide, "pos difference", "diff", keys, rng, resamples, better=True)]
        if BASELINE in set(part["searcher"]):
            wide = part.pivot_table(index=["scenario", "group", "noise", "arrival_h"],
                                    columns="searcher", values="pos").reset_index()
            for other in sorted(set(part["searcher"]) - {BASELINE}):
                d = wide[["scenario", "group", "noise", "arrival_h"]].assign(
                    searcher=f"{other} - {BASELINE}", diff=wide[other] - wide[BASELINE])
                rows += [{**r, "slice": name} for r in _tabulate(
                    d.dropna(subset=["diff"]), "pos difference", "diff", keys, rng, resamples,
                    better=True)]
    cols = ["measure", "slice", "noise", "searcher", "arrival_h", "n_rows", "n_groups",
            "mean", "lo", "hi", "groups_better"]
    out = pd.DataFrame(rows)
    return out[[c for c in cols if c in out]]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("score", help="fly and score rows of the scenario table")
    s.add_argument("--csv", required=True, help="the scenario table")
    s.add_argument("--forcing-dir", required=True, help="data root holding raw/hycom_*, raw/era5_*")
    s.add_argument("--rows", nargs="+", required=True,
                   help="scenario names, or 1-based rows such as 7 or 1-55; several allowed")
    s.add_argument("--noise", nargs="+", choices=sorted(NOISE), default=["rv"])
    s.add_argument("--arrival-h", nargs="+", type=float, default=[1.0, 2.0, 3.0],
                   help="hours from the call to arrival, whole minutes; default 1 2 3")
    s.add_argument("--searchers", nargs="+", choices=SEARCHERS, default=list(SEARCHERS))
    s.add_argument("--particles", type=int, default=10_000)
    s.add_argument("--greedy-headings", type=int, default=GREEDY["headings"])
    s.add_argument("--greedy-decide-s", type=float, default=GREEDY["decide_s"],
                   help="how often greedy chooses a heading; must divide 60 s")
    s.add_argument("--turn-rate", type=float, default=TURN_RATE_DEG_S,
                   help=f"deg/s the helicopter can turn, default {TURN_RATE_DEG_S:.3f} (D032); "
                        f"inf for the pre-D032 referee that turns at once")
    s.add_argument("--out", required=True, help="the experiment's folder")
    m = sub.add_parser("summary", help="tabulate an experiment's folder")
    m.add_argument("folder")
    m.add_argument("--straight-at", type=float, default=STRAIGHT_AT,
                   help=f"straightness splitting straight from turning buoys, default "
                        f"{STRAIGHT_AT} (a curve through about 60 degrees)")
    m.add_argument("--resamples", type=int, default=RESAMPLES)
    args = parser.parse_args(argv)

    if args.command == "summary":
        folder = Path(args.folder)
        flights = read_flights(folder)
        table = summarise(flights, args.straight_at, args.resamples)
        flights.to_parquet(folder / "flights.parquet", index=False)
        table.to_csv(folder / "summary.csv", index=False, float_format="%.5f",
                     lineterminator="\n")
        overall = table[(table["slice"] == "all") & (table["measure"] == "pos")]
        with pd.option_context("display.width", 160, "display.float_format", "{:.4f}".format):
            print(overall.drop(columns=["measure", "slice"]).to_string(index=False))
        skipped = sorted((folder / "flights").glob("*.skipped.json"))
        print(f"\n{len(flights)} flights, {flights['scenario'].nunique()} scenarios, "
              f"{flights['group'].nunique()} groups, {len(skipped)} skipped (forcing gaps) "
              f"-> {folder / 'summary.csv'}")
        return table

    table = read_scenarios(args.csv)
    out = Path(args.out)
    greedy = {"headings": args.greedy_headings, "decide_s": args.greedy_decide_s}
    claim(out, experiment(args.csv, args.noise, args.arrival_h, args.searchers, args.particles,
                          greedy, args.turn_rate))
    (out / "flights").mkdir(exist_ok=True)
    for name in row_names(args.rows, table):
        row = table[table["scenario"] == name].iloc[0]
        t0 = time.perf_counter()
        try:
            flights = score_row(row, args.forcing_dir, args.noise, args.arrival_h,
                                args.searchers, args.particles, greedy, args.turn_rate)
        except ForcingGapError as err:
            # D011: a scenario that crosses a gap in the forcing is excluded, and says why.
            (out / "flights" / f"{name}.skipped.json").write_text(
                json.dumps({"scenario": name, "reason": str(err)}))
            print(f"{name}: skipped, {err}", flush=True)
            continue
        (out / "flights" / f"{name}.jsonl").write_text(
            "".join(json.dumps(f) + "\n" for f in flights))
        best = max(flights, key=lambda f: f["pos"])
        print(f"{name}: {len(flights)} flights in {time.perf_counter() - t0:.1f} s; best pos "
              f"{best['pos']:.3f} ({best['searcher']}, {best['noise']}, {best['arrival_h']:g} h)",
              flush=True)


if __name__ == "__main__":
    main()
