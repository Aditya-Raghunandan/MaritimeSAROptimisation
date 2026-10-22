"""fly.py: fly a v2 model beside the benchmark's searchers; the same flights and summary as sar.rl.fly.

    PYTHONPATH=src python -m sar.rl_v2.fly windows --windows <windows dir> --arrival-min 30 60 120 180 \
        --searchers ppo expanding-square --model <run>/best_model.zip --out <run>/test

Only the PPO policy differs from v1: it hands the model the 2D observation. Everything else, the
referee, the patterns, the JSON and summary.csv, is v1's, imported.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from sar.rl.features import Context, heading_of, time_weighted_return
from sar.rl.fly import PPO, flight_entry, load_policy, summarise, window_record
from sar.rl.windows import is_test, load, parse_windows_name
from sar.rl_v2.features import observe
from sar.search.benchmark import GREEDY, SEARCHERS, searcher


def ppo_policy(model, headings: int, context: Context):
    """The v2 policy as a searcher: the most likely heading from the 2D observation it trained on."""
    def policy(episode) -> float:
        action, _ = model.predict(observe(episode, context), deterministic=True)
        return heading_of(int(action), headings)

    return policy


def fly_window(setup, context: Context, row: dict, names, model=None, headings: int | None = None) -> dict:
    """Every named searcher over one window, as {name: flight entry}."""
    out = {}
    for name in names:
        if name == PPO:
            out[name] = flight_entry(ppo_policy(model, headings, context), None,
                                     f"PPO v2, {headings} headings every 60 s", setup)
        else:
            policy, first, layout = searcher(name, setup, row, setup.arrival_s / 3600.0, GREEDY)
            out[name] = flight_entry(policy, first, layout, setup)
    return out


def score_windows(paths, names, model=None, headings=None, arrivals_min=None, out=None) -> pd.DataFrame:
    """Fly every searcher over every window in `paths`; one row per flight, bundle-format JSON per window."""
    rows = []
    for path in paths:
        windows = load(path)
        params = parse_windows_name(path)
        for k, arrival in enumerate(windows.arrivals_s):
            minutes = round(float(arrival) / 60.0)
            if arrivals_min is not None and minutes not in arrivals_min:
                continue
            setup = windows.setup(k)
            context = Context(windows.lkp, setup.arrival_s, setup.drift_bearing_deg)
            row = {"lat": windows.lkp[0], "lon": windows.lkp[1], "seed": windows.seed}
            flights = fly_window(setup, context, row, names, model, headings)
            if out is not None:
                folder = Path(out) / "flights"
                folder.mkdir(parents=True, exist_ok=True)
                (folder / f"{windows.name}_{minutes}m.json").write_text(
                    json.dumps(window_record(windows, setup, flights), separators=(",", ":")))
            for name, f in flights.items():
                rows.append({"window": windows.name, "lat": params["lat"], "lon": params["lon"],
                             "start": params["start"], "arrival_min": minutes, "searcher": name,
                             "pos": f["python"]["pos"],
                             "time_weighted": time_weighted_return(f["python"]["drain_rate"]),
                             "expected_ttd_s": f["python"]["expected_ttd_s"]})
    return pd.DataFrame(rows)


def _cli(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    w = sub.add_parser("windows", help="score searchers over the test windows, the last week of each month")
    w.add_argument("--windows", required=True, help="directory of windows_*.npz files")
    w.add_argument("--arrival-min", type=int, nargs="+", required=True)
    w.add_argument("--searchers", nargs="+", required=True, choices=[PPO, *SEARCHERS])
    w.add_argument("--model", help="a v2 model.zip or best_model.zip, for ppo")
    w.add_argument("--out", required=True, help="directory for flights.csv, summary.csv, flights/")
    args = parser.parse_args(argv)

    if PPO in args.searchers and not args.model:
        parser.error("--searchers ppo needs --model")
    model, headings = load_policy(args.model) if PPO in args.searchers else (None, None)
    paths = [p for p in sorted(Path(args.windows).glob("windows_*.npz")) if is_test(p)]
    if not paths:
        parser.error(f"no test windows in {args.windows}")
    out = Path(args.out)
    flights = score_windows(paths, args.searchers, model, headings, args.arrival_min, out)
    flights.to_csv(out / "flights.csv", index=False)
    summary = summarise(flights)
    summary.to_csv(out / "summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    _cli()
