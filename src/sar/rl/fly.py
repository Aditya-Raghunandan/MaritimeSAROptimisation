"""fly.py: fly the trained policy beside the benchmark's searchers and write flights in the site's bundle format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sar.pipeline.ensemble import Ensemble
from sar.rl.features import Context, heading_of, observe, time_weighted_return
from sar.rl.windows import is_test, load, parse_windows_name
from sar.search.benchmark import BASELINE, GREEDY, SEARCHERS, recorded, searcher
from sar.search.episode import SearchEpisode, run
from sar.search.patterns import MarkerTrack
from sar.search.platform import STEP_S
from sar.search.scenario import SearchSetup
from sar.utils.geo import offset_position, to_display_longitude

PPO = "ppo"


def load_policy(model_path):
    """The saved model and the heading count it was trained with, from config.json beside it."""
    from stable_baselines3 import PPO as SB3PPO

    config = json.loads((Path(model_path).parent / "config.json").read_text())["arguments"]
    return SB3PPO.load(model_path, device="cpu"), int(config["headings"])


def ppo_policy(model, headings: int, context: Context):
    """The trained policy as a searcher: the most likely heading from the same observation it trained on."""
    def policy(episode) -> float:
        action, _ = model.predict(observe(episode, context), deterministic=True)
        return heading_of(int(action), headings)

    return policy


def _floats(a, digits: int = 9) -> list:
    return [None if not np.isfinite(v) else round(float(v), digits) for v in np.ravel(a)]


def flight_entry(policy, first_bearing, layout: str, setup: SearchSetup) -> dict:
    """One searcher flown and recorded, in scripts/export_scenario_bundles.py's `flights` format."""
    flying, steps = recorded(policy)
    m = run(flying, SearchEpisode(setup.window, setup.marker, target=setup.target))
    target = m.get("target") or {}
    return {"layout": layout, "first_bearing_deg": first_bearing,
            "steps": [{"t_s": _floats(s.t_s), "east_m": _floats(s.east_m),
                       "north_m": _floats(s.north_m)} for s in steps],
            "python": {"pos": m["pos"], "drain_rate": m["removed_per_step"],
                       "expected_ttd_s": m["expected_ttd_s"], "found": target.get("found"),
                       "found_s": target.get("found_s"), "closest_m": target.get("closest_m"),
                       "closest_s": target.get("closest_s")}}


def fly_window(setup: SearchSetup, context: Context, row: dict, names, model=None,
               headings: int | None = None) -> dict:
    """Every named searcher over one window, as {name: flight entry}."""
    out = {}
    for name in names:
        if name == PPO:
            policy = ppo_policy(model, headings, context)
            out[name] = flight_entry(policy, None, f"PPO, {headings} headings every 60 s", setup)
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
                record = window_record(windows, setup, flights)
                folder = Path(out) / "flights"
                folder.mkdir(parents=True, exist_ok=True)
                (folder / f"{windows.name}_{minutes}m.json").write_text(
                    json.dumps(record, separators=(",", ":")))
            for name, f in flights.items():
                rows.append({"window": windows.name, "lat": params["lat"], "lon": params["lon"],
                             "start": params["start"], "arrival_min": minutes, "searcher": name,
                             "pos": f["python"]["pos"],
                             "time_weighted": time_weighted_return(f["python"]["drain_rate"]),
                             "expected_ttd_s": f["python"]["expected_ttd_s"]})
    return pd.DataFrame(rows)


def window_record(windows, setup: SearchSetup, flights: dict) -> dict:
    """A window's marker, datum and flights, keyed as a scenario bundle's window JSON."""
    return {"format": 1, "scenario": windows.name, "arrival_s": setup.arrival_s,
            "arrival_h": setup.arrival_s / 3600.0, "start_utc": str(windows.start),
            "window_start_utc": str(np.datetime_as_string(setup.window.times[0], unit="s")),
            "frames": int(setup.window.lat.shape[0]), "particles": int(setup.window.lat.shape[1]),
            "step_s": STEP_S,
            "marker": {"t_s": _floats(setup.marker.t_s), "lat": _floats(setup.marker.lat, 10),
                       "lon": _floats(setup.marker.lon, 10)},
            "lkp": {"lat": windows.lkp[0], "lon": float(to_display_longitude(windows.lkp[1]))},
            "datum": {"lat": setup.datum[0], "lon": float(to_display_longitude(setup.datum[1]))},
            "drift_bearing_deg": setup.drift_bearing_deg, "flights": flights}


def summarise(flights: pd.DataFrame) -> pd.DataFrame:
    """Mean POS and time-weighted return per searcher and transit, and PPO's paired difference."""
    table = (flights.groupby(["arrival_min", "searcher"])[["pos", "time_weighted"]]
             .mean().reset_index())
    if PPO in set(flights["searcher"]) and BASELINE in set(flights["searcher"]):
        wide = flights.pivot_table(index=["window", "arrival_min"], columns="searcher",
                                   values="pos")
        diff = (wide[PPO] - wide[BASELINE]).groupby(level="arrival_min")
        wins = diff.apply(lambda d: float((d > 0).mean()))
        table = table.merge(pd.DataFrame({"ppo_minus_square": diff.mean(), "ppo_wins": wins})
                            .reset_index(), on="arrival_min", how="left")
    return table


def bundle_setup(record: dict, folder: Path) -> tuple[SearchSetup, Context]:
    """A published scenario window rebuilt from its .f32 cloud, as scripts/export_scenario_bundles.py wrote it."""
    frames, n = int(record["frames"]), int(record["particles"])
    cloud = np.fromfile(folder / record["cloud"], dtype="<f4").reshape(frames, n, 2)
    m = record["marker"]
    marker = MarkerTrack(np.array(m["t_s"], float), np.array(m["lat"], float),
                         np.array(m["lon"], float))
    mlat, mlon = marker.at(np.arange(frames) * STEP_S)
    lat, lon = offset_position(mlat[:, None], mlon[:, None], cloud[..., 0].astype(float),
                               cloud[..., 1].astype(float))
    start = np.datetime64(record["window_start_utc"], "us")
    times = start + (np.arange(frames) * STEP_S * 1e6).astype("timedelta64[us]")
    window = Ensemble(times, lat, lon % 360.0, np.full(n, float(record["weight"])),
                      np.zeros((frames, n), dtype=bool))
    t = record.get("target")
    target = None if t is None else MarkerTrack(np.array(t["t_s"], float),
                                                np.array(t["lat"], float),
                                                np.array(t["lon"], float))
    datum = (float(record["datum"]["lat"]), float(record["datum"]["lon"]) % 360.0)
    bearing = record.get("drift_bearing_deg")
    setup = SearchSetup(window, marker, target, datum, float(record["arrival_s"]), bearing)
    lkp = (float(record["lkp"]["lat"]), float(record["lkp"]["lon"]))
    return setup, Context(lkp, float(record["arrival_s"]), bearing)


def _cli(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    w = sub.add_parser("windows", help="score searchers over the test windows, the last week of each month")
    w.add_argument("--windows", required=True, help="directory of windows_*.npz files")
    w.add_argument("--arrival-min", type=int, nargs="+", required=True)
    w.add_argument("--searchers", nargs="+", required=True, choices=[PPO, *SEARCHERS])
    w.add_argument("--model", help="model.zip or best_model.zip, for ppo")
    w.add_argument("--out", required=True, help="directory for flights.csv, summary.csv, flights/")
    b = sub.add_parser("bundle", help="add the PPO flight to a published scenario window JSON")
    b.add_argument("--model", required=True)
    b.add_argument("--bundle", nargs="+", required=True, help="window JSONs such as S01/rv_2h.json")
    args = parser.parse_args(argv)

    if args.command == "bundle":
        model, headings = load_policy(args.model)
        for path in map(Path, args.bundle):
            record = json.loads(path.read_text())
            setup, context = bundle_setup(record, path.parent)
            record["flights"][PPO] = fly_window(setup, context, {}, [PPO], model, headings)[PPO]
            path.write_text(json.dumps(record, separators=(",", ":")))
            print(f"{path}: ppo pos {record['flights'][PPO]['python']['pos']:.4f}")
        return

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
