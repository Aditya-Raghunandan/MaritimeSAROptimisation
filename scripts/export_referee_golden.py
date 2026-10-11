"""export_referee_golden.py: writes frontend/src/fixtures/referee_golden.json (Stage 3, D031).

The referee (`sar.search.episode`) is the paper's. The site replays the paper's flights and
scores a player's with a JavaScript copy of it (frontend/src/referee.js). Two copies of the
same scorer drift apart silently, so Python is the reference, as it is for the resultant
vector and the patterns: this flies every benchmark searcher over a small real cloud and
records the inputs and the referee's answers, and frontend/tests/referee.test.js feeds the
JavaScript the same inputs and asserts the same answers.

What it records:

    window     60 particles x 46 frames, float64, from the engine under constant forcing with
               the random velocity, so every particle moves differently and the sub-leg
               interpolation is exercised; longitude in the store convention
    marker     the datum marker's track
    target     the real buoy's track, so found / closest approach are checked too
    flights    every searcher (four Coast Guard patterns, greedy, random) as the Waypoints
               it flew each step, with the heading at each, flown by a helicopter that
               turns at TURN_RATE_DEG_S (D032), and the referee's removed_per_step, pos,
               expected time to detection and target result
    instant    the same, flown by the pre-D032 helicopter that turns at once: the site
               still replays format 1 bundles, and the instant referee must stay the same
    player     a scripted "player": a heading command every 5 s, flown by `replay_policy`
               at the turn rate, as the site's PlayerFlight flies and records one
               (frontend/src/playback.js), so a person's flight is scored by the paper's
               referee and the browser's alike
    encoded    one frame as the scenario bundles store it (float32 metres from the marker)
               and the positions Python rebuilds from it, for the bundle decoder

    python scripts/export_referee_golden.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sar.pipeline.forcing import ConstantForcing
from sar.search.benchmark import NOISE, SEARCHERS, recorded, searcher
from sar.search.episode import replay_policy, run
from sar.search.platform import SEARCH_SPEED_MS, SWEEP_WIDTH_M, TURN_RATE_DEG_S
from sar.search.scenario import arrival_heading, scenario_search, search_episode
from sar.search.sweep import relative_m
from sar.utils.geo import M_PER_DEG_LAT, offset_position

OUT = Path("frontend/src/fixtures/referee_golden.json")
FORCING = ConstantForcing(current=(1.0, 0.5), wind=(5.0, 2.0))
PARTICLES, ARRIVAL_H = 60, 1.0


def row() -> pd.Series:
    base = {"scenario": "G01", "start": "2019-06-01T06:00", "lat": 26.5, "lon": -79.0,
            "seed": 11}
    for h in range(1, 7):
        base[f"lat_{h}h"], base[f"lon_{h}h"] = 26.5 + 0.0046 * h, -79.0 + 0.0098 * h
    return pd.Series(base)


def floats(a) -> list:
    return [float(v) for v in np.ravel(a)]


def build() -> dict:
    r = row()
    setup = scenario_search(r, FORCING, round(ARRIVAL_H * 3600), PARTICLES, **NOISE["rv"])
    w = setup.window

    def fly_all(rate):
        flights = {}
        for name in SEARCHERS:
            policy, _, _ = searcher(name, setup, r, ARRIVAL_H)
            flying, steps = recorded(policy)
            m = run(flying, search_episode(setup, rate))
            flights[name] = {
                "steps": [{"t_s": floats(s.t_s), "east_m": floats(s.east_m),
                           "north_m": floats(s.north_m),
                           "heading_deg": None if s.heading_deg is None else floats(s.heading_deg)}
                          for s in steps],
                "removed_per_step": m["removed_per_step"], "pos": m["pos"],
                "expected_ttd_s": m["expected_ttd_s"], "target": m["target"]}
        return flights

    flights = fly_all(TURN_RATE_DEG_S)
    instant = fly_all(float("inf"))

    # A player who circles out from the marker, asking for a new heading every 5 s, with a
    # hard reversal at 10 min so the turn is exercised as well as the gentle drift.
    t = np.arange(0.0, 45 * 60.0, 5.0)
    heading = (90.0 + 2.5 * t / 5.0) % 360.0
    heading[(t >= 600.0) & (t < 660.0)] = (heading[(t >= 600.0) & (t < 660.0)] + 180.0) % 360.0
    record = {"t_s": floats(t), "heading_deg": floats(heading),
              "turn_rate_deg_s": TURN_RATE_DEG_S}
    m = run(replay_policy(record), search_episode(setup))
    player = {"record": record, "removed_per_step": m["removed_per_step"], "pos": m["pos"],
              "target": m["target"]}

    k = 30
    mlat, mlon = setup.marker.at(np.arange(w.lat.shape[0]) * 60.0)
    east, north = relative_m(w.lat[k], w.lon[k], mlat[k], mlon[k])
    e32, n32 = east.astype("<f4"), north.astype("<f4")
    lat, lon = offset_position(mlat[k], mlon[k], e32.astype(float), n32.astype(float))
    return {
        "about": "Written by scripts/export_referee_golden.py. Do not edit by hand.",
        "constants": {"speed_ms": SEARCH_SPEED_MS, "sweep_width_m": SWEEP_WIDTH_M,
                      "m_per_deg_lat": M_PER_DEG_LAT, "step_s": 60.0, "steps": 45,
                      "turn_rate_deg_s": TURN_RATE_DEG_S,
                      "arrival_heading_deg": arrival_heading(setup)},
        "window": {"frames": int(w.lat.shape[0]), "particles": PARTICLES,
                   "lat": floats(w.lat), "lon": floats(w.lon), "weight": floats(w.weight)},
        "marker": {"t_s": floats(setup.marker.t_s), "lat": floats(setup.marker.lat),
                   "lon": floats(setup.marker.lon)},
        "target": {"t_s": floats(setup.target.t_s), "lat": floats(setup.target.lat),
                   "lon": floats(setup.target.lon)},
        "flights": flights,
        "instant": instant,
        "player": player,
        "encoded": {"frame": k, "marker_lat": float(mlat[k]), "marker_lon": float(mlon[k]),
                    "east_f32": floats(e32), "north_f32": floats(n32),
                    "lat": floats(lat), "lon": floats(lon)},
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", default=str(OUT))
    args = p.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = build()
    out.write_text(json.dumps(payload), encoding="utf-8")
    print(f"wrote {out}: {len(payload['flights'])} flights, "
          + ", ".join(f"{k} {v['pos']:.4f}" for k, v in payload["flights"].items()))


if __name__ == "__main__":
    main()
