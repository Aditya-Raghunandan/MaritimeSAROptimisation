"""compare_cloud_paths.py: does HOW the particles move inside the window change POS? (#47)

sigma (D028) is matched so the cloud has the right SPREAD at 4 h. It says nothing about the
PATH each particle takes to get there, and the engine's path is a random walk: a fresh push
of sigma x sqrt(60 s) = 204 m every minute, more than twice the 92.6 m half-strip. A swept
lane fills back in. A real object's departure from the forecast grows as t^1.7-1.9 on the
drifters (D028), closer to a steady wrong velocity than to a random walk.

This flies the Expanding Square and the Sector Search over three synthetic clouds with the
same spread at arrival (sigma sqrt(t), t = 2 h) and, for the moving two, at the end of the
window:

    frozen        every particle stays where it was at arrival
    smooth        every particle moves in a straight line at its own steady velocity
    random walk   the engine's model: a fresh Gaussian push every minute

and prints the mean POS over a few seeds. Marker fixed, cloud centred on it, no current.

    python scripts/compare_cloud_paths.py [--particles 20000] [--seeds 3] [--arrival-h 2]
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from sar.model.position import CALIBRATED_SIGMA
from sar.pipeline.ensemble import Ensemble
from sar.search.episode import STEPS, SearchEpisode, pattern_policy, run
from sar.search.patterns import MarkerTrack, expanding_square, sector_search
from sar.search.platform import STEP_S, SWEEP_WIDTH_M
from sar.utils.geo import offset_position

LAT, LON = 26.5, 281.0
KINDS = ("frozen", "smooth", "random walk")


def cloud(kind: str, n: int, sigma: float, arrival_s: float, rng) -> Ensemble:
    """A cloud of spread sigma sqrt(arrival) at arrival, moving through the window as `kind`."""
    t = np.arange(STEPS + 1) * STEP_S
    start = rng.standard_normal((n, 2)) * sigma * np.sqrt(arrival_s)
    if kind == "random walk":
        kicks = rng.standard_normal((STEPS, n, 2)) * sigma * np.sqrt(STEP_S)
        path = start[None] + np.concatenate([np.zeros((1, n, 2)), np.cumsum(kicks, axis=0)])
    elif kind == "smooth":
        # Var(end) = sigma^2 (arrival + window) for both moving kinds.
        velocity = rng.standard_normal((n, 2)) * sigma / np.sqrt(t[-1])
        path = start[None] + velocity[None] * t[:, None, None]
    else:
        path = np.repeat(start[None], t.size, axis=0)
    lat, lon = offset_position(LAT, LON, path[..., 0], path[..., 1])
    times = np.datetime64("2019-06-01T00:00", "us") + (t * 1e6).astype("timedelta64[us]")
    return Ensemble(times, lat, lon, np.full(n, 1.0 / n), np.zeros(lat.shape, dtype=bool))


def compare(n: int, seeds: int, arrival_s: float, sigma: float) -> dict:
    marker = MarkerTrack.fixed(LAT, LON, STEPS * STEP_S)
    patterns = {"expanding_square": expanding_square(SWEEP_WIDTH_M, 0.0),
                "sector_search": sector_search(first_bearing_deg=0.0)}
    out = {}
    for kind in KINDS:
        out[kind] = {}
        for name, pattern in patterns.items():
            pos = [run(pattern_policy(pattern),
                       SearchEpisode(cloud(kind, n, sigma, arrival_s,
                                           np.random.default_rng(seed)), marker))["pos"]
                   for seed in range(1, seeds + 1)]
            out[kind][name] = {"mean_pos": float(np.mean(pos)), "pos": pos}
    return {"particles": n, "seeds": seeds, "arrival_s": arrival_s, "sigma": sigma,
            "spread_at_arrival_m": sigma * np.sqrt(arrival_s),
            "spread_at_end_m": sigma * np.sqrt(arrival_s + STEPS * STEP_S),
            "results": out}


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--particles", type=int, default=20_000)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--arrival-h", type=float, default=2.0)
    parser.add_argument("--sigma", type=float, default=CALIBRATED_SIGMA)
    args = parser.parse_args(argv)
    result = compare(args.particles, args.seeds, args.arrival_h * 3600.0, args.sigma)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
