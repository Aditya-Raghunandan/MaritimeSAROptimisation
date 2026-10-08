"""episode.py: one search, flown by any searcher, scored by one referee (issue #47).

Every searcher the paper compares -- the Expanding Square, greedy, a random walk, PPO, a
person on the site -- is scored by this code, so a difference in score can only be the
searcher (R5g; docs/ADR004.md, vault D029). Nothing here imports Gymnasium: the RL
environment wraps an episode, and its reward is what `step` returns.

WHAT AN EPISODE IS. The scenario's particle cloud over the 45-minute window, as 46 frames
60 s apart from the helicopter's arrival; each particle carries a weight, its share of the
probability. Every step the searcher says where to fly for the next 60 s, the helicopter
flies it, and every particle it comes within W/2 of loses (pod x) its weight (`sweep`). The
weight removed, summed, is POS: the probability the search found the target. It is never
renormalised (ADR002).

A STEP IS A SHORT PATH (ADR004 row 1). A searcher gives a heading, flown straight for the
step, or `Waypoints`, timed points inside the step. The Expanding Square needs the second:
its first leg lasts 4 s, and joining positions a minute apart would cut across its first
16 minutes (ADR003 section 2). Either way the step becomes sub-legs, each swept on its own.

DETECTION BY CLOSEST APPROACH (row 2). A Gulf Stream particle moves 108 m a step, more than
W/2 = 92.6 m, so each sub-leg is swept against the particles' motion over it (ADR003 section
3). Their positions at a sub-leg's ends are interpolated linearly between the two frames:
exact for Euler-Maruyama, which moves each particle in a straight line over a step.

EVERYONE FLIES ABOUT THE MARKER (row 3). The helicopter's position is an offset in metres
from the datum marker, which drifts with the current alone (`sar.search.datum`), and it
flies `speed_ms` relative to it, as the Coast Guard's patterns are flown (ADR003 row 7). It
starts at the marker. On the ground it is the marker's position plus the offset. A searcher
that flew over the ground instead would spend ~4.9 km of a 125 km window keeping up with a
1.8 m/s current that the patterns are carried by for free.

WHAT IS MEASURED (row 6). POS, and the probability removed in each step: `removed_per_step`,
which the project calls the DRAIN RATE (the share of the probability cleared per minute;
POS is its sum; docs/benchmark.md). The expected time to detection, sum(t x dm) / sum(dm) with t
the END of each sub-leg in seconds since arrival (late by at most one sub-leg, never early);
the distance flown in the marker frame, which is speed x time for every searcher, so a check
and not a comparison. With a target track (the real buoy) also whether and when it came
within W/2, by the site's closest approach, and how close it came.

CLI. One search, its metrics printed as JSON:

    python -m sar.search.episode --spread-km 2 --particles 10000 --seed 1 \\
        --policy expanding-square [--current 1.8 0] [--first-bearing 90]
    python -m sar.search.episode --policy heading --heading 90 --spread-km 2
    python -m sar.search.episode --policy replay --replay flight.json --spread-km 2
    python -m sar.search.episode --csv scenarios.csv --scenario S01 --forcing-dir DATA \\
        --arrival-h 2 --particles 10000 --policy expanding-square
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sar.search.patterns import PATTERNS, MarkerTrack, Pattern
from sar.search.platform import (
    ON_SCENE_WINDOW_S,
    SEARCH_SPEED_MS,
    STEP_S,
    SWEEP_WIDTH_M,
    expanding_square_spacing_m,
)
from sar.search.sweep import first_contact, relative_m, sweep
from sar.utils.geo import east_north, offset_position, to_display_longitude

# R7a's 45-minute window at D009's 60 s step (ADR002, time axis).
STEPS = int(round(ON_SCENE_WINDOW_S / STEP_S))

_TIME_TOLERANCE_S = 1e-6
_FRAME_STEP = np.timedelta64(int(STEP_S * 1_000_000), "us")


@dataclass(frozen=True)
class Waypoints:
    """Where the helicopter is during one step, in metres east and north of the marker.

    t_s are seconds into the step, strictly increasing, the last one the end of the step.
    The start of the step, where the helicopter already is, is not repeated.
    """

    t_s: np.ndarray
    east_m: np.ndarray
    north_m: np.ndarray

    def __post_init__(self):
        arrays = [np.atleast_1d(np.asarray(getattr(self, name), dtype=float))
                  for name in ("t_s", "east_m", "north_m")]
        if any(a.ndim != 1 for a in arrays) or len({a.size for a in arrays}) != 1:
            raise ValueError("waypoint times, east and north must be 1-D and the same length")
        if not all(np.all(np.isfinite(a)) for a in arrays):
            raise ValueError("waypoints must be finite")
        t = arrays[0]
        if t[0] <= 0.0 or np.any(np.diff(t) <= 0.0):
            raise ValueError("waypoint times must be strictly increasing and after the "
                             "start of the step")
        for name, a in zip(("t_s", "east_m", "north_m"), arrays):
            object.__setattr__(self, name, a)


def _read_only(array: np.ndarray) -> np.ndarray:
    view = array.view()
    view.flags.writeable = False
    return view


class SearchEpisode:
    """One search over one scenario's cloud: the referee every searcher is scored by."""

    def __init__(self, window, marker: MarkerTrack, speed_ms: float = SEARCH_SPEED_MS,
                 sweep_width_m: float = SWEEP_WIDTH_M, pod: float = 1.0, steps: int = STEPS,
                 target: MarkerTrack | None = None):
        if isinstance(steps, bool) or not isinstance(steps, (int, np.integer)) or steps < 1:
            raise ValueError(f"steps must be a positive integer, got {steps!r}")
        times = np.asarray(window.times)
        if times.shape != (steps + 1,):
            raise ValueError(f"an episode of {steps} steps needs {steps + 1} frames, "
                             f"got {times.shape[0]}")
        if np.any(np.diff(times.astype("datetime64[us]")) != _FRAME_STEP):
            raise ValueError(f"the window's frames must be {STEP_S:g} s apart")
        speed_ms, sweep_width_m, pod = float(speed_ms), float(sweep_width_m), float(pod)
        if not np.isfinite(speed_ms) or speed_ms < 0.0:
            raise ValueError(f"speed_ms must be zero or positive, got {speed_ms}")
        if not np.isfinite(sweep_width_m) or sweep_width_m <= 0.0:
            raise ValueError(f"sweep_width_m must be positive, got {sweep_width_m}")
        if not 0.0 <= pod <= 1.0:
            raise ValueError(f"pod must lie within 0 to 1, got {pod}")
        duration = steps * STEP_S
        if marker.t_s[0] > _TIME_TOLERANCE_S or marker.t_s[-1] < duration - _TIME_TOLERANCE_S:
            raise ValueError(f"the marker track must cover the episode, 0 to {duration:g} s")

        self.window = window
        self.marker = marker
        self.target = target
        self.speed_ms = speed_ms
        self.sweep_width_m = sweep_width_m
        self.pod = pod
        self.steps = int(steps)
        self.k = 0
        self._weight = np.array(window.weight, dtype=float)
        self._initial = float(np.sum(self._weight))
        self._offset = (0.0, 0.0)
        lat, lon = self.position
        self._track = [(0.0, lat, lon)]
        self._removed: list[float] = []
        self._legs: list[tuple[float, float]] = []   # (end of sub-leg, s since arrival; mass)
        self._distance = 0.0
        self._found_s = None
        self._closest = (np.inf, None)
        self._target_gaps = 0

    # -- the state a searcher reads ----------------------------------------------------

    @property
    def duration_s(self) -> float:
        return self.steps * STEP_S

    @property
    def t_s(self) -> float:
        """Seconds since arrival, now."""
        return self.k * STEP_S

    @property
    def done(self) -> bool:
        return self.k >= self.steps

    @property
    def offset(self) -> tuple[float, float]:
        """(east, north) metres from the marker, now."""
        return self._offset

    @property
    def marker_position(self) -> tuple[float, float]:
        lat, lon = self.marker.at(self.t_s)
        return float(lat), float(lon)

    @property
    def position(self) -> tuple[float, float]:
        """The helicopter's (lat, lon) on the ground, now."""
        lat, lon = offset_position(*self.marker_position, *self._offset)
        return float(lat), float(lon)

    def particles(self) -> tuple[np.ndarray, np.ndarray]:
        """Every particle's (lat, lon) now, read-only."""
        k = min(self.k, self.steps)
        return _read_only(self.window.lat[k]), _read_only(self.window.lon[k])

    @property
    def weight(self) -> np.ndarray:
        """What is left of each particle's probability, read-only."""
        return _read_only(self._weight)

    @property
    def remaining(self) -> float:
        return float(np.sum(self._weight))

    # -- flying ------------------------------------------------------------------------

    def step(self, heading_deg) -> float:
        """Fly straight along a heading, relative to the marker, for one step."""
        if heading_deg is None or not np.isfinite(float(heading_deg)):
            raise ValueError(f"a heading must be a finite number of degrees, got {heading_deg!r}")
        de, dn = east_north(float(heading_deg), self.speed_ms * STEP_S)
        return self.fly(Waypoints([STEP_S], [self._offset[0] + float(de)],
                                  [self._offset[1] + float(dn)]))

    def fly(self, waypoints: Waypoints) -> float:
        """Fly one step through `waypoints`; returns the probability it removed."""
        if self.done:
            raise RuntimeError(f"the episode is over: all {self.steps} steps have been flown")
        if not isinstance(waypoints, Waypoints):
            raise TypeError(f"fly takes Waypoints, got {type(waypoints).__name__}")
        if abs(waypoints.t_s[-1] - STEP_S) > _TIME_TOLERANCE_S:
            raise ValueError(f"waypoints must end at the end of the step, {STEP_S:g} s, "
                             f"not {waypoints.t_s[-1]:g} s")
        t = np.concatenate(([0.0], waypoints.t_s[:-1], [STEP_S]))
        east = np.concatenate(([self._offset[0]], waypoints.east_m))
        north = np.concatenate(([self._offset[1]], waypoints.north_m))
        legs = np.hypot(np.diff(east), np.diff(north))
        allowed = self.speed_ms * np.diff(t)
        if np.any(legs > allowed * (1.0 + 1e-9) + 1e-6):
            fastest = float(np.max(legs / np.maximum(np.diff(t), 1e-12)))
            raise ValueError(f"the path flies {fastest:.2f} m/s, faster than the "
                             f"helicopter's {self.speed_ms:.2f} m/s about the marker")

        t0 = self.t_s
        mlat, mlon = self.marker.at(t0 + t)
        glat, glon = offset_position(mlat, mlon, east, north)
        k = self.k
        lat0, lon0 = self.window.lat[k], self.window.lon[k]
        dlat = self.window.lat[k + 1] - lat0
        dlon = (self.window.lon[k + 1] - lon0 + 180.0) % 360.0 - 180.0

        removed_step = 0.0
        a_lat, a_lon = lat0, lon0
        for i in range(t.size - 1):
            f = t[i + 1] / STEP_S
            b_lat, b_lon = lat0 + f * dlat, lon0 + f * dlon
            self._weight, removed = sweep(a_lat, a_lon, self._weight, (glat[i], glon[i]),
                                          (glat[i + 1], glon[i + 1]), self.sweep_width_m,
                                          self.pod, lat_end=b_lat, lon_end=b_lon)
            self._legs.append((t0 + t[i + 1], removed))
            removed_step += removed
            if self.target is not None:
                self._score_target(t0 + t[i], t0 + t[i + 1], (glat[i], glon[i]),
                                   (glat[i + 1], glon[i + 1]))
            a_lat, a_lon = b_lat, b_lon

        self._distance += float(legs.sum())
        self._offset = (float(east[-1]), float(north[-1]))
        self._track.extend(zip((t0 + t[1:]).tolist(), glat[1:].tolist(), glon[1:].tolist()))
        self._removed.append(removed_step)
        self.k += 1
        return removed_step

    def _score_target(self, ta: float, tb: float, ha, hb) -> None:
        """The site's closest-approach test against the one real target, for one sub-leg."""
        lo, hi = self.target.t_s[0], self.target.t_s[-1]
        if ta < lo - _TIME_TOLERANCE_S or tb > hi + _TIME_TOLERANCE_S:
            self._target_gaps += 1
            return
        tlat, tlon = self.target.at(np.clip([ta, tb], lo, hi))
        r0 = relative_m(ha[0], ha[1], tlat[0], tlon[0])
        r1 = relative_m(hb[0], hb[1], tlat[1], tlon[1])
        contact, closest, tau = first_contact(float(r0[0]), float(r0[1]), float(r1[0]),
                                              float(r1[1]), self.sweep_width_m / 2.0)
        if closest < self._closest[0]:
            self._closest = (closest, ta + tau * (tb - ta))
        if contact is not None and self._found_s is None:
            self._found_s = ta + contact * (tb - ta)

    # -- the result --------------------------------------------------------------------

    def metrics(self) -> dict:
        """POS, expected time to detection and distance, so far; and the target, if any."""
        pos = float(np.sum(self._removed))
        found = sum(m for _, m in self._legs)
        ttd = (sum(t * m for t, m in self._legs) / found) if found > 0.0 else None
        out = {"steps": self.k,
               "elapsed_s": self.t_s,
               "pos": pos,
               "initial_mass": self._initial,
               "remaining": self.remaining,
               "expected_ttd_s": ttd,
               "distance_m": self._distance,
               "removed_per_step": [float(m) for m in self._removed],
               "particles": int(self._weight.size),
               "speed_ms": self.speed_ms,
               "sweep_width_m": self.sweep_width_m,
               "pod": self.pod}
        if self.target is not None:
            closest, when = self._closest
            out["target"] = {"found": self._found_s is not None,
                             "found_s": self._found_s,
                             "closest_m": None if not np.isfinite(closest) else closest,
                             "closest_s": when,
                             "gaps": self._target_gaps}
        return out

    def track(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The flight on the ground, every sub-leg's end: (t since arrival, lat, lon)."""
        t, lat, lon = zip(*self._track)
        return np.array(t), np.array(lat), np.array(lon)


def run(policy, episode: SearchEpisode, decide_every: int = 1) -> dict:
    """Fly a searcher to the end of the episode and return its metrics.

    A policy is any callable taking the episode and returning a heading or `Waypoints`.
    With decide_every = n it is asked every n steps and its heading held in between, each
    step still swept on its own frames. Waypoints cover one step, so they need n = 1.
    """
    if isinstance(decide_every, bool) or not isinstance(decide_every, (int, np.integer)) \
            or decide_every < 1:
        raise ValueError(f"decide_every must be a positive integer, got {decide_every!r}")
    while not episode.done:
        action = policy(episode)
        if isinstance(action, Waypoints):
            if decide_every != 1:
                raise ValueError("waypoints cover one step; decide_every applies to headings")
            episode.fly(action)
        else:
            for _ in range(min(decide_every, episode.steps - episode.k)):
                episode.step(action)
    return episode.metrics()


def heading_policy(heading_deg: float):
    """One fixed heading, every step."""
    heading_deg = float(heading_deg)
    return lambda episode: heading_deg


def pattern_policy(pattern: Pattern):
    """A Coast Guard pattern (`sar.search.patterns`) as a searcher: its waypoints, step by step.

    The pattern is already in metres about the marker, which is the episode's frame. A
    pattern that ends before the episode does (a finished Parallel Track) is refused rather
    than left to hover.
    """
    def policy(episode: SearchEpisode) -> Waypoints:
        if pattern.duration_s < episode.duration_s - _TIME_TOLERANCE_S:
            raise ValueError(f"the {pattern.kind} ends at {pattern.duration_s:g} s, before "
                             f"the episode's {episode.duration_s:g} s")
        t0, t1 = episode.t_s, episode.t_s + STEP_S
        inside = pattern.t_s[(pattern.t_s > t0 + _TIME_TOLERANCE_S)
                             & (pattern.t_s < t1 - _TIME_TOLERANCE_S)]
        times = np.append(inside, t1)
        east, north = pattern.offset_at(times)
        return Waypoints(times - t0, east, north)

    return policy


def replay_policy(record):
    """A recorded flight as a searcher: headings, never positions (ADR004 section 1).

    `record` is a dict, or a path to a JSON file holding one, in either form:

        {"headings_deg": [h0, h1, ...]}               one heading per step
        {"t_s": [0, 12.5, ...], "heading_deg": [...]}  each heading from that time on,
                                                       seconds since arrival, from 0

    A ground path is not replayed: it already holds the current, and flying it about the
    marker would add the current twice.
    """
    if isinstance(record, (str, Path)):
        record = json.loads(Path(record).read_text())
    if "headings_deg" in record:
        headings = [float(h) for h in record["headings_deg"]]

        def per_step(episode: SearchEpisode) -> float:
            if episode.k >= len(headings):
                raise ValueError(f"the record has {len(headings)} headings, the episode "
                                 f"{episode.steps} steps")
            return headings[episode.k]

        return per_step

    if "t_s" not in record or "heading_deg" not in record:
        raise ValueError("a flight record holds 'headings_deg', or 't_s' and 'heading_deg'")
    times = np.asarray(record["t_s"], dtype=float)
    headings = np.asarray(record["heading_deg"], dtype=float)
    if times.ndim != 1 or times.shape != headings.shape or times.size == 0:
        raise ValueError("'t_s' and 'heading_deg' must be 1-D and the same length")
    if abs(times[0]) > _TIME_TOLERANCE_S or np.any(np.diff(times) <= 0.0):
        raise ValueError("'t_s' must start at 0 and increase strictly")

    def timed(episode: SearchEpisode) -> Waypoints:
        t0, t1 = episode.t_s, episode.t_s + STEP_S
        cuts = times[(times > t0 + _TIME_TOLERANCE_S) & (times < t1 - _TIME_TOLERANCE_S)]
        ends = np.append(cuts, t1)
        starts = np.concatenate(([t0], cuts))
        held = headings[np.searchsorted(times, starts + _TIME_TOLERANCE_S, side="right") - 1]
        de, dn = east_north(held, episode.speed_ms * (ends - starts))
        east = episode.offset[0] + np.cumsum(de)
        north = episode.offset[1] + np.cumsum(dn)
        return Waypoints(ends - t0, east, north)

    return timed


def _policy_from(args, first_bearing: float):
    if args.policy == "heading":
        return heading_policy(args.heading)
    if args.policy == "replay":
        if not args.replay:
            raise ValueError("--policy replay needs --replay FILE")
        return replay_policy(args.replay)
    builder = PATTERNS[args.policy]
    if args.policy == "expanding-square":
        pattern = builder(expanding_square_spacing_m(), first_bearing)
    else:
        pattern = builder(first_bearing_deg=first_bearing)
    return pattern_policy(pattern)


def _cli(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--policy", default="expanding-square",
                        choices=["expanding-square", "sector", "heading", "replay"])
    parser.add_argument("--first-bearing", type=float, default=None,
                        help="a pattern's first leg, degrees true; default along the drift")
    parser.add_argument("--heading", type=float, default=0.0, help="for --policy heading")
    parser.add_argument("--replay", help="a flight record, JSON, for --policy replay")
    parser.add_argument("--decide-every", type=int, default=1,
                        help="steps between a heading searcher's decisions, default 1")
    parser.add_argument("--pod", type=float, default=1.0, help="default 1, definite range")
    parser.add_argument("--particles", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=None)
    synthetic = parser.add_argument_group("a synthetic cloud (the default)")
    synthetic.add_argument("--lat", type=float, default=26.5)
    synthetic.add_argument("--lon", type=float, default=-79.0)
    synthetic.add_argument("--spread-km", type=float, default=2.0)
    synthetic.add_argument("--current", nargs=2, type=float, default=(0.0, 0.0),
                           metavar=("U", "V"), help="carries the cloud and the marker, m/s")
    real = parser.add_argument_group("a scenario on the real ocean")
    real.add_argument("--csv", help="the scenario table (scripts/pick_scenario_buoys.py)")
    real.add_argument("--scenario", help="its row, such as S01")
    real.add_argument("--forcing-dir", help="data root holding raw/hycom_* and raw/era5_*")
    real.add_argument("--arrival-h", type=float, default=2.0,
                      help="hours from the call to arrival, whole minutes, default 2")
    args = parser.parse_args(argv)

    from sar.search import scenario as scenarios

    try:
        if args.csv:
            if not (args.scenario and args.forcing_dir):
                raise ValueError("a scenario run needs --csv, --scenario and --forcing-dir")
            row = scenarios.scenario_row(args.csv, args.scenario)
            setup = scenarios.scenario_search(row, args.forcing_dir, args.arrival_h * 3600.0,
                                              args.particles)
            bearing = setup.drift_bearing_deg
            where = {"scenario": args.scenario, "arrival_s": setup.arrival_s}
        else:
            rng = np.random.default_rng(args.seed)
            setup = scenarios.steady_search((args.lat, args.lon), args.spread_km,
                                            args.particles, rng, args.current)
            bearing = setup.drift_bearing_deg
            where = {"synthetic": {"spread_km": args.spread_km, "current_ms": list(args.current)}}
        first = bearing if args.first_bearing is None else args.first_bearing
        first = 0.0 if first is None else first
        policy = _policy_from(args, first)
        episode = SearchEpisode(setup.window, setup.marker, pod=args.pod, target=setup.target)
        metrics = run(policy, episode, args.decide_every)
    except (ValueError, FileNotFoundError, KeyError) as error:
        parser.error(str(error))
    return {**where, "policy": args.policy, "first_bearing_deg": first,
            "datum": {"lat": setup.datum[0], "lon": float(to_display_longitude(setup.datum[1]))},
            **metrics}


if __name__ == "__main__":
    print(json.dumps(_cli(), indent=2))
