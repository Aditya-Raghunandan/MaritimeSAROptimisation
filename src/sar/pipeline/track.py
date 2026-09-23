"""track.py: the drift pipeline, sampling forcing and advancing a position step by step.

    python -m sar.pipeline.track --constant-current <u> <v> --start <iso 8601> \\
        --lat <degrees north> --lon <degrees east> --timestep <seconds> \\
        --duration <seconds> [--constant-wind <u> <v>] [--sigma <m/s^0.5>] \\
        [--seed <int>] [--leeway <fraction>] [--every <n>]
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from sar.model.drift import LEEWAY_COEFFICIENT, calculate_drift
from sar.model.position import (
    DEFAULT_SIGMA,
    INTEGRATION_STEP_SECONDS,
    as_position,
    calculate_position,
)
from sar.pipeline.forcing import ConstantForcing, as_particle_axes

_MICROSECONDS = 1_000_000

# Absorbs the float division in step_count only, not a genuinely ragged duration.
_STEP_TOLERANCE = 1e-9


@dataclass(frozen=True)
class TrackState:
    """The ensemble at one instant, with the forcing belonging to the step that leaves it."""

    step: int
    seconds: float
    time: np.datetime64
    positions: np.ndarray                 # (N, 2), [lat, lon] degrees, longitude 0 to 360
    current: np.ndarray | None = None     # (N, 2), [u, v] m/s; None on the final state
    wind: np.ndarray | None = None
    drift: np.ndarray | None = None

    def rows(self) -> list[dict]:
        """One flat, JSON-serialisable record per particle."""
        out = []
        for k in range(self.positions.shape[0]):
            row = {"step": self.step, "seconds": self.seconds, "time": str(self.time),
                   "particle": k,
                   "lat": float(self.positions[k, 0]), "lon": float(self.positions[k, 1])}
            for name, field in (("current", self.current), ("wind", self.wind),
                                ("drift", self.drift)):
                row[f"{name}_u"] = float(field[k, 0]) if field is not None else None
                row[f"{name}_v"] = float(field[k, 1]) if field is not None else None
            out.append(row)
        return out


class DriftPipeline:
    """Advances an ensemble through a forcing field, one fixed step at a time."""

    def __init__(self, forcing, timestep: float, leeway: float = LEEWAY_COEFFICIENT,
                 sigma: float = DEFAULT_SIGMA, seed=None):
        # timestep has no default: every error in a run is linear in it, so a run must state it.
        if not np.isfinite(timestep) or timestep <= 0.0:
            raise ValueError(f"timestep must be a positive number of seconds, got {timestep}")
        if sigma < 0.0:
            raise ValueError(f"sigma must not be negative, got {sigma}")
        self.forcing = forcing
        self.timestep = float(timestep)
        self.leeway = float(leeway)
        self.sigma = float(sigma)
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def step_count(self, duration: float) -> int:
        """How many steps a duration is, refusing one that does not divide by the step."""
        duration = float(duration)
        if not np.isfinite(duration) or duration < 0.0:
            raise ValueError(f"duration must be a non-negative number of seconds, got {duration}")

        steps = duration / self.timestep
        n = round(steps)
        if abs(steps - n) > _STEP_TOLERANCE * max(1.0, abs(steps)):
            raise ValueError(
                f"a duration of {duration:g} s is {steps:.6f} steps of {self.timestep:g} s, "
                "which is not a whole number: choose a duration that divides by the step"
            )
        return n

    def start_positions(self, lat, lon) -> np.ndarray:
        """The starting ensemble as an (N, 2) array of [lat, lon]; scalars give one particle."""
        lats, lons = as_particle_axes(lat, lon)
        return as_position(np.column_stack((lats, lons)))

    def advance(self, positions: np.ndarray, time):
        """One step: returns the next positions and the current, wind and drift used."""
        current, wind = self.forcing.sample(positions[:, 0], positions[:, 1], time)
        drift = calculate_drift(wind, current, self.leeway)
        moved = calculate_position(positions, drift, self.timestep, self.sigma, self.rng)
        return moved, current, wind, drift

    def track(self, start, lat, lon, duration: float) -> Iterator[TrackState]:
        """Every state from t = 0 to t = duration, so n steps yield n + 1 states."""
        start = np.datetime64(start, "us")
        n_steps = self.step_count(duration)
        positions = self.start_positions(lat, lon)

        def at(k: int) -> tuple[float, np.datetime64]:
            seconds = k * self.timestep
            return seconds, start + np.timedelta64(round(seconds * _MICROSECONDS), "us")

        for k in range(n_steps):
            seconds, time = at(k)
            moved, current, wind, drift = self.advance(positions, time)
            yield TrackState(k, seconds, time, positions, current, wind, drift)
            positions = moved

        seconds, time = at(n_steps)
        yield TrackState(n_steps, seconds, time, positions)

    def describe(self, start, lat, lon, duration: float) -> dict:
        """What the run was asked to do, for the header of an output file or a log."""
        positions = self.start_positions(lat, lon)
        return {"start": str(np.datetime64(start, "us")),
                "start_lat": positions[:, 0].tolist(),
                "start_lon": positions[:, 1].tolist(),
                "duration_s": float(duration),
                "timestep_s": self.timestep,
                "steps": self.step_count(duration),
                "particles": int(positions.shape[0]),
                "leeway": self.leeway,
                "sigma": self.sigma,
                "seed": self.seed,
                "forcing": self.forcing.describe()}


def _cli(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--constant-current", nargs=2, type=float, required=True,
                        metavar=("U", "V"), help="uniform steady current in m/s")
    parser.add_argument("--constant-wind", nargs=2, type=float, metavar=("U", "V"),
                        help="uniform steady 10 m wind in m/s, default calm")
    parser.add_argument("--start", required=True, help="start time, ISO 8601")
    parser.add_argument("--lat", type=float, required=True, help="start latitude, degrees north")
    parser.add_argument("--lon", type=float, required=True, help="start longitude, degrees east")
    parser.add_argument("--duration", type=float, required=True, help="how long to track, seconds")
    parser.add_argument("--timestep", type=float, required=True,
                        help=f"step in seconds, required; D009 fixes {INTEGRATION_STEP_SECONDS:g}")
    parser.add_argument("--leeway", type=float, default=LEEWAY_COEFFICIENT,
                        help=f"leeway coefficient, default {LEEWAY_COEFFICIENT} (D002)")
    parser.add_argument("--sigma", type=float, default=DEFAULT_SIGMA,
                        help=f"D009's sigma in m/s^0.5, default {DEFAULT_SIGMA:g}")
    parser.add_argument("--seed", type=int, help="seed for the sigma draws, for a repeatable run")
    parser.add_argument("--every", type=int, default=1,
                        help="print every Nth state; the first and the last are always printed")
    args = parser.parse_args(argv)

    forcing = ConstantForcing(args.constant_current, args.constant_wind or (0.0, 0.0))
    pipeline = DriftPipeline(forcing, args.timestep, args.leeway, args.sigma, args.seed)
    run = pipeline.describe(args.start, args.lat, args.lon, args.duration)
    rows = [row
            for state in pipeline.track(args.start, args.lat, args.lon, args.duration)
            if state.step % max(1, args.every) == 0 or state.step == run["steps"]
            for row in state.rows()]
    return {"run": run, "track": rows}


if __name__ == "__main__":
    print(json.dumps(_cli(), indent=2))
