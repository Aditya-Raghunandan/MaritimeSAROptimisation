"""track.py: the drift pipeline, sampling forcing and advancing a position step by step.

    python -m sar.pipeline.track (--constant-current <u> <v> | --forcing-dir <data root>) \\
        --start <iso 8601> --lat <degrees north> --lon <degrees east> \\
        --timestep <seconds> --duration <seconds> [--constant-wind <u> <v>] \\
        [--sigma <m/s^0.5> | --sigma-u <m/s> [--memory-h <h>]] [--seed <int>]
        [--leeway <fraction>] [--every <n>]

--forcing-dir reads the real HYCOM current and ERA5 wind under <data root>/raw
(`sar.pipeline.gridded`, issue #88); --constant-current is a uniform steady field.
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
    DEFAULT_SIGMA_U,
    INTEGRATION_STEP_SECONDS,
    MEMORY_TIME_S,
    as_position,
    calculate_position,
    evolve_velocity_error,
    start_velocity_error,
    velocity_memory,
)
from sar.pipeline.forcing import ConstantForcing, as_particle_axes
from sar.pipeline.gridded import GriddedForcing

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
    velocity_error: np.ndarray | None = None   # (N, 2) m/s, the random velocity (D030)

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
    """Advances an ensemble through a forcing field, one fixed step at a time.

    The random term is either a random walk (sigma, m/s^0.5) or a random velocity with
    memory (sigma_u, m/s, and memory_time_s; vault D030), never both. With neither the run
    is deterministic. The velocity error is carried by `track` and handed to `advance`
    explicitly, so `advance` stays a function of what it is given.
    """

    def __init__(self, forcing, timestep: float, leeway: float = LEEWAY_COEFFICIENT,
                 sigma: float = DEFAULT_SIGMA, seed=None, sigma_u: float = DEFAULT_SIGMA_U,
                 memory_time_s: float = MEMORY_TIME_S):
        # timestep has no default: every error in a run is linear in it, so a run must state it.
        if not np.isfinite(timestep) or timestep <= 0.0:
            raise ValueError(f"timestep must be a positive number of seconds, got {timestep}")
        if sigma < 0.0:
            raise ValueError(f"sigma must not be negative, got {sigma}")
        if not np.isfinite(sigma_u) or sigma_u < 0.0:
            raise ValueError(f"sigma_u must not be negative, got {sigma_u}")
        if sigma > 0.0 and sigma_u > 0.0:
            raise ValueError("choose one random term: sigma (a random walk) or sigma_u (a "
                             "random velocity, D030), not both")
        velocity_memory(timestep, memory_time_s)   # refuses a memory time that is not positive
        self.forcing = forcing
        self.timestep = float(timestep)
        self.leeway = float(leeway)
        self.sigma = float(sigma)
        self.sigma_u = float(sigma_u)
        self.memory_time_s = float(memory_time_s)
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

    @property
    def random_term(self) -> str:
        if self.sigma_u > 0.0:
            return "random velocity"
        return "random walk" if self.sigma > 0.0 else "none"

    def start_velocity_error(self, n: int) -> np.ndarray | None:
        """Each particle's velocity error at t = 0 (D030), or None without one."""
        if self.sigma_u == 0.0:
            return None
        return start_velocity_error((int(n), 2), self.sigma_u, self.rng)

    def evolve_velocity_error(self, velocity_error):
        """The velocity error one step on; None stays None."""
        if velocity_error is None:
            return None
        return evolve_velocity_error(velocity_error, self.timestep, self.sigma_u,
                                     self.memory_time_s, self.rng)

    def advance(self, positions: np.ndarray, time, velocity_error=None):
        """One step: returns the next positions and the current, wind and drift used.

        `velocity_error` (N, 2) m/s is added to the drift for this step (D030). The drift
        returned is the deterministic one, current + leeway x wind.
        """
        current, wind = self.forcing.sample(positions[:, 0], positions[:, 1], time)
        drift = calculate_drift(wind, current, self.leeway)
        moving = drift if velocity_error is None else drift + velocity_error
        moved = calculate_position(positions, moving, self.timestep, self.sigma, self.rng)
        # D016: where the forcing is NaN the particle is beached, frozen in place with its mass.
        beached = ~np.isfinite(drift).all(axis=1)
        moved[beached] = positions[beached]
        return moved, current, wind, drift

    def track(self, start, lat, lon, duration: float) -> Iterator[TrackState]:
        """Every state from t = 0 to t = duration, so n steps yield n + 1 states."""
        start = np.datetime64(start, "us")
        n_steps = self.step_count(duration)
        positions = self.start_positions(lat, lon)
        velocity_error = self.start_velocity_error(positions.shape[0])

        def at(k: int) -> tuple[float, np.datetime64]:
            seconds = k * self.timestep
            return seconds, start + np.timedelta64(round(seconds * _MICROSECONDS), "us")

        for k in range(n_steps):
            seconds, time = at(k)
            moved, current, wind, drift = self.advance(positions, time, velocity_error)
            yield TrackState(k, seconds, time, positions, current, wind, drift, velocity_error)
            positions = moved
            velocity_error = self.evolve_velocity_error(velocity_error)

        seconds, time = at(n_steps)
        yield TrackState(n_steps, seconds, time, positions, velocity_error=velocity_error)

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
                "random_term": self.random_term,
                "sigma": self.sigma,
                "sigma_u": self.sigma_u,
                "memory_time_s": self.memory_time_s,
                "seed": self.seed,
                "forcing": self.forcing.describe()}


def _cli(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--constant-current", nargs=2, type=float,
                        metavar=("U", "V"), help="uniform steady current in m/s")
    source.add_argument("--forcing-dir",
                        help="data root holding raw/hycom_* and raw/era5_*: the real forcing")
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
    noise = parser.add_mutually_exclusive_group()
    noise.add_argument("--sigma", type=float, default=DEFAULT_SIGMA,
                       help=f"a random walk, D009's sigma in m/s^0.5, default {DEFAULT_SIGMA:g}")
    noise.add_argument("--sigma-u", type=float, default=DEFAULT_SIGMA_U,
                       help="a random velocity with memory (D030), m/s per axis")
    parser.add_argument("--memory-h", type=float, default=MEMORY_TIME_S / 3600.0,
                        help=f"the random velocity's memory T_L in hours, default "
                             f"{MEMORY_TIME_S / 3600.0:g}")
    parser.add_argument("--seed", type=int, help="seed for the random draws, for a repeatable run")
    parser.add_argument("--every", type=int, default=1,
                        help="print every Nth state; the first and the last are always printed")
    args = parser.parse_args(argv)

    if args.forcing_dir:
        end = np.datetime64(args.start, "us") + np.timedelta64(round(args.duration * 1e6), "us")
        try:
            forcing = GriddedForcing.from_dir(args.forcing_dir, args.start, end)
        except (ValueError, FileNotFoundError) as error:
            parser.error(str(error))
    else:
        forcing = ConstantForcing(args.constant_current, args.constant_wind or (0.0, 0.0))
    try:
        pipeline = DriftPipeline(forcing, args.timestep, args.leeway, args.sigma, args.seed,
                                 args.sigma_u, args.memory_h * 3600.0)
    except ValueError as error:
        parser.error(str(error))
    run = pipeline.describe(args.start, args.lat, args.lon, args.duration)
    rows = [row
            for state in pipeline.track(args.start, args.lat, args.lon, args.duration)
            if state.step % max(1, args.every) == 0 or state.step == run["steps"]
            for row in state.rows()]
    return {"run": run, "track": rows}


if __name__ == "__main__":
    print(json.dumps(_cli(), indent=2))
