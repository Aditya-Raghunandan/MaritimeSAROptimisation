"""position.py: one Euler-Maruyama step of D009, x_{n+1} = x_n + v_d dt + sigma sqrt(dt) Z.

Two random terms live here, and a run uses one of them (D030):

    random walk      a fresh push of sigma sqrt(dt) every step (D009, D028). The cloud's
                     spread grows as sqrt(t).
    random velocity  each particle carries a velocity error u that forgets itself over T_L,
                     a first-order Markov ("random flight") process (Taylor 1921; Griffa
                     1996). x moves by (v_d + u) dt, and u <- a u + sqrt(1 - a^2) sigma_u Z
                     with a = exp(-dt / T_L). The spread grows as t for hours, then as
                     sqrt(t).

The dev drifters say the second (vault D030): their spread at 1-48 h fits it to 2 %, where
one random-walk sigma is off by 50 %, and sigma_u matches HYCOM's measured velocity error.
The model stays three terms (D002): u is eta, with memory.

HOW BIG, BY WATER (vault D033, docs/ADR007.md). One pooled sigma_u made the cloud too wide in
quiet water and far too narrow in the Gulf Stream (vault L23, L35). So sigma_u is sized by
the model's own current at the start, the one thing a forecaster knows when the call comes
in: sigma_u = a + b x speed, capped, with the person's crosswind slide added in quadrature
(`sigma_u_for_current`). sigma_u may be one value or one per particle.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from sar.model.drift import as_vector
from sar.utils.geo import EARTH_RADIUS_M  # the project's one Earth radius

# D009 (vault decision, Aditya, 2026-09-03): one minute per integration step.
INTEGRATION_STEP_SECONDS = 60.0

DEGREES_PER_RADIAN = 180.0 / np.pi

# A project limit, not a published one: 1 / cos(lat) is unusable nearer the pole than this.
POLAR_LIMIT_DEG = 89.0

# The step itself is deterministic unless a caller says otherwise.
DEFAULT_SIGMA = 0.0

# The measured sigma (#89, vault D028 as amended 4 Oct 2026): the real buoy lands inside the
# ensemble's 90 % region 90 % of the time at 4 h on the undrogued dev drifters, the end of the
# longest search (D027), plus a person's crosswind slide. 95 % CI 25.3-27.6. Matched at 4 h
# only: wider than the real error before it, narrower after (docs/sigma-calibration.md).
CALIBRATED_SIGMA = 26.3
CALIBRATION_HORIZON_H = 4

# The random velocity (vault D030, ADR005). Its memory is Taylor's (1921) T_L fitted to the
# undrogued dev drifters' 90 % spread at 1-48 h (scripts/fit_random_velocity.py, 4 Oct 2026):
# 25.7 h, about one inertial period at 26.5 N. sigma_u is matched in the engine at 4 h, as
# sigma was (D028): the ladder (array 58974) puts 90 % of the buoys inside the 90 % region at
# sigma_u* = 0.223 m/s (95 % CI 0.215-0.233), and a person's crosswind slide adds 0.035 m/s
# in quadrature. 95 % CI 0.217-0.236; HYCOM's own velocity error is 0.228 per axis.
DEFAULT_SIGMA_U = 0.0
MEMORY_TIME_S = 25.7 * 3600.0
CALIBRATED_SIGMA_U = 0.226


def as_position(value, name: str = "position") -> np.ndarray:
    """One [lat, lon] pair in degrees, or a stack of them, as a float array."""
    position = np.asarray(value, dtype=float)
    if position.ndim == 0 or position.shape[-1] != 2:
        raise ValueError(f"{name} must be [lat, lon] in degrees, got shape {position.shape}")
    return position


def check_latitude(lat, what: str) -> None:
    """Reject a latitude too near a pole for 1 / cos(lat); NaN passes, since land is NaN."""
    lat = np.asarray(lat, dtype=float)
    if np.any(np.abs(lat) > POLAR_LIMIT_DEG):
        raise ValueError(
            f"{what} is at latitude {float(np.nanmax(np.abs(lat)))}, beyond the "
            f"{POLAR_LIMIT_DEG} degree limit: 1 / cos(lat) is unusable that near a pole"
        )


def random_displacement(shape, timestep: float, sigma: float = DEFAULT_SIGMA,
                        rng=None) -> np.ndarray:
    """D009's sigma sqrt(dt) Z, as [east, north] metres; exactly zero when sigma is zero."""
    if sigma == 0.0:
        return np.zeros(shape)
    if sigma < 0.0:
        raise ValueError(f"sigma must not be negative, got {sigma}")
    rng = np.random.default_rng(rng)
    return sigma * np.sqrt(abs(float(timestep))) * rng.standard_normal(shape)


def velocity_memory(timestep: float, memory_time_s: float = MEMORY_TIME_S) -> float:
    """How much of a velocity error survives one step: exp(-dt / T_L)."""
    timestep, memory_time_s = float(timestep), float(memory_time_s)
    if not np.isfinite(memory_time_s) or memory_time_s <= 0.0:
        raise ValueError(f"the memory time must be a positive number of seconds, "
                         f"got {memory_time_s}")
    return float(np.exp(-abs(timestep) / memory_time_s))


def _check_sigma_u(sigma_u):
    """One sigma_u as a float, or one per particle as an (N, 1) column for an (N, 2) error."""
    if np.ndim(sigma_u) == 0:
        sigma_u = float(sigma_u)
        if not np.isfinite(sigma_u) or sigma_u < 0.0:
            raise ValueError(f"sigma_u must be a non-negative speed in m/s, got {sigma_u}")
        return sigma_u
    values = np.asarray(sigma_u, dtype=float).reshape(-1, 1)
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("every particle's sigma_u must be a non-negative speed in m/s")
    return values


def _is_zero(sigma_u) -> bool:
    return bool(np.all(np.asarray(sigma_u) == 0.0))


def start_velocity_error(shape, sigma_u=DEFAULT_SIGMA_U, rng=None) -> np.ndarray:
    """Each particle's velocity error at the start, [east, north] m/s, N(0, sigma_u^2) per axis.

    The process's own steady state, so a cloud released from one point (D026) spreads in a
    straight line from the first step, as the drifters do. Starting it at zero would grow the
    cloud too slowly through exactly the hours a search happens in. Zeros when sigma_u is 0.
    sigma_u is one value, or one per particle (the rows of `shape`).
    """
    sigma_u = _check_sigma_u(sigma_u)
    if _is_zero(sigma_u):
        return np.zeros(shape)
    return sigma_u * np.random.default_rng(rng).standard_normal(shape)


def evolve_velocity_error(velocity_error, timestep: float = INTEGRATION_STEP_SECONDS,
                          sigma_u=DEFAULT_SIGMA_U,
                          memory_time_s: float = MEMORY_TIME_S, rng=None) -> np.ndarray:
    """The velocity error one step later: a u + sqrt(1 - a^2) sigma_u Z, a = exp(-dt / T_L).

    Exact for the Ornstein-Uhlenbeck process at any step, so the variance stays sigma_u^2
    rather than drifting with dt. At 60 s and T_L = 25.7 h, a = 0.99935 and the fresh part
    is 0.036 sigma_u.
    """
    sigma_u = _check_sigma_u(sigma_u)
    a = velocity_memory(timestep, memory_time_s)
    u = np.asarray(velocity_error, dtype=float)
    if _is_zero(sigma_u):
        return a * u
    noise = np.random.default_rng(rng).standard_normal(u.shape)
    return a * u + np.sqrt(1.0 - a * a) * sigma_u * noise


def calculate_position(position, drift, timestep: float = INTEGRATION_STEP_SECONDS,
                       sigma: float = DEFAULT_SIGMA, rng=None, out=None):
    """The position one step later, [lat, lon] in degrees, longitude wrapped to 0 to 360."""
    # rng: an int reseeds identically on every call, so a loop must pass one live Generator.
    position = as_position(position)
    drift = as_vector(drift, "drift")
    timestep = float(timestep)
    if not np.isfinite(timestep):
        raise ValueError(f"timestep must be a finite number of seconds, got {timestep}")

    lat, lon = position[..., 0], position[..., 1]
    check_latitude(lat, "the position given")

    shape = np.broadcast_shapes(position.shape, drift.shape)
    metres = drift * timestep + random_displacement(shape, timestep, sigma, rng)

    # Flat earth: a metre is 1/R radians of latitude, and 1/(R cos(lat)) of longitude.
    metres_per_degree = EARTH_RADIUS_M / DEGREES_PER_RADIAN
    dlat = metres[..., 1] / metres_per_degree
    dlon = metres[..., 0] / (metres_per_degree * np.cos(lat / DEGREES_PER_RADIAN))
    check_latitude(lat + dlat, "the position after the step")

    if out is None:
        out = np.empty(shape, dtype=float)
    else:
        out = np.asarray(out)
        if out.shape != shape:
            raise ValueError(f"out has shape {out.shape}, but this step produces {shape}")

    # Latitude is not wrapped: a pole crossing also flips longitude, which this cannot model.
    out[..., 0] = lat + dlat
    out[..., 1] = (lon + dlon) % 360.0
    return out


def step_displacement(drift, timestep: float = INTEGRATION_STEP_SECONDS) -> np.ndarray:
    """The deterministic part of one step, as [east, north] metres."""
    return as_vector(drift, "drift") * float(timestep)


def _cli(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--position", nargs=2, type=float, required=True, metavar=("LAT", "LON"),
                        help="starting position in degrees, latitude then longitude")
    parser.add_argument("--drift", nargs=2, type=float, required=True, metavar=("U", "V"),
                        help="drift velocity in m/s, eastward then northward")
    parser.add_argument("--timestep", type=float, default=INTEGRATION_STEP_SECONDS,
                        help=f"step in seconds, default {INTEGRATION_STEP_SECONDS:g} (D009)")
    parser.add_argument("--sigma", type=float, default=DEFAULT_SIGMA,
                        help=f"D009's sigma in m/s^0.5, default {DEFAULT_SIGMA:g}")
    parser.add_argument("--seed", type=int, help="seed for the sigma draw, for a repeatable step")
    args = parser.parse_args(argv)

    moved = calculate_position(args.position, args.drift, args.timestep, args.sigma, args.seed)
    metres = step_displacement(args.drift, args.timestep)
    return {
        "position_deg": list(args.position),
        "drift_ms": list(args.drift),
        "timestep_s": args.timestep,
        "sigma": args.sigma,
        "seed": args.seed,
        "drift_displacement_m": [float(metres[0]), float(metres[1])],
        "drift_distance_m": float(np.hypot(metres[0], metres[1])),
        "moved_deg": [float(moved[0]), float(moved[1])],
        "delta_deg": [float(moved[0] - args.position[0]), float(moved[1] - args.position[1])],
    }


if __name__ == "__main__":
    print(json.dumps(_cli(), indent=2))
