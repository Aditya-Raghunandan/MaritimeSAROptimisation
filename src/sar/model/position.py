"""position.py: one Euler-Maruyama step of D009, x_{n+1} = x_n + v_d dt + sigma sqrt(dt) Z."""

from __future__ import annotations

import argparse
import json

import numpy as np

from sar.model.drift import as_vector

# D009 (vault decision, Aditya, 2026-09-03): one minute per integration step.
INTEGRATION_STEP_SECONDS = 60.0

# GRS80 mean radius R1 = (2a + b) / 3; Moritz (2000), J. Geodesy 74(1), 128-133.
# The same number as R_EARTH_M in frontend/src/geo.js, so both measure alike.
EARTH_RADIUS_M = 6371008.8

DEGREES_PER_RADIAN = 180.0 / np.pi

# A project limit, not a published one: 1 / cos(lat) is unusable nearer the pole than this.
POLAR_LIMIT_DEG = 89.0

# D009's sigma is unpinned until the Monte Carlo ticket measures it, so the step is
# deterministic unless a caller says otherwise.
DEFAULT_SIGMA = 0.0


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


def calculate_position(position, drift, timestep: float = INTEGRATION_STEP_SECONDS,
                       sigma: float = DEFAULT_SIGMA, rng=None, out=None):
    """The position one step later, [lat, lon] in degrees, longitude wrapped to 0 to 360."""
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
