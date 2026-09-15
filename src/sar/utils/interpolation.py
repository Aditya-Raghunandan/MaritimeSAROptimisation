"""interpolation.py: linear interpolation between two known timesteps (issue #3).

The drift-model integrator steps every minute (see docs/linear_time_interpolation.md), but
the forcing it reads does not arrive at that cadence: wind is hourly, current is
3-hourly. Both fields need the same interpolation logic to be broken down into per-minute
pieces between two known field values, so this module provides one small function for a
single value or vector, and one wrapper for interpolating several points at once, for
example the four spatial grid corners used by the resultant-vector interpolation work.

CLI:
    python -m sar.utils.interpolation --t1 1.0 --t2 3.0 --pieces 5
    python -m sar.utils.interpolation --t1 1.0,2.0 --t2 3.0,4.0 --pieces 5
"""

import argparse

import numpy as np


def interpolate_series(value_t1, value_t2, pieces: int) -> np.ndarray:
    """Linearly interpolate from value_t1 up to, but not including, value_t2.

    Works for a scalar (returns shape (pieces,)) or a vector such as [u, v]
    (returns shape (pieces, 2)); value_t1 and value_t2 must have the same shape.

    Parameters
    ----------
    value_t1 : the field value at the first timestep.
    value_t2 : the field value at the second timestep.
    pieces : how many equal steps to split the interval into.

    Returns
    -------
    An array of length pieces. Index 0 equals value_t1; value_t2 itself is not
    included. The step size is (value_t2 - value_t1) / pieces.
    """
    value_t1 = np.asarray(value_t1, dtype=float)
    value_t2 = np.asarray(value_t2, dtype=float)
    if pieces < 1:
        raise ValueError(f"pieces must be at least 1, got {pieces}")

    step = (value_t2 - value_t1) / pieces
    indices = np.arange(pieces).reshape((pieces,) + (1,) * value_t1.ndim)
    return value_t1 + indices * step


def interpolate_grid_series(points_t1, points_t2, pieces: int) -> np.ndarray:
    """Linearly interpolate several points at once, each with its own [u, v].

    points_t1 and points_t2 are each shape (N, 2): N points (for example the four
    spatial grid corners used elsewhere for bilinear interpolation), each with an
    eastward/northward pair at t1 and t2. Returns shape (pieces, N, 2): the same
    per-minute breakdown as interpolate_series, applied to every point at once.

    This is a thin wrapper around interpolate_series, which already generalises to
    this shape by broadcasting; the wrapper's job is validating that both inputs are
    genuinely shaped (N, 2) and agree with each other, since a shape mismatch here
    would otherwise fail silently or with a confusing broadcasting error deeper in
    the call stack.
    """
    points_t1 = np.asarray(points_t1, dtype=float)
    points_t2 = np.asarray(points_t2, dtype=float)
    if points_t1.shape != points_t2.shape:
        raise ValueError(
            f"points_t1 and points_t2 must have the same shape, "
            f"got {points_t1.shape} and {points_t2.shape}"
        )
    if points_t1.ndim != 2 or points_t1.shape[1] != 2:
        raise ValueError(f"expected shape (N, 2), got {points_t1.shape}")

    return interpolate_series(points_t1, points_t2, pieces)


def _parse_value(raw: str):
    parts = [float(p) for p in raw.split(",")]
    return parts[0] if len(parts) == 1 else parts


def main() -> None:
    p = argparse.ArgumentParser(description="Linearly interpolate between two timesteps")
    p.add_argument("--t1", required=True, help="value at t1, e.g. 1.0 or 1.0,2.0")
    p.add_argument("--t2", required=True, help="value at t2, e.g. 3.0 or 3.0,4.0")
    p.add_argument("--pieces", required=True, type=int, help="number of steps")
    args = p.parse_args()

    result = interpolate_series(_parse_value(args.t1), _parse_value(args.t2), args.pieces)
    print(result)


if __name__ == "__main__":
    main()
