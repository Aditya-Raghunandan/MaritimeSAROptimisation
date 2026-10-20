"""drift_c2.py: sar.model.drift with the current doubled, v_d = 2 v_c + alpha v_w10, for the c2 experiment."""

from __future__ import annotations

import argparse
import json

import numpy as np

from sar.model.drift import LEEWAY_COEFFICIENT, as_vector

CURRENT_FACTOR = 2.0   # project choice: the c2 experiment's weight on the current, against D002's 1


def calculate_drift(wind, current, leeway: float = LEEWAY_COEFFICIENT, out=None) -> np.ndarray:
    """Drift velocity in m/s, as [u, v]: CURRENT_FACTOR x current + leeway x 10 m wind; NaN propagates."""
    wind = as_vector(wind, "wind")
    current = as_vector(current, "current")
    if out is None:
        return CURRENT_FACTOR * current + leeway * wind
    np.multiply(wind, leeway, out=out)
    out += CURRENT_FACTOR * current
    return out


def _cli(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--wind", nargs=2, type=float, required=True, metavar=("U", "V"),
                        help="10 m wind components in m/s, eastward then northward")
    parser.add_argument("--current", nargs=2, type=float, required=True, metavar=("U", "V"),
                        help="surface current components in m/s, eastward then northward")
    parser.add_argument("--leeway", type=float, default=LEEWAY_COEFFICIENT,
                        help=f"leeway coefficient, default {LEEWAY_COEFFICIENT} (D002)")
    args = parser.parse_args(argv)

    drift = calculate_drift(args.wind, args.current, args.leeway)
    return {
        "wind_ms": list(args.wind),
        "current_ms": list(args.current),
        "current_factor": CURRENT_FACTOR,
        "leeway": args.leeway,
        "drift_ms": [float(drift[0]), float(drift[1])],
        "drift_speed_ms": float(np.hypot(drift[0], drift[1])),
    }


if __name__ == "__main__":
    print(json.dumps(_cli(), indent=2))
