"""drift.py: drift velocity from the current and the 10 m wind (issue #5).

The deterministic part of the drift equation frozen by D002:

    v_d = v_c + alpha * v_w10 + eta

with the stochastic term eta left out. eta is not a property of a position; it is drawn
per particle per step, so it belongs to the Euler-Maruyama update in the engine (D009),
where it enters as sigma * sqrt(dt) * Z. What this module returns is the bracket that
update multiplies by dt, which keeps it pure and testable against hand-computed numbers.

CLI. One particle, printed as JSON, for checking the arithmetic by hand:

    python -m sar.model.drift --wind 7 7 --current 1.2 0.9
    python -m sar.model.drift --wind 7 7 --current 1.2 0.9 --leeway 0.04

See docs/pr_drift_calculator.md for the theory, the citations behind alpha and the
limitation the missing crosswind term carries.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

# Leeway coefficient for a person in water, frozen by D002 and cited to Allen 2000
# (DTIC ADA376479): a downwind slope of 1.93 % of wind speed, 2.7 % for a survival suit.
# R1c's range is 1 to 4 %, which the sensitivity sweep covers by passing its own value.
LEEWAY_COEFFICIENT = 0.02


def as_vector(value, name: str) -> np.ndarray:
    """One [u, v] pair, or a stack of them, as a float array.

    Accepts anything array-like whose trailing axis is the two components, so a single
    particle is shape (2,) and an ensemble is (N, 2). Raises ValueError otherwise,
    because a three-element vector here means a caller has confused this with a position
    or has passed speed and direction.
    """
    vector = np.asarray(value, dtype=float)
    if vector.ndim == 0 or vector.shape[-1] != 2:
        raise ValueError(f"{name} must be [u, v] in m/s, got shape {vector.shape}")
    return vector


def calculate_drift(wind, current, leeway: float = LEEWAY_COEFFICIENT, out=None) -> np.ndarray:
    """Drift velocity in m/s, as [u, v], from the 10 m wind and the surface current.

    wind must be the wind at 10 m, which is the height alpha is defined against; a wind
    from any other height silently rescales the whole leeway term. Both inputs are in
    m/s and are broadcast against each other, so one wind field can be applied to an
    ensemble of particles and the shape of the result follows the wider of the two.

    NaN propagates. HYCOM writes NaN for land, and a particle over land is a beaching
    question for the engine (D016), so a missing current must come back as a missing
    drift rather than as still water.
    """
    wind = as_vector(wind, "wind")
    current = as_vector(current, "current")
    if out is None:
        return current + leeway * wind
    np.multiply(wind, leeway, out=out)
    np.add(out, current, out=out)
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
    leeway_term = args.leeway * np.asarray(args.wind, dtype=float)
    return {
        "wind_ms": list(args.wind),
        "current_ms": list(args.current),
        "leeway": args.leeway,
        "leeway_term_ms": [float(leeway_term[0]), float(leeway_term[1])],
        "drift_ms": [float(drift[0]), float(drift[1])],
        "drift_speed_ms": float(np.hypot(drift[0], drift[1])),
    }


if __name__ == "__main__":
    print(json.dumps(_cli(), indent=2))
