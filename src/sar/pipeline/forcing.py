"""forcing.py: the one interface the engine reads current and wind through (D009)."""

from __future__ import annotations

import numpy as np

from sar.utils.geo import to_store_longitude


def as_particle_axes(lats, lons) -> tuple[np.ndarray, np.ndarray]:
    """Latitudes and longitudes as two matching 1-D arrays over particles, longitude 0 to 360."""
    lats = np.atleast_1d(np.asarray(lats, dtype=float))
    lons = np.atleast_1d(np.asarray(to_store_longitude(lons), dtype=float))
    if lats.ndim != 1 or lons.ndim != 1:
        raise ValueError(f"lats and lons must be 1-D over particles, got {lats.shape} "
                         f"and {lons.shape}")
    if lats.size != lons.size:
        raise ValueError(f"{lats.size} latitudes and {lons.size} longitudes do not pair up")
    return lats, lons


class ConstantForcing:
    """A uniform, steady current and wind, whose answer can be worked out on paper."""

    def __init__(self, current=(0.0, 0.0), wind=(0.0, 0.0)):
        self.current = np.asarray(current, dtype=float)
        self.wind = np.asarray(wind, dtype=float)
        for name, vector in (("current", self.current), ("wind", self.wind)):
            if vector.shape != (2,):
                raise ValueError(f"{name} must be one [u, v] pair in m/s, got {vector.shape}")

    def sample(self, lats, lons, time) -> tuple[np.ndarray, np.ndarray]:
        """The current and the wind at every particle, each (N, 2) in m/s."""
        lats, _ = as_particle_axes(lats, lons)
        return np.tile(self.current, (lats.size, 1)), np.tile(self.wind, (lats.size, 1))

    def describe(self) -> dict:
        """What this backend is, for the header of an output file."""
        return {"backend": "constant",
                "current_ms": self.current.tolist(),
                "wind_ms": self.wind.tolist()}
