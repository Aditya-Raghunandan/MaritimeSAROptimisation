"""track_c2.py: the drift pipeline stepped with sar.model.drift_c2, the current doubled."""

from __future__ import annotations

import numpy as np

from sar.model.drift_c2 import CURRENT_FACTOR, calculate_drift
from sar.model.position import calculate_position
from sar.pipeline.track import DriftPipeline


class DriftPipelineC2(DriftPipeline):
    """DriftPipeline with one change: each step's drift is drift_c2's, 2 x current + leeway x wind."""

    def advance(self, positions: np.ndarray, time, velocity_error=None):
        current, wind = self.forcing.sample(positions[:, 0], positions[:, 1], time)
        drift = calculate_drift(wind, current, self.leeway)
        moving = drift if velocity_error is None else drift + velocity_error
        moved = calculate_position(positions, moving, self.timestep, self.sigma, self.rng)
        beached = ~np.isfinite(drift).all(axis=1)   # D016: beached where the forcing is NaN
        moved[beached] = positions[beached]
        return moved, current, wind, drift

    def describe(self, start, lat, lon, duration: float) -> dict:
        return {**super().describe(start, lat, lon, duration), "current_factor": CURRENT_FACTOR}
