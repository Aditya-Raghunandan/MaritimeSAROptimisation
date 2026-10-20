"""datum_c2.py: sar.search.datum's marker and the drift bearing, with drift_c2's doubled current."""

from __future__ import annotations

import numpy as np

from sar.model.drift import LEEWAY_COEFFICIENT
from sar.model.drift_c2 import calculate_drift
from sar.pipeline.track_c2 import DriftPipelineC2
from sar.search.datum import _TIME_TOLERANCE_S
from sar.search.patterns import MarkerTrack
from sar.search.platform import ON_SCENE_WINDOW_S, STEP_S
from sar.utils.geo import to_store_longitude


def drift_track(lat: float, lon: float, start, duration_s: float, forcing,
                leeway: float, timestep: float = STEP_S) -> MarkerTrack:
    """sar.search.datum.drift_track through DriftPipelineC2: whole steps, then one short step."""
    duration_s = float(duration_s)
    if not np.isfinite(duration_s) or duration_s <= 0:
        raise ValueError(f"duration must be a positive number of seconds, got {duration_s}")
    if leeway < 0:
        raise ValueError(f"leeway must not be negative, got {leeway}")

    start = np.datetime64(start, "us")
    whole = int(np.floor(duration_s / timestep + _TIME_TOLERANCE_S))
    pipe = DriftPipelineC2(forcing, timestep=timestep, leeway=leeway, sigma=0.0)
    states = list(pipe.track(start, lat, to_store_longitude(lon), whole * timestep))
    t = [s.seconds for s in states]
    pos = [s.positions[0] for s in states]

    rest = duration_s - whole * timestep
    if rest > _TIME_TOLERANCE_S:
        tail = DriftPipelineC2(forcing, timestep=rest, leeway=leeway, sigma=0.0)
        when = start + np.timedelta64(round(t[-1] * 1e6), "us")
        moved = tail.advance(np.atleast_2d(pos[-1]), when)[0]
        t.append(duration_s)
        pos.append(moved[0])

    pos = np.array(pos)
    return MarkerTrack(np.array(t, dtype=float), pos[:, 0], pos[:, 1])


def marker_track(datum_lat: float, datum_lon: float, drop_time,
                 duration_s: float = ON_SCENE_WINDOW_S, forcing=None) -> MarkerTrack:
    """The datum marker from the drop onwards, drifting with the doubled current alone."""
    if forcing is None:
        raise ValueError("marker_track needs a forcing backend")
    return drift_track(datum_lat, datum_lon, drop_time, duration_s, forcing, leeway=0.0)


def drift_bearing(forcing, lat: float, lon: float, time,
                  leeway: float = LEEWAY_COEFFICIENT) -> float | None:
    """The target's drift at a point under drift_c2, degrees true; None where it is NaN or zero."""
    current, wind = forcing.sample([lat], [lon], np.datetime64(time, "us"))
    u, v = calculate_drift(wind[0], current[0], leeway)
    if not (np.isfinite(u) and np.isfinite(v)) or (u == 0.0 and v == 0.0):
        return None
    return float(np.degrees(np.arctan2(u, v)) % 360.0)
