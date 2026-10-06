"""scenario.py: a scenario's cloud, marker and buoy, ready to search (issue #47).

The episode (`sar.search.episode`) needs the cloud every 60 s through the 45-minute window,
the datum marker's track, and, to score "found", the real buoy. This builds all three.

THE CLOUD EVERY MINUTE, FROM THE SAME SEED (ADR004 row 4). A scenario run saves the cloud
every 5 minutes (scripts/run_scenarios.sbatch). `search_window` runs the scenario again with
the same seed and keeps every step of the window. `run_ensemble` draws its random numbers
once per step whether or not the step is saved, so this is the same cloud, bit for bit,
only seen more often: exact, and about 2 s at 10^4 particles.

Under the old random walk (sigma = 26.3) this was necessary: a straight line between
5-minute snapshots missed a particle by ~222 m, 2.4 times the 92.6 m half-strip. Under the
random velocity (D030, ADR005), a particle's velocity error barely changes in 5 minutes, and
the same straight line misses by under 1 m (measured 6 Oct, steady forcing). So 5-minute
snapshots are now close enough to sweep between too; re-running stays because it is exact.

THE COMMON DATUM (D004). The marker is dropped at the cloud's centroid at arrival, the same
point for every searcher, and drifts with the current alone (`sar.search.datum`).

ARRIVAL IS A CHOICE, NOT A FLIGHT (ADR004 row 5). It is given in seconds after the call, a
whole number of minutes, and the window must end by 4 h, just after the longest search
(D027). Since D030 the cloud itself is calibrated to 24 h, so this is a guard on the
scenario, not a limit of the cloud.

STEADY CLOUDS. `steady_search` is a synthetic cloud carried by a uniform current, with its
marker, for tests and the CLI.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from sar.model.drift import LEEWAY_COEFFICIENT
from sar.model.position import CALIBRATED_SIGMA_U, CALIBRATION_HORIZON_H, DEFAULT_SIGMA
from sar.pipeline.ensemble import Ensemble, run_ensemble, synthetic_cloud
from sar.pipeline.gridded import GriddedForcing
from sar.search.datum import marker_track
from sar.search.episode import STEPS
from sar.search.patterns import MarkerTrack
from sar.search.platform import STEP_S
from sar.utils.geo import offset_position, to_store_longitude

# The window must end by 4 h, just after the longest search (D027).
HORIZON_S = CALIBRATION_HORIZON_H * 3600.0

# The scenario table's truth: the buoy every hour to 6 h (scripts/pick_scenario_buoys.py).
TRUTH_HOURS = (1, 2, 3, 4, 5, 6)

DEFAULT_TIME = "2019-01-01T00:00"


@dataclass(frozen=True)
class SearchSetup:
    """Everything an episode needs, plus where the search started from."""

    window: Ensemble
    marker: MarkerTrack
    target: MarkerTrack | None
    datum: tuple[float, float]          # (lat, lon), longitude 0 to 360
    arrival_s: float                    # seconds after the call
    drift_bearing_deg: float | None     # the target's drift at the datum, for a first leg


def check_arrival(arrival_s, steps: int = STEPS, horizon_s: float = HORIZON_S) -> int:
    """Arrival as whole steps after the call, refusing one the cloud cannot be trusted for."""
    arrival_s = float(arrival_s)
    whole = round(arrival_s / STEP_S)
    if not np.isfinite(arrival_s) or arrival_s < 0 or abs(arrival_s - whole * STEP_S) > 1e-6:
        raise ValueError(f"arrival must be a whole number of minutes after the call, "
                         f"got {arrival_s:g} s")
    if (whole + steps) * STEP_S > horizon_s + 1e-6:
        latest = (horizon_s - steps * STEP_S) / 60.0
        raise ValueError(f"arrival at {arrival_s / 60:g} min ends the window past "
                         f"{horizon_s / 3600:g} h, after the longest search (D027); arrive by "
                         f"{latest:g} min")
    return int(whole)


def centroid(lat, lon, weight) -> tuple[float, float]:
    """The weighted mean position, longitude 0 to 360."""
    lat, lon, weight = (np.asarray(a, dtype=float) for a in (lat, lon, weight))
    ref = lon[0]
    dlon = (lon - ref + 180.0) % 360.0 - 180.0
    total = float(np.sum(weight))
    return (float(np.sum(weight * lat) / total),
            float(to_store_longitude(ref + np.sum(weight * dlon) / total)))


def drift_bearing(forcing, lat: float, lon: float, time,
                  leeway: float = LEEWAY_COEFFICIENT) -> float | None:
    """The target's drift at a point, degrees true: doctrine's first leg (p. 3-26)."""
    current, wind = forcing.sample([lat], [lon], np.datetime64(time, "us"))
    u, v = current[0] + leeway * wind[0]
    if not (np.isfinite(u) and np.isfinite(v)) or (u == 0.0 and v == 0.0):
        return None
    return float(np.degrees(np.arctan2(u, v)) % 360.0)


def search_window(forcing, start, lat: float, lon: float, seed, particles: int,
                  arrival_s: float, sigma_u: float = CALIBRATED_SIGMA_U,
                  steps: int = STEPS, sigma: float = DEFAULT_SIGMA) -> SearchSetup:
    """A scenario's cloud through the search window, every minute, and its marker.

    The scenario's own run again (one start point, D026; its seed), kept at every step and
    sliced to the steps + 1 frames from arrival. Memory is the whole run at every step:
    39 MB at 10^4 particles for 4 h, 386 MB at 10^5. `sigma` with sigma_u = 0 is the old
    random walk (D028), for comparing the two noise models on the same seeds.
    """
    a = check_arrival(arrival_s, steps)
    run = run_ensemble(forcing, particles, start, lat, lon, (a + steps) * STEP_S, STEP_S,
                       datum_sigma_km=0.0, seed=seed, sigma=sigma, sigma_u=sigma_u)
    cut = slice(a, a + steps + 1)
    window = Ensemble(run.times[cut], run.lat[cut], run.lon[cut], run.weight, run.beached[cut])
    datum = centroid(window.lat[0], window.lon[0], window.weight)
    marker = marker_track(*datum, window.times[0], steps * STEP_S, forcing)
    return SearchSetup(window, marker, None, datum, a * STEP_S,
                       drift_bearing(forcing, *datum, window.times[0]))


def read_scenarios(path) -> pd.DataFrame:
    """The scenario table written by scripts/pick_scenario_buoys.py."""
    return pd.read_csv(path)


def scenario_row(path, name: str) -> pd.Series:
    """One row of the scenario table by its name, such as S01."""
    table = read_scenarios(path)
    rows = table[table["scenario"] == name]
    if len(rows) != 1:
        raise ValueError(f"{name!r} is not a scenario in {path}")
    return rows.iloc[0]


def truth_track(row, arrival_s: float) -> MarkerTrack:
    """The real buoy, at the start and every hour after, in seconds since arrival."""
    hours = [0] + [h for h in TRUTH_HOURS if np.isfinite(float(row[f"lat_{h}h"]))]
    lat = [float(row["lat"])] + [float(row[f"lat_{h}h"]) for h in hours[1:]]
    lon = [float(row["lon"])] + [float(row[f"lon_{h}h"]) for h in hours[1:]]
    t = np.array(hours, dtype=float) * 3600.0 - float(arrival_s)
    return MarkerTrack(t, np.array(lat), to_store_longitude(np.array(lon)))


def scenario_search(row, forcing, arrival_s: float, particles: int,
                    sigma_u: float = CALIBRATED_SIGMA_U, steps: int = STEPS,
                    sigma: float = DEFAULT_SIGMA) -> SearchSetup:
    """A row of the scenario table, ready to search: cloud, marker, and the buoy as target.

    `forcing` is a backend, or a data root holding raw/hycom_* and raw/era5_*.
    """
    start = np.datetime64(str(row["start"]), "us")
    if isinstance(forcing, (str, Path)):
        a = check_arrival(arrival_s, steps)
        end = start + np.timedelta64(int((a + steps) * STEP_S), "s")
        forcing = GriddedForcing.from_dir(forcing, start, end)
    setup = search_window(forcing, start, float(row["lat"]), float(row["lon"]),
                          int(row["seed"]), particles, arrival_s, sigma_u, steps, sigma)
    return replace(setup, target=truth_track(row, setup.arrival_s))


def steady_window(centre, spread_km: float, particles: int, rng, current=(0.0, 0.0),
                  steps: int = STEPS, time=DEFAULT_TIME) -> Ensemble:
    """A synthetic Gaussian cloud carried rigidly by a uniform current, steps + 1 frames."""
    cloud = synthetic_cloud(centre, spread_km, particles, rng, time)
    t = np.arange(steps + 1) * STEP_S
    east, north = float(current[0]) * t, float(current[1]) * t
    lat, lon = offset_position(cloud.lat[0][None, :], cloud.lon[0][None, :],
                               east[:, None], north[:, None])
    times = cloud.times[0] + (t * 1e6).astype("timedelta64[us]")
    return Ensemble(times, lat, to_store_longitude(lon), cloud.weight,
                    np.zeros(lat.shape, dtype=bool))


def steady_marker(lat: float, lon: float, current=(0.0, 0.0),
                  steps: int = STEPS) -> MarkerTrack:
    """A marker carried by the same uniform current, from (lat, lon)."""
    t = np.arange(steps + 1) * STEP_S
    mlat, mlon = offset_position(np.full(t.size, float(lat)), np.full(t.size, float(lon)),
                                 float(current[0]) * t, float(current[1]) * t)
    return MarkerTrack(t, mlat, to_store_longitude(mlon))


def steady_search(centre, spread_km: float, particles: int, rng, current=(0.0, 0.0),
                  steps: int = STEPS) -> SearchSetup:
    """A steady synthetic cloud with its marker at the centroid: no forcing, no target."""
    window = steady_window(centre, spread_km, particles, rng, current, steps)
    datum = centroid(window.lat[0], window.lon[0], window.weight)
    u, v = float(current[0]), float(current[1])
    bearing = None if u == 0.0 and v == 0.0 else float(np.degrees(np.arctan2(u, v)) % 360.0)
    return SearchSetup(window, steady_marker(*datum, current, steps), None, datum, 0.0,
                       bearing)
