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

THE ARRIVAL HEADING (D032). A helicopter that cannot turn on the spot arrives pointing
somewhere, and that is a fact about the search, not about the searcher, so every searcher
arrives on the same one: along the target's drift at the datum, which is where the Coast
Guard lays the first leg of its Expanding Square and Sector (Addendum p. 3-26); north if
there is no drift. `search_episode` builds the referee with it.

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
from sar.pipeline.ensemble import Ensemble, resolve_sigma_u, run_ensemble, synthetic_cloud
from sar.pipeline.gridded import GriddedForcing
from sar.search.datum import marker_track
from sar.search.episode import STEPS, SearchEpisode
from sar.search.patterns import (
    MarkerTrack,
    Pattern,
    expanding_square,
    parallel_track,
    sector_search,
    trackline_return,
)
from sar.search.platform import (
    ON_SCENE_WINDOW_S,
    STEP_S,
    TURN_RATE_DEG_S,
    expanding_square_spacing_m,
    sector_radius_m,
)
from sar.utils.geo import (
    M_PER_DEG_LAT,
    metres_per_degree_lon,
    offset_position,
    to_store_longitude,
)

# The window must end by 4 h, just after the longest search (D027).
HORIZON_S = CALIBRATION_HORIZON_H * 3600.0

# The scenario table's truth: the buoy every hour to 6 h (scripts/pick_scenario_buoys.py).
TRUTH_HOURS = (1, 2, 3, 4, 5, 6)

# The Coast Guard patterns, and the two laid out along the datum line rather than the drift.
DOCTRINAL = ("expanding-square", "sector", "parallel", "trackline")
ALONG_THE_DATUM_LINE = ("parallel", "trackline")

# A datum line shorter than this has no direction worth flying along (frontend/src/searchRun.js).
MIN_DATUM_LINE_M = 50.0

# How many windows of an endless pattern to draw, so the autopilot never runs out (D032).
DRAWN_WINDOWS = 2

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
    sigma_u: float | None = None        # the random velocity's size the cloud was run with
    start_current_ms: float | None = None   # the current at the start, when it sized sigma_u


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
                  arrival_s: float, sigma_u: float | str = CALIBRATED_SIGMA_U,
                  steps: int = STEPS, sigma: float = DEFAULT_SIGMA) -> SearchSetup:
    """A scenario's cloud through the search window, every minute, and its marker.

    The scenario's own run again (one start point, D026; its seed), kept at every step and
    sliced to the steps + 1 frames from arrival. Memory is the whole run at every step:
    39 MB at 10^4 particles for 4 h, 386 MB at 10^5. `sigma` with sigma_u = 0 is the old
    random walk (D028), for comparing the two noise models on the same seeds.

    sigma_u = "by-current" sizes the random velocity by the model's current at the start
    point at the call (D033): the scenario's own LKP and time, what a forecaster has.
    """
    a = check_arrival(arrival_s, steps)
    sigma_u, speed = resolve_sigma_u(sigma_u, forcing, lat, lon, start)
    run = run_ensemble(forcing, particles, start, lat, lon, (a + steps) * STEP_S, STEP_S,
                       datum_sigma_km=0.0, seed=seed, sigma=sigma, sigma_u=sigma_u)
    cut = slice(a, a + steps + 1)
    window = Ensemble(run.times[cut], run.lat[cut], run.lon[cut], run.weight, run.beached[cut])
    datum = centroid(window.lat[0], window.lon[0], window.weight)
    marker = marker_track(*datum, window.times[0], steps * STEP_S, forcing)
    return SearchSetup(window, marker, None, datum, a * STEP_S,
                       drift_bearing(forcing, *datum, window.times[0]), sigma_u, speed)


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
                    sigma_u: float | str = CALIBRATED_SIGMA_U, steps: int = STEPS,
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


def arrival_heading(setup: SearchSetup) -> float:
    """The heading every searcher arrives on: along the drift at the datum, else north (D032)."""
    return 0.0 if setup.drift_bearing_deg is None else float(setup.drift_bearing_deg)


def search_episode(setup: SearchSetup, turn_rate_deg_s: float = TURN_RATE_DEG_S,
                   **kw) -> SearchEpisode:
    """The referee for a scenario: its cloud, marker and buoy, arriving on its drift."""
    return SearchEpisode(setup.window, setup.marker, target=setup.target,
                         heading_deg=arrival_heading(setup), turn_rate_deg_s=turn_rate_deg_s,
                         **kw)


def datum_line(lkp, datum) -> tuple[float | None, float]:
    """The drift model's prediction as a line, last known position to datum: (bearing, metres).

    The Addendum's "datum line" (§H.7.3.1). The same flat-earth formula as the site's
    `datumLine` (frontend/src/searchRun.js), cos of the mid latitude, so both lay the Parallel
    Track and the Trackline the same way. The bearing is None for a line with no length.
    """
    (lat0, lon0), (lat1, lon1) = (tuple(float(v) for v in p) for p in (lkp, datum))
    dlon = (lon1 - lon0 + 180.0) % 360.0 - 180.0
    east = dlon * float(metres_per_degree_lon(0.5 * (lat0 + lat1)))
    north = (lat1 - lat0) * M_PER_DEG_LAT
    length = float(np.hypot(east, north))
    bearing = None if length == 0.0 else float(np.degrees(np.arctan2(east, north)) % 360.0)
    return bearing, length


def doctrinal_searcher(name: str, setup: SearchSetup, lkp) -> tuple[Pattern, float, str]:
    """A Coast Guard pattern laid out for one scenario, as the site lays it: (pattern, bearing, why).

    One rule for the scorer, the site's bundles and the golden fixture, so a pattern is the
    same pattern wherever it is flown:

      * Expanding Square and Sector: first leg along the target's drift at the datum
        (Addendum p. 3-26), north if there is no drift. Square spacing from W (p. 3-23).
      * Parallel Track and Trackline: along the datum line, last known position to datum,
        when it is over 50 m long (§H.7.3.9); otherwise along the drift, as above.
      * Trackline half-length: the datum line's length, at least a Sector radius, so a
        near-still datum still gets a real line: the buoy may have drifted anywhere from not
        at all to twice as far as predicted (frontend/src/searchRun.js).
      * The patterns that go on for ever (all but the Parallel Track, whose area is the
        window's) are drawn for DRAWN_WINDOWS windows. The autopilot rounds corners and so
        gets along the drawing faster than the clock (D032); it must not run out of legs.
        A helicopter that turns at once flies only the first window, as before.
    """
    if name not in DOCTRINAL:
        raise ValueError(f"{name!r} is not a Coast Guard pattern; one of {', '.join(DOCTRINAL)}")
    drift = setup.drift_bearing_deg
    bearing, why = (0.0, "no drift at the datum; north") if drift is None \
        else (float(drift), "along the drift")
    line_bearing, line_m = datum_line(lkp, setup.datum)
    if name in ALONG_THE_DATUM_LINE and line_bearing is not None and line_m > MIN_DATUM_LINE_M:
        bearing, why = line_bearing, "along the datum line"
    drawn = DRAWN_WINDOWS * ON_SCENE_WINDOW_S
    if name == "expanding-square":
        pattern = expanding_square(expanding_square_spacing_m(), bearing, duration_s=drawn)
    elif name == "sector":
        pattern = sector_search(first_bearing_deg=bearing, duration_s=drawn)
    elif name == "parallel":
        pattern = parallel_track(first_bearing_deg=bearing)
    else:
        pattern = trackline_return(max(line_m, sector_radius_m()), first_bearing_deg=bearing,
                                   duration_s=drawn)
    return pattern, bearing, why


def straightness(row, hours: int = 4) -> float:
    """How far the buoy got over `hours`, divided by how far it travelled along its hourly fixes.

    1 is a straight line; 0.90 a quarter-circle bend; 0.71 a right-angle turn halfway; 0.64
    a half circle; 0 back where it started. NaN if the buoy did not move or a fix is missing.
    Flat earth at the start's latitude, which is exact enough over a few kilometres.
    """
    lat = np.array([float(row["lat"])] + [float(row[f"lat_{h}h"]) for h in range(1, hours + 1)])
    lon = np.array([float(row["lon"])] + [float(row[f"lon_{h}h"]) for h in range(1, hours + 1)])
    if not (np.all(np.isfinite(lat)) and np.all(np.isfinite(lon))):
        return float("nan")
    east = ((lon - lon[0] + 180.0) % 360.0 - 180.0) * float(metres_per_degree_lon(lat[0]))
    north = (lat - lat[0]) * M_PER_DEG_LAT
    path = float(np.sum(np.hypot(np.diff(east), np.diff(north))))
    if path == 0.0:
        return float("nan")
    return float(np.hypot(east[-1], north[-1]) / path)


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
