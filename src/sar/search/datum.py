"""datum.py: where the search starts, and where its marker drifts (issue #48).

Two points are drifted, and the difference between them is doctrine, not detail:

    the DATUM    the last known position carried forward to the moment the helicopter
                 arrives, by the current AND the leeway. Addendum p. 3-21: "Drift is
                 estimated as the vector sum of the total water current ... and leeway".
                 This is where the helicopter flies to and drops its marker.
    the MARKER   the smoke float or datum marker buoy dropped at the datum, drifting from
                 then on with the water current only -- marker buoys "are tools for
                 determining total water current" (p. 3-21). The pattern is flown about it
                 (`sar.search.patterns`).

Both use the drift pipeline exactly as the ensemble does -- `DriftPipeline`, Euler steps of
60 s -- with sigma = 0, so the datum is the ensemble's centroid when the ensemble has no
spread (D026). In the paper the datum is the real ensemble's centroid at arrival (D004);
this module is what the site and the tests use until the ensemble exists.

A duration need not divide by the step: a transit of 47 min 13 s is whole steps and then
one short step, because `DriftPipeline` refuses a ragged duration and a search timeline is
ragged by nature.

CLI. One datum and its marker, printed as JSON:

    python -m sar.search.datum --lat 26.5 --lon -79.0 --start 2019-06-01T06:00 \\
        --elapsed-min 77 --constant-current 1.8 0.0 --constant-wind 5.0 5.0
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from sar.model.drift import LEEWAY_COEFFICIENT
from sar.pipeline.forcing import ConstantForcing
from sar.pipeline.track import DriftPipeline
from sar.search.patterns import MarkerTrack
from sar.search.platform import ON_SCENE_WINDOW_S, STEP_S
from sar.utils.geo import to_display_longitude, to_store_longitude

_TIME_TOLERANCE_S = 1e-9


def drift_track(lat: float, lon: float, start, duration_s: float, forcing,
                leeway: float, timestep: float = STEP_S) -> MarkerTrack:
    """One point drifted for `duration_s` with no random term, as (t, lat, lon).

    Whole steps of `timestep`, then one shorter step for any remainder. Longitude comes
    back in the store's 0 to 360, whatever it went in as.
    """
    duration_s = float(duration_s)
    if not np.isfinite(duration_s) or duration_s <= 0:
        raise ValueError(f"duration must be a positive number of seconds, got {duration_s}")
    if leeway < 0:
        raise ValueError(f"leeway must not be negative, got {leeway}")

    start = np.datetime64(start, "us")
    whole = int(np.floor(duration_s / timestep + _TIME_TOLERANCE_S))
    pipe = DriftPipeline(forcing, timestep=timestep, leeway=leeway, sigma=0.0)
    states = list(pipe.track(start, lat, to_store_longitude(lon), whole * timestep))
    t = [s.seconds for s in states]
    pos = [s.positions[0] for s in states]

    rest = duration_s - whole * timestep
    if rest > _TIME_TOLERANCE_S:
        tail = DriftPipeline(forcing, timestep=rest, leeway=leeway, sigma=0.0)
        when = start + np.timedelta64(round(t[-1] * 1e6), "us")
        moved = tail.advance(np.atleast_2d(pos[-1]), when)[0]
        t.append(duration_s)
        pos.append(moved[0])

    pos = np.array(pos)
    return MarkerTrack(np.array(t, dtype=float), pos[:, 0], pos[:, 1])


def datum_at(lkp_lat: float, lkp_lon: float, report_time, elapsed_s: float, forcing,
             leeway: float = LEEWAY_COEFFICIENT) -> tuple[float, float]:
    """The datum at arrival: the last known position drifted by current and leeway.

    `leeway` is the TARGET's: 0.02 for a person in water (D002). A drogued drifter buoy is
    built to follow the water, so the site uses 0 for one.
    """
    track = drift_track(lkp_lat, lkp_lon, report_time, elapsed_s, forcing, leeway)
    return float(track.lat[-1]), float(track.lon[-1])


def marker_track(datum_lat: float, datum_lon: float, drop_time,
                 duration_s: float = ON_SCENE_WINDOW_S, forcing=None) -> MarkerTrack:
    """The datum marker from the drop onwards, drifting with the current alone."""
    if forcing is None:
        raise ValueError("marker_track needs a forcing backend; a marker in still water "
                         "is MarkerTrack.fixed")
    return drift_track(datum_lat, datum_lon, drop_time, duration_s, forcing, leeway=0.0)


def _cli(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lat", type=float, required=True, help="last known latitude")
    parser.add_argument("--lon", type=float, required=True, help="last known longitude")
    parser.add_argument("--start", required=True, help="report time, ISO 8601")
    parser.add_argument("--elapsed-min", type=float, required=True,
                        help="minutes from the report to arrival on scene")
    parser.add_argument("--constant-current", nargs=2, type=float, required=True,
                        metavar=("U", "V"), help="uniform surface current, m/s")
    parser.add_argument("--constant-wind", nargs=2, type=float, default=(0.0, 0.0),
                        metavar=("U", "V"), help="uniform 10 m wind, m/s")
    parser.add_argument("--leeway", type=float, default=LEEWAY_COEFFICIENT,
                        help=f"the target's leeway, default {LEEWAY_COEFFICIENT} (D002)")
    parser.add_argument("--window-min", type=float, default=ON_SCENE_WINDOW_S / 60.0,
                        help="time on scene in minutes, default 45 (R7a)")
    args = parser.parse_args(argv)

    forcing = ConstantForcing(current=args.constant_current, wind=args.constant_wind)
    elapsed = args.elapsed_min * 60.0
    lat, lon = datum_at(args.lat, args.lon, args.start, elapsed, forcing, args.leeway)
    drop = np.datetime64(args.start, "us") + np.timedelta64(round(elapsed * 1e6), "us")
    marker = marker_track(lat, lon, drop, args.window_min * 60.0, forcing)
    return {
        "datum": {"lat": lat, "lon": float(to_display_longitude(lon))},
        "marker_end": {"lat": float(marker.lat[-1]),
                       "lon": float(to_display_longitude(marker.lon[-1]))},
        "elapsed_s": elapsed,
        "window_s": args.window_min * 60.0,
        "forcing": forcing.describe(),
        "leeway": args.leeway,
    }


if __name__ == "__main__":
    print(json.dumps(_cli(), indent=2))
