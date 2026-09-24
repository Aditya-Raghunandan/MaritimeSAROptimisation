"""export_search_golden.py: writes frontend/src/fixtures/search_golden.json (issues #44, #48).

The doctrinal search lives twice: in `src/sar/search/` for the paper, and in the browser
for the site, which is static and has no server to ask. Two implementations of the same
pattern drift apart silently, and a pattern that is subtly wrong still looks like a
pattern on screen. So Python is the reference, as it is for the resultant vector
(`scripts/export_resultant_golden.py`): this runs it over a handful of cases and records
the inputs and the answers, and the frontend tests feed the JavaScript the same inputs and
assert the same answers.

What it records:

    platform   every constant in `sar.search.platform`, so `frontend/src/geo.js` cannot
               quote a different sweep width (issue #44's parity criterion)
    patterns   Expanding Square and Sector Search waypoints, and offsets and headings
               sampled along them, at several first bearings and windows, including ones
               that end mid-leg
    on_ground  a pattern carried by a drifting marker at 17, 26.5 and 36 N
    drift      one point drifted by `sar.search.datum.drift_track` under constant forcing,
               including a ragged duration that ends with a short step
    transit    base-to-datum distances and the time to arrive

Longitude is written in the frontend's DISPLAY convention, -180 to 180.

    python scripts/export_search_golden.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sar.pipeline.forcing import ConstantForcing
from sar.search.datum import drift_track, marker_track
from sar.search.patterns import expanding_square, on_ground, sector_search
from sar.search.platform import (
    NM_M,
    SEARCH_SPEED_MS,
    SWEEP_WIDTH_M,
    constants,
    great_circle_m,
    transit_time_s,
)
from sar.utils.geo import to_display_longitude

OUT = Path("frontend/src/fixtures/search_golden.json")
START = "2019-06-01T06:00"
V, S = SEARCH_SPEED_MS, SWEEP_WIDTH_M

PATTERN_CASES = [
    ("expanding_square", {"spacing_m": S, "first_bearing_deg": 0.0, "duration_s": 600.0}),
    ("expanding_square", {"spacing_m": S, "first_bearing_deg": 37.0, "duration_s": 2700.0}),
    ("expanding_square", {"spacing_m": S, "first_bearing_deg": 290.0,
                          "duration_s": 1.5 * S / V}),
    ("sector_search", {"radius_m": None, "first_bearing_deg": 0.0, "duration_s": 540.0}),
    ("sector_search", {"radius_m": None, "first_bearing_deg": 10.0, "duration_s": 1080.0}),
    ("sector_search", {"radius_m": 1000.0, "first_bearing_deg": 200.0, "duration_s": 700.0}),
]

BUILDERS = {"expanding_square": expanding_square, "sector_search": sector_search}


def disp(lon) -> list[float]:
    return [float(x) for x in np.atleast_1d(to_display_longitude(lon))]


def pattern_case(kind: str, args: dict) -> dict:
    p = BUILDERS[kind](speed_ms=V, **args)
    t = np.linspace(0.0, p.duration_s, 13)
    east, north = p.offset_at(t)
    return {
        "kind": kind,
        "args": {**args, "speed_ms": V},
        "waypoints": {"east_m": p.east_m.tolist(), "north_m": p.north_m.tolist(),
                      "t_s": p.t_s.tolist(), "heading_deg": p.heading_deg.tolist()},
        "samples": {"t_s": t.tolist(), "east_m": east.tolist(), "north_m": north.tolist(),
                    "heading_deg": p.heading_at(t).tolist()},
        "length_m": p.length_m,
    }


def on_ground_case(lat: float, lon: float) -> dict:
    p = expanding_square(S, 45.0, V, 600.0)
    forcing = ConstantForcing(current=(1.8, 0.3), wind=(0.0, 0.0))
    marker = marker_track(lat, lon, START, p.duration_s, forcing)
    t = np.linspace(0.0, p.duration_s, 11)
    glat, glon = on_ground(p, marker, t)
    return {
        "pattern": {"kind": "expanding_square",
                    "args": {"spacing_m": S, "first_bearing_deg": 45.0, "speed_ms": V,
                             "duration_s": 600.0}},
        "marker": {"t_s": marker.t_s.tolist(), "lat": marker.lat.tolist(),
                   "lon": disp(marker.lon)},
        "t_s": t.tolist(), "lat": glat.tolist(), "lon": disp(glon),
    }


def drift_case(lat, lon, current, wind, leeway, duration_s) -> dict:
    track = drift_track(lat, lon, START, duration_s, ConstantForcing(current, wind), leeway)
    return {
        "start": {"lat": lat, "lon": float(to_display_longitude(lon))},
        "current_ms": list(current), "wind_ms": list(wind), "leeway": leeway,
        "duration_s": duration_s,
        "track": {"t_s": track.t_s.tolist(), "lat": track.lat.tolist(),
                  "lon": disp(track.lon)},
    }


def build() -> dict:
    base = (25.8, -80.1)   # a base near Miami, display longitude
    datums = [(25.8, -79.5), (26.9, -78.0), (29.0, -76.5)]
    return {
        "generated_by": "scripts/export_search_golden.py",
        "reference": "src/sar/search/",
        "note": ("Python is the reference implementation. The frontend tests assert the "
                 "browser port reproduces every case here. Regenerate with: "
                 "python scripts/export_search_golden.py"),
        "tolerance_m": 1e-6,
        "tolerance_deg": 1e-10,
        "tolerance_s": 1e-6,
        "platform": constants(),
        "patterns": [pattern_case(kind, args) for kind, args in PATTERN_CASES],
        "on_ground": [on_ground_case(lat, 281.0) for lat in (17.0, 26.5, 36.0)],
        "drift": [
            drift_case(26.5, -79.0, (1.0, 0.0), (0.0, 0.0), 0.0, 600.0),
            drift_case(26.5, -79.0, (1.0, 0.0), (0.0, 0.0), 0.0, 90.0),
            drift_case(26.5, -79.0, (1.8, 0.0), (5.0, 5.0), 0.02, 4620.0),
            drift_case(35.5, -75.0, (0.4, 1.1), (-8.0, 3.0), 0.02, 3000.0),
        ],
        "transit": [
            {"base": {"lat": base[0], "lon": base[1]}, "datum": {"lat": la, "lon": lo},
             "distance_m": float(great_circle_m(base[0], base[1], la, lo)),
             "time_s": transit_time_s(float(great_circle_m(base[0], base[1], la, lo)))}
            for la, lo in datums
        ],
        "radius_of_action_nm": 300.0,
        "nm_m": NM_M,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Write the search golden fixture for the browser")
    p.add_argument("--out", default=str(OUT))
    args = p.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = build()
    out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"wrote {out}\n  {len(payload['patterns'])} patterns, "
          f"{len(payload['on_ground'])} on-ground, {len(payload['drift'])} drift, "
          f"{len(payload['transit'])} transit cases")


if __name__ == "__main__":
    main()
