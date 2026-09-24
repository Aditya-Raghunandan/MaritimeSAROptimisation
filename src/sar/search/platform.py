"""platform.py: the search helicopter's numbers, each with its source (issue #44).

ADR002 asks for exactly one Python home for the sweep width and never a second; this is
it, together with everything else the doctrinal search needs to turn a 45-minute window
into a track: how fast the helicopter searches, how fast it transits, how far it can go,
and how long it takes to get airborne. The frontend reads the same numbers through
`frontend/src/fixtures/search_golden.json` (`scripts/export_search_golden.py`), and a test
holds `frontend/src/geo.js`'s `SWEEP_WIDTH_M` to the value here.

THE SOURCE. Everything except the step and the on-scene window comes from one document,
cited below by table and printed page:

    U.S. Coast Guard, "U.S. Coast Guard Addendum to the United States National Search and
    Rescue Supplement (NSS) to the International Aeronautical and Maritime Search and
    Rescue Manual (IAMSAR)", COMDTINST M16130.2F, 7 January 2013.

It is the manual the Coast Guard plans its own searches with, which is why it and not a
secondary source sets the baseline the paper compares against. The vault decision is D027;
the engineering record is docs/ADR003.md.

WHAT THE SWEEP WIDTH ASSUMES. 0.1 NM is the Addendum's *uncorrected* figure, and the
corrections for this scenario are all 1.0 only under these conditions: a person in the
water, from a helicopter at 300 to 1,000 ft, visibility at least 3 NM, winds up to 15 kt or
seas up to 3 ft, at 90 kt, normal crew fatigue, and no flotation device. Stronger wind
halves it (Table H-10) and a lifejacket multiplies it by four (p. H-41). Both are
scenario-dependent corrections that this module does not apply.

CLI. Prints every constant with its source, and what one on-scene window allows:

    python -m sar.search.platform
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from sar.model.position import EARTH_RADIUS_M

# The international nautical mile, exact by definition (1929), and the knot it defines.
NM_M = 1852.0
KNOT_MS = NM_M / 3600.0

# Addendum App. H, Tables H-15 and H-16, p. H-44, "Person in Water", uncorrected visual
# sweep width from a helicopter: 0.1 NM at 300, 500, 750 and 1,000 ft for visibility 3 NM
# and above (and at 1 NM from 300 ft). Correction factors 1.0: Table H-9 p. H-40 (90 kt),
# Table H-10 p. H-41 (winds 0-15 kt or seas 0-3 ft), p. H-41(c) (normal fatigue). Table H-8
# p. H-40 recommends 200-500 ft for persons over water.
SWEEP_WIDTH_M = 0.1 * NM_M

# Addendum Table H-9, p. H-40: the helicopter speed at which the correction factor for a
# person in water is 1.0, and the speed of the worked example on p. 3-23. The Addendum
# gives no MH-60T-specific search speed; this is the speed its sweep widths are stated at.
SEARCH_SPEED_MS = 90.0 * KNOT_MS

# Addendum Table 5-3, p. 5-17, "H-60": cruise speed 125 KTAS, radius of action 300 NM,
# maximum endurance 6 h. The manufacturer quotes 170 kt cruise; the SAR manual's figure is
# used because it is the one search planners are told to plan with.
TRANSIT_SPEED_MS = 125.0 * KNOT_MS
RADIUS_OF_ACTION_M = 300.0 * NM_M
MAX_ENDURANCE_S = 6.0 * 3600.0

# Addendum, "SAR readiness", p. PPO-7: units maintain B-0 readiness, "a suitable SAR
# resource ready to proceed within 30 minutes of notification of a distress".
LAUNCH_DELAY_S = 30.0 * 60.0

# A project requirement, not a published figure: R7a/R7b fix a 45-minute on-scene window,
# which ADR002's time axis turns into an episode of exactly 45 steps.
ON_SCENE_WINDOW_S = 45.0 * 60.0

# D009 and ADR002: the integration step, and the search agent's decision interval.
STEP_S = 60.0

SOURCES = {
    "sweep_width_m": "Addendum App. H Tables H-15/H-16 p. H-44: PIW, helicopter, 0.1 NM",
    "search_speed_ms": "Addendum Table H-9 p. H-40: 90 kt, speed correction 1.0",
    "transit_speed_ms": "Addendum Table 5-3 p. 5-17: H-60 cruise 125 KTAS",
    "radius_of_action_m": "Addendum Table 5-3 p. 5-17: H-60 radius of action 300 NM",
    "max_endurance_s": "Addendum Table 5-3 p. 5-17: H-60 maximum endurance 6 h",
    "launch_delay_s": "Addendum p. PPO-7: B-0 readiness, ready within 30 minutes",
    "on_scene_window_s": "project requirement R7a/R7b (45 minutes)",
    "step_s": "D009 / ADR002 (60 s)",
}


def track_length_m(speed_ms: float = SEARCH_SPEED_MS,
                   window_s: float = ON_SCENE_WINDOW_S) -> float:
    """How far the helicopter flies searching in one on-scene window."""
    if speed_ms <= 0 or window_s <= 0:
        raise ValueError(f"speed and window must be positive, got {speed_ms} and {window_s}")
    return float(speed_ms * window_s)


def expanding_square_spacing_m(sweep_width_m: float = SWEEP_WIDTH_M) -> float:
    """Track spacing for an Expanding Square: the sweep width itself.

    Addendum p. 3-23 (a)-(b): the default initial search has an average coverage of 1.0,
    which "for an expanding square search (SS) ... means the track spacing should equal
    the sweep width".
    """
    if sweep_width_m <= 0:
        raise ValueError(f"sweep width must be positive, got {sweep_width_m}")
    return float(sweep_width_m)


def sector_radius_m(speed_ms: float = SEARCH_SPEED_MS,
                    sweep_width_m: float = SWEEP_WIDTH_M) -> float:
    """Minimum Sector Search radius for an aircraft.

    Addendum p. 3-23 (d): "the distance the aircraft can cover in one minute at search
    speed, or twice the sweep width, whichever is larger" -- 1.5 NM at 90 kt.
    """
    if speed_ms <= 0 or sweep_width_m <= 0:
        raise ValueError(f"speed and sweep width must be positive, got {speed_ms} "
                         f"and {sweep_width_m}")
    return float(max(speed_ms * 60.0, 2.0 * sweep_width_m))


def great_circle_m(lat1, lon1, lat2, lon2):
    """Great-circle distance in metres by the haversine formula. Scalar or array.

    The same formula and radius as `distance` in `frontend/src/geo.js`, so the site and
    this module agree on how far a base is from a datum.
    """
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(np.asarray(lon2, dtype=float) - np.asarray(lon1, dtype=float))
    h = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2.0 * EARTH_RADIUS_M * np.arcsin(np.minimum(1.0, np.sqrt(h)))


def transit_time_s(distance_m: float, speed_ms: float = TRANSIT_SPEED_MS,
                   launch_delay_s: float = LAUNCH_DELAY_S) -> float:
    """Seconds from the call to arriving on scene: the launch delay, then the flight out.

    Refuses a datum beyond the H-60's radius of action, because a search the helicopter
    could not fly is not a baseline.
    """
    if distance_m < 0:
        raise ValueError(f"distance must not be negative, got {distance_m}")
    if distance_m > RADIUS_OF_ACTION_M:
        raise ValueError(f"datum is {distance_m / NM_M:.0f} NM from the base, beyond the "
                         f"H-60's {RADIUS_OF_ACTION_M / NM_M:.0f} NM radius of action")
    if speed_ms <= 0:
        raise ValueError(f"transit speed must be positive, got {speed_ms}")
    return float(launch_delay_s + distance_m / speed_ms)


def constants() -> dict:
    """Every constant by name, for the CLI and the frontend fixture."""
    return {
        "nm_m": NM_M,
        "knot_ms": KNOT_MS,
        "sweep_width_m": SWEEP_WIDTH_M,
        "search_speed_ms": SEARCH_SPEED_MS,
        "transit_speed_ms": TRANSIT_SPEED_MS,
        "radius_of_action_m": RADIUS_OF_ACTION_M,
        "max_endurance_s": MAX_ENDURANCE_S,
        "launch_delay_s": LAUNCH_DELAY_S,
        "on_scene_window_s": ON_SCENE_WINDOW_S,
        "step_s": STEP_S,
    }


def _cli(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.parse_args(argv)
    length = track_length_m()
    return {
        "constants": constants(),
        "sources": SOURCES,
        "one_window": {
            "track_length_m": length,
            "track_length_nm": length / NM_M,
            "area_swept_km2": length * SWEEP_WIDTH_M / 1e6,
            "expanding_square_spacing_m": expanding_square_spacing_m(),
            "sector_radius_m": sector_radius_m(),
        },
    }


if __name__ == "__main__":
    print(json.dumps(_cli(), indent=2))
