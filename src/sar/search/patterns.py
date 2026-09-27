"""patterns.py: the Coast Guard's search patterns, flown about a drifting marker.

Issues #48 (Expanding Square), #63 (Sector Search), #75 (Parallel Track, Trackline).
These are the doctrinal baseline the paper compares every other searcher against, so they
follow the manual the Coast Guard plans with -- the USCG Addendum, COMDTINST M16130.2F
(2013), cited by page -- rather than a textbook drawing of the shapes.

WHAT A PATTERN IS HERE. Waypoints in metres EAST and NORTH OF A DATUM MARKER, each with
the time the helicopter reaches it at constant search speed. Where the helicopter actually
is, on the ground, is the marker's position at that moment plus the offset (`on_ground`).

WHY RELATIVE TO A MARKER, NOT TO THE GROUND. Addendum p. 3-24: "The objective is to
perform an accurate search pattern relative to the search object ... For aircraft SRUs,
the same effect may be obtained by deploying a smoke float at datum and flying the search
pattern relative to that object." A marker drifts with the water -- marker buoys "are
tools for determining total water current" (p. 3-21) -- so a pattern flown about it keeps
up with the current. It does NOT keep up with leeway: a person in the water is also pushed
by the wind and slowly leaves a marker-centred pattern, about 540 m in 45 minutes at a
10 m/s wind and 2 % leeway, six times the 92.6 m half-width. That gap is real doctrine,
not a modelling shortcut, and it is part of what the paper measures.

WHY WAYPOINTS AND NOT 60 s SAMPLES. At 0.1 NM spacing and 90 kt the Expanding Square's
first leg takes 4 s, and legs outlast a 60 s step only after about 16 minutes. Joining
positions sampled every 60 s would cut straight across the first third of the pattern and
sweep water it never flew over. The sweep must follow the waypoints.

GEOMETRY, from the Addendum:

    Expanding Square (SS), p. 3-26: begins at datum; "the first leg is normally in the
    direction of the search object's drift"; "all course changes are 90 degrees to the
    right"; legs S, S, 2S, 2S, 3S, 3S ... (Figure 3-2). Track spacing S = sweep width for
    coverage 1.0 (p. 3-23).

    Sector Search (VS), pp. 3-27 to 3-28: three equilateral triangles with one corner at
    datum; "all turns in this pattern are 120 degrees to the right"; "all legs of the
    search pattern are equal to the chosen radius", so the aircraft passes straight through
    datum between triangles and one pattern is nine legs, 9R of track (p. 3-28). "A second
    pattern is started with the heading of the new first leg 30 degrees to the right of the
    final course of the first pattern." The Addendum describes the second pattern only;
    applying the same rule to every later pattern is this project's extension, stated so it
    can be argued with.

    Parallel Track (PS), §H.7.3.2 (a), pp. H-117 to H-118 and Figure H-34: straight legs
    "along the major axis" of a rectangle, S apart, the first starting 1/2 S inside a
    corner. The rectangle is centred on the marker with its major axis along the target's
    drift, as §H.7.3.9 directs in a current. Its default size is the square one window can
    cover at coverage 1.0: the search effort Z = W x V x T (p. H-40 (g)), 4.8 km a side.

    Trackline Return (TSR), §H.7.3.1 (b), p. H-116 and Figure H-31: along "the track or
    datum line", starting 1/2 S off it; "the SRU runs up one side and down the other, ending
    one track space from where it began". The line is centred on the marker. With time left
    after one return, each further pass steps out by S on both sides -- this repetition is
    the project's extension, like the Sector Search's.

    Both new patterns begin by flying from the marker to their start point, as a helicopter
    arriving at the datum would.

TURNS ARE INSTANTANEOUS. A real helicopter at 90 kt turns on a radius of several hundred
metres, larger than the Expanding Square's first legs. The manual draws the patterns with
square corners and so does this; it is a limitation, recorded in the vault's register.

CLI. One pattern, printed as JSON:

    python -m sar.search.patterns expanding-square --first-bearing 45
    python -m sar.search.patterns sector --first-bearing 45 --duration-min 18
    python -m sar.search.patterns parallel --first-bearing 30
    python -m sar.search.patterns trackline --first-bearing 30 --half-length-m 6000
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import numpy as np

from sar.search.platform import (
    ON_SCENE_WINDOW_S,
    SEARCH_SPEED_MS,
    SWEEP_WIDTH_M,
    expanding_square_spacing_m,
    search_effort_m2,
    sector_radius_m,
)
from sar.utils.geo import east_north, offset_position

# One Sector Search pattern, as headings relative to its first leg: out, across, back
# through datum, then the same for the next two triangles (Addendum pp. 3-27 to 3-28).
SECTOR_LEGS = (0.0, 120.0, 240.0, 240.0, 0.0, 120.0, 120.0, 240.0, 0.0)
SECTOR_ROTATION_DEG = 30.0

# Absorbs float error when a leg ends exactly at the end of the window.
_TIME_TOLERANCE_S = 1e-9


@dataclass(frozen=True)
class Pattern:
    """A flown pattern: waypoints relative to the marker, and when each is reached."""

    kind: str
    east_m: np.ndarray       # (K,) metres east of the marker, K waypoints, first at 0
    north_m: np.ndarray      # (K,) metres north of the marker
    t_s: np.ndarray          # (K,) seconds from the start, strictly increasing, t_s[0] = 0
    heading_deg: np.ndarray  # (K - 1,) the heading of each leg, degrees true
    speed_ms: float

    @property
    def duration_s(self) -> float:
        return float(self.t_s[-1])

    @property
    def length_m(self) -> float:
        return float(np.hypot(np.diff(self.east_m), np.diff(self.north_m)).sum())

    def _check_time(self, t) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        if np.any(t < -_TIME_TOLERANCE_S) or np.any(t > self.duration_s + _TIME_TOLERANCE_S):
            raise ValueError(f"t must lie within the pattern, 0 to {self.duration_s:g} s")
        return t

    def offset_at(self, t):
        """(east, north) metres from the marker at time t. Scalar or array."""
        t = self._check_time(t)
        return np.interp(t, self.t_s, self.east_m), np.interp(t, self.t_s, self.north_m)

    def heading_at(self, t):
        """The heading of the leg being flown at time t. At a waypoint, the leg leaving it."""
        t = self._check_time(t)
        k = np.searchsorted(self.t_s, t, side="right") - 1
        return self.heading_deg[np.clip(k, 0, self.heading_deg.size - 1)]


@dataclass(frozen=True)
class MarkerTrack:
    """Where the datum marker is, from the moment it is dropped. Longitude 0 to 360."""

    t_s: np.ndarray   # seconds from the drop, strictly increasing
    lat: np.ndarray
    lon: np.ndarray

    def __post_init__(self):
        t = np.asarray(self.t_s, dtype=float)
        if t.ndim != 1 or t.size < 2 or np.any(np.diff(t) <= 0):
            raise ValueError("a marker track needs two or more strictly increasing times")
        if np.shape(self.lat) != t.shape or np.shape(self.lon) != t.shape:
            raise ValueError("marker times, latitudes and longitudes must be the same length")

    @classmethod
    def fixed(cls, lat: float, lon: float, duration_s: float) -> MarkerTrack:
        """A marker that does not move: the ground-stabilised case, kept for tests."""
        return cls(np.array([0.0, float(duration_s)]), np.full(2, float(lat)),
                   np.full(2, float(lon)))

    def at(self, t):
        """(lat, lon) of the marker at t seconds after the drop, interpolated linearly."""
        t = np.asarray(t, dtype=float)
        lo, hi = self.t_s[0] - _TIME_TOLERANCE_S, self.t_s[-1] + _TIME_TOLERANCE_S
        if np.any(t < lo) or np.any(t > hi):
            raise ValueError(f"t must lie within the marker track, {self.t_s[0]:g} to "
                             f"{self.t_s[-1]:g} s")
        return np.interp(t, self.t_s, self.lat), np.interp(t, self.t_s, self.lon)


def _check(name: str, value: float) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive number, got {value}")
    return value


def _fly(kind: str, legs, speed_ms: float, duration_s: float) -> Pattern:
    """Fly (heading, length) legs from the marker until the window ends, mid-leg if need be."""
    east, north, times, headings = [0.0], [0.0], [0.0], []
    for heading, length in legs:
        left = duration_s - times[-1]
        if left <= _TIME_TOLERANCE_S:
            break
        flown = min(length, left * speed_ms)
        de, dn = east_north(heading, flown)
        east.append(east[-1] + float(de))
        north.append(north[-1] + float(dn))
        times.append(times[-1] + flown / speed_ms)
        headings.append(heading % 360.0)
    return Pattern(kind, np.array(east), np.array(north), np.array(times),
                   np.array(headings), speed_ms)


def expanding_square(spacing_m: float = SWEEP_WIDTH_M, first_bearing_deg: float = 0.0,
                     speed_ms: float = SEARCH_SPEED_MS,
                     duration_s: float = ON_SCENE_WINDOW_S) -> Pattern:
    """The Expanding Square (SS): legs S, S, 2S, 2S, ..., every turn 90 degrees right."""
    spacing_m = _check("spacing_m", spacing_m)
    speed_ms = _check("speed_ms", speed_ms)
    duration_s = _check("duration_s", duration_s)

    def legs():
        k = 0
        while True:
            yield first_bearing_deg + 90.0 * k, (k // 2 + 1) * spacing_m
            k += 1

    return _fly("expanding_square", legs(), speed_ms, duration_s)


def sector_search(radius_m: float | None = None, first_bearing_deg: float = 0.0,
                  speed_ms: float = SEARCH_SPEED_MS,
                  duration_s: float = ON_SCENE_WINDOW_S) -> Pattern:
    """The Sector Search (VS): nine legs of R per pattern, each pattern 30 degrees right."""
    speed_ms = _check("speed_ms", speed_ms)
    radius_m = _check("radius_m", sector_radius_m(speed_ms) if radius_m is None else radius_m)
    duration_s = _check("duration_s", duration_s)

    def legs():
        p = 0
        while True:
            base = first_bearing_deg + SECTOR_ROTATION_DEG * p
            for turn in SECTOR_LEGS:
                yield base + turn, radius_m
            p += 1

    return _fly("sector_search", legs(), speed_ms, duration_s)


def _legs_through(points):
    """(heading, length) legs visiting each point in turn, starting from the marker."""
    here = (0.0, 0.0)
    for east, north in points:
        de, dn = east - here[0], north - here[1]
        length = float(np.hypot(de, dn))
        if length > 1e-9:
            yield float(np.degrees(np.arctan2(de, dn)) % 360.0), length
        here = (east, north)


def _axes(bearing_deg: float):
    """Unit vectors along a bearing and across it to the right, as (east, north)."""
    ue, un = east_north(bearing_deg, 1.0)
    ve, vn = east_north(bearing_deg + 90.0, 1.0)
    return (float(ue), float(un)), (float(ve), float(vn))


def _point(u, v, along: float, across: float):
    return along * u[0] + across * v[0], along * u[1] + across * v[1]


def parallel_track(length_m: float | None = None, width_m: float | None = None,
                   spacing_m: float = SWEEP_WIDTH_M, first_bearing_deg: float = 0.0,
                   speed_ms: float = SEARCH_SPEED_MS,
                   duration_s: float = ON_SCENE_WINDOW_S) -> Pattern:
    """Parallel Track (PS): legs along the major axis, S apart, over a rectangle on the marker.

    `first_bearing_deg` is the major axis. Length and width default to the square one window
    covers at coverage 1.0 (the search effort Z = W x V x T).
    """
    spacing_m = _check("spacing_m", spacing_m)
    speed_ms = _check("speed_ms", speed_ms)
    duration_s = _check("duration_s", duration_s)
    side = float(np.sqrt(search_effort_m2(spacing_m, speed_ms, duration_s)))
    length_m = _check("length_m", side if length_m is None else length_m)
    width_m = _check("width_m", side if width_m is None else width_m)
    if length_m < spacing_m or width_m < spacing_m:
        raise ValueError("a parallel track area must be at least one track spacing each way")

    u, v = _axes(first_bearing_deg)
    # Nearest, not floor: floor left the far strip unflown when the width was 25.98 S.
    legs = max(1, int(np.floor(width_m / spacing_m + 0.5)))
    near, far = -length_m / 2 + spacing_m / 2, length_m / 2 - spacing_m / 2

    def points():
        for k in range(legs):
            across = -width_m / 2 + spacing_m / 2 + k * spacing_m
            start, end = (near, far) if k % 2 == 0 else (far, near)
            yield _point(u, v, start, across)
            yield _point(u, v, end, across)

    return _fly("parallel_track", _legs_through(points()), speed_ms, duration_s)


def trackline_return(half_length_m: float, spacing_m: float = SWEEP_WIDTH_M,
                     first_bearing_deg: float = 0.0, speed_ms: float = SEARCH_SPEED_MS,
                     duration_s: float = ON_SCENE_WINDOW_S) -> Pattern:
    """Trackline Return (TSR): up one side of a line and down the other, then wider.

    The line runs through the marker along `first_bearing_deg`, `half_length_m` each way.
    Pass k flies both sides at (k + 1/2) S off the line, so the first pass is the Addendum's
    TSR exactly and later passes widen it by S.
    """
    half_length_m = _check("half_length_m", half_length_m)
    spacing_m = _check("spacing_m", spacing_m)
    speed_ms = _check("speed_ms", speed_ms)
    duration_s = _check("duration_s", duration_s)
    u, v = _axes(first_bearing_deg)
    h = half_length_m

    def points():
        k = 0
        while True:
            side = 1.0 if k % 2 == 0 else -1.0
            off = (k + 0.5) * spacing_m
            yield _point(u, v, -h, side * off)
            yield _point(u, v, h, side * off)
            yield _point(u, v, h, -side * off)
            yield _point(u, v, -h, -side * off)
            k += 1

    return _fly("trackline_return", _legs_through(points()), speed_ms, duration_s)


PATTERNS = {"expanding-square": expanding_square, "sector": sector_search,
            "parallel": parallel_track, "trackline": trackline_return}


def on_ground(pattern: Pattern, marker: MarkerTrack, t):
    """The helicopter's (lat, lon) at t seconds into the pattern: the marker plus the offset."""
    mlat, mlon = marker.at(t)
    east, north = pattern.offset_at(t)
    return offset_position(mlat, mlon, east, north)


def ground_track(pattern: Pattern, marker: MarkerTrack, every_s: float = 10.0):
    """The ground path as (t, lat, lon): every waypoint, plus a point every `every_s`.

    Between waypoints the offset is a straight line but the marker keeps drifting, so the
    ground path is only piecewise straight at the scale of the extra points. Ten seconds
    at 90 kt is 463 m of leg, during which a 1.8 m/s marker moves 18 m.
    """
    every_s = _check("every_s", every_s)
    regular = np.arange(0.0, pattern.duration_s, every_s)
    t = np.union1d(regular, pattern.t_s)
    lat, lon = on_ground(pattern, marker, t)
    return t, lat, lon


def describe(pattern: Pattern, sweep_width_m: float = SWEEP_WIDTH_M) -> dict:
    """What a pattern is, in numbers a person can check against the manual."""
    return {
        "kind": pattern.kind,
        "legs": int(pattern.heading_deg.size),
        "duration_s": pattern.duration_s,
        "length_m": pattern.length_m,
        "area_swept_km2_ignoring_overlap": pattern.length_m * sweep_width_m / 1e6,
        "extent_m": {"east": [float(pattern.east_m.min()), float(pattern.east_m.max())],
                     "north": [float(pattern.north_m.min()), float(pattern.north_m.max())]},
        "first_waypoints": [
            {"east_m": float(e), "north_m": float(n), "t_s": float(t)}
            for e, n, t in list(zip(pattern.east_m, pattern.north_m, pattern.t_s))[:6]
        ],
    }


def _cli(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pattern", choices=sorted(PATTERNS))
    parser.add_argument("--first-bearing", type=float, default=0.0,
                        help="first leg, degrees true; doctrine flies it along the drift")
    parser.add_argument("--spacing-m", type=float, default=None,
                        help=f"track spacing for the square, parallel and trackline patterns, "
                             f"default the sweep width "
                             f"{expanding_square_spacing_m():.1f} m (Addendum p. 3-23)")
    parser.add_argument("--radius-m", type=float, default=None,
                        help=f"Sector Search radius, default {sector_radius_m():.0f} m, "
                             "one minute at search speed (Addendum p. 3-23)")
    parser.add_argument("--length-m", type=float, default=None,
                        help="Parallel Track length along its axis, default the effort square")
    parser.add_argument("--width-m", type=float, default=None,
                        help="Parallel Track width across it, default the effort square")
    parser.add_argument("--half-length-m", type=float, default=None,
                        help=f"Trackline half-length each side of the marker, default "
                             f"{sector_radius_m():.0f} m")
    parser.add_argument("--speed-ms", type=float, default=SEARCH_SPEED_MS,
                        help=f"search speed, default {SEARCH_SPEED_MS:.2f} m/s (90 kt)")
    parser.add_argument("--duration-min", type=float, default=ON_SCENE_WINDOW_S / 60.0,
                        help="time on scene in minutes, default 45 (R7a)")
    args = parser.parse_args(argv)

    duration_s = args.duration_min * 60.0
    if args.pattern == "expanding-square":
        spacing = expanding_square_spacing_m() if args.spacing_m is None else args.spacing_m
        pattern = expanding_square(spacing, args.first_bearing, args.speed_ms, duration_s)
    elif args.pattern == "sector":
        pattern = sector_search(args.radius_m, args.first_bearing, args.speed_ms, duration_s)
    elif args.pattern == "parallel":
        spacing = expanding_square_spacing_m() if args.spacing_m is None else args.spacing_m
        pattern = parallel_track(args.length_m, args.width_m, spacing, args.first_bearing,
                                 args.speed_ms, duration_s)
    else:
        spacing = expanding_square_spacing_m() if args.spacing_m is None else args.spacing_m
        half = sector_radius_m() if args.half_length_m is None else args.half_length_m
        pattern = trackline_return(half, spacing, args.first_bearing, args.speed_ms, duration_s)
    return describe(pattern)


if __name__ == "__main__":
    print(json.dumps(_cli(), indent=2))
