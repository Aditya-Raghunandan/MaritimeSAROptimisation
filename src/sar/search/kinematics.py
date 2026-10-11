"""kinematics.py: how the helicopter moves. It cannot turn on the spot (D032, docs/ADR006.md).

Until D032 every searcher turned instantly: a heading searcher could reverse between two
minutes and the Coast Guard's patterns were flown with square corners. A real helicopter at
90 kt turns as an aeroplane does, on an arc. This module is the one place that says how.

THE MODEL: A DUBINS VEHICLE. The helicopter flies at a constant speed V about the datum
marker (ADR004 row 3) and its heading changes at most omega degrees a second, so the
tightest path it can fly is an arc of radius r = V / omega (Dubins 1957: such a vehicle's
paths are arcs of radius r and straight lines). omega is 7.0 deg/s, a 30 degree bank at
90 kt, r = 379 m (`sar.search.platform`, D032). An infinite omega is the old referee, kept
so "as the manual draws it" stays a reported comparison.

TWO WAYS A PATH IS MADE.

    steer    A COMMAND: turn to a heading (greedy, the random floor, a person's keys), or
             turn by so many degrees, right positive (the trained agent's action). The
             helicopter turns at omega until the command is met, then flies straight. A
             heading command turns the short way round, and right at exactly 180 degrees.

    follow   A DRAWN PATH, a pattern's legs from the manual, flown by an autopilot: the L1
             guidance law of Park, Deyst and How (2004), the path follower ArduPilot and
             PX4 fly. Each instant it picks the point where the drawn path leaves a circle
             of radius L1 about the helicopter, and turns toward it at a rate proportional
             to the sine of the angle off its nose: 2 V sin(eta) / L1, never faster than
             omega. Nothing is planned ahead; the path emerges from correcting toward that
             moving point. Corners within its reach are rounded; legs closer together
             than the turn diameter (the Parallel Track's 185 m against 757 m) are joined
             by a loop. L1 = r, so a 90 degree corner starts to turn where a fly-by turn of
             radius r would (r tan 45 = r).

CHORDS. The referee sweeps straight sub-legs (ADR004 row 2), so an arc is cut into pieces
of at most ARC_PIECE_DEG of turn, each swept as its chord. The chord of a 10 degree piece
lies within r (1 - cos 5 deg) = 1.4 m of the arc at r = 379 m, against a 92.6 m half-width.

THE REFEREE'S CHECK, `check_turns`. As it already refuses a path faster than the
helicopter, the referee refuses one that turns faster than it, or that changes speed.
Every waypoint carries the heading at that moment, and each chord must be one a helicopter
at speed V turning at most omega could fly: see `check_turns`. A square corner fails.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass

import numpy as np

from sar.utils.geo import east_north

# The most turn swept as one straight chord: 1.4 m from the arc at r = 379 m.
ARC_PIECE_DEG = 10.0
# The autopilot's clock: how often it re-aims. Its arcs are integrated exactly in between.
AUTOPILOT_DT_S = 0.5
# A chord may turn no more than this, so its direction is never ambiguous.
MAX_CHORD_TURN_DEG = 30.0
# Below this a turn rate is no turn (rad/s): the autopilot's residue on a straight leg.
_STRAIGHT_RAD_S = 1e-12
# The check's slack: floating point, never a loophole of any size that matters.
_REL_TOL = 1e-9
_ABS_TOL_DEG = 1e-6
_ABS_TOL_M = 1e-6


def is_instant(turn_rate_deg_s: float) -> bool:
    """True for the old referee's helicopter, which turns at once."""
    return math.isinf(turn_rate_deg_s)


def check_turn_rate(turn_rate_deg_s) -> float:
    rate = float(turn_rate_deg_s)
    if math.isnan(rate) or rate <= 0.0:
        raise ValueError(f"the turn rate must be positive degrees a second (or inf for a "
                         f"helicopter that turns at once), got {turn_rate_deg_s!r}")
    return rate


def normalise(heading_deg: float) -> float:
    """A heading in [0, 360)."""
    h = float(heading_deg) % 360.0
    return 0.0 if h == 360.0 else h


def shortest_turn(from_deg: float, to_deg: float) -> float:
    """The turn from one heading to another the short way, in (-180, 180]: right positive.

    Exactly opposite is +180, a turn to the right, so a command always means one path.
    """
    d = (float(to_deg) - float(from_deg)) % 360.0
    return d - 360.0 if d > 180.0 else d


def bearing(de: float, dn: float) -> float:
    """The compass bearing of an (east, north) step, degrees in [0, 360)."""
    return normalise(math.degrees(math.atan2(de, dn)))


def arc_point(east: float, north: float, heading_deg: float, rate_deg_s: float, t: float,
              speed_ms: float) -> tuple[float, float, float]:
    """Where a helicopter turning at a constant rate is after t seconds: (east, north, heading).

    Exact: with psi(t) = psi0 + w t the helicopter ends on the chord of its arc, which points
    along psi0 + w t / 2 and is V t sin(w t / 2) / (w t / 2) long. Written as a chord rather
    than as (V / w)(cos psi0 - cos psi), it stays exact as w goes to 0, where V / w would
    multiply a cancellation error by a huge number; at w = 0 it is the straight line.
    """
    psi0 = math.radians(heading_deg)
    half = math.radians(rate_deg_s) * t / 2.0
    length = speed_ms * t * (math.sin(half) / half if half != 0.0 else 1.0)
    along = psi0 + half
    return (east + length * math.sin(along), north + length * math.cos(along),
            normalise(heading_deg + rate_deg_s * t))


@dataclass(frozen=True)
class Path:
    """A flown path: chord ends in time, metres about the marker, and the heading there."""

    t_s: np.ndarray
    east_m: np.ndarray
    north_m: np.ndarray
    heading_deg: np.ndarray


def steer(east: float, north: float, heading_deg: float, turn_deg: float, duration_s: float,
          speed_ms: float, turn_rate_deg_s: float) -> Path:
    """Turn by `turn_deg` (right positive) at the turn rate, then fly straight, for a while.

    Returns the chord ends after the start, at t in (0, duration_s]: the arc cut into equal
    pieces of at most ARC_PIECE_DEG, then one straight piece if any time is left. A turn the
    time is too short for is flown for all of it and the rest is not carried over.
    """
    duration_s = float(duration_s)
    if not duration_s > 0.0:
        raise ValueError(f"a steer needs a positive duration, got {duration_s}")
    turn_deg = float(turn_deg)
    if not math.isfinite(turn_deg):
        raise ValueError(f"a turn must be a finite number of degrees, got {turn_deg!r}")
    if is_instant(turn_rate_deg_s):
        h = normalise(heading_deg + turn_deg)
        de, dn = east_north(h, speed_ms * duration_s)
        return Path(np.array([duration_s]), np.array([east + float(de)]),
                    np.array([north + float(dn)]), np.array([h]))

    omega = float(turn_rate_deg_s)
    sign = 1.0 if turn_deg > 0.0 else -1.0
    full = abs(turn_deg) / omega                     # seconds the whole turn would take
    tau = min(full, duration_s)
    t, e, n, h = [], [], [], []
    if tau > 0.0:
        turned = abs(turn_deg) if full <= duration_s else omega * duration_s
        pieces = max(1, math.ceil(turned / ARC_PIECE_DEG - 1e-9))
        for k in range(1, pieces + 1):
            tk = tau * k / pieces
            ek, nk, hk = arc_point(east, north, heading_deg, sign * omega, tk, speed_ms)
            t.append(tk)
            e.append(ek)
            n.append(nk)
            h.append(hk)
        if full <= duration_s:
            h[-1] = normalise(heading_deg + turn_deg)     # exact, not w x tau rounded
    if tau < duration_s:
        e0, n0, h0 = (e[-1], n[-1], h[-1]) if t else (east, north, normalise(heading_deg))
        ek, nk, hk = arc_point(e0, n0, h0, 0.0, duration_s - tau, speed_ms)
        t.append(duration_s)
        e.append(ek)
        n.append(nk)
        h.append(hk)
    return Path(np.array(t), np.array(e), np.array(n), np.array(h))


def steer_to(east: float, north: float, heading_deg: float, to_heading_deg: float,
             duration_s: float, speed_ms: float, turn_rate_deg_s: float) -> Path:
    """Turn to a heading the short way round, then straight: a heading command."""
    return steer(east, north, heading_deg, shortest_turn(heading_deg, to_heading_deg),
                 duration_s, speed_ms, turn_rate_deg_s)


# -- the autopilot ------------------------------------------------------------------------

@dataclass(frozen=True)
class Flown:
    """A drawn path flown by the autopilot: its state every AUTOPILOT_DT_S.

    rate_deg_s[k] is the turn rate held from t_s[k] to t_s[k + 1].
    """

    t_s: np.ndarray
    east_m: np.ndarray
    north_m: np.ndarray
    heading_deg: np.ndarray
    rate_deg_s: np.ndarray

    def path(self, t0: float, t1: float) -> Path:
        """The chords flown in (t0, t1], each turning one way and at most ARC_PIECE_DEG."""
        a = _tick(self.t_s, t0)
        b = _tick(self.t_s, t1)
        keep = chord_ends(self.rate_deg_s, a, b, self.t_s[1] - self.t_s[0])
        return Path(self.t_s[keep], self.east_m[keep], self.north_m[keep], self.heading_deg[keep])


def _tick(t_s: np.ndarray, t: float) -> int:
    k = int(round(t / (t_s[1] - t_s[0])))
    if k < 0 or k >= t_s.size or abs(t_s[k] - t) > 1e-6:
        raise ValueError(f"{t:g} s is not one of the autopilot's ticks")
    return k


def _sign(rate_deg_s: float) -> int:
    if abs(math.radians(rate_deg_s)) <= _STRAIGHT_RAD_S:
        return 0
    return 1 if rate_deg_s > 0 else -1


def chord_ends(rate_deg_s, a: int, b: int, dt_s: float) -> list[int]:
    """Which ticks after `a` up to `b` to keep, so each chord turns one way, at most a piece.

    A chord grows while its turn keeps one sign (straight joins either) and its total turn
    stays within ARC_PIECE_DEG; `b` is always kept.
    """
    keep = []
    sign, turned = 0, 0.0
    for k in range(a, b):
        s = _sign(rate_deg_s[k])
        d = abs(rate_deg_s[k]) * dt_s
        if k > a and ((s != 0 and sign != 0 and s != sign) or turned + d > ARC_PIECE_DEG + 1e-9):
            keep.append(k)
            sign, turned = 0, 0.0
        if s != 0:
            sign = s
        turned += d
    keep.append(b)
    return keep


class _Polyline:
    """A drawn path as straight segments, measured by distance along it."""

    def __init__(self, east_m, north_m):
        pts = [(float(e), float(n)) for e, n in zip(east_m, north_m)]
        self.p = [pts[0]]
        for q in pts[1:]:
            if math.hypot(q[0] - self.p[-1][0], q[1] - self.p[-1][1]) > 1e-9:
                self.p.append(q)
        if len(self.p) < 2:
            raise ValueError("a path to follow needs at least two distinct points")
        self.d = []          # unit direction of each segment
        self.length = []
        self.cum = [0.0]     # distance along the path to the start of each segment
        for (ax, ay), (bx, by) in zip(self.p[:-1], self.p[1:]):
            L = math.hypot(bx - ax, by - ay)
            self.d.append(((bx - ax) / L, (by - ay) / L))
            self.length.append(L)
            self.cum.append(self.cum[-1] + L)
        self.total = self.cum[-1]

    def segment_at(self, s: float) -> int:
        """The segment holding distance s: the last whose start is at or before it."""
        return min(max(bisect_right(self.cum, s) - 1, 0), len(self.length) - 1)

    def point(self, s: float) -> tuple[float, float]:
        s = min(max(s, 0.0), self.total)
        i = self.segment_at(s)
        u = s - self.cum[i]
        return self.p[i][0] + u * self.d[i][0], self.p[i][1] + u * self.d[i][1]

    def project(self, x: float, y: float, s_lo: float, s_hi: float) -> tuple[float, float]:
        """The nearest point to (x, y) between s_lo and s_hi along the path: (s, distance)."""
        s_hi = min(s_hi, self.total)
        best_s, best_d = s_lo, math.inf
        i = self.segment_at(s_lo)
        while i < len(self.length) and self.cum[i] <= s_hi:
            lo = max(s_lo, self.cum[i]) - self.cum[i]
            hi = min(s_hi, self.cum[i + 1]) - self.cum[i]
            ax, ay = self.p[i]
            dx, dy = self.d[i]
            u = min(max((x - ax) * dx + (y - ay) * dy, lo), hi)
            dist = math.hypot(x - (ax + u * dx), y - (ay + u * dy))
            if dist < best_d:
                best_s, best_d = self.cum[i] + u, dist
            i += 1
        return best_s, best_d

    def exit(self, x: float, y: float, s0: float, radius: float) -> float:
        """The first point after s0 where the path leaves the circle of `radius` about (x, y).

        The end of the path if it never leaves; s0 itself if s0 is already outside.
        """
        i = self.segment_at(s0)
        u0 = s0 - self.cum[i]
        while i < len(self.length):
            ax, ay = self.p[i]
            dx, dy = self.d[i]
            # |A + u D|^2 = radius^2 with A = start - helicopter: u^2 + 2 b u + c = 0.
            b = (ax - x) * dx + (ay - y) * dy
            c = (ax - x) ** 2 + (ay - y) ** 2 - radius * radius
            disc = b * b - c
            if disc < 0.0:
                return self.cum[i] + u0                      # this segment is all outside
            u_out = -b + math.sqrt(disc)
            if u_out < u0:
                return self.cum[i] + u0                      # already outside at u0
            if u_out <= self.length[i]:
                return self.cum[i] + u_out
            i += 1
            u0 = 0.0
        return self.total


def follow(east_m, north_m, heading_deg: float, duration_s: float, speed_ms: float,
           turn_rate_deg_s: float, l1_m: float | None = None,
           dt_s: float = AUTOPILOT_DT_S) -> Flown:
    """Fly a drawn path from its first point with the L1 autopilot, for `duration_s`.

    Each tick: (1) where along the path the helicopter is, the nearest point a little ahead
    of the last; (2) the reference point, where the path leaves a circle of radius L1 about
    the helicopter (the nearest point if the path is farther than L1); (3) eta, the angle
    from the nose to that point; (4) turn at 2 V sin(eta) / L1, full rate when it is behind
    (|eta| > 90 deg), never faster than omega; (5) fly the tick's arc exactly. At the end of
    the path it keeps turning toward the last point, circling it.
    """
    if is_instant(turn_rate_deg_s):
        raise ValueError("an instantly turning helicopter flies the drawn path itself")
    omega = check_turn_rate(turn_rate_deg_s)
    speed_ms = float(speed_ms)
    if not speed_ms > 0.0:
        raise ValueError(f"the autopilot needs a positive speed, got {speed_ms}")
    line = _Polyline(east_m, north_m)
    l1 = speed_ms / math.radians(omega) if l1_m is None else float(l1_m)
    ticks = int(round(duration_s / dt_s))
    if ticks < 1 or abs(ticks * dt_s - duration_s) > 1e-9:
        raise ValueError(f"the duration must be whole ticks of {dt_s:g} s, got {duration_s:g}")

    x, y = line.p[0]
    psi = normalise(heading_deg)
    s = 0.0
    window = 2.0 * l1 + speed_ms * dt_s
    t_s = [0.0]
    es, ns, hs, rates = [x], [y], [psi], []
    for k in range(ticks):
        s, dist = line.project(x, y, s, s + window)
        ref = line.point(line.exit(x, y, s, l1)) if dist < l1 else line.point(s)
        de, dn = ref[0] - x, ref[1] - y
        if math.hypot(de, dn) <= 1e-9:
            rate = 0.0
        else:
            eta = math.radians(shortest_turn(psi, bearing(de, dn)))
            push = math.sin(eta) if abs(eta) <= math.pi / 2 else math.copysign(1.0, eta)
            rate = math.degrees(2.0 * speed_ms * push / l1)
            rate = max(-omega, min(omega, rate))
        x, y, psi = arc_point(x, y, psi, rate, dt_s, speed_ms)
        t_s.append((k + 1) * dt_s)
        es.append(x)
        ns.append(y)
        hs.append(psi)
        rates.append(rate)
    rates.append(0.0)
    return Flown(np.array(t_s), np.array(es), np.array(ns), np.array(hs), np.array(rates))


# -- the referee's check -------------------------------------------------------------------

def check_turns(t_s, east_m, north_m, heading_deg, start: tuple[float, float, float],
                speed_ms: float, turn_rate_deg_s: float) -> None:
    """Refuse a path the helicopter could not fly at its speed and turn rate. Raises ValueError.

    `start` is (east, north, heading) at time 0; the waypoints follow at t_s > 0, each with
    the heading at that moment. Each chord, dt long, turns a = |dh| degrees between its
    ends, one way. A helicopter at constant speed V turning at most omega can fly it only if:

      1. a <= omega dt (and a <= MAX_CHORD_TURN_DEG, so the direction is unambiguous);
      2. with the turn made at full rate in tau = a / omega and the rest of the time
         straight, its length lies between the straight time split half before and half
         after the turn, V (dt - tau) cos(a / 2) + 2 (V / w) sin(a / 2), and all of it on
         the turn's bisector, V (dt - tau) + 2 (V / w) sin(a / 2): so the helicopter
         neither slows down nor speeds up;
      3. it points off the start heading by between a tau / (2 dt) and a (1 - tau / (2 dt)),
         the average heading of the earliest and the latest the turn can be made, to within
         a^3 / 5 rad, the gap between an average heading and the direction of the chord.

    An arc at a constant rate meets all three exactly. A square corner fails the first; a
    corner disguised by slowing down fails the second; a turn made at once and then hidden
    in a long straight chord fails the third.
    """
    omega = check_turn_rate(turn_rate_deg_s)
    w = math.radians(omega)
    v = float(speed_ms)
    e0, n0, h0 = (float(x) for x in start)
    t0 = 0.0
    for t, e, n, h in zip(t_s, east_m, north_m, heading_deg):
        t, e, n, h = float(t), float(e), float(n), float(h)
        dt = t - t0
        dh = shortest_turn(h0, h)
        a = abs(dh)
        if a > MAX_CHORD_TURN_DEG + _ABS_TOL_DEG:
            raise ValueError(f"a chord ending at {t:g} s turns {a:.1f} degrees; cut turns into "
                             f"pieces of at most {MAX_CHORD_TURN_DEG:g}")
        if a > omega * dt * (1.0 + _REL_TOL) + _ABS_TOL_DEG:
            raise ValueError(f"the path turns {a:.2f} degrees in {dt:.3g} s ending at {t:g} s, "
                             f"faster than the helicopter's {omega:.2f} deg/s")
        a_rad = math.radians(a)
        tau = min(a_rad / w, dt)
        arc = 2.0 * v / w * math.sin(a_rad / 2.0) if a_rad > 0.0 else 0.0
        shortest = v * (dt - tau) * math.cos(a_rad / 2.0) + arc
        longest = v * (dt - tau) + arc
        c = math.hypot(e - e0, n - n0)
        if c < shortest * (1.0 - _REL_TOL) - _ABS_TOL_M or c > longest * (1.0 + _REL_TOL) + _ABS_TOL_M:
            raise ValueError(f"the chord ending at {t:g} s is {c:.1f} m; at {v:.1f} m/s turning "
                             f"{a:.1f} degrees it must be {shortest:.1f} to {longest:.1f} m")
        if c > _ABS_TOL_M:
            side = 1.0 if dh >= 0.0 else -1.0
            off = side * shortest_turn(h0, bearing(e - e0, n - n0))
            frac = tau / (2.0 * dt) if dt > 0.0 else 0.5
            slack = math.degrees(a_rad ** 3 / 5.0) + _ABS_TOL_DEG
            if not a * frac - slack <= off <= a * (1.0 - frac) + slack:
                raise ValueError(f"the chord ending at {t:g} s points {off:.2f} degrees into a "
                                 f"{a:.2f} degree turn; flown at {omega:.2f} deg/s it must point "
                                 f"{a * frac:.2f} to {a * (1.0 - frac):.2f}: a corner, not a turn")
        e0, n0, h0, t0 = e, n, h, t
