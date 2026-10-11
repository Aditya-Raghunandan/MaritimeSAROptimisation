"""greedy.py: two searchers that do not learn, the greedy search and the random floor (issue #49).

Both are scored by the same referee as everything else (`sar.search.episode`, D029), so a
difference in score can only be the searcher.

GREEDY: FLY WHERE THE MAP IS FULLEST, ONE STEP AT A TIME. Every minute it bins what is left
of the probability into the episode map's 250 m cells (D007), in the marker's frame, and
tries `headings` evenly spaced compass headings. For each it adds up the map's mass in the
cells the next `lookahead` minutes of straight flight would pass through, and flies the
heading with the most. D015 makes greedy the control PPO must beat, and D023 places it in
the map / no-learning cell: the DECISION reads the map, but the score still comes from
`sweep()` on the particles, exactly as for every other searcher.

  * A cell counts if the straight path passes within half a cell of its centre. That is
    wider than the 92.6 m half-strip, so the estimate is a little generous, but it is the
    same for every heading, and it is the map, not the particles, that greedy may read.
  * The map is the current frame's: greedy does not predict where particles will be a
    minute on. Relative to the marker they move slowly (the marker rides the same current).
  * Ties go to the first heading, north first and then clockwise, so the same inputs always
    give the same track.
  * When no heading reaches any mass (a cleared patch), it flies toward the centre of mass
    of what is left, rather than holding north into empty sea.

HOW OFTEN IT DECIDES, `decide_s`. At the referee's 60 s step a heading is a 2.78 km straight
leg. On S01 at 1 h (7 Oct, scratch probe) eight such headings cleared 0.367 against the
Expanding Square's 0.831: legs on a 45-degree lattice cannot lie side by side, so it kept
re-flying the same line. 16 headings gave 0.775; deciding every 10 s with 8 gave 0.835. So
`decide_s` may divide the minute: greedy then returns the step as `Waypoints`, choosing a
heading every `decide_s` and crossing off, on its own copy of the map, the cells each
sub-leg has just passed over, so it does not chase mass it has already taken. The episode
still sweeps the particles itself. The values are chosen on dev and frozen before any test
set is opened.

IT PLANS THE TURN IT CAN FLY (D032). A helicopter that turns at 7 deg/s cannot fly a
straight leg off in any direction: to reverse it spends 26 s on a 379 m half circle. So for
each candidate heading greedy scores the path the referee will actually fly, the turn to
that heading at the turn rate (`sar.search.kinematics.steer_to`) and then straight to the
end of its reach, by the cells within half a cell of any of its chords. It still chooses a
compass heading, and the referee flies the turn. On synthetic clouds (9 Oct, vault notebook)
the unchanged greedy lost 9 points of POS to the turn and this one 0.1. For a helicopter
that turns at once the straight leg is that path, and greedy is exactly what it was.

RANDOM: THE FLOOR. A fresh uniformly random heading every minute, from its own seed: the
helicopter does a random walk about the marker. #49 and R5f: anything worth reporting must
beat it. It is called "random" in the results, never "random walk", because the old noise
model of the drift engine already has that name (D028).
"""

from __future__ import annotations

import numpy as np

from sar.search.episode import Waypoints
from sar.search.kinematics import shortest_turn, steer, steer_to
from sar.search.platform import STEP_S
from sar.search.sweep import relative_m

# The episode map's cell (D007, ADR002 section 4): what greedy is allowed to read.
CELL_M = 250.0


def _positive_int(name: str, value) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


def remaining_map(episode, cell_m: float = CELL_M) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """What is left of the probability, binned into cells about the marker.

    Returns (east, north, mass): the centres in metres east and north of the marker of
    every cell holding any probability now, and that probability. Empty cells are left
    out. The cells are anchored on the marker, so a cell edge runs through it.
    """
    if not cell_m > 0:
        raise ValueError(f"cell_m must be positive, got {cell_m}")
    lat, lon = episode.particles()
    weight = np.asarray(episode.weight, dtype=float)
    live = weight > 0.0
    if not np.any(live):
        empty = np.zeros(0)
        return empty, empty, empty
    east, north = relative_m(lat[live], lon[live], *episode.marker_position)
    cells = np.floor(np.column_stack([east, north]) / cell_m).astype(np.int64)
    keys, inverse = np.unique(cells, axis=0, return_inverse=True)
    mass = np.bincount(inverse.ravel(), weights=weight[live], minlength=len(keys))
    centres = (keys + 0.5) * cell_m
    return centres[:, 0], centres[:, 1], mass


def _distance_to_segment(px, py, ax, ay, bx, by) -> np.ndarray:
    dx, dy = bx - ax, by - ay
    t = np.clip(((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy), 0.0, 1.0)
    return np.hypot(px - (ax + t * dx), py - (ay + t * dy))


def greedy_policy(headings: int = 8, lookahead: int = 1, cell_m: float = CELL_M,
                  decide_s: float = STEP_S):
    """Every `decide_s`, the heading whose next `lookahead` decisions pass over the most map.

    At decide_s = 60 s (the referee's step) it returns a heading; below, `Waypoints`.
    """
    headings = _positive_int("headings", headings)
    lookahead = _positive_int("lookahead", lookahead)
    if not cell_m > 0:
        raise ValueError(f"cell_m must be positive, got {cell_m}")
    decide_s = float(decide_s)
    per_step = STEP_S / decide_s if decide_s > 0 else 0.0
    if not (per_step >= 1 and abs(per_step - round(per_step)) < 1e-9):
        raise ValueError(f"decide_s must divide the {STEP_S:g} s step, got {decide_s:g}")
    per_step = int(round(per_step))
    bearings = np.arange(headings) * (360.0 / headings)
    b = np.radians(bearings)
    unit_e, unit_n = np.sin(b), np.cos(b)

    def choose(east, north, mass, x0, y0, reach) -> float:
        """The best heading from (x0, y0) over this map."""
        if mass.size == 0 or not mass.sum() > 0.0:
            return float(bearings[0])
        scores = np.array([
            mass[_distance_to_segment(east, north, x0, y0, x0 + reach * ue, y0 + reach * un)
                 <= 0.5 * cell_m].sum()
            for ue, un in zip(unit_e, unit_n)])
        best = int(np.argmax(scores))
        if scores[best] > 0.0:
            return float(bearings[best])
        total = float(mass.sum())
        ce, cn = float(np.sum(mass * east) / total), float(np.sum(mass * north) / total)
        if ce == x0 and cn == y0:
            return float(bearings[0])
        return float(np.degrees(np.arctan2(ce - x0, cn - y0)) % 360.0)

    def flown(x0, y0, psi0, to, speed, rate):
        """The chords the helicopter flies from (x0, y0) on psi0 to a heading, to its reach."""
        p = steer_to(x0, y0, psi0, to, decide_s, speed, rate)
        e = np.concatenate(([x0], p.east_m))
        n = np.concatenate(([y0], p.north_m))
        if lookahead > 1:
            more = speed * decide_s * (lookahead - 1)
            h = np.radians(p.heading_deg[-1])
            e = np.append(e, e[-1] + more * np.sin(h))
            n = np.append(n, n[-1] + more * np.cos(h))
        return e, n

    def passed(east, north, e, n) -> np.ndarray:
        """Which cells lie within half a cell of any chord of the path (e, n)."""
        hit = np.zeros(east.size, dtype=bool)
        for i in range(e.size - 1):
            if e[i] == e[i + 1] and n[i] == n[i + 1]:
                continue
            hit |= _distance_to_segment(east, north, e[i], n[i], e[i + 1], n[i + 1]) <= 0.5 * cell_m
        return hit

    def choose_flyable(east, north, mass, x0, y0, psi0, speed, rate) -> float:
        """The best heading from (x0, y0) on psi0, scoring the turn and leg it would fly."""
        if mass.size == 0 or not mass.sum() > 0.0:
            return float(bearings[0])
        scores = np.array([mass[passed(east, north, *flown(x0, y0, psi0, b, speed, rate))].sum()
                           for b in bearings])
        best = int(np.argmax(scores))
        if scores[best] > 0.0:
            return float(bearings[best])
        total = float(mass.sum())
        ce, cn = float(np.sum(mass * east) / total), float(np.sum(mass * north) / total)
        if ce == x0 and cn == y0:
            return float(bearings[0])
        return float(np.degrees(np.arctan2(ce - x0, cn - y0)) % 360.0)

    def policy(episode):
        east, north, mass = remaining_map(episode, cell_m)
        x, y = episode.offset
        leg = episode.speed_ms * decide_s
        if not episode.instant:
            speed, rate = episode.speed_ms, episode.turn_rate_deg_s
            psi = episode.heading_deg
            if per_step == 1:
                return choose_flyable(east, north, mass, x, y, psi, speed, rate)
            mass = mass.copy()
            t, e, n, h = [], [], [], []
            for k in range(per_step):
                to = choose_flyable(east, north, mass, x, y, psi, speed, rate)
                p = steer(x, y, psi, shortest_turn(psi, to), decide_s, speed, rate)
                if mass.size:
                    mass[passed(east, north, np.concatenate(([x], p.east_m)),
                                np.concatenate(([y], p.north_m)))] = 0.0
                t.extend((p.t_s + k * decide_s).tolist())
                e.extend(p.east_m.tolist())
                n.extend(p.north_m.tolist())
                h.extend(p.heading_deg.tolist())
                x, y, psi = float(p.east_m[-1]), float(p.north_m[-1]), float(p.heading_deg[-1])
            t[-1] = STEP_S
            return Waypoints(np.array(t), np.array(e), np.array(n), np.array(h))
        if per_step == 1:
            return choose(east, north, mass, x, y, leg * lookahead)
        mass = mass.copy()
        ends_e, ends_n = [], []
        for _ in range(per_step):
            heading = np.radians(choose(east, north, mass, x, y, leg * lookahead))
            nx, ny = x + leg * np.sin(heading), y + leg * np.cos(heading)
            if mass.size:
                mass[_distance_to_segment(east, north, x, y, nx, ny) <= 0.5 * cell_m] = 0.0
            x, y = nx, ny
            ends_e.append(x)
            ends_n.append(y)
        return Waypoints(decide_s * np.arange(1, per_step + 1), np.array(ends_e),
                         np.array(ends_n))

    return policy


def random_heading_policy(seed):
    """A uniformly random heading every minute, from its own seed: the floor (#49, R5f)."""
    rng = np.random.default_rng(seed)
    return lambda episode: float(rng.uniform(0.0, 360.0))
