"""Tests for sar.search.kinematics: the turn, the autopilot, and the referee's check (D032)."""

import math

import numpy as np
import pytest

from sar.search.kinematics import (
    ARC_PIECE_DEG,
    arc_point,
    check_turns,
    chord_ends,
    follow,
    shortest_turn,
    steer,
    steer_to,
)
from sar.search.patterns import expanding_square, parallel_track, sector_search, trackline_return
from sar.search.platform import (
    MAX_BANK_DEG,
    SEARCH_SPEED_MS,
    STANDARD_RATE_DEG_S,
    SWEEP_WIDTH_M,
    TURN_RADIUS_M,
    TURN_RATE_DEG_S,
    turn_radius_m,
    turn_rate_deg_s,
)

V = SEARCH_SPEED_MS
W = TURN_RATE_DEG_S
R = TURN_RADIUS_M


class TestTheNumbers:
    def test_a_30_degree_bank_at_90_kt_is_7_degrees_a_second_on_379_m(self):
        assert MAX_BANK_DEG == 30.0
        assert W == pytest.approx(7.0065, abs=1e-4)
        assert R == pytest.approx(378.62, abs=0.01)

    def test_standard_rate_at_90_kt_is_a_14_degree_bank_on_884_m(self):
        # The FAA's 3 deg/s, the other side of the bracket.
        bank = math.degrees(math.atan(V * math.radians(STANDARD_RATE_DEG_S) / 9.80665))
        assert bank == pytest.approx(13.9, abs=0.05)
        assert turn_radius_m(STANDARD_RATE_DEG_S) == pytest.approx(884.3, abs=0.1)
        assert turn_rate_deg_s(bank) == pytest.approx(STANDARD_RATE_DEG_S, abs=1e-9)

    @pytest.mark.parametrize("bank", [0.0, 90.0, -5.0])
    def test_a_bank_outside_0_to_90_is_refused(self, bank):
        with pytest.raises(ValueError):
            turn_rate_deg_s(bank)


class TestTheTurn:
    def test_the_short_way_round_and_right_at_180(self):
        assert shortest_turn(350.0, 10.0) == pytest.approx(20.0)
        assert shortest_turn(10.0, 350.0) == pytest.approx(-20.0)
        assert shortest_turn(0.0, 180.0) == 180.0
        assert shortest_turn(90.0, 270.0) == 180.0

    def test_an_arc_is_exact_and_a_tiny_rate_is_a_straight_line(self):
        e, n, h = arc_point(0.0, 0.0, 0.0, W, 180.0 / W, V)
        assert (e, n) == pytest.approx((2 * R, 0.0), abs=1e-9)
        assert h == pytest.approx(180.0)
        # A rate of 1e-15 deg/s must not blow up through V / w.
        assert arc_point(0.0, 0.0, 0.0, 1e-15, 10.0, V)[:2] == pytest.approx((0.0, 10 * V), abs=1e-9)

    def test_steer_turns_at_the_rate_then_flies_straight(self):
        p = steer(0.0, 0.0, 0.0, 90.0, 60.0, V, W)
        tau = 90.0 / W
        assert p.t_s[-1] == 60.0 and p.heading_deg[-1] == 90.0
        assert np.diff(p.t_s[p.t_s <= tau + 1e-9]).max() <= ARC_PIECE_DEG / W + 1e-9
        assert (p.east_m[-1], p.north_m[-1]) == pytest.approx((R + V * (60 - tau), R), abs=1e-6)

    def test_the_arc_is_cut_into_pieces_of_at_most_10_degrees(self):
        p = steer(0.0, 0.0, 0.0, 175.0, 60.0, V, W)
        assert np.max(np.abs(np.diff(np.concatenate(([0.0], p.heading_deg))))) <= ARC_PIECE_DEG + 1e-9

    def test_an_instant_helicopter_flies_one_straight_leg(self):
        p = steer_to(0.0, 0.0, 0.0, 90.0, 60.0, V, math.inf)
        assert p.t_s.tolist() == [60.0]
        assert (p.east_m[0], p.north_m[0]) == pytest.approx((V * 60, 0.0), abs=1e-9)

    def test_every_steer_passes_the_referee(self):
        rng = np.random.default_rng(1)
        for _ in range(200):
            h0, turn, dur = rng.uniform(0, 360), rng.uniform(-400, 400), rng.uniform(0.5, 60)
            p = steer(10.0, -5.0, h0, turn, dur, V, W)
            check_turns(p.t_s, p.east_m, p.north_m, p.heading_deg, (10.0, -5.0, h0), V, W)


class TestTheCheck:
    def test_a_square_corner_fails(self):
        with pytest.raises(ValueError):
            check_turns([10, 20], [V * 10, V * 10], [0, V * 10], [90, 0], (0, 0, 90), V, W)

    def test_a_corner_hidden_by_claiming_no_turn_fails(self):
        with pytest.raises(ValueError, match="corner"):
            check_turns([10, 20], [V * 10, V * 10], [0, V * 10], [90, 90], (0, 0, 90), V, W)

    def test_a_corner_hidden_by_slowing_down_fails(self):
        # 45 degrees in 20 s is slow enough, but this chord is the turn made at once, then
        # flown straight at 0.71 V: no helicopter at V gets there.
        with pytest.raises(ValueError):
            check_turns([20], [V * 10], [V * 10], [45], (0, 0, 90), V, W)

    def test_hovering_may_turn_at_the_rate(self):
        check_turns([4.0], [0.0], [0.0], [25.0], (0.0, 0.0, 0.0), 0.0, W)
        with pytest.raises(ValueError, match="faster"):
            check_turns([1.0], [0.0], [0.0], [20.0], (0.0, 0.0, 0.0), 0.0, W)


def mean_offset_from_drawn(pattern, flown):
    """The mean distance from the flown path to the nearest point of the drawn legs, m."""
    from sar.search.kinematics import _Polyline
    line = _Polyline(pattern.east_m, pattern.north_m)
    return float(np.mean([line.project(x, y, 0.0, line.total)[1]
                          for x, y in zip(flown.east_m[::4], flown.north_m[::4])]))


class TestTheAutopilot:
    PATTERNS = {
        "expanding-square": lambda: expanding_square(SWEEP_WIDTH_M, 30.0, duration_s=5400.0),
        "parallel": lambda: parallel_track(first_bearing_deg=30.0),
        "sector": lambda: sector_search(first_bearing_deg=30.0, duration_s=5400.0),
        "trackline": lambda: trackline_return(3000.0, first_bearing_deg=30.0, duration_s=5400.0),
    }

    @pytest.mark.parametrize("name", list(PATTERNS))
    def test_every_minute_it_flies_passes_the_referee(self, name):
        f = follow(*self._legs(name), 30.0, 2700.0, V, W)
        assert np.max(np.abs(f.rate_deg_s)) <= W + 1e-12
        for k in range(45):
            p = f.path(60.0 * k, 60.0 * (k + 1))
            start = (f.east_m[120 * k], f.north_m[120 * k], f.heading_deg[120 * k])
            check_turns(p.t_s - 60.0 * k, p.east_m, p.north_m, p.heading_deg, start, V, W)

    @pytest.mark.parametrize("name, within", [("expanding-square", 25.0), ("parallel", 30.0),
                                              ("sector", 45.0), ("trackline", 20.0)])
    def test_it_keeps_close_to_the_drawn_legs(self, name, within):
        # Measured 9 Oct at L1 = r: 17.5, 21.5, 38.3 and 13.9 m (D032).
        pattern = self.PATTERNS[name]()
        f = follow(pattern.east_m, pattern.north_m, 30.0, 2700.0, V, W)
        assert mean_offset_from_drawn(pattern, f) < within

    def test_on_a_long_leg_it_settles_onto_the_line(self):
        f = follow([0.0, 0.0], [0.0, 20_000.0], 90.0, 300.0, V, W)
        assert abs(f.east_m[-1]) < 0.01 and f.heading_deg[-1] == pytest.approx(0.0, abs=1e-6)

    def test_the_same_inputs_fly_the_same_path(self):
        a = follow(*self._legs("parallel"), 30.0, 600.0, V, W)
        b = follow(*self._legs("parallel"), 30.0, 600.0, V, W)
        assert np.array_equal(a.east_m, b.east_m) and np.array_equal(a.heading_deg, b.heading_deg)

    def test_chords_turn_one_way_and_at_most_a_piece(self):
        f = follow(*self._legs("sector"), 30.0, 600.0, V, W)
        keep = chord_ends(f.rate_deg_s, 0, 1200, 0.5)
        start = 0
        for k in keep:
            r = f.rate_deg_s[start:k]
            assert np.sum(np.abs(r)) * 0.5 <= ARC_PIECE_DEG + 1e-9
            assert not (np.any(r > 1e-9) and np.any(r < -1e-9))
            start = k

    def test_an_instant_helicopter_has_no_autopilot(self):
        with pytest.raises(ValueError):
            follow([0.0, 0.0], [0.0, 1000.0], 0.0, 60.0, V, math.inf)

    def _legs(self, name):
        p = self.PATTERNS[name]()
        return p.east_m, p.north_m
