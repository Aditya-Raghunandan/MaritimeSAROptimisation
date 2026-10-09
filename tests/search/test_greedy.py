"""Tests for sar.search.greedy: issue #49's acceptance criteria, on synthetic clouds."""

import math

import numpy as np
import pytest

from sar.search.episode import SearchEpisode, Waypoints, run
from sar.search.greedy import greedy_policy, random_heading_policy, remaining_map
from sar.search.scenario import steady_marker, steady_search, steady_window
from sar.utils.geo import offset_position

LAT, LON = 26.5, 281.0


def east_of_marker(east_m: float, spread_km: float = 0.3, particles: int = 400, seed: int = 1,
                   **kw):
    """A cloud `east_m` due east of a still marker; the helicopter arrives pointing north."""
    lat, lon = offset_position(LAT, LON, east_m, 0.0)
    window = steady_window((float(lat), float(lon)), spread_km, particles,
                           np.random.default_rng(seed))
    return SearchEpisode(window, steady_marker(LAT, LON), **kw)


class TestTheMap:
    def test_it_holds_all_the_remaining_probability(self):
        ep = east_of_marker(2000.0)
        east, north, mass = remaining_map(ep)
        assert mass.sum() == pytest.approx(ep.remaining)
        assert np.average(east, weights=mass) == pytest.approx(2000.0, abs=60.0)
        assert abs(np.average(north, weights=mass)) < 60.0

    def test_cells_are_on_a_250_m_lattice_about_the_marker(self):
        east, north, _ = remaining_map(east_of_marker(2000.0))
        assert np.allclose((east / 250.0 - 0.5) % 1.0, 0.0)
        assert np.allclose((north / 250.0 - 0.5) % 1.0, 0.0)


class TestGreedy:
    def test_with_all_the_mass_due_east_the_first_heading_is_east(self):
        ep = east_of_marker(2000.0)
        assert greedy_policy()(ep) == 90.0

    def test_beyond_one_step_it_still_heads_for_the_mass(self):
        # 6 km east: no heading reaches it in a minute, so it flies to the centre of mass.
        ep = east_of_marker(6000.0)
        assert greedy_policy()(ep) == pytest.approx(90.0, abs=1.0)

    def test_ties_break_the_same_way_every_time(self):
        def flight():
            setup = steady_search((LAT, LON), 2.0, 300, np.random.default_rng(3))
            ep = SearchEpisode(setup.window, setup.marker)
            run(greedy_policy(), ep)
            return ep.track()

        (t1, lat1, lon1), (t2, lat2, lon2) = flight(), flight()
        assert np.array_equal(lat1, lat2) and np.array_equal(lon1, lon2)

    def test_over_20_seeds_greedy_beats_the_random_floor(self):
        greedy, floor = [], []
        for seed in range(20):
            setup = steady_search((LAT, LON), 2.0, 500, np.random.default_rng(seed))
            greedy.append(run(greedy_policy(), SearchEpisode(setup.window, setup.marker))["pos"])
            floor.append(run(random_heading_policy(seed),
                             SearchEpisode(setup.window, setup.marker))["pos"])
        margin = np.mean(greedy) - np.mean(floor)
        print(f"\nmean POS over 20 seeds: greedy {np.mean(greedy):.3f}, random "
              f"{np.mean(floor):.3f}, margin {margin:+.3f}")
        assert margin > 0.0

    def test_deciding_inside_the_minute_turns_and_the_referee_accepts_it(self):
        ep = east_of_marker(1000.0)
        wp = greedy_policy(decide_s=10.0)(ep)
        assert isinstance(wp, Waypoints) and wp.heading_deg is not None
        assert wp.t_s[-1] == 60.0 and wp.heading_deg[0] < 90.0      # still turning east
        ep.fly(wp)
        assert ep.k == 1

    def test_it_scores_the_turn_it_would_fly(self):
        # Mass 1 km due south of a helicopter pointing north: reversing costs a 757 m wide
        # half circle, so the leg it can fly reaches the mass only by turning, and it does.
        lat, lon = offset_position(LAT, LON, 0.0, -1200.0)
        window = steady_window((float(lat), float(lon)), 0.2, 400, np.random.default_rng(2))
        ep = SearchEpisode(window, steady_marker(LAT, LON))
        heading = greedy_policy()(ep)
        assert 90.0 <= heading <= 270.0
        ep.step(heading)
        assert ep.remaining < 1.0

    def test_an_instant_helicopter_decides_inside_the_minute_as_before(self):
        ep = east_of_marker(1000.0, turn_rate_deg_s=math.inf)
        wp = greedy_policy(decide_s=10.0)(ep)
        assert isinstance(wp, Waypoints) and list(wp.t_s) == [10, 20, 30, 40, 50, 60]
        assert wp.east_m[0] == pytest.approx(ep.speed_ms * 10.0)
        assert wp.north_m[0] == pytest.approx(0.0, abs=1e-9)
        ep.fly(wp)                                     # within the helicopter's speed
        assert ep.k == 1

    def test_it_does_not_chase_what_it_has_just_crossed_off(self):
        # A tight blob 500 m east is crossed off within two 463 m sub-legs; it then stops
        # flying east instead of carrying on into empty sea.
        wp = greedy_policy(decide_s=10.0)(east_of_marker(500.0, spread_km=0.05,
                                                         turn_rate_deg_s=math.inf))
        assert wp.east_m[0] > 0.0
        assert wp.east_m[-1] < 2.5 * 463.0

    @pytest.mark.parametrize("bad", [0.0, 7.0, 90.0, -10.0])
    def test_the_decision_interval_must_divide_the_minute(self, bad):
        with pytest.raises(ValueError, match="divide"):
            greedy_policy(decide_s=bad)

    @pytest.mark.parametrize("bad", [0, -1, 1.5, True])
    def test_headings_and_lookahead_must_be_positive_integers(self, bad):
        with pytest.raises(ValueError, match="headings"):
            greedy_policy(headings=bad)
        with pytest.raises(ValueError, match="lookahead"):
            greedy_policy(lookahead=bad)


class TestRandomFloor:
    def test_the_same_seed_flies_the_same_headings(self):
        ep = east_of_marker(0.0)
        a, b = random_heading_policy(7), random_heading_policy(7)
        assert [a(ep) for _ in range(5)] == [b(ep) for _ in range(5)]

    def test_headings_are_compass_bearings(self):
        policy = random_heading_policy([1, 120])
        headings = [policy(None) for _ in range(200)]
        assert min(headings) >= 0.0 and max(headings) < 360.0
