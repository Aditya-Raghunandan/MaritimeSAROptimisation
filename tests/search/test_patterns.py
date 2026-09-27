"""Tests for sar.search.patterns. Hand-computed values only, no files."""

import numpy as np
import pytest

from sar.search.patterns import (
    SECTOR_LEGS,
    MarkerTrack,
    _cli,
    describe,
    expanding_square,
    ground_track,
    on_ground,
    parallel_track,
    sector_search,
    trackline_return,
)
from sar.search.platform import NM_M, SEARCH_SPEED_MS, SWEEP_WIDTH_M
from sar.utils.geo import M_PER_DEG_LAT, metres_per_degree_lon

S = SWEEP_WIDTH_M
V = SEARCH_SPEED_MS


def leg_lengths(p):
    return np.hypot(np.diff(p.east_m), np.diff(p.north_m))


def turns(p):
    return np.diff(p.heading_deg) % 360.0


class TestExpandingSquare:
    def test_legs_are_s_s_2s_2s_3s_3s(self):
        p = expanding_square(S, 0.0, V, 600.0)
        assert leg_lengths(p)[:6] == pytest.approx([S, S, 2 * S, 2 * S, 3 * S, 3 * S])

    def test_every_turn_is_ninety_degrees_right(self):
        p = expanding_square(S, 37.0, V, 600.0)
        assert np.all(np.isclose(turns(p), 90.0))

    def test_starts_at_the_marker_on_the_first_bearing(self):
        p = expanding_square(S, 90.0, V, 600.0)
        assert (p.east_m[0], p.north_m[0], p.t_s[0]) == (0.0, 0.0, 0.0)
        assert p.heading_deg[0] == 90.0
        assert p.east_m[1] == pytest.approx(S) and p.north_m[1] == pytest.approx(0.0, abs=1e-9)

    def test_adjacent_parallel_legs_are_one_spacing_apart(self):
        # First bearing north: legs 1, 3, 5 ... run north-south at east = 0, S, -S, 2S, -2S,
        # 3S, so sorted they are consecutive multiples of S, each exactly S from the next.
        p = expanding_square(S, 0.0, V, 2000.0)
        east_of_north_south_legs = np.sort(p.east_m[1::2][:6])
        assert east_of_north_south_legs / S == pytest.approx([-2, -1, 0, 1, 2, 3])
        assert np.diff(east_of_north_south_legs) == pytest.approx([S] * 5)

    def test_total_distance_is_speed_times_duration(self):
        p = expanding_square(S, 0.0, V, 2700.0)
        assert p.length_m == pytest.approx(V * 2700.0)
        assert p.duration_s == pytest.approx(2700.0)

    def test_the_window_cuts_the_last_leg(self):
        # Legs S and S take 2 S / V; stopping halfway through the second leaves it half flown.
        p = expanding_square(S, 0.0, V, 1.5 * S / V)
        assert leg_lengths(p) == pytest.approx([S, 0.5 * S])

    def test_non_positive_inputs_raise(self):
        for kwargs in ({"spacing_m": 0.0}, {"speed_ms": -1.0}, {"duration_s": 0.0}):
            with pytest.raises(ValueError, match="must be a positive number"):
                expanding_square(**kwargs)


class TestSectorSearch:
    def test_one_pattern_is_nine_legs_of_the_radius(self):
        r = 1.5 * NM_M
        p = sector_search(r, 0.0, V, 9 * r / V)
        assert p.heading_deg.size == 9
        assert leg_lengths(p) == pytest.approx([r] * 9)

    def test_the_manuals_worked_example_takes_nine_minutes(self):
        # Addendum p. 3-23: radius 1.5 NM at 90 kt, "about 9 minutes", 9R of track.
        p = sector_search(None, 0.0, V, 540.0)
        assert p.length_m == pytest.approx(9 * 1.5 * NM_M)
        assert p.heading_deg.size == 9

    def test_it_returns_to_datum_after_every_triangle(self):
        r = 1000.0
        p = sector_search(r, 20.0, V, 9 * r / V)
        for k in (3, 6, 9):
            assert np.hypot(p.east_m[k], p.north_m[k]) == pytest.approx(0.0, abs=1e-6)

    def test_turns_are_120_right_and_straight_through_datum(self):
        p = sector_search(1000.0, 0.0, V, 9000.0 / V)
        assert turns(p) == pytest.approx([120, 120, 0, 120, 120, 0, 120, 120])

    def test_six_spokes_sixty_degrees_apart(self):
        r = 1000.0
        p = sector_search(r, 0.0, V, 9 * r / V)
        corners = [k for k in range(1, 9) if k not in (3, 6)]
        bearings = np.degrees(np.arctan2(p.east_m[corners], p.north_m[corners])) % 360
        assert np.sort(np.round(bearings)) == pytest.approx([0, 60, 120, 180, 240, 300])

    def test_the_second_pattern_starts_thirty_degrees_right(self):
        r = 1000.0
        p = sector_search(r, 10.0, V, 18 * r / V)
        assert p.heading_deg[8] == pytest.approx(10.0)      # the first pattern's final course
        assert p.heading_deg[9] == pytest.approx(40.0)

    def test_sector_legs_constant_is_nine_long(self):
        assert len(SECTOR_LEGS) == 9


class TestPatternQueries:
    def test_offset_is_linear_between_waypoints(self):
        p = expanding_square(S, 0.0, V, 600.0)
        mid = 0.5 * (p.t_s[1] + p.t_s[2])
        east, north = p.offset_at(mid)
        assert east == pytest.approx(0.5 * (p.east_m[1] + p.east_m[2]))
        assert north == pytest.approx(0.5 * (p.north_m[1] + p.north_m[2]))

    def test_heading_at_a_waypoint_is_the_leg_leaving_it(self):
        p = expanding_square(S, 0.0, V, 600.0)
        assert p.heading_at(p.t_s[1]) == pytest.approx(90.0)
        assert p.heading_at(p.duration_s) == pytest.approx(p.heading_deg[-1])

    def test_times_outside_the_pattern_raise(self):
        p = expanding_square(S, 0.0, V, 600.0)
        with pytest.raises(ValueError, match="within the pattern"):
            p.offset_at(601.0)
        with pytest.raises(ValueError, match="within the pattern"):
            p.heading_at(-1.0)


class TestOnTheGround:
    @pytest.mark.parametrize("lat", [17.0, 26.5, 36.0])
    def test_the_same_shape_in_metres_at_every_latitude(self, lat):
        p = expanding_square(S, 0.0, V, 600.0)
        marker = MarkerTrack.fixed(lat, 281.0, p.duration_s)
        glat, glon = on_ground(p, marker, p.t_s)
        east = (glon - 281.0) * metres_per_degree_lon(lat)
        north = (glat - lat) * M_PER_DEG_LAT
        assert east == pytest.approx(p.east_m, abs=1e-6)
        assert north == pytest.approx(p.north_m, abs=1e-6)

    def test_a_drifting_marker_carries_the_whole_pattern(self):
        # A marker moving 1 m/s east: the ground path is the pattern shifted by 1 m/s x t.
        p = expanding_square(S, 0.0, V, 600.0)
        lat = 26.5
        dlon = 600.0 / metres_per_degree_lon(lat)
        marker = MarkerTrack(np.array([0.0, 600.0]), np.array([lat, lat]),
                             np.array([281.0, 281.0 + dlon]))
        glat, glon = on_ground(p, marker, p.t_s)
        east = (glon - 281.0) * metres_per_degree_lon(lat)
        assert east == pytest.approx(p.east_m + 1.0 * p.t_s, abs=1e-3)
        assert (glat - lat) * M_PER_DEG_LAT == pytest.approx(p.north_m, abs=1e-6)

    def test_ground_track_holds_every_waypoint(self):
        p = expanding_square(S, 0.0, V, 600.0)
        t, _, _ = ground_track(p, MarkerTrack.fixed(26.5, 281.0, p.duration_s), every_s=10.0)
        assert set(np.round(p.t_s, 9)) <= set(np.round(t, 9))
        assert np.all(np.diff(t) > 0)

    def test_marker_track_rejects_bad_times(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            MarkerTrack(np.array([0.0, 0.0]), np.array([1.0, 1.0]), np.array([1.0, 1.0]))
        with pytest.raises(ValueError, match="same length"):
            MarkerTrack(np.array([0.0, 1.0]), np.array([1.0]), np.array([1.0, 1.0]))

    def test_marker_outside_its_track_raises(self):
        marker = MarkerTrack.fixed(26.5, 281.0, 60.0)
        with pytest.raises(ValueError, match="within the marker track"):
            marker.at(61.0)


class TestDescribeAndCLI:
    def test_describe_reports_legs_and_length(self):
        out = describe(sector_search(None, 0.0, V, 540.0))
        assert out["legs"] == 9
        assert out["length_m"] == pytest.approx(9 * 1.5 * NM_M)

    def test_cli_runs_both_patterns(self):
        es = _cli(["expanding-square", "--first-bearing", "45"])
        assert es["kind"] == "expanding_square"
        assert es["length_m"] == pytest.approx(V * 2700.0)
        vs = _cli(["sector", "--duration-min", "9"])
        assert vs["legs"] == 9


def unit(bearing):
    b = np.radians(bearing)
    return np.array([np.sin(b), np.cos(b)]), np.array([np.sin(b + np.pi / 2), np.cos(b + np.pi / 2)])


def along_across(p, bearing):
    u, v = unit(bearing)
    xy = np.column_stack([p.east_m, p.north_m])
    return xy @ u, xy @ v


class TestParallelTrack:
    def test_the_default_area_is_what_one_window_covers(self):
        # Z = W x V x T, 23.15 km2: a square 4.81 km a side (Addendum p. H-40 (g)).
        side = np.sqrt(S * V * 2700)
        p = parallel_track()
        _, c = along_across(p, 0.0)
        legs = round(side / S)                      # 26 legs across 25.98 spacings
        assert c[1:].max() - c[1:].min() == pytest.approx((legs - 1) * S)
        # The outermost legs sit about half a spacing inside the edges, so none is left unflown.
        assert side / 2 - c[1:].max() == pytest.approx(S / 2, abs=S / 4)

    def test_first_leg_starts_half_a_spacing_inside_the_corner(self):
        p = parallel_track(4000.0, 2000.0, S, 30.0, V, 2700.0)
        a, c = along_across(p, 30.0)
        assert (a[1], c[1]) == pytest.approx((-2000 + S / 2, -1000 + S / 2))

    def test_legs_run_along_the_axis_and_are_a_spacing_apart(self):
        p = parallel_track(4000.0, 2000.0, S, 30.0, V, 2700.0)
        a, c = along_across(p, 30.0)
        # Waypoints 1, 2 are the ends of leg 1; 3, 4 of leg 2; and so on.
        assert c[1] == pytest.approx(c[2]) and c[3] == pytest.approx(c[4])
        assert c[3] - c[1] == pytest.approx(S)
        assert abs(a[2] - a[1]) == pytest.approx(4000 - S)

    def test_starts_by_flying_from_the_marker_to_the_start_point(self):
        p = parallel_track(4000.0, 2000.0, S, 0.0, V, 2700.0)
        assert (p.east_m[0], p.north_m[0]) == (0.0, 0.0)
        assert p.t_s[1] == pytest.approx(np.hypot(p.east_m[1], p.north_m[1]) / V)

    def test_a_finished_area_ends_the_pattern_early(self):
        p = parallel_track(1000.0, 1000.0, S, 0.0, V, 2700.0)
        assert p.duration_s < 2700.0

    def test_too_small_an_area_raises(self):
        with pytest.raises(ValueError, match="one track spacing"):
            parallel_track(100.0, 1000.0, S, 0.0, V, 2700.0)


class TestTracklineReturn:
    def test_first_pass_is_up_one_side_and_down_the_other(self):
        # Addendum Figure H-31: the CSP 1/2 S off the line; up one side, down the other.
        p = trackline_return(3000.0, S, 45.0, V, 2700.0)
        a, c = along_across(p, 45.0)
        assert (a[1], c[1]) == pytest.approx((-3000.0, S / 2))
        assert (a[2], c[2]) == pytest.approx((3000.0, S / 2))
        assert (a[3], c[3]) == pytest.approx((3000.0, -S / 2))
        assert (a[4], c[4]) == pytest.approx((-3000.0, -S / 2))

    def test_it_ends_the_first_return_one_track_space_from_where_it_began(self):
        p = trackline_return(3000.0, S, 0.0, V, 2700.0)
        assert np.hypot(p.east_m[4] - p.east_m[1], p.north_m[4] - p.north_m[1]) == pytest.approx(S)

    def test_each_later_pass_widens_by_a_spacing(self):
        p = trackline_return(3000.0, S, 0.0, V, 2700.0)
        _, c = along_across(p, 0.0)
        assert sorted({round(abs(x) / S, 6) for x in c[1:13]}) == pytest.approx([0.5, 1.5, 2.5])

    def test_it_flies_the_whole_window(self):
        p = trackline_return(3000.0, S, 0.0, V, 2700.0)
        assert p.length_m == pytest.approx(V * 2700.0)

    @pytest.mark.parametrize("lat", [17.0, 36.0])
    def test_the_same_shape_in_metres_at_every_latitude(self, lat):
        p = trackline_return(2000.0, S, 60.0, V, 900.0)
        glat, glon = on_ground(p, MarkerTrack.fixed(lat, 281.0, p.duration_s), p.t_s)
        east = (glon - 281.0) * metres_per_degree_lon(lat)
        assert east == pytest.approx(p.east_m, abs=1e-6)

    def test_bad_inputs_raise(self):
        with pytest.raises(ValueError, match="positive number"):
            trackline_return(0.0)
