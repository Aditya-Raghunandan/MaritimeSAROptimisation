"""Tests for sar.search.platform. Hand-computed values only, no files."""

import numpy as np
import pytest

from sar.search.platform import (
    KNOT_MS,
    LAUNCH_DELAY_S,
    NM_M,
    ON_SCENE_WINDOW_S,
    RADIUS_OF_ACTION_M,
    SEARCH_SPEED_MS,
    SOURCES,
    STEP_S,
    SWEEP_WIDTH_M,
    TRANSIT_SPEED_MS,
    _cli,
    constants,
    expanding_square_spacing_m,
    great_circle_m,
    sector_radius_m,
    track_length_m,
    transit_time_s,
)


class TestTheNumbersAreTheManualsNumbers:
    def test_the_sweep_width_is_a_tenth_of_a_nautical_mile(self):
        assert SWEEP_WIDTH_M == pytest.approx(185.2)

    def test_a_knot_is_one_nautical_mile_an_hour(self):
        assert KNOT_MS * 3600 == pytest.approx(NM_M)

    def test_search_speed_is_ninety_knots(self):
        assert SEARCH_SPEED_MS == pytest.approx(46.3)

    def test_transit_speed_is_the_h60_cruise(self):
        assert TRANSIT_SPEED_MS / KNOT_MS == pytest.approx(125.0)

    def test_radius_of_action_is_three_hundred_nautical_miles(self):
        assert RADIUS_OF_ACTION_M == pytest.approx(555_600.0)

    def test_the_window_and_the_step_make_forty_five_steps(self):
        assert ON_SCENE_WINDOW_S / STEP_S == 45

    def test_launch_delay_is_the_b0_thirty_minutes(self):
        assert LAUNCH_DELAY_S == 1800

    def test_every_constant_has_a_source(self):
        assert set(constants()) - {"nm_m", "knot_ms"} == set(SOURCES)


class TestTrackLength:
    def test_one_window_at_search_speed(self):
        # 46.3 m/s x 2700 s, which is 67.5 NM.
        assert track_length_m() == pytest.approx(125_010.0)
        assert track_length_m() / NM_M == pytest.approx(67.5)

    def test_non_positive_inputs_raise(self):
        with pytest.raises(ValueError, match="must be positive"):
            track_length_m(0.0)
        with pytest.raises(ValueError, match="must be positive"):
            track_length_m(window_s=-1.0)


class TestPatternSizes:
    def test_expanding_square_spacing_is_the_sweep_width(self):
        assert expanding_square_spacing_m() == SWEEP_WIDTH_M

    def test_sector_radius_is_one_minute_at_ninety_knots(self):
        # The Addendum's own worked example, p. 3-23: 1.5 NM at 90 kt.
        assert sector_radius_m() == pytest.approx(1.5 * NM_M)

    def test_sector_radius_falls_back_to_twice_the_sweep_width(self):
        assert sector_radius_m(speed_ms=1.0) == pytest.approx(2 * SWEEP_WIDTH_M)

    def test_non_positive_inputs_raise(self):
        with pytest.raises(ValueError):
            expanding_square_spacing_m(0.0)
        with pytest.raises(ValueError):
            sector_radius_m(speed_ms=0.0)


class TestTransit:
    def test_one_hundred_nautical_miles(self):
        # 100 NM at 125 kt is 0.8 h = 2880 s, after the 30-minute launch.
        assert transit_time_s(100 * NM_M) == pytest.approx(1800 + 2880)

    def test_beyond_the_radius_of_action_raises(self):
        with pytest.raises(ValueError, match="radius of action"):
            transit_time_s(RADIUS_OF_ACTION_M + 1.0)

    def test_negative_distance_raises(self):
        with pytest.raises(ValueError, match="must not be negative"):
            transit_time_s(-1.0)


class TestGreatCircle:
    def test_one_degree_of_latitude(self):
        # 2 pi R / 360 on the GRS80 mean radius.
        assert great_circle_m(26.0, 280.0, 27.0, 280.0) == pytest.approx(111_195.08, abs=0.01)

    def test_is_symmetric_and_zero_on_itself(self):
        a = great_circle_m(26.5, 281.0, 27.1, 282.3)
        assert great_circle_m(27.1, 282.3, 26.5, 281.0) == pytest.approx(a)
        assert great_circle_m(26.5, 281.0, 26.5, 281.0) == 0.0

    def test_accepts_arrays(self):
        d = great_circle_m(np.array([26.0, 26.0]), 280.0, np.array([27.0, 28.0]), 280.0)
        assert d.shape == (2,)


class TestCLI:
    def test_prints_constants_sources_and_the_window(self):
        out = _cli([])
        assert out["constants"]["sweep_width_m"] == pytest.approx(185.2)
        assert "H-44" in out["sources"]["sweep_width_m"]
        assert out["one_window"]["track_length_nm"] == pytest.approx(67.5)
