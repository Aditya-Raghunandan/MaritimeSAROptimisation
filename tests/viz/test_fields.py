"""Tests for the generic field renderers. No network, no real data.

Every figure function returns a dict of numbers, and the numbers are what these
tests check -- the PNG is only checked for existing. That mirrors the rule
`scripts/check_forcing_pair.py` set: the picture is not the evidence, the
measurement is.

Inputs are chosen so the answer is known analytically rather than recorded from
a previous run, which would only prove the code still does what it did.
"""

import numpy as np
import pandas as pd
import pytest

from sar.viz.fields import (
    _display_axes,
    _vector_mean,
    seasonal_cycle,
    speed_histogram,
    speed_map,
    wind_rose,
)


class TestDisplayAxes:
    def test_converts_longitude_and_keeps_it_sorted(self):
        lat = np.array([17.0, 18.0])
        lon = np.array([278.0, 290.0, 297.0])
        field = np.arange(6, dtype=float).reshape(2, 3)

        out_lat, out_lon, out_field = _display_axes(lat, lon, field)

        assert out_lon == pytest.approx([-82.0, -70.0, -63.0])
        assert (np.diff(out_lon) > 0).all()
        np.testing.assert_allclose(out_lat, lat)
        np.testing.assert_allclose(out_field, field)

    def test_reorders_the_field_to_match_a_wrapped_axis(self):
        """A box spanning the meridian unsorts on conversion, and pcolormesh on an
        unsorted axis draws a mirrored smear rather than raising."""
        lon = np.array([350.0, 355.0, 5.0])          # -10, -5, +5 after conversion
        field = np.array([[1.0, 2.0, 3.0]])

        _, out_lon, out_field = _display_axes(np.array([0.0]), lon, field)

        assert (np.diff(out_lon) > 0).all()
        assert out_lon == pytest.approx([-10.0, -5.0, 5.0])
        assert out_field[0] == pytest.approx([1.0, 2.0, 3.0])


class TestVectorMean:
    def test_a_wind_from_the_north_blows_southward(self):
        """dir_from 0 deg means FROM the north, so v is negative and u is zero."""
        u, v = _vector_mean([0.0], [10.0])
        assert u == pytest.approx(0.0, abs=1e-9)
        assert v == pytest.approx(-10.0)

    def test_a_wind_from_the_east_blows_westward(self):
        u, v = _vector_mean([90.0], [10.0])
        assert u == pytest.approx(-10.0)
        assert v == pytest.approx(0.0, abs=1e-9)

    def test_opposing_winds_cancel(self):
        u, v = _vector_mean([0.0, 180.0], [10.0, 10.0])
        assert u == pytest.approx(0.0, abs=1e-9)
        assert v == pytest.approx(0.0, abs=1e-9)


class TestWindRose:
    def test_a_constant_direction_is_perfectly_constant(self, tmp_path):
        nums = wind_rose(np.full(100, 90.0), np.full(100, 5.0),
                         path=tmp_path / "r.png", title="t")

        assert nums["directional_constancy"] == pytest.approx(1.0)
        assert nums["mean_direction_from_deg"] == pytest.approx(90.0)
        assert nums["steadiness"] == pytest.approx(1.0)
        assert (tmp_path / "r.png").exists()

    def test_uniformly_spread_directions_have_no_constancy(self, tmp_path):
        nums = wind_rose(np.arange(0, 360, 1.0), path=tmp_path / "r.png", title="t")

        assert nums["directional_constancy"] == pytest.approx(0.0, abs=1e-6)

    def test_the_mean_direction_wraps_correctly_around_north(self, tmp_path):
        """Averaging 350 and 10 arithmetically gives 180 -- the exact opposite.

        Compared circularly: the answer is north, and the function may report it
        as either 0 or 359.999..., both of which are the same bearing.
        """
        nums = wind_rose(np.array([350.0, 10.0]), path=tmp_path / "r.png", title="t")

        got = nums["mean_direction_from_deg"]
        assert min(got, 360.0 - got) == pytest.approx(0.0, abs=1e-6)
        assert 0.0 <= got < 360.0

    def test_steadiness_falls_below_constancy_when_the_fast_winds_oppose(self, tmp_path):
        """Two equal-and-opposite winds: the directions are balanced, so the vector
        mean is zero however fast they blow."""
        nums = wind_rose(np.array([0.0, 180.0]), np.array([20.0, 20.0]),
                         path=tmp_path / "r.png", title="t")

        assert nums["steadiness"] == pytest.approx(0.0, abs=1e-9)

    def test_no_finite_directions_raises(self, tmp_path):
        with pytest.raises(ValueError, match="no finite directions"):
            wind_rose(np.full(5, np.nan), path=tmp_path / "r.png", title="t")


class TestSpeedHistogram:
    def test_measures_the_fraction_above_the_threshold(self, tmp_path):
        speeds = np.array([1.0, 5.0, 11.0, 12.0])       # two of four above 10

        nums = speed_histogram(speeds, path=tmp_path / "h.png", title="t")

        assert nums["fraction_above_threshold"] == pytest.approx(0.5)
        assert nums["max_ms"] == pytest.approx(12.0)
        assert nums["n"] == 4

    def test_leeway_is_two_percent_of_the_mean(self, tmp_path):
        nums = speed_histogram(np.full(10, 10.0), path=tmp_path / "h.png", title="t")

        assert nums["leeway_at_mean_ms"] == pytest.approx(0.2)

    def test_ignores_nans_rather_than_propagating_them(self, tmp_path):
        nums = speed_histogram(np.array([1.0, np.nan, 3.0]),
                               path=tmp_path / "h.png", title="t")

        assert nums["n"] == 2
        assert nums["mean_ms"] == pytest.approx(2.0)

    def test_no_finite_speeds_raises(self, tmp_path):
        with pytest.raises(ValueError, match="no finite speeds"):
            speed_histogram(np.full(5, np.nan), path=tmp_path / "h.png", title="t")


class TestSeasonalCycle:
    def test_finds_the_windiest_and_calmest_months(self, tmp_path):
        idx = pd.date_range("2021-01-01", periods=365, freq="D")
        # July windy, January calm, everything else in between
        speed = np.where(idx.month == 7, 9.0, np.where(idx.month == 1, 1.0, 5.0))

        nums = seasonal_cycle(idx, speed, path=tmp_path / "s.png", title="t")

        assert nums["windiest_month"] == 7
        assert nums["calmest_month"] == 1
        assert nums["seasonal_range_ms"] == pytest.approx(8.0)

    def test_a_flat_record_has_no_seasonal_range(self, tmp_path):
        idx = pd.date_range("2021-01-01", periods=365, freq="D")

        nums = seasonal_cycle(idx, np.full(365, 4.0), path=tmp_path / "s.png", title="t")

        assert nums["seasonal_range_ms"] == pytest.approx(0.0)

    def test_zero_scatter_gives_nan_not_infinity(self, tmp_path):
        """An unguarded divide returns inf, which reads as an overwhelming seasonal
        signal -- the exact opposite of what no variance at all means."""
        import math

        idx = pd.date_range("2021-01-01", periods=365, freq="D")

        nums = seasonal_cycle(idx, np.full(365, 4.0), path=tmp_path / "s.png", title="t")

        assert math.isnan(nums["range_over_within_month_sd"])
        assert nums["within_month_sd_ms"] == pytest.approx(0.0)


class TestSpeedMap:
    def test_measures_the_field_and_locates_its_maximum(self, tmp_path):
        lat = np.array([17.0, 18.0, 19.0])
        lon = np.array([278.0, 279.0])
        u = np.zeros((3, 2))
        v = np.zeros((3, 2))
        u[2, 1] = 3.0
        v[2, 1] = 4.0                                   # speed 5 at (19 N, 279 E)

        nums = speed_map(lat, lon, u, v, path=tmp_path / "m.png", title="t")

        assert nums["speed_max_ms"] == pytest.approx(5.0)
        assert nums["max_at_lat"] == pytest.approx(19.0)
        assert nums["max_at_lon"] == pytest.approx(-81.0)     # 279 E in display frame
        assert nums["cells"] == 6
        assert (tmp_path / "m.png").exists()

    def test_reports_the_land_mask_fraction(self, tmp_path):
        """HYCOM carries NaN over land -- about 11.8 % of the box. Wind carries none,
        so this number distinguishes the two products without naming either."""
        u = np.array([[1.0, np.nan], [1.0, 1.0]])
        v = np.zeros((2, 2))

        nums = speed_map(np.array([17.0, 18.0]), np.array([278.0, 279.0]), u, v,
                         path=tmp_path / "m.png", title="t")

        assert nums["nan_fraction"] == pytest.approx(0.25)
