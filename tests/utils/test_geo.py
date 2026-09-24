"""Tests for the D020 coordinate conventions. No network: synthetic grids only."""

import numpy as np
import pytest
import xarray as xr

from sar.utils.geo import (
    M_PER_DEG_LAT,
    assert_conventions,
    east_north,
    metres_per_degree_lon,
    normalise_grid,
    offset_position,
    regular_axis_step,
    to_display_longitude,
    to_store_longitude,
)


def era5_shaped():
    """An ERA5-shaped dataset: latitude/longitude, descending lat, -180..180 lon."""
    lat = np.arange(36.0, 16.9, -0.25)      # DESCENDING, as ARCO serves it
    lon = np.arange(-82.0, -62.9, 0.25)     # -180..180, as wind.py used to write
    data = np.random.default_rng(0).normal(size=(lat.size, lon.size))
    return xr.Dataset(
        {"u10": (("latitude", "longitude"), data)},
        coords={"latitude": lat, "longitude": lon},
    )


def hycom_shaped():
    """A HYCOM-shaped dataset: lat/lon, ascending, 0-360. Already compliant."""
    lat = np.arange(17.0, 36.01, 0.04)
    lon = np.arange(278.0, 297.01, 0.08)
    data = np.zeros((lat.size, lon.size))
    return xr.Dataset({"water_u": (("lat", "lon"), data)}, coords={"lat": lat, "lon": lon})


def test_store_longitude_wraps_west_to_0_360():
    assert to_store_longitude(-82.0) == pytest.approx(278.0)
    assert to_store_longitude(-63.0) == pytest.approx(297.0)


def test_display_longitude_is_the_inverse():
    for v in (-82.0, -63.0, -0.5, 0.0, 179.9):
        assert to_display_longitude(to_store_longitude(v)) == pytest.approx(v)


def test_normalise_renames_axes():
    out = normalise_grid(era5_shaped())
    assert "lat" in out.coords and "lon" in out.coords
    assert "latitude" not in out.coords and "longitude" not in out.coords


def test_normalise_sorts_latitude_ascending():
    """The failure this guards is silent: sel(lat=slice(17, 36)) on a descending
    axis returns an EMPTY selection and does not raise."""
    out = normalise_grid(era5_shaped())
    assert (np.diff(out.lat.values) > 0).all()
    assert out.sel(lat=slice(20, 30)).sizes["lat"] > 0


def test_normalise_converts_longitude_to_0_360():
    out = normalise_grid(era5_shaped())
    assert out.lon.values.min() >= 0.0 and out.lon.values.max() < 360.0
    assert out.lon.values.min() == pytest.approx(278.0)


def test_normalise_does_not_reorder_the_data_away_from_its_coords():
    """Sorting must carry the values with the axis, not just relabel it."""
    src = era5_shaped()
    probe = float(src["u10"].sel(latitude=36.0, longitude=-82.0))
    out = normalise_grid(src)
    assert float(out["u10"].sel(lat=36.0, lon=278.0)) == pytest.approx(probe)


def test_normalise_is_idempotent_and_leaves_hycom_alone():
    once = normalise_grid(hycom_shaped())
    twice = normalise_grid(once)
    assert once.equals(twice)


def test_the_two_fetchers_agree_after_normalisation():
    """The actual point of D020: wind and current become the same kind of object."""
    w = normalise_grid(era5_shaped())
    c = normalise_grid(hycom_shaped())
    assert set(w.sizes) >= {"lat", "lon"} and set(c.sizes) >= {"lat", "lon"}
    for d in ("lat", "lon"):
        assert (np.diff(w[d].values) > 0).all() and (np.diff(c[d].values) > 0).all()
    assert w.lon.values.min() >= 0 and c.lon.values.min() >= 0


def test_assert_conventions_rejects_minus180_longitude():
    with pytest.raises(AssertionError, match="0-360"):
        assert_conventions(era5_shaped().rename({"latitude": "lat", "longitude": "lon"}))


def test_assert_conventions_rejects_descending_latitude():
    ds = era5_shaped().rename({"latitude": "lat", "longitude": "lon"})
    ds = ds.assign_coords(lon=to_store_longitude(ds.lon))
    with pytest.raises(AssertionError, match="ascending"):
        assert_conventions(ds)


def test_assert_conventions_accepts_a_normalised_dataset():
    assert_conventions(normalise_grid(era5_shaped()))
    assert_conventions(normalise_grid(hycom_shaped()))


def test_normalise_raises_on_a_dataset_with_no_recognisable_axes():
    ds = xr.Dataset({"x": (("a", "b"), np.zeros((2, 2)))}, coords={"a": [1, 2], "b": [3, 4]})
    with pytest.raises(KeyError):
        normalise_grid(ds)


class TestRegularAxisStep:
    """The one definition of a regular axis, shared by the engine, the exporter and the
    table pivot. Its job is to accept float32 coordinates and refuse a real irregularity."""

    def test_returns_the_step_of_a_clean_axis(self):
        assert regular_axis_step(np.arange(0.0, 1.0, 0.25)) == pytest.approx(0.25)

    def test_accepts_the_float32_longitude_the_hycom_archive_stores(self):
        # The real axis from data/current/current_2019-01-01_2019-01-03.txt: a true 0.08
        # degree grid whose float32 storage gives gaps of 0.079956 to 0.080018. Comparing
        # consecutive gaps rejected this, which is why that file could not be read.
        axis = np.float32(278.0 + 0.08 * np.arange(238)).astype(float)
        assert regular_axis_step(axis, "lon") == pytest.approx(0.08, abs=1e-6)
        gaps = np.diff(axis)
        assert not np.allclose(gaps, gaps[0], rtol=0, atol=1e-9)   # the old test failed here

    def test_accepts_the_float32_latitude_too(self):
        axis = np.float32(17.0 + 0.04 * np.arange(476)).astype(float)
        assert regular_axis_step(axis, "lat") == pytest.approx(0.04, abs=1e-6)

    def test_accepts_a_descending_axis_by_returning_a_negative_step(self):
        assert regular_axis_step([36.0, 35.75, 35.5, 35.25]) == pytest.approx(-0.25)

    def test_refuses_a_genuinely_irregular_axis(self):
        with pytest.raises(ValueError, match="not regularly spaced"):
            regular_axis_step([0.0, 0.25, 0.75, 1.0])

    def test_refuses_a_stretched_axis_even_though_every_gap_grows_smoothly(self):
        # A curvilinear grid: each gap differs only slightly from its neighbour, so a
        # pairwise check can pass it, while the fitted line cannot.
        axis = np.cumsum(np.linspace(0.08, 0.09, 200))
        with pytest.raises(ValueError, match="not regularly spaced"):
            regular_axis_step(axis, "lon")

    def test_says_how_far_off_it_was_and_what_was_allowed(self):
        with pytest.raises(ValueError, match="of a step"):
            regular_axis_step([0.0, 0.25, 0.75, 1.0], "lat")

    def test_needs_at_least_two_points(self):
        with pytest.raises(ValueError, match="at least 2 points"):
            regular_axis_step([1.0])

    def test_refuses_an_axis_that_does_not_move(self):
        with pytest.raises(ValueError, match="zero step"):
            regular_axis_step([5.0, 5.0, 5.0])


class TestEastNorth:
    def test_compass_points(self):
        for bearing, expected in ((0, (0, 1)), (90, (1, 0)), (180, (0, -1)), (270, (-1, 0))):
            e, n = east_north(bearing, 1.0)
            assert (e, n) == pytest.approx(expected, abs=1e-12)

    def test_is_clockwise_from_north(self):
        e, n = east_north(45.0, np.sqrt(2.0))
        assert (e, n) == pytest.approx((1.0, 1.0))


class TestOffsetPosition:
    def test_metres_become_degrees_at_the_reference_latitude(self):
        lat, lon = offset_position(26.5, 281.0, 1000.0, 2000.0)
        assert (lat - 26.5) * M_PER_DEG_LAT == pytest.approx(2000.0)
        assert (lon - 281.0) * metres_per_degree_lon(26.5) == pytest.approx(1000.0)

    def test_accepts_arrays(self):
        lat, lon = offset_position(np.array([17.0, 36.0]), 281.0, 100.0, 0.0)
        assert lat.shape == lon.shape == (2,)
