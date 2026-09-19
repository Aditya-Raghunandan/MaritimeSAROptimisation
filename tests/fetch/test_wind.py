"""Unit tests for sar.fetch.wind.

No network: `open_wind_store` is monkeypatched to return a small synthetic
dataset shaped like ARCO-ERA5 -- full CF variable names, `latitude`/`longitude`
axes, latitude DESCENDING and longitude 0-360, which are the three things about
that store that have each caused a real bug.
"""

import numpy as np
import pytest
import xarray as xr

from sar.fetch import wind


@pytest.fixture
def arco_shaped() -> xr.Dataset:
    """A dataset with ARCO-ERA5's names and axis orders, not ours.

    latitude descends (90 -> -90, as ERA5 serves it) and longitude is 0-360, so
    a test that passes here would also have caught both of the traps documented
    in `fetch_wind_box`.
    """
    lat = np.array([30.0, 25.0, 20.0])          # DESCENDING, on purpose
    lon = np.array([280.0, 290.0, 300.0])       # 0-360, on purpose
    time = np.array(
        ["2019-01-01T00:00", "2019-01-01T01:00", "2019-01-01T02:00"],
        dtype="datetime64[ns]",
    )

    shape = (len(time), len(lat), len(lon))
    u = np.arange(np.prod(shape), dtype="float64").reshape(shape)

    return xr.Dataset(
        {
            "10m_u_component_of_wind": (("time", "latitude", "longitude"), u),
            "10m_v_component_of_wind": (("time", "latitude", "longitude"), -u),
            "mean_sea_level_pressure": (("time", "latitude", "longitude"), u + 101325.0),
        },
        coords={"time": time, "latitude": lat, "longitude": lon},
    )


class TestFetchWind:
    def test_returns_exactly_u10_v10_and_time(self, monkeypatch, arco_shaped):
        monkeypatch.setattr(wind, "open_wind_store", lambda: arco_shaped)
        result = wind.fetch_wind("2019-01-01", "00:00", lat=25.0, lon=-70.0)
        assert set(result) == {"u10", "v10", "time"}
        assert isinstance(result["u10"], float)
        assert isinstance(result["v10"], float)

    def test_renames_the_full_cf_names_to_u10_v10(self, monkeypatch, arco_shaped):
        # "10u"/"10v" are the CDS/GRIB short codes and do not exist in this
        # store; selecting them raises KeyError. The rename is the whole point.
        monkeypatch.setattr(wind, "open_wind_store", lambda: arco_shaped)
        result = wind.fetch_wind("2019-01-01", "00:00", lat=25.0, lon=-70.0)
        expected = arco_shaped["10m_u_component_of_wind"].sel(
            time="2019-01-01T00:00", latitude=25.0, longitude=290.0
        ).item()
        assert result["u10"] == pytest.approx(expected)
        assert result["v10"] == pytest.approx(-expected)

    def test_converts_longitude_to_the_stores_0_360_convention(
        self, monkeypatch, arco_shaped
    ):
        # -70 must land on 290. Without the conversion, nearest-neighbour on an
        # 0-360 axis returns the 280 cell (the closest to zero) and answers
        # confidently with the wrong cell rather than raising.
        monkeypatch.setattr(wind, "open_wind_store", lambda: arco_shaped)
        at_minus_70 = wind.fetch_wind("2019-01-01", "00:00", lat=25.0, lon=-70.0)
        at_290 = wind.fetch_wind("2019-01-01", "00:00", lat=25.0, lon=290.0)
        assert at_minus_70["u10"] == pytest.approx(at_290["u10"])

        on_280 = arco_shaped["10m_u_component_of_wind"].sel(
            time="2019-01-01T00:00", latitude=25.0, longitude=280.0
        ).item()
        assert at_minus_70["u10"] != pytest.approx(on_280)

    def test_nearest_neighbour_in_space(self, monkeypatch, arco_shaped):
        monkeypatch.setattr(wind, "open_wind_store", lambda: arco_shaped)
        # lat 26 is nearer 25 than 30; lon -69 (-> 291) is nearer 290 than 300.
        result = wind.fetch_wind("2019-01-01", "00:00", lat=26.0, lon=-69.0)
        expected = arco_shaped["10m_u_component_of_wind"].sel(
            time="2019-01-01T00:00", latitude=25.0, longitude=290.0
        ).item()
        assert result["u10"] == pytest.approx(expected)

    def test_nearest_neighbour_in_time_and_reports_what_it_picked(
        self, monkeypatch, arco_shaped
    ):
        monkeypatch.setattr(wind, "open_wind_store", lambda: arco_shaped)
        # 01:20 is nearer 01:00 than 02:00. The caller is told which it got.
        result = wind.fetch_wind("2019-01-01", "01:20", lat=25.0, lon=-70.0)
        assert result["time"] == np.datetime64("2019-01-01T01:00")

    def test_descending_latitude_does_not_break_the_lookup(
        self, monkeypatch, arco_shaped
    ):
        # slice() on a descending axis silently returns empty; sel(nearest)
        # does not. This pins the distinction so the point path is not
        # "fixed" into using a slice later.
        monkeypatch.setattr(wind, "open_wind_store", lambda: arco_shaped)
        result = wind.fetch_wind("2019-01-01", "00:00", lat=30.0, lon=-70.0)
        expected = arco_shaped["10m_u_component_of_wind"].sel(
            time="2019-01-01T00:00", latitude=30.0, longitude=290.0
        ).item()
        assert result["u10"] == pytest.approx(expected)

    def test_matches_the_box_pull_at_the_same_point(self, monkeypatch, arco_shaped):
        """The two code paths must agree, or the engine and the archive diverge."""
        monkeypatch.setattr(wind, "open_wind_store", lambda: arco_shaped)
        point = wind.fetch_wind("2019-01-01", "00:00", lat=25.0, lon=-70.0)
        box = wind.fetch_wind_box(
            "2019-01-01", "2019-01-02",
            lat_bounds=(20.0, 30.0), lon_bounds=(-80.0, -60.0),
        )
        from_box = box["u10"].sel(
            time=np.datetime64("2019-01-01T00:00"), lat=25.0, lon=290.0
        ).item()
        assert point["u10"] == pytest.approx(from_box)


class TestFetchWindBox:
    def test_normalises_to_the_d020_convention(self, monkeypatch, arco_shaped):
        monkeypatch.setattr(wind, "open_wind_store", lambda: arco_shaped)
        sub = wind.fetch_wind_box(
            "2019-01-01", "2019-01-02",
            lat_bounds=(20.0, 30.0), lon_bounds=(-80.0, -60.0),
        )
        assert set(sub.dims) == {"time", "lat", "lon"}
        assert (np.diff(sub["lat"].values) > 0).all()     # ascending now
        assert (np.diff(sub["lon"].values) > 0).all()

    def test_all_nan_window_is_refused(self, monkeypatch, arco_shaped):
        # ARCO's time axis is padded to 2050; a slice past the valid range
        # returns all-NaN rather than raising.
        blank = arco_shaped.copy(deep=True)
        blank["10m_u_component_of_wind"].values[:] = np.nan
        monkeypatch.setattr(wind, "open_wind_store", lambda: blank)
        with pytest.raises(SystemExit, match="all-NaN"):
            wind.fetch_wind_box(
                "2019-01-01", "2019-01-02",
                lat_bounds=(20.0, 30.0), lon_bounds=(-80.0, -60.0),
            )
