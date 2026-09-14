"""Unit tests for sar.fetch.current.

These never touch the real HYCOM server: open_current_dataset is monkeypatched
to return a small synthetic xr.Dataset with the same dimension names and
conventions (ascending lat, 0-360 lon, depth level 0 = 0 m) as the real store,
so the tests are fast, deterministic, and runnable offline / in CI.
"""

import numpy as np
import pytest
import xarray as xr

from sar.fetch import current


@pytest.fixture
def synthetic_dataset() -> xr.Dataset:
    """A small dataset shaped like the real HYCOM store.

    depth = [0, 2, 4]     (level 0 is 0 m, as D019 requires)
    lat   = [10, 20, 30]  (ascending, unlike ERA5)
    lon   = [260, 270, 280, 290, 300]   (0-360 convention)
    time  = three 3-hourly steps starting 2019-01-01T00:00

    water_u/water_v are filled with distinct values per grid point so a
    nearest-neighbour selection can be checked exactly.
    """
    depth = np.array([0.0, 2.0, 4.0])
    lat = np.array([10.0, 20.0, 30.0])
    lon = np.array([260.0, 270.0, 280.0, 290.0, 300.0])
    time = np.array(
        ["2019-01-01T00:00", "2019-01-01T03:00", "2019-01-01T06:00"],
        dtype="datetime64[ns]",
    )

    shape = (len(time), len(depth), len(lat), len(lon))
    size = np.prod(shape)
    water_u = np.arange(size, dtype="float64").reshape(shape)
    water_v = -water_u

    return xr.Dataset(
        {
            "water_u": (("time", "depth", "lat", "lon"), water_u),
            "water_v": (("time", "depth", "lat", "lon"), water_v),
        },
        coords={"time": time, "depth": depth, "lat": lat, "lon": lon},
    )


class TestToStoreLongitude:
    def test_negative_longitude_wraps_to_0_360(self):
        assert current.to_store_longitude(-70.0) == pytest.approx(290.0)

    def test_zero_is_unchanged(self):
        assert current.to_store_longitude(0.0) == pytest.approx(0.0)

    def test_positive_longitude_unchanged(self):
        assert current.to_store_longitude(120.0) == pytest.approx(120.0)

    def test_minus_180_wraps_to_180(self):
        assert current.to_store_longitude(-180.0) == pytest.approx(180.0)


class TestOpenCurrentDataset:
    def test_asserts_depth_level_0_is_zero(self, monkeypatch, synthetic_dataset):
        bad = synthetic_dataset.assign_coords(depth=[1.0, 2.0, 4.0])
        monkeypatch.setattr(xr, "open_dataset", lambda *a, **k: bad)
        with pytest.raises(AssertionError, match="depth level 0"):
            current.open_current_dataset()

    def test_passes_through_a_valid_dataset(self, monkeypatch, synthetic_dataset):
        monkeypatch.setattr(xr, "open_dataset", lambda *a, **k: synthetic_dataset)
        ds = current.open_current_dataset()
        assert ds["depth"].values[0] == 0.0

    def test_drops_the_unparseable_tau_variable(self, monkeypatch, synthetic_dataset):
        captured = {}

        def fake_open_dataset(url, drop_variables=None):
            captured["drop_variables"] = drop_variables
            return synthetic_dataset

        monkeypatch.setattr(xr, "open_dataset", fake_open_dataset)
        current.open_current_dataset()
        assert captured["drop_variables"] == ["tau"]


class TestFetchCurrent:
    def test_returns_expected_keys_and_types(self, monkeypatch, synthetic_dataset):
        monkeypatch.setattr(current, "open_current_dataset", lambda: synthetic_dataset)
        result = current.fetch_current("2019-01-01", "00:00", lat=10.0, lon=-100.0)
        assert set(result) == {"water_u", "water_v", "time"}
        assert isinstance(result["water_u"], float)
        assert isinstance(result["water_v"], float)

    def test_exact_grid_point_returns_exact_value(self, monkeypatch, synthetic_dataset):
        monkeypatch.setattr(current, "open_current_dataset", lambda: synthetic_dataset)
        # lon=-100 -> 260 in store convention, an exact grid point.
        result = current.fetch_current("2019-01-01", "00:00", lat=10.0, lon=-100.0)
        expected = synthetic_dataset["water_u"].sel(
            time="2019-01-01T00:00", depth=0.0, lat=10.0, lon=260.0
        ).item()
        assert result["water_u"] == pytest.approx(expected)
        assert result["water_v"] == pytest.approx(-expected)

    def test_nearest_neighbour_off_grid(self, monkeypatch, synthetic_dataset):
        monkeypatch.setattr(current, "open_current_dataset", lambda: synthetic_dataset)
        # lat=11 is closer to 10 than 20; lon=-99 (-> 261) closer to 260 than 270.
        result = current.fetch_current("2019-01-01", "01:00", lat=11.0, lon=-99.0)
        expected = synthetic_dataset["water_u"].sel(
            time="2019-01-01T00:00", depth=0.0, lat=10.0, lon=260.0
        ).item()
        assert result["water_u"] == pytest.approx(expected)


class TestFetchCurrentBox:
    def test_subsets_expected_shape(self, monkeypatch, synthetic_dataset):
        monkeypatch.setattr(current, "open_current_dataset", lambda: synthetic_dataset)
        sub = current.fetch_current_box(
            "2019-01-01", "2019-01-01T06:00",
            lat_bounds=(10.0, 20.0), lon_bounds=(-100.0, -80.0),
        )
        assert sub.sizes["lat"] == 2   # 10, 20
        assert sub.sizes["lon"] == 3   # 260, 270, 280
        assert "water_u" in sub and "water_v" in sub

    def test_empty_latitude_raises(self, monkeypatch, synthetic_dataset):
        monkeypatch.setattr(current, "open_current_dataset", lambda: synthetic_dataset)
        with pytest.raises(AssertionError, match="latitude"):
            current.fetch_current_box(
                "2019-01-01", "2019-01-01T06:00",
                lat_bounds=(40.0, 50.0), lon_bounds=(-100.0, -80.0),
            )

    def test_empty_time_range_raises(self, monkeypatch, synthetic_dataset):
        monkeypatch.setattr(current, "open_current_dataset", lambda: synthetic_dataset)
        with pytest.raises(AssertionError, match="no timesteps"):
            current.fetch_current_box(
                "2020-01-01", "2020-01-02",
                lat_bounds=(10.0, 20.0), lon_bounds=(-100.0, -80.0),
            )
