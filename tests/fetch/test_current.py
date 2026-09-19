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


@pytest.fixture
def box_dataset() -> xr.Dataset:
    """A dataset covering the real D014 box, with a land mask.

    Deliberately spans exactly 2021-01-05 -> 2021-01-08 at 3-hourly cadence, so
    the end-exclusive boundary check below is the same 24-timestep case the
    issue states, and 2021-01-05 is the date of the already-passing
    figures/report/current_quiver_20210105.png.

    Three rows of latitude are NaN, standing in for the Florida/Carolina coast:
    3/20 = 15 % of the box, inside the [0.02, 0.40] land-mask bound.
    """
    depth = np.array([0.0, 2.0, 4.0])
    lat = np.arange(17.0, 37.0, 1.0)          # 20 rows, ascending
    lon = np.arange(278.0, 298.0, 1.0)        # 20 cols, 0-360
    time = np.arange(
        np.datetime64("2021-01-05T00:00"),
        np.datetime64("2021-01-08T00:00"),
        np.timedelta64(3, "h"),
    ).astype("datetime64[ns]")

    shape = (len(time), len(depth), len(lat), len(lon))
    water_u = np.arange(np.prod(shape), dtype="float64").reshape(shape)
    water_v = -water_u.copy()
    water_u[:, :, :3, :] = np.nan
    water_v[:, :, :3, :] = np.nan

    ds = xr.Dataset(
        {
            "water_u": (("time", "depth", "lat", "lon"), water_u),
            "water_v": (("time", "depth", "lat", "lon"), water_v),
        },
        coords={"time": time, "depth": depth, "lat": lat, "lon": lon},
    )
    for v in ("water_u", "water_v"):
        ds[v].attrs["units"] = "m/s"
    return ds


class TestEstimateBytes:
    def test_three_days_is_twenty_four_timesteps(self):
        # 3-hourly, so 3 days is 24 steps, not 72.
        assert current.estimate_bytes("2021-01-05", "2021-01-08") == 24 * current.BYTES_PER_TIMESTEP

    def test_one_year_is_about_2_6_gb(self):
        got = current.estimate_bytes("2019-01-01", "2020-01-01")
        assert 2.4e9 < got < 2.8e9

    def test_five_years_is_about_12_9_gb(self):
        got = current.estimate_bytes("2019-01-01", "2024-01-01")
        assert 12.0e9 < got < 13.5e9

    def test_reversed_range_is_zero_not_negative(self):
        assert current.estimate_bytes("2021-01-08", "2021-01-05") == 0


class TestWriteCurrentNetcdf:
    def test_writes_exactly_the_expected_filename_and_nothing_else(
        self, monkeypatch, box_dataset, tmp_path
    ):
        monkeypatch.setattr(current, "open_current_dataset", lambda: box_dataset)
        path = current.write_current_netcdf("2021-01-05", "2021-01-08", tmp_path)

        assert path == tmp_path / "raw" / "hycom_17-36N_82-63W_20210105-20210108.nc"
        assert path.exists()
        assert [p.name for p in (tmp_path / "raw").iterdir()] == [path.name]
        assert [p.name for p in tmp_path.iterdir()] == ["raw"]

    def test_round_trip_satisfies_d020_conventions(self, monkeypatch, box_dataset, tmp_path):
        monkeypatch.setattr(current, "open_current_dataset", lambda: box_dataset)
        path = current.write_current_netcdf("2021-01-05", "2021-01-08", tmp_path)

        from sar.utils.geo import assert_conventions

        with xr.open_dataset(path) as ds:
            assert_conventions(ds)                      # raises if it is not
            assert set(ds.dims) == {"time", "lat", "lon"}
            assert (np.diff(ds["lat"].values) > 0).all()
            assert (np.diff(ds["lon"].values) > 0).all()
            assert ds["lon"].values.min() >= 0.0 and ds["lon"].values.max() < 360.0

    def test_depth_is_selected_not_left_as_a_length_one_dimension(
        self, monkeypatch, box_dataset, tmp_path
    ):
        monkeypatch.setattr(current, "open_current_dataset", lambda: box_dataset)
        path = current.write_current_netcdf("2021-01-05", "2021-01-08", tmp_path)
        with xr.open_dataset(path) as ds:
            assert "depth" not in ds.dims

    def test_end_is_exclusive(self, monkeypatch, box_dataset, tmp_path):
        monkeypatch.setattr(current, "open_current_dataset", lambda: box_dataset)
        path = current.write_current_netcdf("2021-01-05", "2021-01-08", tmp_path)
        with xr.open_dataset(path) as ds:
            assert ds.sizes["time"] == 24
            assert ds["time"].values.max() < np.datetime64("2021-01-08")

    def test_units_survive_the_round_trip(self, monkeypatch, box_dataset, tmp_path):
        monkeypatch.setattr(current, "open_current_dataset", lambda: box_dataset)
        path = current.write_current_netcdf("2021-01-05", "2021-01-08", tmp_path)
        with xr.open_dataset(path) as ds:
            assert ds["water_u"].attrs["units"] == "m/s"
            assert ds["water_v"].attrs["units"] == "m/s"

    def test_missing_units_are_stamped_rather_than_lost(
        self, monkeypatch, box_dataset, tmp_path
    ):
        for v in ("water_u", "water_v"):
            box_dataset[v].attrs.pop("units")
        monkeypatch.setattr(current, "open_current_dataset", lambda: box_dataset)
        path = current.write_current_netcdf("2021-01-05", "2021-01-08", tmp_path)
        with xr.open_dataset(path) as ds:
            assert ds["water_u"].attrs["units"] == "m/s"

    def test_land_mask_survives_the_write(self, monkeypatch, box_dataset, tmp_path):
        monkeypatch.setattr(current, "open_current_dataset", lambda: box_dataset)
        path = current.write_current_netcdf("2021-01-05", "2021-01-08", tmp_path)
        with xr.open_dataset(path) as ds:
            nan_fraction = float(np.isnan(ds["water_u"].values).mean())
        assert current.NAN_FRACTION_MIN <= nan_fraction <= current.NAN_FRACTION_MAX

    # ---- the guards. Every one of these gets a test that fires it. ----

    def test_oversized_range_refuses_without_force(self, monkeypatch, box_dataset, tmp_path):
        monkeypatch.setattr(current, "open_current_dataset", lambda: box_dataset)
        with pytest.raises(SystemExit, match="over the"):
            current.write_current_netcdf("2019-01-01", "2024-01-01", tmp_path)
        assert not (tmp_path / "raw").exists()

    def test_wrong_units_raise(self, monkeypatch, box_dataset, tmp_path):
        box_dataset["water_u"].attrs["units"] = "cm/s"
        monkeypatch.setattr(current, "open_current_dataset", lambda: box_dataset)
        with pytest.raises(ValueError, match="cm/s"):
            current.write_current_netcdf("2021-01-05", "2021-01-08", tmp_path)

    def test_all_finite_box_is_refused_as_a_missing_land_mask(
        self, monkeypatch, box_dataset, tmp_path
    ):
        # No NaN anywhere means the box is not where the caller thinks it is --
        # the 0-360 convention being the usual reason.
        filled = box_dataset.fillna(0.0)
        for v in ("water_u", "water_v"):
            filled[v].attrs["units"] = "m/s"
        monkeypatch.setattr(current, "open_current_dataset", lambda: filled)
        with pytest.raises(SystemExit, match="land mask"):
            current.write_current_netcdf("2021-01-05", "2021-01-08", tmp_path)

    def test_mostly_nan_box_is_refused(self, monkeypatch, box_dataset, tmp_path):
        box_dataset["water_u"].values[:] = np.nan
        box_dataset["water_v"].values[:] = np.nan
        monkeypatch.setattr(current, "open_current_dataset", lambda: box_dataset)
        with pytest.raises(SystemExit, match="land mask"):
            current.write_current_netcdf("2021-01-05", "2021-01-08", tmp_path)


class TestFetchCurrentRange:
    """Long windows must be split into requests the server will actually serve.

    A half-year in one OPeNDAP call is 1,448 timesteps and times out with a 500
    after about half an hour, having returned nothing. The per-timestep rate
    measured on a 4-day request gave no warning, because the rate and the
    largest answerable request are different limits.
    """

    def test_splits_a_long_window_into_pieces(self, monkeypatch, box_dataset):
        calls = []
        real = current.fetch_current_box

        def spy(start, end, *a, **k):
            calls.append((start[:10], end[:10]))
            return real(start, end, *a, **k)

        monkeypatch.setattr(current, "open_current_dataset", lambda: box_dataset)
        monkeypatch.setattr(current, "fetch_current_box", spy)
        current.fetch_current_range("2021-01-05", "2021-01-08",
                                    (17.0, 36.0), (-82.0, -63.0),
                                    chunk_days=1, verbose=False)
        assert calls == [("2021-01-05", "2021-01-06"),
                         ("2021-01-06", "2021-01-07"),
                         ("2021-01-07", "2021-01-08")]

    def test_the_pieces_reassemble_into_the_whole_window(self, monkeypatch, box_dataset):
        monkeypatch.setattr(current, "open_current_dataset", lambda: box_dataset)
        whole = current.fetch_current_box("2021-01-05", "2021-01-08",
                                          (17.0, 36.0), (-82.0, -63.0))
        split = current.fetch_current_range("2021-01-05", "2021-01-08",
                                            (17.0, 36.0), (-82.0, -63.0),
                                            chunk_days=1, verbose=False)
        assert split.sizes["time"] == whole.sizes["time"] == 24
        assert np.array_equal(split["time"].values, whole["time"].values)
        assert np.allclose(split["water_u"].values, whole["water_u"].values, equal_nan=True)

    def test_a_window_shorter_than_one_chunk_is_a_single_request(
        self, monkeypatch, box_dataset
    ):
        calls = []
        real = current.fetch_current_box

        def spy(start, end, *a, **k):
            calls.append(start)
            return real(start, end, *a, **k)

        monkeypatch.setattr(current, "open_current_dataset", lambda: box_dataset)
        monkeypatch.setattr(current, "fetch_current_box", spy)
        current.fetch_current_range("2021-01-05", "2021-01-08",
                                    (17.0, 36.0), (-82.0, -63.0),
                                    chunk_days=30, verbose=False)
        assert len(calls) == 1

    def test_a_ragged_last_piece_is_not_truncated(self, monkeypatch, box_dataset):
        # 3 days in 2-day pieces: the second piece is one day, not two.
        monkeypatch.setattr(current, "open_current_dataset", lambda: box_dataset)
        out = current.fetch_current_range("2021-01-05", "2021-01-08",
                                          (17.0, 36.0), (-82.0, -63.0),
                                          chunk_days=2, verbose=False)
        assert out.sizes["time"] == 24
        assert out["time"].values.max() < np.datetime64("2021-01-08")

    def test_overlapping_pieces_are_caught_rather_than_silently_duplicated(
        self, monkeypatch, box_dataset
    ):
        """If a boundary ever goes wrong, the archive must not gain a duplicate hour."""
        monkeypatch.setattr(current, "open_current_dataset", lambda: box_dataset)
        # Force every piece to cover the same window, which is what a broken
        # boundary would look like.
        monkeypatch.setattr(current, "fetch_current_box",
                            lambda *a, **k: _fixed_window(box_dataset))
        with pytest.raises(AssertionError, match="duplicate timestamps"):
            current.fetch_current_range("2021-01-05", "2021-01-08",
                                        (17.0, 36.0), (-82.0, -63.0),
                                        chunk_days=1, verbose=False)


def _fixed_window(ds):
    """The same three timesteps every time, standing in for a boundary bug."""
    return ds[["water_u", "water_v"]].isel(time=slice(0, 3)).sel(depth=0.0)


class TestResume:
    """A bulk pull from this endpoint WILL be resubmitted, so it must not redo work."""

    def test_an_existing_file_is_skipped(self, monkeypatch, box_dataset, tmp_path, capsys):
        monkeypatch.setattr(current, "open_current_dataset", lambda: box_dataset)
        first = current.write_current_netcdf("2021-01-05", "2021-01-08", tmp_path)
        stamp = first.stat().st_mtime_ns

        # A second call must not re-fetch: fail loudly if it tries.
        monkeypatch.setattr(current, "fetch_current_range",
                            lambda *a, **k: pytest.fail("re-fetched an existing window"))
        again = current.write_current_netcdf("2021-01-05", "2021-01-08", tmp_path)
        assert again == first
        assert again.stat().st_mtime_ns == stamp
        assert "skip" in capsys.readouterr().out

    def test_force_re_fetches(self, monkeypatch, box_dataset, tmp_path):
        monkeypatch.setattr(current, "open_current_dataset", lambda: box_dataset)
        path = current.write_current_netcdf("2021-01-05", "2021-01-08", tmp_path)
        calls = []
        real = current.fetch_current_range
        monkeypatch.setattr(current, "fetch_current_range",
                            lambda *a, **k: (calls.append(1), real(*a, **k))[1])
        current.write_current_netcdf("2021-01-05", "2021-01-08", tmp_path, force=True)
        assert len(calls) == 1
        assert path.exists()
