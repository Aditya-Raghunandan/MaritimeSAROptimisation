"""Unit tests for sar.viz.archive.

No network. Synthetic NetCDF files are written into a tmp_path archive that
looks like `<data>/raw/era5_*.nc`, because the thing under test is precisely
how several files on disk become one published store.
"""

import json

import numpy as np
import pytest
import xarray as xr

from sar.viz import archive


def _wind_file(path, start, hours, *, lat_desc=False, lon_signed=False):
    """Write one era5-shaped NetCDF. Defaults to the D020 convention.

    `lat_desc`/`lon_signed` reproduce the pre-D020 local slices, which is the
    mixed archive `open_forcing` has to cope with on the way in.
    """
    lat = np.arange(17.0, 36.25, 0.25)
    lon = np.arange(278.0, 297.25, 0.25)
    time = np.arange(np.datetime64(start), np.datetime64(start) + np.timedelta64(hours, "h"),
                     np.timedelta64(1, "h"))

    shape = (time.size, lat.size, lon.size)
    # Smoothly varying rather than random, so a stride is checkable by value.
    u = np.arange(np.prod(shape), dtype="float32").reshape(shape)
    ds = xr.Dataset(
        {"u10": (("lat", "lon", "time"), u.transpose(1, 2, 0)),
         "v10": (("lat", "lon", "time"), -u.transpose(1, 2, 0))},
        coords={"lat": lat, "lon": lon, "time": time},
    ).transpose("time", "lat", "lon")

    if lon_signed:
        ds = ds.assign_coords(lon=((ds.lon.values + 180.0) % 360.0) - 180.0).sortby("lon")
        ds = ds.rename({"lat": "latitude", "lon": "longitude"})
    if lat_desc:
        name = "latitude" if lon_signed else "lat"
        ds = ds.isel({name: slice(None, None, -1)})

    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)
    return ds


@pytest.fixture
def archive_dir(tmp_path):
    """Two adjacent, non-overlapping wind files: 48 h in total."""
    _wind_file(tmp_path / "raw" / "era5_17-36N_82-63W_20210101-20210102.nc",
               "2021-01-01T00", 24)
    _wind_file(tmp_path / "raw" / "era5_17-36N_82-63W_20210102-20210103.nc",
               "2021-01-02T00", 24)
    return tmp_path


class TestOpenArchive:
    def test_concatenates_files_into_one_time_axis(self, archive_dir):
        ds, names = archive.open_archive(archive_dir, "wind")
        assert ds.sizes["time"] == 48
        assert len(names) == 2
        assert (np.diff(ds["time"].values) == np.timedelta64(1, "h")).all()

    def test_time_axis_comes_back_sorted(self, archive_dir):
        ds, _ = archive.open_archive(archive_dir, "wind")
        assert (np.diff(ds["time"].values) > np.timedelta64(0, "s")).all()

    def test_normalises_pre_d020_files_rather_than_rejecting_them(self, tmp_path):
        # The local slices are descending-latitude, -180..180, latitude/longitude.
        _wind_file(tmp_path / "raw" / "era5_old.nc", "2021-01-01T00", 6,
                   lat_desc=True, lon_signed=True)
        with pytest.warns(UserWarning, match="before D020"):
            ds, _ = archive.open_archive(tmp_path, "wind")
        assert set(ds.dims) == {"time", "lat", "lon"}
        assert (np.diff(ds["lat"].values) > 0).all()
        assert ds["lon"].values.min() >= 0.0

    def test_overlapping_files_are_deduplicated_and_named(self, archive_dir):
        # A re-fetch of day 2 that runs one hour long, i.e. a real overlap.
        _wind_file(archive_dir / "raw" / "era5_17-36N_82-63W_20210102-20210103b.nc",
                   "2021-01-02T00", 24)
        with pytest.warns(UserWarning, match="duplicate timestamps") as record:
            ds, _ = archive.open_archive(archive_dir, "wind")
        assert ds.sizes["time"] == 48                      # not 72
        assert (np.diff(ds["time"].values) > np.timedelta64(0, "s")).all()
        # The point of the warning is that it says WHICH files.
        assert "20210102-20210103b.nc" in str(record[0].message)

    def test_empty_archive_raises(self, tmp_path):
        (tmp_path / "raw").mkdir()
        with pytest.raises(FileNotFoundError, match="era5_"):
            archive.open_archive(tmp_path, "wind")

    def test_file_named_for_a_product_it_does_not_contain_raises(self, tmp_path):
        # A hycom_*.nc holding wind variables: the glob finds it, so the error
        # has to come from the contents rather than from the filename.
        _wind_file(tmp_path / "raw" / "hycom_17-36N_82-63W_20210101-20210102.nc",
                   "2021-01-01T00", 6)
        with pytest.raises(KeyError, match="water_u"):
            archive.open_archive(tmp_path, "current")


class TestToDisplayGrid:
    def test_converts_to_minus_180_180_and_sorts(self, archive_dir):
        ds, _ = archive.open_archive(archive_dir, "wind")
        out = archive.to_display_grid(ds)
        lon = out["lon"].values
        assert lon.min() >= -180.0 and lon.max() < 180.0
        assert (np.diff(lon) > 0).all()
        assert lon[0] == pytest.approx(-82.0)

    def test_values_follow_their_cell_across_the_conversion(self, archive_dir):
        ds, _ = archive.open_archive(archive_dir, "wind")
        out = archive.to_display_grid(ds)
        # 290 in store convention is -70 in display convention: same cell.
        assert out["u10"].sel(lon=-70.0).values == pytest.approx(
            ds["u10"].sel(lon=290.0).values
        )


class TestWriteZarrTier:
    def test_stride_one_keeps_every_frame(self, archive_dir, tmp_path):
        ds = archive.to_display_grid(archive.open_archive(archive_dir, "wind")[0])
        info = archive.write_zarr_tier(ds, tmp_path / "w.zarr", 1, level=1)
        assert info["frames"] == 48
        assert info["step_seconds"] == 3600

    def test_stride_six_keeps_every_sixth_frame(self, archive_dir, tmp_path):
        ds = archive.to_display_grid(archive.open_archive(archive_dir, "wind")[0])
        info = archive.write_zarr_tier(ds, tmp_path / "w.zarr", 6, level=1)
        assert info["frames"] == 8
        assert info["step_seconds"] == 6 * 3600

    def test_subsampling_is_a_stride_not_an_average(self, archive_dir, tmp_path):
        """A 24 h mean of a rotating wind vector is ~0; the tier must show a real hour."""
        ds = archive.to_display_grid(archive.open_archive(archive_dir, "wind")[0])
        archive.write_zarr_tier(ds, tmp_path / "w.zarr", 6, level=1)
        published = xr.open_zarr(tmp_path / "w.zarr", consolidated=False)
        for k in range(published.sizes["time"]):
            assert published["u10"].isel(time=k, lat=3, lon=4).values == pytest.approx(
                ds["u10"].isel(time=6 * k, lat=3, lon=4).values
            )
        published.close()

    def test_round_trips_through_zarr_unchanged(self, archive_dir, tmp_path):
        ds = archive.to_display_grid(archive.open_archive(archive_dir, "wind")[0])
        archive.write_zarr_tier(ds, tmp_path / "w.zarr", 1, level=1)
        published = xr.open_zarr(tmp_path / "w.zarr", consolidated=False)
        assert np.allclose(published["u10"].values, ds["u10"].values.astype("float32"))
        assert np.array_equal(published["lon"].values, ds["lon"].values)
        published.close()

    def test_chunks_the_whole_spatial_box(self, archive_dir, tmp_path):
        ds = archive.to_display_grid(archive.open_archive(archive_dir, "wind")[0])
        info = archive.write_zarr_tier(ds, tmp_path / "w.zarr", 1, time_chunk=24, level=1)
        assert info["chunks"] == {"time": 24, "lat": ds.sizes["lat"], "lon": ds.sizes["lon"]}

    def test_time_chunk_is_clamped_to_a_short_tier(self, archive_dir, tmp_path):
        # A daily tier of an 8-day archive has 8 frames; asking for 48 must not
        # produce a chunk larger than the array.
        ds = archive.to_display_grid(archive.open_archive(archive_dir, "wind")[0])
        info = archive.write_zarr_tier(ds, tmp_path / "w.zarr", 24, level=1)
        assert info["frames"] == 2
        assert info["chunks"]["time"] == 2

    def test_records_the_compression_it_used(self, archive_dir, tmp_path):
        ds = archive.to_display_grid(archive.open_archive(archive_dir, "wind")[0])
        info = archive.write_zarr_tier(ds, tmp_path / "w.zarr", 1, level=5)
        assert info["compression"] == "zstd-5"
        assert info["compression_ratio"] > 1.0
        assert info["bytes"] > 0

    def test_zero_stride_raises(self, archive_dir, tmp_path):
        ds = archive.to_display_grid(archive.open_archive(archive_dir, "wind")[0])
        with pytest.raises(ValueError, match="stride must be"):
            archive.write_zarr_tier(ds, tmp_path / "w.zarr", 0)

    def test_stride_longer_than_the_record_gives_one_frame_not_an_error(
        self, archive_dir, tmp_path
    ):
        # "daily" over an eight-hour archive is one frame, not a failure.
        ds = archive.to_display_grid(archive.open_archive(archive_dir, "wind")[0])
        info = archive.write_zarr_tier(ds.isel(time=slice(0, 2)), tmp_path / "w.zarr", 99,
                                       level=1)
        assert info["frames"] == 1
        assert info["step_seconds"] == 0

    def test_empty_dataset_raises(self, archive_dir, tmp_path):
        ds = archive.to_display_grid(archive.open_archive(archive_dir, "wind")[0])
        with pytest.raises(ValueError, match="no timesteps"):
            archive.write_zarr_tier(ds.isel(time=slice(0, 0)), tmp_path / "w.zarr", 1)


class TestVerifyTier:
    def test_reports_a_match_for_a_faithful_store(self, archive_dir, tmp_path):
        ds = archive.to_display_grid(archive.open_archive(archive_dir, "wind")[0])
        archive.write_zarr_tier(ds, tmp_path / "w.zarr", 1, level=1)
        result = archive.verify_tier(tmp_path / "w.zarr", ds, "u10")
        assert result["match"] is True
        assert result["published"] == pytest.approx(result["source"])

    def test_catches_a_corrupted_store(self, archive_dir, tmp_path):
        """If this cannot fail, it is not a check."""
        ds = archive.to_display_grid(archive.open_archive(archive_dir, "wind")[0])
        archive.write_zarr_tier(ds, tmp_path / "w.zarr", 1, level=1)
        shifted = ds.copy(deep=True)
        shifted["u10"] = shifted["u10"] + 1000.0
        result = archive.verify_tier(tmp_path / "w.zarr", shifted, "u10")
        assert result["match"] is False


class TestExportArchive:
    def test_writes_every_tier_and_a_manifest(self, archive_dir, tmp_path):
        out = tmp_path / "published"
        m = archive.export_archive(archive_dir, out, "wind",
                                   tiers={"hourly": 1, "6-hourly": 6}, level=1)
        assert set(m["tiers"]) == {"hourly", "6-hourly"}
        assert (out / "wind_hourly.zarr").is_dir()
        assert (out / "wind_6-hourly.zarr").is_dir()
        assert (out / "wind_archive.json").is_file()
        on_disk = json.loads((out / "wind_archive.json").read_text(encoding="utf-8"))
        assert on_disk["tiers"]["hourly"]["frames"] == 48

    def test_every_tier_verifies_against_the_source(self, archive_dir, tmp_path):
        m = archive.export_archive(archive_dir, tmp_path / "p", "wind",
                                   tiers={"hourly": 1, "daily": 24}, level=1)
        assert all(t["verified"]["match"] for t in m["tiers"].values())

    def test_manifest_records_the_display_convention_not_the_stored_one(
        self, archive_dir, tmp_path
    ):
        m = archive.export_archive(archive_dir, tmp_path / "p", "wind",
                                   tiers={"hourly": 1}, level=1)
        assert m["grid"]["lon0"] == pytest.approx(-82.0)
        assert m["bbox"] == pytest.approx([17.0, -82.0, 36.0, -63.0])
        assert "-180..180" in m["longitude_convention"]

    def test_manifest_names_its_sources(self, archive_dir, tmp_path):
        m = archive.export_archive(archive_dir, tmp_path / "p", "wind",
                                   tiers={"hourly": 1}, level=1)
        assert len(m["provenance"]["source_files"]) == 2
        assert m["provenance"]["source_frames"] == 48

    def test_unknown_product_raises(self, archive_dir, tmp_path):
        with pytest.raises(ValueError, match="unknown product"):
            archive.export_archive(archive_dir, tmp_path / "p", "swell")


class TestTimeAxisSpec:
    """A gapped axis must not be publishable by accident.

    `grid_spec` in sar.viz.export already refuses an irregular lat/lon axis.
    This is the same guard on the third axis, which did not have one until the
    real local archive -- eight days of January plus two of March -- produced a
    manifest claiming 264 hourly frames ending on a date 51 days out.
    """

    def _axis(self, *segments):
        parts = [
            np.arange(np.datetime64(s), np.datetime64(s) + np.timedelta64(h, "h"),
                      np.timedelta64(1, "h"))
            for s, h in segments
        ]
        return np.concatenate(parts)

    def test_regular_axis_is_described_by_start_and_step(self):
        spec = archive.time_axis_spec(self._axis(("2021-01-01T00", 48)))
        assert spec["regular"] is True
        assert spec["frames"] == 48
        assert spec["step_seconds"] == 3600
        assert spec["gaps"] is None

    def test_gapped_axis_raises_by_default(self):
        times = self._axis(("2021-01-01T00", 24), ("2021-03-01T00", 24))
        with pytest.raises(ValueError, match="time axis is not regular"):
            archive.time_axis_spec(times)

    def test_the_error_names_the_largest_gap_and_where_it_is(self):
        times = self._axis(("2021-01-01T00", 24), ("2021-03-01T00", 24))
        with pytest.raises(ValueError) as e:
            archive.time_axis_spec(times)
        assert "2021-01-01T23:00:00Z" in str(e.value)     # the frame before the hole
        assert "1393 h" in str(e.value)

    def test_allow_gaps_publishes_but_marks_it(self):
        times = self._axis(("2021-01-01T00", 24), ("2021-03-01T00", 24))
        spec = archive.time_axis_spec(times, allow_gaps=True)
        assert spec["regular"] is False
        assert len(spec["gaps"]) == 1
        assert spec["gaps"][0]["gap_hours"] == pytest.approx(1393.0)

    def test_single_frame_axis_is_trivially_regular(self):
        spec = archive.time_axis_spec(self._axis(("2021-01-01T00", 1)))
        assert spec["regular"] is True
        assert spec["step_seconds"] == 0


class TestGappedArchiveIsNotPublished:
    def test_export_refuses_a_gapped_archive(self, tmp_path):
        _wind_file(tmp_path / "raw" / "era5_a.nc", "2021-01-01T00", 24)
        _wind_file(tmp_path / "raw" / "era5_b.nc", "2021-03-01T00", 24)
        with pytest.raises(ValueError, match="time axis is not regular"):
            archive.export_archive(tmp_path, tmp_path / "p", "wind",
                                   tiers={"hourly": 1}, level=1)

    def test_nothing_is_written_when_the_axis_is_refused(self, tmp_path):
        """Discovering a bad axis must cost nothing, not a gigabyte of upload."""
        _wind_file(tmp_path / "raw" / "era5_a.nc", "2021-01-01T00", 24)
        _wind_file(tmp_path / "raw" / "era5_b.nc", "2021-03-01T00", 24)
        out = tmp_path / "p"
        with pytest.raises(ValueError):
            archive.export_archive(tmp_path, out, "wind", tiers={"hourly": 1}, level=1)
        assert not (out / "wind_hourly.zarr").exists()

    def test_allow_gaps_lets_it_through(self, tmp_path):
        _wind_file(tmp_path / "raw" / "era5_a.nc", "2021-01-01T00", 24)
        _wind_file(tmp_path / "raw" / "era5_b.nc", "2021-03-01T00", 24)
        m = archive.export_archive(tmp_path, tmp_path / "p", "wind",
                                   tiers={"hourly": 1}, level=1, allow_gaps=True)
        assert m["tiers"]["hourly"]["regular"] is False


class TestRegulariseTime:
    """The source really is missing timesteps; publishing them honestly.

    HYCOM's study window has five irregular steps -- four of 6 h and one of
    12 h against a 3-hourly cadence, which D011 records as 7 missing steps in
    14,608. Interpolating across them would invent current fields in the very
    store other results are checked against.
    """

    def _gapped(self, drop):
        """A 3-hourly dataset with some timesteps removed."""
        time = np.arange(np.datetime64("2021-01-01T00"),
                         np.datetime64("2021-01-03T00"),
                         np.timedelta64(3, "h")).astype("datetime64[ns]")
        keep = np.array([k for k in range(time.size) if k not in drop])
        lat = np.arange(17.0, 19.0, 1.0)
        lon = np.arange(278.0, 280.0, 1.0)
        u = np.arange(time.size * lat.size * lon.size, dtype="float32").reshape(
            (time.size, lat.size, lon.size))
        ds = xr.Dataset(
            {"water_u": (("time", "lat", "lon"), u),
             "water_v": (("time", "lat", "lon"), -u)},
            coords={"time": time, "lat": lat, "lon": lon},
        )
        return ds.isel(time=keep), time

    def test_a_gap_free_axis_is_returned_untouched(self):
        ds, _ = self._gapped(drop=[])
        out, info = archive.regularise_time(ds)
        assert info["inserted"] == 0
        assert out.sizes["time"] == ds.sizes["time"]

    def test_missing_steps_come_back_as_nan(self):
        ds, full = self._gapped(drop=[3])
        out, info = archive.regularise_time(ds)
        assert info["inserted"] == 1
        assert out.sizes["time"] == full.size
        assert np.array_equal(out["time"].values, full)
        assert np.isnan(out["water_u"].isel(time=3).values).all()

    def test_a_double_gap_inserts_both(self):
        # A 12 h hole in a 3-hourly axis is three missing steps, not one.
        ds, full = self._gapped(drop=[4, 5, 6])
        out, info = archive.regularise_time(ds)
        assert info["inserted"] == 3
        assert out.sizes["time"] == full.size

    def test_the_real_values_are_not_moved(self):
        """The whole point: nothing is interpolated and nothing shifts."""
        ds, _ = self._gapped(drop=[2, 5])
        out, _ = archive.regularise_time(ds)
        for t in ds["time"].values:
            assert np.array_equal(out["water_u"].sel(time=t).values,
                                  ds["water_u"].sel(time=t).values)

    def test_the_axis_is_regular_afterwards(self):
        ds, _ = self._gapped(drop=[1, 4, 5])
        out, _ = archive.regularise_time(ds)
        # This is the property that lets the client reconstruct timestamps.
        archive.time_axis_spec(out["time"].values)      # raises if not regular

    def test_a_cadence_that_is_simply_wrong_raises(self):
        # A step that is not a whole multiple of the modal one is not a gap --
        # it means the cadence itself is wrong, and filling would hide that.
        time = np.array(["2021-01-01T00", "2021-01-01T03", "2021-01-01T07"],
                        dtype="datetime64[ns]")
        ds = xr.Dataset({"water_u": (("time",), np.zeros(3, dtype="float32"))},
                        coords={"time": time})
        with pytest.raises(ValueError, match="not whole multiples"):
            archive.regularise_time(ds)


class TestPerProductChunking:
    """A chunk is one HTTP GET, so it decides what the browser downloads.

    HYCOM is 476 x 238 against ERA5's 77 x 77 -- 19x the cells per timestep --
    so the same chunk length is 1.1 MB for wind and 29 MB for current. 29 MB to
    show one frame is unusable, and that, not disk space, is the real
    constraint. Shortening the current chunk fixes it at FULL spatial
    resolution; halving the grid would have thrown away half the data to solve
    the same problem worse.
    """

    def test_current_chunks_are_shorter_than_wind_chunks(self):
        assert archive.TIME_CHUNK["current"] < archive.TIME_CHUNK["wind"]

    def test_the_chunks_land_near_the_same_download_size(self):
        wind = archive.TIME_CHUNK["wind"] * 77 * 77 * 2 * 4
        current = archive.TIME_CHUNK["current"] * 476 * 238 * 2 * 4
        # Within an order of magnitude of each other, where the naive shared
        # chunk was 26x apart.
        assert current / wind < 10

    def test_an_unknown_product_falls_back_rather_than_raising(self):
        assert archive.TIME_CHUNK.get("swell", archive.DEFAULT_TIME_CHUNK) == 48

    def test_the_tier_records_the_chunk_it_used(self, archive_dir, tmp_path):
        m = archive.export_archive(archive_dir, tmp_path / "w", "wind",
                                   tiers={"hourly": 1}, level=1)
        assert m["tiers"]["hourly"]["chunks"]["time"] == archive.TIME_CHUNK["wind"]

    def test_full_spatial_resolution_is_preserved(self, archive_dir, tmp_path):
        """Nothing is decimated in space -- the whole point of the revision."""
        ds = archive.to_display_grid(archive.open_archive(archive_dir, "wind")[0])
        info = archive.write_zarr_tier(ds, tmp_path / "w.zarr", 1, level=1)
        assert info["chunks"]["lat"] == ds.sizes["lat"]
        assert info["chunks"]["lon"] == ds.sizes["lon"]
