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
