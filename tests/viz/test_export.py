"""Tests for the frontend bundle. No network.

The one that matters most is the round trip: a cell reconstructed purely from
the manifest's grid arithmetic must equal the source, because that is the
contract the browser relies on. The client never sees coordinates -- it
rebuilds them from `lat0 + i * dlat` -- so if the manifest and the buffer
disagree the map is wrong in a way that still looks like weather.
"""

import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from sar.utils.geo import to_display_longitude
from sar.viz.export import export_window, grid_spec, write_field


def wind_shaped(n_time: int = 6, start: str = "2021-01-01") -> xr.Dataset:
    """A D020-compliant wind file: lat/lon ascending, longitude 0-360."""
    lat = np.arange(17.0, 18.01, 0.25)          # 5
    lon = np.arange(278.0, 279.01, 0.25)        # 5
    time = pd.date_range(start, periods=n_time, freq="h")
    rng = np.random.default_rng(0)
    shape = (time.size, lat.size, lon.size)
    return xr.Dataset(
        {
            "u10": (("time", "lat", "lon"), rng.normal(size=shape)),
            "v10": (("time", "lat", "lon"), rng.normal(size=shape)),
        },
        coords={"time": time, "lat": lat, "lon": lon},
    )


def write_archive(data_dir, ds, name="era5_17-36N_82-63W_20210101-20210102.nc"):
    raw = data_dir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(raw / name)
    return raw / name


class TestGridSpec:
    def test_describes_a_regular_grid_in_display_longitude(self):
        spec = grid_spec(np.array([17.0, 17.25, 17.5]), np.array([278.0, 278.25]))

        assert spec["lat0"] == pytest.approx(17.0)
        assert spec["dlat"] == pytest.approx(0.25)
        assert spec["nlat"] == 3
        assert spec["lon0"] == pytest.approx(-82.0)      # 278 E, converted for the client
        assert spec["nlon"] == 2

    def test_an_irregular_axis_raises(self):
        """The client rebuilds coordinates from ONE step. On an irregular axis
        that is simply wrong, and a wrong map still looks like a map."""
        with pytest.raises(ValueError, match="lon is not regularly spaced"):
            grid_spec(np.array([17.0, 17.25]), np.array([278.0, 278.25, 279.0]))

    def test_an_irregular_latitude_raises(self):
        with pytest.raises(ValueError, match="lat is not regularly spaced"):
            grid_spec(np.array([17.0, 17.25, 18.0]), np.array([278.0, 278.25]))

    def test_a_single_point_axis_raises(self):
        with pytest.raises(ValueError, match="at least 2 points"):
            grid_spec(np.array([17.0]), np.array([278.0, 278.25]))


class TestWriteField:
    def test_writes_interleaved_little_endian_float32(self, tmp_path):
        ds = wind_shaped(n_time=2)
        enc = write_field(ds, "u10", "v10", tmp_path / "wind.f32")

        expected = 2 * 5 * 5 * 2 * 4
        assert enc["bytes"] == expected
        assert enc["dtype"] == "float32"
        assert enc["components"] == 2
        assert enc["order"] == ["u", "v"]

        buf = np.fromfile(tmp_path / "wind.f32", dtype="<f4").reshape(2, 5, 5, 2)
        assert buf[1, 2, 3, 0] == pytest.approx(ds.u10.values[1, 2, 3], abs=1e-6)
        assert buf[1, 2, 3, 1] == pytest.approx(ds.v10.values[1, 2, 3], abs=1e-6)


class TestExportWindow:
    def test_round_trips_a_cell_through_the_manifest_arithmetic(self, tmp_path):
        """The contract the browser depends on, end to end."""
        ds = wind_shaped(n_time=6)
        write_archive(tmp_path, ds)
        out = tmp_path / "bundle"

        manifest = export_window(tmp_path, "2021-01-01", "2021-01-01T04:00:00", out)

        g = manifest["layers"][0]["grid"]
        buf = np.fromfile(out / "wind.f32", dtype="<f4").reshape(
            manifest["clock"]["frames"], g["nlat"], g["nlon"], 2)

        t, j, i = 2, 3, 1
        lat = g["lat0"] + j * g["dlat"]
        lon = g["lon0"] + i * g["dlon"]

        assert lat == pytest.approx(float(ds.lat.values[j]))
        assert lon == pytest.approx(float(to_display_longitude(ds.lon.values)[i]))
        assert buf[t, j, i, 0] == pytest.approx(float(ds.u10.values[t, j, i]), abs=1e-6)

    def test_the_clock_is_end_exclusive(self, tmp_path):
        write_archive(tmp_path, wind_shaped(n_time=6))

        manifest = export_window(tmp_path, "2021-01-01", "2021-01-01T04:00:00",
                                 tmp_path / "bundle")

        assert manifest["clock"]["frames"] == 4
        assert manifest["clock"]["step_seconds"] == 3600

    def test_records_provenance_including_the_arrival_convention(self, tmp_path):
        """A published map has to be traceable to the file and commit behind it."""
        write_archive(tmp_path, wind_shaped())

        manifest = export_window(tmp_path, "2021-01-01", "2021-01-01T03:00:00",
                                 tmp_path / "bundle")

        assert manifest["provenance"]["source_file"].startswith("era5_")
        assert manifest["provenance"]["arrival_convention"] == "d020"
        assert manifest["provenance"]["script"] == "sar.viz.export"

    def test_manifest_is_valid_json_with_one_field_layer(self, tmp_path):
        write_archive(tmp_path, wind_shaped())
        out = tmp_path / "bundle"

        export_window(tmp_path, "2021-01-01", "2021-01-01T03:00:00", out)

        m = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        assert [layer["type"] for layer in m["layers"]] == ["field"]
        assert m["layers"][0]["source"] == "wind.f32"
        assert m["layers"][0]["units"] == "m/s"

    def test_bbox_is_in_display_longitude(self, tmp_path):
        """GeoJSON and every web map require -180..180; D020 stores 0-360. This
        is the one boundary where the conversion happens."""
        write_archive(tmp_path, wind_shaped())

        m = export_window(tmp_path, "2021-01-01", "2021-01-01T03:00:00", tmp_path / "bundle")

        lat_min, lon_min, lat_max, lon_max = m["bbox"]
        assert lon_min == pytest.approx(-82.0)
        assert lon_max < 0
        assert lat_min < lat_max

    def test_an_unknown_product_raises(self, tmp_path):
        write_archive(tmp_path, wind_shaped())
        with pytest.raises(ValueError, match="unknown product"):
            export_window(tmp_path, "2021-01-01", "2021-01-01T03:00:00",
                          tmp_path / "bundle", product="swell")

    def test_a_window_no_file_covers_raises(self, tmp_path):
        write_archive(tmp_path, wind_shaped())
        with pytest.raises(ValueError, match="no era5_.*covers"):
            export_window(tmp_path, "2023-06-01", "2023-06-02", tmp_path / "bundle")

    def test_an_empty_archive_raises(self, tmp_path):
        (tmp_path / "raw").mkdir()
        with pytest.raises(FileNotFoundError, match="no era5_"):
            export_window(tmp_path, "2021-01-01", "2021-01-02", tmp_path / "bundle")
