"""Tests for the read half. No network, no fixture files: everything synthetic.

The load-bearing case is `TestOpenForcing::test_accepts_a_pre_d020_file_and_says_so`.
The local archive holds files in BOTH conventions -- pre-D020 ones written
before 2026-09-15 and compliant ones written after -- so a loader that only
asserts would reject half of it, and one that silently repaired would hide that
half forever.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from sar.utils.data_io import (
    ARRIVAL_D020,
    ARRIVAL_PRE_D020,
    load_drifters,
    open_box_means,
    open_forcing,
    select_window,
)


def era5_shaped(n_time: int = 48, start: str = "2021-01-01") -> xr.Dataset:
    """A pre-D020 wind file: latitude/longitude, descending lat, -180..180 lon.

    Shaped like the real local slices, verified 2026-09-16 against
    raw/era5_17-36N_82-63W_20210101-20210108.nc.
    """
    lat = np.arange(36.0, 16.9, -0.25)
    lon = np.arange(-82.0, -62.9, 0.25)
    time = pd.date_range(start, periods=n_time, freq="h")
    shape = (time.size, lat.size, lon.size)
    rng = np.random.default_rng(0)
    return xr.Dataset(
        {
            "u10": (("time", "latitude", "longitude"), rng.normal(size=shape)),
            "v10": (("time", "latitude", "longitude"), rng.normal(size=shape)),
        },
        coords={"time": time, "latitude": lat, "longitude": lon},
    )


def hycom_shaped(n_time: int = 8, start: str = "2021-01-01") -> xr.Dataset:
    """A D020-compliant current file: lat/lon, ascending, 0-360, 3-hourly."""
    lat = np.arange(17.0, 36.01, 0.5)
    lon = np.arange(278.0, 297.01, 0.5)
    time = pd.date_range(start, periods=n_time, freq="3h")
    shape = (time.size, lat.size, lon.size)
    return xr.Dataset(
        {
            "water_u": (("time", "lat", "lon"), np.zeros(shape)),
            "water_v": (("time", "lat", "lon"), np.zeros(shape)),
        },
        coords={"time": time, "lat": lat, "lon": lon},
    )


def drifter_frame() -> pd.DataFrame:
    """Two buoys shaped like the GDP CSV: ERDDAP names, Kelvin sst, a 6 h gap.

    Buoy A loses its drogue partway through, so it exercises the per-observation
    rule. Buoy B has a gap wide enough to split its track into two segments.
    """
    t = pd.date_range("2021-06-01", periods=6, freq="h")
    a = pd.DataFrame({
        "ID": "A", "time": t,
        "latitude": np.linspace(30.0, 30.5, 6),
        "longitude": np.linspace(-70.0, -69.5, 6),
        "sst": [298.15] * 5 + [1273.15],                  # last is a fill value
        "drogue_lost_date": pd.Timestamp("2021-06-01 03:00"),
    })
    t_b = list(pd.date_range("2021-06-01", periods=3, freq="h"))
    t_b += list(pd.date_range("2021-06-01 09:00", periods=3, freq="h"))   # 6 h gap
    b = pd.DataFrame({
        "ID": "B", "time": t_b,
        "latitude": np.linspace(25.0, 25.5, 6),
        "longitude": np.linspace(-75.0, -74.5, 6),
        "sst": 300.15,
        "drogue_lost_date": pd.NaT,
    })
    return pd.concat([a, b], ignore_index=True)


def box_mean_frame(year: int, n: int = 24) -> pd.DataFrame:
    idx = pd.date_range(f"{year}-01-01", periods=n, freq="h", name="time")
    return pd.DataFrame(
        {
            "u10": 1.0, "v10": 2.0, "speed": 2.236, "dir_from_deg": 206.6,
            "msl_hpa": 1018.0, "msl_spread_hpa": 4.0, "hour_utc": idx.hour,
        },
        index=idx,
    )


def write_gdp_csv(path, df, *, units_row: bool = False):
    """Write a drifter CSV, optionally with ERDDAP's units line under the header.

    The real file on disk HAS that line -- verified 2026-09-16 -- so both shapes
    have to load.
    """
    df.to_csv(path, index=False)
    if units_row:
        lines = path.read_text(encoding="utf-8").splitlines()
        units = ",UTC,degrees_north,degrees_east,Kelvin,UTC"
        out = [lines[0], units, *lines[1:]]
        path.write_text("\n".join(out) + "\n", encoding="utf-8")


class TestOpenForcing:
    def test_accepts_a_pre_d020_file_and_says_so(self, tmp_path):
        """The whole point: the local archive is pre-D020 and must still load."""
        p = tmp_path / "era5_old.nc"
        era5_shaped().to_netcdf(p)

        with pytest.warns(UserWarning, match="before D020"):
            ds = open_forcing(p)

        assert ds.attrs["sar_arrival_convention"] == ARRIVAL_PRE_D020
        assert set(ds.coords) >= {"lat", "lon", "time"}
        assert (np.diff(ds.lat.values) > 0).all()
        assert ds.lon.values.min() >= 0.0 and ds.lon.values.max() < 360.0

    def test_accepts_a_compliant_file_without_warning(self, tmp_path):
        p = tmp_path / "hycom_new.nc"
        hycom_shaped().to_netcdf(p)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            ds = open_forcing(p)

        assert ds.attrs["sar_arrival_convention"] == ARRIVAL_D020

    def test_repairs_longitude_to_the_store_convention(self, tmp_path):
        """-82 W must come back as 278 E, not as -82."""
        p = tmp_path / "era5_old.nc"
        era5_shaped().to_netcdf(p)

        with pytest.warns(UserWarning):
            ds = open_forcing(p)

        assert ds.lon.values.min() == pytest.approx(278.0)
        assert ds.lon.values.max() == pytest.approx(297.0)

    def test_records_the_source_path(self, tmp_path):
        p = tmp_path / "hycom_new.nc"
        hycom_shaped().to_netcdf(p)
        assert open_forcing(p).attrs["sar_source_path"] == str(p)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="no forcing file"):
            open_forcing(tmp_path / "absent.nc")


class TestSelectWindow:
    def test_end_is_exclusive(self, tmp_path):
        """The off-by-one that gave an eight-day file 192 steps instead of 168."""
        p = tmp_path / "w.nc"
        era5_shaped(n_time=72, start="2021-01-01").to_netcdf(p)
        with pytest.warns(UserWarning):
            ds = open_forcing(p)

        sub = select_window(ds, "2021-01-01", "2021-01-02")

        assert sub.sizes["time"] == 24
        assert sub.time.values.max() < np.datetime64("2021-01-02")

    def test_start_is_inclusive(self, tmp_path):
        p = tmp_path / "w.nc"
        era5_shaped(n_time=72).to_netcdf(p)
        with pytest.warns(UserWarning):
            ds = open_forcing(p)

        sub = select_window(ds, "2021-01-02", "2021-01-03")

        assert sub.time.values.min() == np.datetime64("2021-01-02T00:00:00")

    def test_window_outside_the_file_raises(self, tmp_path):
        p = tmp_path / "w.nc"
        era5_shaped(n_time=24).to_netcdf(p)
        with pytest.warns(UserWarning):
            ds = open_forcing(p)

        with pytest.raises(ValueError, match="no timesteps"):
            select_window(ds, "2023-01-01", "2023-01-02")


class TestLoadDrifters:
    def test_renames_axes_and_converts_longitude(self, tmp_path):
        p = tmp_path / "gdp.csv"
        drifter_frame().to_csv(p, index=False)

        df = load_drifters(p)

        assert {"lat", "lon"} <= set(df.columns)
        assert "latitude" not in df.columns
        assert df.lon.min() >= 0.0 and df.lon.max() < 360.0
        assert df.lon.min() == pytest.approx(285.0)      # -75 E

    def test_masks_the_sst_fill_values(self, tmp_path):
        """1273.15 K is 1000 C. ~99.9 % of values are plausible, so a naive
        mean looks almost right -- which is why this must be masked, not trusted."""
        p = tmp_path / "gdp.csv"
        drifter_frame().to_csv(p, index=False)

        df = load_drifters(p)

        assert df.sst_c.max() < 40.0
        assert df.sst_c.isna().sum() == 1
        assert df.sst_c.dropna().iloc[0] == pytest.approx(25.0)

    def test_undrogued_is_per_observation_not_per_buoy(self, tmp_path):
        """Buoy A is drogued before 03:00 and undrogued after. Both states, one buoy."""
        p = tmp_path / "gdp.csv"
        drifter_frame().to_csv(p, index=False)

        df = load_drifters(p)
        a = df[df.ID == "A"]

        assert a.undrogued.any() and not a.undrogued.all()
        assert not df[df.ID == "B"].undrogued.any()

    def test_a_gap_over_three_hours_splits_the_segment(self, tmp_path):
        p = tmp_path / "gdp.csv"
        drifter_frame().to_csv(p, index=False)

        df = load_drifters(p)

        assert df[df.ID == "A"].segment_id.nunique() == 1
        assert df[df.ID == "B"].segment_id.nunique() == 2
        assert df.segment_id.nunique() == 3

    def test_out_of_box_rows_are_dropped(self, tmp_path):
        raw = drifter_frame()
        raw.loc[raw.ID == "B", "latitude"] = 5.0          # south of the box
        p = tmp_path / "gdp.csv"
        raw.to_csv(p, index=False)

        df = load_drifters(p)

        assert set(df.ID) == {"A"}

    def test_in_box_false_keeps_everything(self, tmp_path):
        raw = drifter_frame()
        raw.loc[raw.ID == "B", "latitude"] = 5.0
        p = tmp_path / "gdp.csv"
        raw.to_csv(p, index=False)

        assert set(load_drifters(p, in_box=False).ID) == {"A", "B"}

    def test_nothing_in_the_box_raises(self, tmp_path):
        raw = drifter_frame()
        raw["latitude"] = 5.0
        p = tmp_path / "gdp.csv"
        raw.to_csv(p, index=False)

        with pytest.raises(ValueError, match="inside the study box"):
            load_drifters(p)

    def test_missing_columns_raise(self, tmp_path):
        p = tmp_path / "gdp.csv"
        drifter_frame().drop(columns=["latitude"]).to_csv(p, index=False)

        with pytest.raises(ValueError, match="missing"):
            load_drifters(p)

    def test_reads_a_file_carrying_erddap_units_row(self, tmp_path):
        """The real 108.8 MB file has it. Left in, every numeric column loads as
        object dtype and the first comparison raises TypeError."""
        p = tmp_path / "gdp.csv"
        write_gdp_csv(p, drifter_frame(), units_row=True)

        df = load_drifters(p)

        assert len(df) == 12
        assert df.lat.dtype.kind == "f"
        assert df.lon.dtype.kind == "f"

    def test_a_file_without_the_units_row_keeps_every_observation(self, tmp_path):
        """Detected, not assumed -- a blind skiprows=[1] would eat a real row."""
        p = tmp_path / "gdp.csv"
        write_gdp_csv(p, drifter_frame(), units_row=False)

        assert len(load_drifters(p)) == 12

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="no drifter CSV"):
            load_drifters(tmp_path / "absent.csv")


class TestOpenBoxMeans:
    def test_concatenates_a_directory_in_time_order(self, tmp_path):
        box_mean_frame(2021).to_parquet(tmp_path / "wind_box_mean_20210101-20220101.parquet")
        box_mean_frame(2020).to_parquet(tmp_path / "wind_box_mean_20200101-20210101.parquet")

        df = open_box_means(tmp_path)

        assert len(df) == 48
        assert df.index.is_monotonic_increasing
        assert df.index[0].year == 2020

    def test_accepts_a_single_file(self, tmp_path):
        p = tmp_path / "wind_box_mean_20210101-20220101.parquet"
        box_mean_frame(2021).to_parquet(p)

        assert len(open_box_means(p)) == 24

    def test_overlapping_files_raise(self, tmp_path):
        """Yearly files are written end-exclusive, so they abut. An overlap must
        not be silently dropped."""
        box_mean_frame(2021).to_parquet(tmp_path / "wind_box_mean_20210101-20220101.parquet")
        box_mean_frame(2021).to_parquet(tmp_path / "wind_box_mean_20210101-20220102.parquet")

        with pytest.raises(ValueError, match="duplicate timestamps"):
            open_box_means(tmp_path)

    def test_the_overlap_error_names_the_files_smallest_first(self, tmp_path):
        """The real directory held five yearly files plus three 10 Sep test slices,
        312 duplicate hours, and nothing wrong with any single file. The message
        has to point at the leftover, not at a phantom boundary bug."""
        box_mean_frame(2021, n=24).to_parquet(tmp_path / "wind_box_mean_20210101-20220101.parquet")
        box_mean_frame(2021, n=6).to_parquet(tmp_path / "wind_box_mean_20210101-20210102.parquet")

        with pytest.raises(ValueError) as e:
            open_box_means(tmp_path)

        msg = str(e.value)
        assert "wind_box_mean_20210101-20210102.parquet" in msg
        assert "wind_box_mean_20210101-20220101.parquet" in msg
        # smallest first: the leftover slice is listed above the yearly file
        assert msg.index("20210101-20210102") < msg.index("20210101-20220101")

    def test_empty_directory_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="no wind_box_mean"):
            open_box_means(tmp_path)
