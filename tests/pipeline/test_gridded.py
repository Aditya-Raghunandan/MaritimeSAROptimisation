"""Tests for sar.pipeline.gridded (issue #88). Synthetic HYCOM- and ERA5-shaped files only.

The data are float32 like the real files. The default field is linear in grid index and in
hours, with dyadic coefficients, so every stored value is exact in float32 and the right
answer anywhere is known on paper: bilinear-in-space, linear-in-time interpolation is exact
for such a field.
"""

import json

import numpy as np
import pytest
import xarray as xr

from sar.fetch.current import box_tag
from sar.model.interpolate import CURRENT_VARS, WIND_VARS, OutOfCoverageError, sample_field
from sar.pipeline import ensemble, track
from sar.pipeline.forcing import ConstantForcing
from sar.pipeline.gridded import (
    ForcingGapError,
    GriddedForcing,
    RegularAxis,
    TimeAxis,
    _cli,
    as_ns,
    cell_weights,
    usable_corners,
)
from sar.pipeline.track import DriftPipeline

TAG = box_tag()
T0 = np.datetime64("2021-01-05T00:00", "ns")
HYCOM = (17.0, 0.04, 30, 278.0, 0.08, 25)       # lat0, dlat, nlat, lon0, dlon, nlon
ERA5 = (17.0, 0.25, 8, 278.0, 0.25, 9)          # reaches 280.0 E; HYCOM stops at 279.92
I0, J0 = 10, 12                                  # the cell the land tests are built around


def hours(t):
    return (np.asarray(t, dtype="datetime64[ns]") - T0) / np.timedelta64(1, "h")


def times(start, n, step_h):
    return np.datetime64(start, "ns") + np.arange(n) * np.timedelta64(step_h, "h")


def linear(grid, t):
    """0.25 + i/8 + j/16 + hours/32 at every grid point: exact in float32."""
    nlat, nlon = grid[2], grid[5]
    i = np.arange(nlat)[None, :, None]
    j = np.arange(nlon)[None, None, :]
    return 0.25 + 0.125 * i + 0.0625 * j + 0.03125 * hours(t)[:, None, None]


def truth(grid, lat, lon, t):
    lat0, dlat, _, lon0, dlon, _ = grid
    u = (0.25 + 0.125 * (np.asarray(lat) - lat0) / dlat
         + 0.0625 * (np.asarray(lon) - lon0) / dlon + 0.03125 * hours(t))
    return np.stack([u, 1.0 - 0.5 * u], axis=-1)


def dataset(grid, t, names, u):
    lat0, dlat, nlat, lon0, dlon, nlon = grid
    u = np.asarray(u, dtype=np.float32)
    return xr.Dataset(
        {names[0]: (("time", "lat", "lon"), u),
         names[1]: (("time", "lat", "lon"), (1.0 - 0.5 * u).astype(np.float32))},
        coords={"time": t, "lat": lat0 + dlat * np.arange(nlat),
                "lon": lon0 + dlon * np.arange(nlon)})


def write(ds, root, prefix, name_dates):
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    path = raw / f"{prefix}_{TAG}_{name_dates}.nc"
    ds.to_netcdf(path)
    return path


def current_file(root, u=None, t=None, name_dates="20210105-20210106"):
    t = times(T0, 9, 3) if t is None else t
    u = linear(HYCOM, t) if u is None else u
    return write(dataset(HYCOM, t, CURRENT_VARS, u), root, "hycom", name_dates)


def wind_file(root, u=None, t=None, name_dates="20210105-20210106"):
    t = times(T0, 25, 1) if t is None else t
    u = linear(ERA5, t) if u is None else u
    return write(dataset(ERA5, t, WIND_VARS, u), root, "era5", name_dates)


@pytest.fixture
def root(tmp_path):
    current_file(tmp_path)
    wind_file(tmp_path)
    return tmp_path


@pytest.fixture
def forcing(root):
    with GriddedForcing.from_dir(root, "2021-01-05T00:00", "2021-01-06T00:00") as f:
        yield f


def cell_point(fy, fx, i=I0, j=J0):
    """The position fy and fx of the way across HYCOM cell (i, j)."""
    return (np.array([HYCOM[0] + (i + fy) * HYCOM[1]]),
            np.array([HYCOM[3] + (j + fx) * HYCOM[4]]))


def with_cell(corner_values, t=None):
    """A HYCOM field that is linear everywhere except cell (I0, J0)'s four corners,
    constant in time there; None marks a corner as land."""
    t = times(T0, 9, 3) if t is None else t
    u = linear(HYCOM, t)
    for (di, dj), value in zip(((0, 0), (0, 1), (1, 0), (1, 1)), corner_values):
        u[:, I0 + di, J0 + dj] = np.nan if value is None else value
    return u


def sample_current(tmp_path, u, lat, lon, when="2021-01-05T01:30"):
    current_file(tmp_path, u)
    wind_file(tmp_path)
    with GriddedForcing.from_dir(tmp_path, when, when) as f:
        return f.sample(lat, lon, when)[0]


class TestAsNs:
    def test_integer_nanoseconds(self):
        assert as_ns("1970-01-01T00:00:01") == 1_000_000_000

    def test_microseconds_and_strings_agree(self):
        assert as_ns(np.datetime64("2021-01-05T07:30", "us")) == as_ns("2021-01-05T07:30")

    def test_nat_raises(self):
        with pytest.raises(ValueError, match="real instant"):
            as_ns(np.datetime64("NaT", "ns"))


class TestRegularAxis:
    def test_origin_step_size(self):
        axis = RegularAxis.from_values(17.0 + 0.04 * np.arange(30), "lat")
        assert (axis.origin, axis.size) == (17.0, 30)
        assert axis.step == pytest.approx(0.04) and axis.last == pytest.approx(18.16)

    def test_descending_raises(self):
        with pytest.raises(ValueError, match="ascend"):
            RegularAxis.from_values(np.arange(5.0)[::-1], "lat")

    def test_inside_allows_edge_rounding_but_not_nan(self):
        axis = RegularAxis.from_values(np.arange(5.0), "x")
        assert axis.inside(np.array([-1e-10, 0.0, 4.0, 4.0 + 1e-10])).all()
        assert not axis.inside(np.array([-0.1, 4.1, np.nan])).any()

    def test_locate_interior_line_and_last_line(self):
        axis = RegularAxis.from_values(np.arange(5.0), "x")
        i0, f = axis.locate(np.array([1.25, 2.0, 4.0, -1e-10]))
        assert i0.tolist() == [1, 2, 3, 0]
        assert f == pytest.approx([0.25, 0.0, 1.0, 0.0])


class TestTimeAxis:
    def test_joins_files_and_remembers_where_each_step_is(self):
        axis = TimeAxis([times(T0, 3, 3), times(T0 + np.timedelta64(9, "h"), 2, 3)], ["a", "b"])
        assert axis.file.tolist() == [0, 0, 0, 1, 1]
        assert axis.local.tolist() == [0, 1, 2, 0, 1]

    def test_bracket_is_exact_in_integers(self):
        axis = TimeAxis([times(T0, 3, 3)], ["a"])
        assert axis.bracket(as_ns("2021-01-05T01:00")) == (0, pytest.approx(1 / 3, abs=0))
        assert axis.bracket(as_ns("2021-01-05T03:00")) == (1, 0.0)
        assert axis.bracket(as_ns("2021-01-05T06:00")) == (2, 0.0)

    def test_outside_raises(self):
        axis = TimeAxis([times(T0, 3, 3)], ["a"])
        with pytest.raises(OutOfCoverageError, match="outside the files"):
            axis.bracket(as_ns("2021-01-05T06:01"))

    def test_overlapping_files_are_refused_by_name(self):
        with pytest.raises(ValueError, match="a and b overlap"):
            TimeAxis([times(T0, 3, 3), times(T0 + np.timedelta64(6, "h"), 2, 3)], ["a", "b"])

    def test_non_increasing_times_are_refused(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            TimeAxis([np.array([T0, T0])], ["a"])

    def test_gaps_lists_only_the_long_ones_inside_the_window(self):
        t = np.concatenate([times(T0, 3, 3), times(T0 + np.timedelta64(12, "h"), 3, 3)])
        axis = TimeAxis([t], ["a"])
        gaps = axis.gaps(as_ns("2021-01-05T00:00"), as_ns("2021-01-05T18:00"), 3 * 3600)
        assert gaps == [("2021-01-05T06:00:00.000000000", "2021-01-05T12:00:00.000000000", 6.0)]
        assert axis.gaps(as_ns("2021-01-05T12:00"), as_ns("2021-01-05T18:00"), 3 * 3600) == []


class TestCellWeights:
    def test_the_worked_example(self):
        w = cell_weights(np.array([0.25]), np.array([0.625]))[:, 0]
        assert w.tolist() == [0.28125, 0.46875, 0.09375, 0.15625]

    def test_always_sum_to_one(self):
        rng = np.random.default_rng(0)
        assert cell_weights(rng.random(100), rng.random(100)).sum(axis=0) == pytest.approx(1.0)


class TestUsableCorners:
    fy, fx = np.array([0.2]), np.array([0.3])          # nearest corner: lower-left

    def test_all_wet(self):
        usable, beached = usable_corners(np.ones((4, 1), bool), self.fy, self.fx)
        assert usable.all() and not beached.any()

    def test_nearest_dry_is_beached(self):
        wet = np.array([[False], [True], [True], [True]])
        assert usable_corners(wet, self.fy, self.fx)[1].tolist() == [True]

    def test_diagonal_is_dropped_behind_two_dry_edges(self):
        wet = np.array([[True], [False], [False], [True]])
        usable, beached = usable_corners(wet, self.fy, self.fx)
        assert usable[:, 0].tolist() == [True, False, False, False] and not beached[0]

    def test_diagonal_counts_through_one_wet_edge(self):
        wet = np.array([[True], [True], [False], [True]])
        assert usable_corners(wet, self.fy, self.fx)[0][:, 0].tolist() == [True, True, False, True]

    def test_nearest_follows_the_half_cell(self):
        wet = np.array([[False], [False], [False], [True]])   # only upper-right wet
        _, beached = usable_corners(wet, np.array([0.5, 0.49]), np.array([0.5, 0.5]))
        assert beached.tolist() == [False, True]


class TestSampleExact:
    def test_a_linear_field_is_reproduced_exactly(self, forcing):
        rng = np.random.default_rng(1)
        lat = rng.uniform(17.0, 18.16, 1000)
        lon = rng.uniform(278.0, 279.92, 1000)
        for when in ("2021-01-05T00:00", "2021-01-05T07:30", "2021-01-05T13:17",
                     "2021-01-06T00:00"):
            current, wind = forcing.sample(lat, lon, when)
            assert current == pytest.approx(truth(HYCOM, lat, lon, np.datetime64(when)),
                                             abs=1e-9)
            assert wind == pytest.approx(truth(ERA5, lat, lon, np.datetime64(when)), abs=1e-9)

    def test_matches_sample_field_on_a_random_field(self, tmp_path):
        rng = np.random.default_rng(2)
        t = times(T0, 9, 3)
        u = rng.normal(0.0, 1.0, (9, HYCOM[2], HYCOM[5]))
        current_file(tmp_path, u)
        wind_file(tmp_path)
        ds = xr.open_dataset(tmp_path / "raw" / f"hycom_{TAG}_20210105-20210106.nc")
        with GriddedForcing.from_dir(tmp_path, "2021-01-05", "2021-01-06") as f:
            for _ in range(300):
                lat, lon = rng.uniform(17.0, 18.16), rng.uniform(278.0, 279.92)
                when = t[0] + np.timedelta64(int(rng.integers(0, 24 * 3600)), "s")
                ref = sample_field(ds, CURRENT_VARS, lat, lon, when)
                ours = f.sample([lat], [lon], when)[0][0]
                assert ours == pytest.approx(ref.weights @ ref.corners, abs=1e-9)
        ds.close()

    def test_a_grid_point_at_a_stored_time_is_the_stored_value(self, forcing):
        lat = np.array([17.0 + 0.04 * 7, 17.0 + 0.04 * 29, 17.0])
        lon = np.array([278.0 + 0.08 * 3, 278.0 + 0.08 * 24, 278.0 + 0.08 * 24])
        current, _ = forcing.sample(lat, lon, "2021-01-05T06:00")
        stored = linear(HYCOM, times("2021-01-05T06:00", 1, 3))[0]
        assert current[:, 0] == pytest.approx([stored[7, 3], stored[29, 24], stored[0, 24]])

    def test_across_two_files(self, tmp_path):
        day = np.timedelta64(1, "D")
        current_file(tmp_path, t=times(T0, 8, 3), name_dates="20210105-20210106")
        current_file(tmp_path, t=times(T0 + day, 8, 3), name_dates="20210106-20210107")
        wind_file(tmp_path, t=times(T0, 48, 1), name_dates="20210105-20210107")
        with GriddedForcing.from_dir(tmp_path, "2021-01-05T12:00", "2021-01-06T12:00") as f:
            assert len(f.current.datasets) == 2
            lat, lon = np.array([17.5]), np.array([279.0])
            when = "2021-01-05T22:30"                # between 21:00 and the next file's 00:00
            assert f.sample(lat, lon, when)[0] == pytest.approx(
                truth(HYCOM, lat, lon, np.datetime64(when)), abs=1e-9)

    def test_longitude_in_either_convention(self, forcing):
        a = forcing.sample([17.5], [279.3], "2021-01-05T02:00")
        b = forcing.sample([17.5], [279.3 - 360.0], "2021-01-05T02:00")
        assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])

    def test_returns_new_arrays_every_call(self, forcing):
        a, _ = forcing.sample([17.5], [279.0], "2021-01-05T02:00")
        b, _ = forcing.sample([17.5], [279.0], "2021-01-05T02:00")
        assert not np.shares_memory(a, b)


class TestLand:
    def test_the_worked_example(self, tmp_path):
        lat, lon = cell_point(0.25, 0.625)
        u = sample_current(tmp_path, with_cell((1.2, 1.4, 1.0, 1.3)), lat, lon)
        assert u[0, 0] == pytest.approx(1.290625, abs=1e-6)

    def test_a_dry_far_corner_is_dropped_and_the_rest_renormalised(self, tmp_path):
        lat, lon = cell_point(0.25, 0.625)
        u = sample_current(tmp_path, with_cell((1.2, 1.4, 1.0, None)), lat, lon)
        assert u[0, 0] == pytest.approx(1.0875 / 0.84375, abs=1e-6)

    def test_a_dry_nearest_corner_beaches(self, tmp_path):
        lat, lon = cell_point(0.25, 0.625)          # nearest corner: lower-right
        u = sample_current(tmp_path, with_cell((1.2, None, 1.0, 1.3)), lat, lon)
        assert np.isnan(u).all()

    def test_no_leak_across_a_diagonal_barrier(self, tmp_path):
        # Plain renormalising would give (0.3025 - 0.2025) / 0.505 = 0.198.
        lat, lon = cell_point(0.45, 0.45)
        u = sample_current(tmp_path, with_cell((1.0, None, None, -1.0)), lat, lon)
        assert u[0, 0] == pytest.approx(1.0, abs=1e-6)

    def test_a_one_cell_channel_is_one_dimensional(self, tmp_path):
        u = with_cell((1.0, 2.0, None, None))
        u[:, I0 + 1, :] = np.nan
        lat, lon = cell_point(0.3, 0.4)
        assert sample_current(tmp_path, u, lat, lon)[0, 0] == pytest.approx(1.4, abs=1e-6)

    def test_the_midline_decides_beaching(self, tmp_path):
        u = with_cell((1.0, 2.0, None, None))
        u[:, I0 + 1, :] = np.nan
        lat = np.array([cell_point(0.49, 0.4)[0][0], cell_point(0.51, 0.4)[0][0]])
        lon = np.full(2, cell_point(0.49, 0.4)[1][0])
        u = sample_current(tmp_path, u, lat, lon)
        assert np.isfinite(u[0]).all() and np.isnan(u[1]).all()

    def test_a_lone_sea_point_returns_its_own_value(self, tmp_path):
        lat, lon = cell_point(0.2, 0.3)
        u = sample_current(tmp_path, with_cell((0.7, None, None, None)), lat, lon)
        assert u[0, 0] == pytest.approx(0.7, abs=1e-6)

    def test_land_only_in_the_next_slice_does_not_count_on_the_stored_step(self, tmp_path):
        t = times(T0, 9, 3)
        u = linear(HYCOM, t)
        u[1, I0, J0] = np.nan                        # dry at 03:00 only
        current_file(tmp_path, u)
        wind_file(tmp_path)
        lat, lon = cell_point(0.1, 0.1)
        with GriddedForcing.from_dir(tmp_path, "2021-01-05", "2021-01-06") as f:
            assert np.isfinite(f.sample(lat, lon, "2021-01-05T00:00")[0]).all()
            assert np.isnan(f.sample(lat, lon, "2021-01-05T01:30")[0]).all()

    def test_a_beached_particle_leaves_the_others_alone(self, tmp_path):
        lat_land, lon_land = cell_point(0.25, 0.625)
        lat = np.array([lat_land[0], 17.5, 17.9])
        lon = np.array([lon_land[0], 279.0, 278.4])
        u = sample_current(tmp_path, with_cell((1.2, None, 1.0, 1.3)), lat, lon)
        when = np.datetime64("2021-01-05T01:30")
        assert np.isnan(u[0]).all()
        assert u[1:] == pytest.approx(truth(HYCOM, lat[1:], lon[1:], when), abs=1e-9)

    @pytest.mark.filterwarnings("ignore:All-NaN slice")
    def test_every_answer_lies_between_its_wet_corners(self, tmp_path):
        rng = np.random.default_rng(3)
        t = times(T0, 9, 3)
        u = rng.normal(0.0, 1.0, (9, HYCOM[2], HYCOM[5])).astype(np.float32)  # as stored
        u[:, rng.random((HYCOM[2], HYCOM[5])) < 0.25] = np.nan
        lat = rng.uniform(17.0, 18.16 - 1e-9, 3000)
        lon = rng.uniform(278.0, 279.92 - 1e-9, 3000)
        got = sample_current(tmp_path, u, lat, lon)[:, 0]
        i = np.floor((lat - 17.0) / 0.04).astype(int)
        j = np.floor((lon - 278.0) / 0.08).astype(int)
        cells = np.stack([u[k][[i, i, i + 1, i + 1], [j, j + 1, j, j + 1]] for k in (0, 1)])
        lo, hi = np.nanmin(cells, axis=(0, 1)), np.nanmax(cells, axis=(0, 1))
        finite = np.isfinite(got)
        assert finite.sum() > 1500 and (~finite).sum() > 300
        assert np.all(got[finite] >= lo[finite] - 1e-9)
        assert np.all(got[finite] <= hi[finite] + 1e-9)


class TestOutside:
    def test_outside_is_nan_for_both_and_flagged(self, forcing):
        lat, lon = np.array([16.9, 17.5, np.nan]), np.array([279.0, 277.9, 279.0])
        current, wind = forcing.sample(lat, lon, "2021-01-05T02:00")
        assert np.isnan(current).all() and np.isnan(wind).all()
        assert forcing.outside(lat, lon).all()

    def test_the_strip_only_the_wind_covers_is_outside(self, forcing):
        # HYCOM stops at 279.92 E here, as it stops at 296.96 E in the real archive.
        assert forcing.outside([17.5], [279.95]).tolist() == [True]
        assert np.isnan(forcing.sample([17.5], [279.95], "2021-01-05T02:00")[1]).all()

    def test_a_beached_particle_is_not_outside(self, tmp_path):
        current_file(tmp_path, with_cell((1.2, None, 1.0, 1.3)))
        wind_file(tmp_path)
        lat, lon = cell_point(0.25, 0.625)
        with GriddedForcing.from_dir(tmp_path, "2021-01-05", "2021-01-06") as f:
            assert f.outside(lat, lon).tolist() == [False]


class TestGapsAndCoverage:
    def gapped(self, tmp_path, gap_h):
        t = np.concatenate([times(T0, 3, 3), times(T0 + np.timedelta64(6 + gap_h, "h"), 4, 3)])
        current_file(tmp_path, t=t)
        wind_file(tmp_path)

    def test_a_six_hour_gap_is_blended_and_recorded(self, tmp_path):
        self.gapped(tmp_path, 6)                     # 06:00 then 12:00
        with GriddedForcing.from_dir(tmp_path, "2021-01-05T00:00", "2021-01-05T18:00") as f:
            assert [g["hours"] for g in f.gaps_blended] == [6.0]
            assert f.describe()["gaps_blended"][0]["from"].startswith("2021-01-05T06:00")
            lat, lon = np.array([17.5]), np.array([279.0])
            assert f.sample(lat, lon, "2021-01-05T09:00")[0] == pytest.approx(
                truth(HYCOM, lat, lon, np.datetime64("2021-01-05T09:00")), abs=1e-9)

    def test_a_twelve_hour_gap_is_refused_for_the_window(self, tmp_path):
        self.gapped(tmp_path, 12)                    # 06:00 then 18:00
        with pytest.raises(ForcingGapError, match="12 h gap"):
            GriddedForcing.from_dir(tmp_path, "2021-01-05T00:00", "2021-01-05T20:00")

    def test_a_window_that_misses_the_gap_is_fine_but_sampling_into_it_is_not(self, tmp_path):
        self.gapped(tmp_path, 12)
        with GriddedForcing.from_dir(tmp_path, "2021-01-05T00:00", "2021-01-05T06:00") as f:
            with pytest.raises(ForcingGapError, match="D011"):
                f.sample([17.5], [279.0], "2021-01-05T09:00")

    def test_outside_the_files_in_time_is_refused(self, root):
        with pytest.raises(OutOfCoverageError):
            GriddedForcing.from_dir(root, "2021-01-05T00:00", "2021-01-06T01:00")
        with GriddedForcing.from_dir(root, "2021-01-05", "2021-01-06") as f:
            with pytest.raises(OutOfCoverageError):
                f.sample([17.5], [279.0], "2021-01-06T00:01")

    def test_an_end_before_the_start_is_refused(self, forcing):
        with pytest.raises(ValueError, match="before start"):
            forcing.require("2021-01-05T06:00", "2021-01-05T03:00")


class TestFiles:
    def test_picks_only_the_files_the_window_needs(self, tmp_path):
        day = np.timedelta64(1, "D")
        for d in range(3):
            start = str(np.datetime64("2021-01-05") + np.timedelta64(d, "D")).replace("-", "")
            end = str(np.datetime64("2021-01-05") + np.timedelta64(d + 1, "D")).replace("-", "")
            current_file(tmp_path, t=times(T0 + d * day, 8, 3), name_dates=f"{start}-{end}")
        wind_file(tmp_path, t=times(T0, 72, 1), name_dates="20210105-20210108")
        with GriddedForcing.from_dir(tmp_path, "2021-01-07T03:00", "2021-01-07T09:00") as f:
            assert [p[-20:] for p in f.describe()["current"]["files"]] == [
                "20210106-20210107.nc", "20210107-20210108.nc"]

    def test_a_name_that_says_inclusive_is_read_by_its_times(self, tmp_path):
        current_file(tmp_path)
        wind_file(tmp_path, name_dates="20210105-20210105")   # really runs to 06T00
        with GriddedForcing.from_dir(tmp_path, "2021-01-05T20:00", "2021-01-05T23:30") as f:
            assert np.isfinite(f.sample([17.5], [279.0], "2021-01-05T23:30")[1]).all()

    def test_no_files_for_the_window_raises(self, root):
        with pytest.raises(FileNotFoundError, match="covers"):
            GriddedForcing.from_dir(root, "2022-01-05", "2022-01-06")

    def test_other_boxes_are_ignored(self, root):
        other = root / "raw" / "hycom_0-10N_0-10W_20210105-20210106.nc"
        other.write_bytes(b"not a netcdf file")
        with GriddedForcing.from_dir(root, "2021-01-05", "2021-01-06") as f:
            assert len(f.current.datasets) == 1

    def test_overlapping_files_are_refused(self, root):
        current_file(root, t=times(T0 + np.timedelta64(12, "h"), 4, 3),
                     name_dates="20210105-20210107")
        with pytest.raises(ValueError, match="overlap"):
            GriddedForcing.from_dir(root, "2021-01-05", "2021-01-06")

    def test_files_on_different_grids_are_refused(self, tmp_path):
        a = current_file(tmp_path)
        other = (17.0, 0.04, 30, 278.04, 0.08, 25)
        t = times(T0 + np.timedelta64(1, "D"), 4, 3)
        b = write(dataset(other, t, CURRENT_VARS, linear(other, t)), tmp_path, "hycom",
                  "20210106-20210107")
        w = wind_file(tmp_path)
        with pytest.raises(ValueError, match="different grids"):
            GriddedForcing([a, b], [w])

    def test_slices_are_read_once_per_bracket(self, forcing):
        start = np.datetime64("2021-01-05T00:00", "us")
        for k in range(9 * 60):                      # nine hours of 60 s steps
            forcing.sample([17.5], [279.0], start + np.timedelta64(60 * k, "s"))
        assert forcing.current.loads == 4            # 00, 03, 06, 09
        assert forcing.wind.loads == 10              # 00 to 09

    def test_describe_is_json(self, forcing):
        d = json.loads(json.dumps(forcing.describe()))
        assert d["backend"] == "gridded" and d["current"]["grid"]["lat"][2] == 30


class TestWithTheEngine:
    def test_a_uniform_field_drifts_exactly_like_constant_forcing(self, tmp_path):
        t_current, t_wind = times(T0, 9, 3), times(T0, 25, 1)
        current_file(tmp_path, np.full((9, HYCOM[2], HYCOM[5]), 0.25),
                     t=t_current)                    # v = 1 - 0.5 u = 0.875
        wind_file(tmp_path, np.full((25, ERA5[2], ERA5[5]), 4.0), t=t_wind)   # v = -1
        constant = ConstantForcing(current=(0.25, 0.875), wind=(4.0, -1.0))
        with GriddedForcing.from_dir(tmp_path, "2021-01-05T00:00", "2021-01-05T06:00") as f:
            a = [s.positions for s in DriftPipeline(f, 60.0).track(
                "2021-01-05T00:00", 17.3, 278.5, 6 * 3600)]
        b = [s.positions for s in DriftPipeline(constant, 60.0).track(
            "2021-01-05T00:00", 17.3, 278.5, 6 * 3600)]
        assert np.allclose(a, b, rtol=0, atol=1e-12)

    def test_a_current_into_land_freezes_the_particle_in_the_land_pixel(self, tmp_path):
        u = np.full((9, HYCOM[2], HYCOM[5]), 1.0)
        u[:, :, 20:] = np.nan                        # land from column 20 eastward
        current_file(tmp_path, u)                    # v = 1 - 0.5 u = 0.5
        wind_file(tmp_path, np.full((25, ERA5[2], ERA5[5]), 2.0))    # v = 0: no leeway north
        with GriddedForcing.from_dir(tmp_path, "2021-01-05T00:00", "2021-01-05T12:00") as f:
            lons = np.array([s.positions[0, 1] for s in DriftPipeline(f, 60.0).track(
                "2021-01-05T00:00", 17.3, 278.0 + 0.08 * 17, 12 * 3600)])
        frozen = lons >= 278.0 + 0.08 * 19.5
        assert frozen.any() and np.all(lons[frozen] == lons[frozen][0])
        assert lons[frozen][0] - (278.0 + 0.08 * 19.5) < 0.01


class TestCommandLines:
    def test_gridded_cli_counts_and_checks(self, root):
        out = _cli(["--forcing-dir", str(root), "--time", "2021-01-05T07:30",
                    "--particles", "400", "--seed", "1", "--check", "20"])
        assert out["sea"] + out["beached"] + out["outside"] == 400
        assert out["check_against_sample_field"]["current"]["within_bound"]
        assert out["check_against_sample_field"]["wind"]["within_bound"]

    def test_gridded_cli_refuses_no_particles(self, root):
        with pytest.raises(SystemExit):
            _cli(["--forcing-dir", str(root), "--time", "2021-01-05T07:30", "--particles", "0"])

    def test_track_runs_on_the_real_forcing_flag(self, root):
        out = track._cli(["--forcing-dir", str(root), "--start", "2021-01-05T01:00",
                          "--lat", "17.5", "--lon", "-80.8", "--duration", "3600",
                          "--timestep", "60", "--every", "30"])
        assert out["run"]["forcing"]["backend"] == "gridded" and len(out["track"]) == 3

    @pytest.mark.parametrize("flags", [[], ["--constant-current", "1", "0", "--forcing-dir", "x"]])
    def test_track_needs_exactly_one_forcing(self, flags):
        with pytest.raises(SystemExit):
            track._cli(["--start", "2021-01-05T01:00", "--lat", "17.5", "--lon", "-80.8",
                        "--duration", "60", "--timestep", "60", *flags])

    def test_the_ensemble_runs_on_the_real_forcing_flag(self, root, tmp_path):
        path = ensemble._cli(["--lat", "17.5", "--lon", "-80.8", "--datum-sigma-km", "0",
                              "--start", "2021-01-05T01:00", "--timestep", "60",
                              "--duration", "1h", "--particles", "5", "--sigma", "0.1",
                              "--seed", "1", "--forcing-dir", str(root),
                              "--out", str(tmp_path / "out")])
        sidecar = json.loads(path.with_suffix(".json").read_text())
        assert sidecar["pipeline"]["forcing"]["backend"] == "gridded"
