"""Tests for sar.model.interpolate (issue #10). Synthetic grids only, no network."""

import json

import numpy as np
import pytest
import xarray as xr

from sar.model import interpolate as ip
from sar.utils.interpolation import interpolate_series

A, B, C, E = 0.3, 2.0, -1.5, 0.05      # linear test field: a + b*lon_off + c*lat_off + e*hours


def make_ds(lat0, dlat, n_lat, lon0, dlon, n_lon, hours, names, dt_hours):
    """A dataset shaped like the stored files, holding a linear field in space and time."""
    lat = lat0 + dlat * np.arange(n_lat)
    lon = lon0 + dlon * np.arange(n_lon)
    time = np.datetime64("2021-01-05T00:00") + np.arange(hours) * np.timedelta64(dt_hours, "h")
    h = np.arange(hours) * dt_hours
    u = A + B * (lon[None, None, :] - lon0) + C * (lat[None, :, None] - lat0) + E * h[:, None, None]
    data = {names[0]: (("time", "lat", "lon"), u), names[1]: (("time", "lat", "lon"), 2.0 * u)}
    return xr.Dataset(data, coords={"time": time, "lat": lat, "lon": lon})


@pytest.fixture
def hycom():
    return make_ds(17.0, 0.04, 40, 278.0, 0.08, 40, 4, ip.CURRENT_VARS, 3)


@pytest.fixture
def era5():
    return make_ds(17.0, 0.25, 12, 278.0, 0.25, 12, 10, ip.WIND_VARS, 1)


def linear_truth(ds, lat, lon, hours):
    return A + B * (lon - ds.lon.values[0]) + C * (lat - ds.lat.values[0]) + E * hours


class TestGridStep:
    def test_returns_each_grid_step(self, hycom, era5):
        assert ip.grid_step(hycom.lat.values) == pytest.approx(0.04)
        assert ip.grid_step(hycom.lon.values) == pytest.approx(0.08)
        assert ip.grid_step(era5.lat.values) == pytest.approx(0.25)

    def test_uneven_axis_raises(self):
        with pytest.raises(ValueError, match="not regularly spaced"):
            ip.grid_step([0.0, 1.0, 3.0])

    def test_single_point_raises(self):
        with pytest.raises(ValueError, match="at least 2 points"):
            ip.grid_step([1.0])

    def test_accepts_a_float32_axis_as_the_real_archive_stores_it(self):
        """HYCOM writes float32 coordinates, so a true 0.08 grid arrives with ragged gaps.

        This is the check that rejected the project's only real current file. The axis
        below is the genuine one from data/current/current_2019-01-01_2019-01-03.txt,
        reconstructed the same way: a 0.08 step rounded through float32 at 278 to 297.
        """
        axis = np.float32(278.0 + 0.08 * np.arange(238)).astype(float)
        assert ip.grid_step(axis, "lon") == pytest.approx(0.08, abs=1e-6)

    def test_names_the_axis_it_rejected(self):
        with pytest.raises(ValueError, match="lon is not regularly spaced"):
            ip.grid_step([0.0, 0.08, 0.30], "lon")


class TestNearestIndex:
    def test_index_and_signed_offset(self, hycom):
        i, d = ip.nearest_index(hycom.lat.values, 17.0 + 0.04 * 5 + 0.01, "lat")
        assert i == 5
        assert d == pytest.approx(0.01)

    def test_offset_is_negative_when_below(self, hycom):
        i, d = ip.nearest_index(hycom.lat.values, 17.0 + 0.04 * 5 - 0.01, "lat")
        assert i == 5
        assert d == pytest.approx(-0.01)

    def test_rounds_to_the_closer_line(self, hycom):
        i, _ = ip.nearest_index(hycom.lat.values, 17.0 + 0.04 * 5 + 0.03, "lat")
        assert i == 6

    def test_outside_the_axis_raises_naming_it(self, hycom):
        with pytest.raises(ip.OutOfCoverageError, match="lat"):
            ip.nearest_index(hycom.lat.values, 10.0, "lat")
        with pytest.raises(ip.OutOfCoverageError):
            ip.nearest_index(hycom.lat.values, 30.0, "lat")

    def test_exactly_on_the_last_line_is_inside(self, hycom):
        i, d = ip.nearest_index(hycom.lat.values, hycom.lat.values[-1], "lat")
        assert i == hycom.lat.size - 1
        assert d == 0.0


class TestNeighbourDirection:
    @pytest.mark.parametrize(
        "dy, dx, expected",
        [(0.01, -0.01, (1, -1)), (0.01, 0.01, (1, 1)), (-0.01, -0.01, (-1, -1)), (-0.01, 0.01, (-1, 1))],
    )
    def test_each_axis_gets_its_own_sign(self, dy, dx, expected):
        assert ip.neighbour_direction(dy, dx) == expected

    def test_exactly_zero_counts_as_positive(self):
        assert ip.neighbour_direction(0.0, 0.0) == (1, 1)


class TestKeepInside:
    def test_leaves_an_inward_sign_alone(self):
        assert ip.keep_inside(5, 1, 10) == 1
        assert ip.keep_inside(5, -1, 10) == -1

    def test_flips_a_sign_that_points_off_either_end(self):
        assert ip.keep_inside(9, 1, 10) == -1
        assert ip.keep_inside(0, -1, 10) == 1


class TestCellCorners:
    @pytest.mark.parametrize("sy, sx", [(1, 1), (1, -1), (-1, 1), (-1, -1)])
    def test_the_four_corners_follow_the_signs(self, sy, sx):
        got = ip.cell_corners(5, 6, sy, sx, (20, 20))
        assert got == [(5, 6), (5, 6 + sx), (5 + sy, 6), (5 + sy, 6 + sx)]

    def test_a_neighbour_off_the_grid_raises(self):
        with pytest.raises(ip.OutOfCoverageError):
            ip.cell_corners(0, 3, -1, 1, (10, 10))
        with pytest.raises(ip.OutOfCoverageError):
            ip.cell_corners(3, 9, 1, 1, (10, 10))

    def test_never_raises_for_a_position_inside_the_grid(self, hycom):
        lat, lon = hycom.lat.values, hycom.lon.values
        for la in (lat[0], lat[-1], 0.5 * (lat[0] + lat[-1])):
            for lo in (lon[0], lon[-1], 0.5 * (lon[0] + lon[-1])):
                ip.sample_field(hycom, ip.CURRENT_VARS, la, lo, "2021-01-05T00:00")


class TestBilinearWeights:
    def test_sum_to_one_and_stay_in_range(self):
        rng = np.random.default_rng(0)
        for fy, fx in rng.uniform(0.0, 0.5, size=(1000, 2)):
            w = ip.bilinear_weights(fy, fx)
            assert w.sum() == pytest.approx(1.0)
            assert (w >= 0).all() and (w <= 1).all()

    def test_on_the_nearest_point_only_it_counts(self):
        assert ip.bilinear_weights(0.0, 0.0).tolist() == [1.0, 0.0, 0.0, 0.0]

    def test_at_the_cell_centre_all_four_are_equal(self):
        assert ip.bilinear_weights(0.5, 0.5) == pytest.approx([0.25] * 4)

    def test_out_of_range_offset_raises(self):
        with pytest.raises(ValueError):
            ip.bilinear_weights(1.5, 0.2)


class TestBracketTime:
    def test_fraction_between_two_steps(self, hycom):
        k, frac = ip.bracket_time(hycom.time.values, "2021-01-05T04:30")
        assert (k, frac) == (1, pytest.approx(0.5))

    def test_exactly_on_a_step(self, hycom):
        assert ip.bracket_time(hycom.time.values, "2021-01-05T03:00") == (1, 0.0)

    def test_the_last_step_has_zero_fraction(self, hycom):
        assert ip.bracket_time(hycom.time.values, "2021-01-05T09:00") == (3, 0.0)

    def test_outside_the_file_raises(self, hycom):
        with pytest.raises(ip.OutOfCoverageError):
            ip.bracket_time(hycom.time.values, "2021-01-06T00:00")


class TestTimeBlend:
    def test_matches_interpolate_series_at_each_piece(self):
        c1, c2 = np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([[3.0, 6.0], [5.0, 0.0]])
        series = interpolate_series(c1, c2, 6)
        for m in range(6):
            assert ip.time_blend(c1, c2, m / 6) == pytest.approx(series[m])

    def test_out_of_range_fraction_raises(self):
        with pytest.raises(ValueError):
            ip.time_blend([1.0], [2.0], 1.2)


class TestResultantVector:
    def test_two_corner_example(self):
        u, v = ip.resultant_vector([[1.0, 0.0], [0.0, 1.0]], [0.5, 0.5])
        assert (u, v) == pytest.approx((0.5, 0.5))
        speed, bearing = ip.speed_direction(u, v)
        assert speed == pytest.approx(0.7071, abs=1e-4)
        assert bearing == pytest.approx(45.0)

    def test_opposing_corners_cancel_to_zero_not_one(self):
        u, v = ip.resultant_vector([[1.0, 0.0], [-1.0, 0.0]], [0.5, 0.5])
        assert ip.speed_direction(u, v)[0] == pytest.approx(0.0)

    def test_bearings_across_the_wrap_do_not_average_to_south(self):
        s = np.sin(np.radians([350.0, 10.0]))
        c = np.cos(np.radians([350.0, 10.0]))
        u, v = ip.resultant_vector(np.column_stack([s, c]), [0.5, 0.5])
        _, bearing = ip.speed_direction(u, v)
        assert bearing == pytest.approx(0.0, abs=1e-9) or bearing == pytest.approx(360.0)


class TestSpeedDirection:
    @pytest.mark.parametrize(
        "u, v, bearing", [(0, 1, 0), (1, 0, 90), (0, -1, 180), (-1, 0, 270)]
    )
    def test_compass_bearing_is_clockwise_from_north(self, u, v, bearing):
        assert ip.speed_direction(u, v)[1] == pytest.approx(bearing)

    def test_zero_vector_has_no_bearing(self):
        assert ip.speed_direction(0.0, 0.0) == (0.0, None)


class TestSampleField:
    def test_a_linear_field_is_reproduced_exactly_on_both_grids(self, hycom, era5):
        rng = np.random.default_rng(1)
        for ds, names, dt in ((hycom, ip.CURRENT_VARS, 3.0), (era5, ip.WIND_VARS, 1.0)):
            lat0, lat1 = ds.lat.values[[0, -1]]
            lon0, lon1 = ds.lon.values[[0, -1]]
            hours = (ds.sizes["time"] - 1) * dt
            for _ in range(200):
                la, lo, h = rng.uniform(lat0, lat1), rng.uniform(lon0, lon1), rng.uniform(0, hours)
                t = ds.time.values[0] + np.timedelta64(int(h * 3600), "s")
                s = ip.sample_field(ds, names, la, lo, t)
                u, v = ip.resultant_vector(s.corners, s.weights)
                truth = linear_truth(ds, la, lo, int(h * 3600) / 3600.0)
                assert u == pytest.approx(truth, abs=1e-9)
                assert v == pytest.approx(2.0 * truth, abs=1e-9)

    def test_on_a_grid_point_returns_the_grid_value(self, hycom):
        la, lo = hycom.lat.values[7], hycom.lon.values[9]
        s = ip.sample_field(hycom, ip.CURRENT_VARS, la, lo, "2021-01-05T03:00")
        assert s.weights.tolist() == [1.0, 0.0, 0.0, 0.0]
        u, _ = ip.resultant_vector(s.corners, s.weights)
        assert u == hycom["water_u"].isel(time=1, lat=7, lon=9).item()

    def test_on_a_grid_line_is_the_two_point_interpolation(self, hycom):
        la = hycom.lat.values[7]
        lo = 0.5 * (hycom.lon.values[9] + hycom.lon.values[10])
        s = ip.sample_field(hycom, ip.CURRENT_VARS, la, lo, "2021-01-05T03:00")
        u, _ = ip.resultant_vector(s.corners, s.weights)
        expected = hycom["water_u"].isel(time=1, lat=7, lon=[9, 10]).mean().item()
        assert u == pytest.approx(expected)

    def test_the_same_offset_gives_different_fractions_on_the_two_grids(self, hycom, era5):
        # 0.02 deg in both axes: half a step in latitude for HYCOM, far less for ERA5.
        h = ip.sample_field(hycom, ip.CURRENT_VARS, 17.0 + 0.04 * 5 + 0.01, 278.0 + 0.08 * 5 + 0.01, "2021-01-05T00:00")
        e = ip.sample_field(era5, ip.WIND_VARS, 17.0 + 0.25 * 5 + 0.01, 278.0 + 0.25 * 5 + 0.01, "2021-01-05T00:00")
        assert h.fy == pytest.approx(0.25) and h.fx == pytest.approx(0.125)
        assert e.fy == pytest.approx(0.04) and e.fx == pytest.approx(0.04)

    def test_longitude_given_as_negative_matches_0_360(self, hycom):
        a = ip.sample_field(hycom, ip.CURRENT_VARS, 18.0, 280.03, "2021-01-05T00:00")
        b = ip.sample_field(hycom, ip.CURRENT_VARS, 18.0, 280.03 - 360.0, "2021-01-05T00:00")
        assert a.corners == pytest.approx(b.corners)

    def test_outside_the_grid_raises_naming_the_axis(self, hycom):
        with pytest.raises(ip.OutOfCoverageError, match="lon"):
            ip.sample_field(hycom, ip.CURRENT_VARS, 18.0, 250.0, "2021-01-05T00:00")

    def test_space_then_time_equals_time_then_space(self, hycom):
        t = "2021-01-05T04:30"
        blended = ip.sample_field(hycom, ip.CURRENT_VARS, 18.013, 280.031, t)
        early = ip.sample_field(hycom, ip.CURRENT_VARS, 18.013, 280.031, "2021-01-05T03:00")
        late = ip.sample_field(hycom, ip.CURRENT_VARS, 18.013, 280.031, "2021-01-05T06:00")
        space_first = ip.resultant_vector(blended.corners, blended.weights)
        time_first = 0.5 * (np.array(ip.resultant_vector(early.corners, early.weights))
                            + np.array(ip.resultant_vector(late.corners, late.weights)))
        assert space_first == pytest.approx(tuple(time_first), abs=1e-12)


class TestMissingCorners:
    def test_one_missing_corner_raises_and_says_one_of_four(self, hycom):
        hycom["water_u"][:, 8, 9] = np.nan
        with pytest.raises(ip.MissingCornerError, match=r"1 of 4"):
            ip.sample_field(hycom, ip.CURRENT_VARS, hycom.lat.values[8] + 0.01,
                            hycom.lon.values[9] + 0.01, "2021-01-05T00:00")

    def test_all_missing_raises_and_says_all_four(self, hycom):
        hycom["water_u"][:, 8:10, 9:11] = np.nan
        with pytest.raises(ip.MissingCornerError, match=r"all 4"):
            ip.sample_field(hycom, ip.CURRENT_VARS, hycom.lat.values[8] + 0.01,
                            hycom.lon.values[9] + 0.01, "2021-01-05T00:00")

    def test_a_missing_corner_in_an_unused_time_slice_is_not_read(self, hycom):
        hycom["water_u"][2, 8, 9] = np.nan
        ip.sample_field(hycom, ip.CURRENT_VARS, hycom.lat.values[8] + 0.01,
                        hycom.lon.values[9] + 0.01, "2021-01-05T00:00")


class TestInterpolationUncertainty:
    def test_identical_corners_have_no_spread_and_full_coherence(self):
        corners = np.tile([0.4, 0.2], (4, 1))
        out = ip.interpolation_uncertainty(corners, [0.25] * 4)
        assert out["sigma_spatial_ms"] == pytest.approx(0.0)
        assert out["coherence"] == pytest.approx(1.0)
        assert out["speed_loss_ms"] == pytest.approx(0.0, abs=1e-12)
        assert out["direction_spread_deg"] == pytest.approx(0.0, abs=1e-6)

    def test_opposing_pairs_have_zero_coherence_and_no_direction_spread(self):
        corners = [[1.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]]
        out = ip.interpolation_uncertainty(corners, [0.25] * 4)
        assert out["coherence"] == pytest.approx(0.0)
        assert out["direction_spread_deg"] is None

    def test_product_sigma_adds_in_quadrature(self):
        corners = [[0.0, 0.0], [0.6, 0.0], [0.0, 0.0], [0.6, 0.0]]
        base = ip.interpolation_uncertainty(corners, [0.25] * 4)
        out = ip.interpolation_uncertainty(corners, [0.25] * 4, product_sigma=0.4)
        assert out["sigma_total_ms"] == pytest.approx(np.hypot(base["sigma_spatial_ms"], 0.4))

    def test_negative_product_sigma_raises(self):
        with pytest.raises(ValueError):
            ip.interpolation_uncertainty([[0.0, 0.0]] * 4, [0.25] * 4, product_sigma=-1.0)

    def test_true_error_on_a_quadratic_field_stays_below_the_bound(self):
        k = 40.0                                     # f = k * x^2, so f_xx = 2k
        rng = np.random.default_rng(2)
        for dlon in (0.02, 0.04, 0.08):
            worst = 0.0
            for _ in range(200):
                x0, fx = rng.uniform(0.0, 1.0), rng.uniform(0.0, 1.0)
                f = lambda x: k * x**2
                bilinear = (1 - fx) * f(x0) + fx * f(x0 + dlon)
                worst = max(worst, abs(bilinear - f(x0 + fx * dlon)))
            assert worst <= ip.bilinear_error_bound(dlon, 0.04, 2 * k, 0.0) + 1e-12

    def test_corner_scatter_grows_with_cell_size(self):
        k = 40.0
        sigmas = []
        for dlon in (0.02, 0.04, 0.08):
            corners = np.array([[k * x**2, 0.0] for x in (0.0, dlon, 0.0, dlon)])
            sigmas.append(ip.interpolation_uncertainty(corners, [0.25] * 4)["sigma_spatial_ms"])
        assert sigmas[0] < sigmas[1] < sigmas[2]


class TestResultantAt:
    T = "2021-01-05T04:30"

    def test_includes_wind_when_asked_and_round_trips_through_json(self, hycom, era5):
        out = ip.resultant_at(18.013, 280.031, self.T, hycom, era5, include_wind=True)
        assert set(out) == {"lat", "lon", "time", "current", "wind"}
        assert json.loads(json.dumps(out)) == out

    def test_leaves_the_wind_key_out_when_excluded(self, hycom):
        out = ip.resultant_at(18.013, 280.031, self.T, hycom, None, include_wind=False)
        assert "wind" not in out

    def test_wind_requested_without_a_wind_dataset_raises(self, hycom):
        with pytest.raises(ValueError, match="wind"):
            ip.resultant_at(18.013, 280.031, self.T, hycom, None, include_wind=True)

    def test_default_follows_the_module_flag(self, hycom, era5, monkeypatch):
        monkeypatch.setattr(ip, "INCLUDE_WIND", False)
        out = ip.resultant_at(18.013, 280.031, self.T, hycom, era5, include_wind=ip.INCLUDE_WIND)
        assert "wind" not in out


class TestCommandLine:
    def _files(self, tmp_path, hycom, era5):
        hp, ep = tmp_path / "hycom_test.nc", tmp_path / "era5_test.nc"
        hycom.to_netcdf(hp)
        era5.to_netcdf(ep)
        return str(hp), str(ep)

    def _run(self, monkeypatch, capsys, argv):
        monkeypatch.setattr("sys.argv", ["interpolate", *argv])
        ip.main()
        return capsys.readouterr().out

    def test_prints_json_with_wind(self, tmp_path, hycom, era5, monkeypatch, capsys):
        hp, ep = self._files(tmp_path, hycom, era5)
        out = self._run(monkeypatch, capsys, ["--current", hp, "--wind", ep, "--lat", "18.013",
                                              "--lon", "280.031", "--time", "2021-01-05T04:30", "--with-wind"])
        assert "wind" in json.loads(out)

    def test_no_wind_flag_omits_it(self, tmp_path, hycom, era5, monkeypatch, capsys):
        hp, ep = self._files(tmp_path, hycom, era5)
        out = self._run(monkeypatch, capsys, ["--current", hp, "--wind", ep, "--lat", "18.013",
                                              "--lon", "280.031", "--time", "2021-01-05T04:30", "--no-wind"])
        assert "wind" not in json.loads(out)


class TestDiagram:
    def test_writes_a_png(self, tmp_path):
        from sar.model.interpolate import plot_resultant_cell, synthetic_sample, wrong_average

        s = synthetic_sample()
        u, v = ip.resultant_vector(s.corners, s.weights)
        wu, wv = wrong_average(s.corners, s.weights)
        assert np.hypot(wu, wv) > np.hypot(u, v)          # averaging speeds overstates
        path = tmp_path / "diagram.png"
        plot_resultant_cell(s, str(path))
        assert path.stat().st_size > 1000

    @pytest.mark.parametrize(
        "corner, quadrant",
        [((0.0, 1.0), "north"), ((1.0, 0.0), "east"),
         ((0.0, -1.0), "south"), ((-1.0, 0.0), "west")],
    )
    def test_the_bearing_panel_draws_in_every_quadrant(self, tmp_path, corner, quadrant):
        """The bearing arcs convert compass degrees to matplotlib's own angles.

        Getting that conversion backwards mirrors the picture, and a resultant in one
        quadrant alone cannot show it, so all four are drawn. A westward current is the
        case that first drew outside its axes.
        """
        from sar.model.interpolate import plot_resultant_cell, synthetic_sample

        s = synthetic_sample()
        flat = ip.Sample(
            s.lat, s.lon, s.i, s.j, s.sy, s.sx, s.dy, s.dx, s.dlat, s.dlon, s.fy, s.fx,
            np.tile(corner, (4, 1)).astype(float), s.weights,
        )
        path = tmp_path / f"{quadrant}.png"
        plot_resultant_cell(flat, str(path))
        assert path.stat().st_size > 1000

    def test_the_bearing_panel_survives_a_zero_resultant(self, tmp_path):
        """Two opposing pairs cancel, so there is no bearing and no direction spread."""
        from sar.model.interpolate import plot_resultant_cell, synthetic_sample

        s = synthetic_sample()
        opposed = ip.Sample(
            s.lat, s.lon, s.i, s.j, s.sy, s.sx, s.dy, s.dx, s.dlat, s.dlon, s.fy, s.fx,
            np.array([[1.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]]),
            np.array([0.25] * 4),
        )
        assert ip.interpolation_uncertainty(opposed.corners,
                                            opposed.weights)["direction_spread_deg"] is None
        path = tmp_path / "zero.png"
        plot_resultant_cell(opposed, str(path))
        assert path.stat().st_size > 1000


class TestMatplotlibIsOptional:
    """Only the drawing needs matplotlib, and a missing one must say how to get it.

    A virtual environment made before matplotlib was added to pyproject.toml does not
    have it, and the bare ModuleNotFoundError names neither the package nor the cure.
    """

    def test_the_json_path_does_not_import_matplotlib(self, hycom, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def refuse_matplotlib(name, *args, **kwargs):
            if name.split(".")[0] == "matplotlib":
                raise ModuleNotFoundError("No module named 'matplotlib'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", refuse_matplotlib)
        out = ip.resultant_at(18.013, 280.031, "2021-01-05T04:30", hycom, None,
                              include_wind=False)
        assert out["current"]["speed"] >= 0

    def test_drawing_without_matplotlib_says_how_to_install_it(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def refuse_matplotlib(name, *args, **kwargs):
            if name.split(".")[0] == "matplotlib":
                raise ModuleNotFoundError("No module named 'matplotlib'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", refuse_matplotlib)
        with pytest.raises(SystemExit, match="pip install"):
            ip._pyplot()
