"""Tests for sar.search.scenario. Constant forcing and synthetic rows, no files."""

import numpy as np
import pandas as pd
import pytest

from sar.pipeline.ensemble import run_ensemble
from sar.pipeline.forcing import ConstantForcing
from sar.search.episode import SearchEpisode, pattern_policy, run
from sar.search.patterns import expanding_square
from sar.search.platform import STEP_S, SWEEP_WIDTH_M
from sar.search.scenario import (
    centroid,
    check_arrival,
    datum_line,
    doctrinal_searcher,
    drift_bearing,
    scenario_row,
    scenario_search,
    search_window,
    steady_search,
    straightness,
    truth_track,
)
from sar.search.sweep import relative_m
from sar.utils.geo import offset_position

START = "2019-06-01T06:00"
LAT, LON = 26.5, -79.0
FORCING = ConstantForcing(current=(1.0, 0.5), wind=(5.0, 2.0))


class TestArrival:
    def test_the_window_must_end_by_4_h(self):
        assert check_arrival(3 * 3600 + 15 * 60) == 195
        with pytest.raises(ValueError, match="4 h"):
            check_arrival(3 * 3600 + 16 * 60)

    def test_arrival_is_whole_minutes(self):
        assert check_arrival(0.0) == 0
        with pytest.raises(ValueError, match="whole number of minutes"):
            check_arrival(90.5)
        with pytest.raises(ValueError, match="whole number of minutes"):
            check_arrival(-60.0)


class TestSearchWindow:
    def test_every_fifth_minute_is_the_5_minute_run_bit_for_bit(self):
        # The same seed draws the same pushes, saved or not (ADR004 section 3).
        setup = search_window(FORCING, START, LAT, LON, 7, 300, 3600.0, sigma_u=0.226)
        five = run_ensemble(FORCING, 300, START, LAT, LON, (60 + 45) * 60.0, STEP_S,
                            datum_sigma_km=0.0, seed=7, save_every=300.0, sigma_u=0.226)
        rows = slice(12, 22)                 # 60, 65, ..., 105 min
        assert np.array_equal(setup.window.times[::5], five.times[rows])
        assert np.array_equal(setup.window.lat[::5], five.lat[rows])
        assert np.array_equal(setup.window.lon[::5], five.lon[rows])

    def test_it_is_46_frames_from_arrival_a_minute_apart(self):
        setup = search_window(FORCING, START, LAT, LON, 1, 50, 7200.0)
        assert setup.window.lat.shape == (46, 50)
        assert setup.window.times[0] == np.datetime64("2019-06-01T08:00", "us")
        assert np.all(np.diff(setup.window.times) == np.timedelta64(60, "s"))
        assert setup.arrival_s == 7200.0

    def test_the_marker_starts_at_the_centroid_and_drifts_with_the_current_only(self):
        setup = search_window(FORCING, START, LAT, LON, 2, 200, 3600.0)
        assert setup.datum == pytest.approx(centroid(setup.window.lat[0], setup.window.lon[0],
                                                     setup.window.weight))
        assert (setup.marker.lat[0], setup.marker.lon[0]) == pytest.approx(setup.datum)
        east, north = relative_m(setup.marker.lat[-1], setup.marker.lon[-1], *setup.datum)
        # 1.0 and 0.5 m/s for 2,700 s, no leeway: the marker does not feel the wind.
        assert (east, north) == pytest.approx((2700.0, 1350.0), rel=1e-3)

    def test_the_first_leg_is_the_targets_drift(self):
        setup = search_window(FORCING, START, LAT, LON, 3, 20, 0.0)
        # current + 2 % of the wind: (1.1, 0.54).
        assert setup.drift_bearing_deg == pytest.approx(np.degrees(np.arctan2(1.1, 0.54)))

    def test_the_old_random_walk_is_the_5_minute_run_too(self):
        # sigma with sigma_u = 0 rebuilds a D028 run, so both noise models share the seeds.
        setup = search_window(FORCING, START, LAT, LON, 7, 300, 3600.0, sigma_u=0.0,
                              sigma=26.3)
        five = run_ensemble(FORCING, 300, START, LAT, LON, (60 + 45) * 60.0, STEP_S,
                            datum_sigma_km=0.0, seed=7, save_every=300.0, sigma=26.3)
        assert np.array_equal(setup.window.lat[::5], five.lat[slice(12, 22)])
        assert np.array_equal(setup.window.lon[::5], five.lon[slice(12, 22)])

    def test_both_random_terms_at_once_is_refused(self):
        with pytest.raises(ValueError):
            search_window(FORCING, START, LAT, LON, 7, 10, 0.0, sigma_u=0.226, sigma=26.3)

    def test_it_searches(self):
        setup = search_window(FORCING, START, LAT, LON, 4, 2000, 3600.0)
        m = run(pattern_policy(expanding_square(SWEEP_WIDTH_M, setup.drift_bearing_deg)),
                SearchEpisode(setup.window, setup.marker))
        assert 0.0 < m["pos"] < 1.0


def row(**changes):
    base = {"scenario": "S01", "start": START, "lat": LAT, "lon": LON, "seed": 5}
    for h in range(1, 7):
        base[f"lat_{h}h"] = LAT + 0.01 * h
        base[f"lon_{h}h"] = LON + 0.02 * h
    base.update(changes)
    return pd.Series(base)


class TestScenarioRow:
    def test_the_truth_is_the_buoy_every_hour_in_seconds_since_arrival(self):
        target = truth_track(row(), 7200.0)
        assert list(target.t_s) == [-7200.0 + 3600.0 * h for h in range(7)]
        assert target.lat[3] == pytest.approx(LAT + 0.03)
        assert target.lon[0] == pytest.approx(360.0 + LON)

    def test_a_missing_hour_is_left_out(self):
        assert truth_track(row(lat_6h=np.nan, lon_6h=np.nan), 0.0).t_s[-1] == 5 * 3600.0

    def test_a_row_becomes_a_search_with_the_buoy_as_target(self):
        setup = scenario_search(row(), FORCING, 3600.0, 100)
        assert setup.target is not None and setup.window.lat.shape == (46, 100)
        m = run(pattern_policy(expanding_square(SWEEP_WIDTH_M, 0.0)),
                SearchEpisode(setup.window, setup.marker, target=setup.target))
        assert m["target"]["gaps"] == 0
        assert m["target"]["closest_m"] is not None

    def test_a_row_is_found_by_name(self, tmp_path):
        path = tmp_path / "scenarios.csv"
        pd.DataFrame([row(), row(scenario="S02")]).to_csv(path, index=False)
        assert scenario_row(path, "S02")["scenario"] == "S02"
        with pytest.raises(ValueError, match="S03"):
            scenario_row(path, "S03")


class TestSteady:
    def test_a_point_cloud_rides_with_its_marker(self):
        setup = steady_search((LAT, LON), 0.0, 10, np.random.default_rng(0), (1.8, -0.4))
        mlat, mlon = setup.marker.at(np.arange(46) * 60.0)
        assert np.allclose(setup.window.lat[:, 0], mlat, atol=1e-12)
        assert np.allclose(setup.window.lon[:, 0], mlon, atol=1e-12)

    def test_drift_bearing_of_still_water_is_none(self):
        assert drift_bearing(ConstantForcing(), LAT, LON, START) is None


def path_row(east_km, north_km):
    """A row whose buoy is at these (east, north) km from the start at hours 1 to 4."""
    base = {"lat": LAT, "lon": LON}
    for h, (e, n) in enumerate(zip(east_km, north_km), start=1):
        lat, lon = offset_position(LAT, LON, e * 1000.0, n * 1000.0)
        base[f"lat_{h}h"], base[f"lon_{h}h"] = float(lat), float(lon)
    return pd.Series(base)


class TestDatumLine:
    def test_north_and_east(self):
        lat, lon = offset_position(LAT, LON, 0.0, 1000.0)
        bearing, length = datum_line((LAT, LON), (lat, lon))
        assert bearing == pytest.approx(0.0, abs=1e-6) and length == pytest.approx(1000.0, rel=1e-6)
        lat, lon = offset_position(LAT, LON, 1000.0, 0.0)
        bearing, length = datum_line((LAT, LON), (lat, lon))
        assert bearing == pytest.approx(90.0, abs=1e-6) and length == pytest.approx(1000.0, rel=1e-4)

    def test_either_longitude_convention(self):
        lat, lon = offset_position(LAT, LON + 360.0, -500.0, -500.0)
        bearing, length = datum_line((LAT, LON), (lat, lon))
        assert bearing == pytest.approx(225.0, abs=0.01) and length == pytest.approx(707.1, rel=1e-3)

    def test_no_length_no_bearing(self):
        assert datum_line((LAT, LON), (LAT, LON)) == (None, 0.0)


class TestDoctrinalSearcher:
    def setup_method(self):
        # Drift due east; the datum line from a start 3 km south-west of the datum runs NE.
        self.setup = steady_search((LAT, LON), 1.0, 50, np.random.default_rng(1),
                                   current=(1.0, 0.0))
        self.lkp = offset_position(*self.setup.datum, -2121.3, -2121.3)

    def test_the_square_and_the_sector_fly_along_the_drift(self):
        for name in ("expanding-square", "sector"):
            pattern, bearing, why = doctrinal_searcher(name, self.setup, self.lkp)
            assert bearing == pytest.approx(90.0) and why == "along the drift"
            assert pattern.duration_s >= 45 * 60.0

    def test_parallel_and_trackline_fly_along_the_datum_line(self):
        for name in ("parallel", "trackline"):
            _, bearing, why = doctrinal_searcher(name, self.setup, self.lkp)
            assert bearing == pytest.approx(45.0, abs=0.05) and why == "along the datum line"

    def test_a_datum_line_under_50_m_falls_back_to_the_drift(self):
        near = offset_position(*self.setup.datum, 0.0, -40.0)
        _, bearing, why = doctrinal_searcher("parallel", self.setup, near)
        assert bearing == pytest.approx(90.0) and why == "along the drift"

    def test_the_trackline_is_at_least_as_long_as_the_datum_line(self):
        far = offset_position(*self.setup.datum, -8000.0, 0.0)
        long_line, _, _ = doctrinal_searcher("trackline", self.setup, far)
        short_line, _, _ = doctrinal_searcher("trackline", self.setup, self.lkp)
        reach = lambda p: np.max(np.hypot(*p.offset_at(p.t_s)))   # noqa: E731
        assert reach(long_line) > 7900.0 > reach(short_line)

    def test_no_drift_means_north(self):
        still = steady_search((LAT, LON), 1.0, 50, np.random.default_rng(1))
        _, bearing, why = doctrinal_searcher("sector", still, still.datum)
        assert bearing == 0.0 and why.startswith("no drift")

    def test_only_coast_guard_patterns(self):
        with pytest.raises(ValueError, match="greedy"):
            doctrinal_searcher("greedy", self.setup, self.lkp)


class TestStraightness:
    def test_a_straight_line_is_1(self):
        assert straightness(path_row([1, 2, 3, 4], [0, 0, 0, 0])) == pytest.approx(1.0)

    def test_a_right_angle_halfway_is_0_71(self):
        assert straightness(path_row([1, 2, 2, 2], [0, 0, 1, 2])) == pytest.approx(
            np.sqrt(2) / 2, rel=1e-3)

    def test_back_where_it_started_is_0(self):
        assert straightness(path_row([1, 2, 1, 0], [0, 0, 0, 0])) == pytest.approx(0.0, abs=1e-9)

    def test_no_motion_or_a_missing_fix_is_nan(self):
        assert np.isnan(straightness(path_row([0, 0, 0, 0], [0, 0, 0, 0])))
        missing = path_row([1, 2, 3, 4], [0, 0, 0, 0])
        missing["lat_3h"] = np.nan
        assert np.isnan(straightness(missing))
