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
    drift_bearing,
    scenario_row,
    scenario_search,
    search_window,
    steady_search,
    truth_track,
)
from sar.search.sweep import relative_m

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
