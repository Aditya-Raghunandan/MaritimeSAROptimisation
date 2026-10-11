"""Tests for sar.rl.windows: a 5-minute run cut into the referee's 1-minute search windows."""

import numpy as np
import pytest
from conftest import ARRIVALS_S, FORCING, make_run

from sar.rl.windows import build, frames_at, load, parse_windows_name, save, windows_name
from sar.search.episode import STEPS
from sar.search.sweep import relative_m


class TestNames:
    def test_the_name_round_trips_to_the_run_and_its_transits(self):
        run, _ = make_run()
        name = windows_name(run, ARRIVALS_S)
        assert name == "windows_2516N08006W_20190101T1700_N600_dt60s_T105m_seed7_arr30-45-60m"
        back = parse_windows_name(f"{name}.npz")
        assert back["arrivals_min"] == [30, 45, 60]
        assert back["start"] == "2019-01-01T17:00" and back["seed"] == 7

    def test_a_stranger_is_refused(self):
        with pytest.raises(ValueError, match="not a windows file name"):
            parse_windows_name("ensemble_2516N08006W_20190101T1700_N600_dt60s_T4h_seed7.npz")


class TestFrames:
    def test_a_saved_time_is_the_save_and_halfway_is_the_mean(self):
        t = np.array([0.0, 300.0])
        lat = np.array([[1.0, 2.0], [3.0, 6.0]])
        lon = np.array([[359.9, 10.0], [0.1, 20.0]])
        beached = np.zeros((2, 2), dtype=bool)
        la, lo, _ = frames_at(t, lat, lon, beached, [0.0, 150.0, 300.0])
        assert la.tolist() == [[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]]
        assert lo[1] == pytest.approx([0.0, 15.0], abs=1e-9)

    def test_times_outside_the_saves_are_refused(self):
        with pytest.raises(ValueError, match="outside the saves"):
            frames_at(np.array([0.0, 300.0]), np.zeros((2, 1)), np.zeros((2, 1)),
                      np.zeros((2, 1), dtype=bool), [400.0])


class TestBuild:
    def test_each_window_is_46_frames_a_minute_apart_from_its_arrival(self, windows):
        for k, arrival in enumerate(ARRIVALS_S):
            setup = windows.setup(k)
            seconds = (setup.window.times - windows.start) / np.timedelta64(1, "s")
            assert seconds.tolist() == (arrival + 60.0 * np.arange(STEPS + 1)).tolist()
            assert setup.marker.t_s.tolist() == (60.0 * np.arange(STEPS + 1)).tolist()
            assert setup.arrival_s == arrival

    def test_the_marker_starts_at_the_cloud_centroid(self, windows):
        setup = windows.setup(1)
        assert (float(setup.marker.lat[0]), float(setup.marker.lon[0])) == pytest.approx(setup.datum)

    def test_five_minute_saves_rebuild_the_minute_by_minute_cloud_to_a_metre_rms(self, windows):
        _, every_minute = make_run(save_every=60.0)
        start = int(ARRIVALS_S[0] / 60.0)
        frames = every_minute.lat[start:start + STEPS + 1], every_minute.lon[start:start + STEPS + 1]
        window = windows.setup(0).window
        miss = np.hypot(*relative_m(window.lat, window.lon, *frames))
        assert np.sqrt(np.mean(miss ** 2)) < 1.0
        assert miss.max() < 10.0                  # against a 92.6 m half-strip

    def test_the_drift_bearing_is_the_current_plus_leeway(self, windows):
        u, v = 0.3 + 0.02 * 4.0, 1.5 + 0.02 * -2.0
        assert windows.drift_bearing_deg[0] == pytest.approx(np.degrees(np.arctan2(u, v)))

    def test_a_window_past_the_end_of_the_run_is_refused(self):
        run, ens = make_run(duration=3600.0)
        with pytest.raises(ValueError, match="needs"):
            build(ens, run, FORCING, ARRIVALS_S)

    def test_save_and_load_round_trip(self, windows, tmp_path):
        back = load(save(windows, tmp_path))
        assert back.name == windows.name and back.lkp == windows.lkp
        assert np.array_equal(back.lat, windows.lat)
        assert np.array_equal(back.marker_lon, windows.marker_lon)
        assert back.setup(2).window.times[0] == windows.setup(2).window.times[0]
