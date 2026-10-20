"""Tests for the c2 experiment's doubled current: drift_c2, DriftPipelineC2, ensemble_c2, datum_c2 and windows_c2."""

import numpy as np
import pytest

from sar.model import drift, drift_c2
from sar.pipeline import ensemble, ensemble_c2
from sar.pipeline.forcing import ConstantForcing
from sar.pipeline.track import DriftPipeline
from sar.pipeline.track_c2 import DriftPipelineC2
from sar.rl import windows, windows_c2
from sar.search import datum, datum_c2

CURRENT, WIND = np.array([1.2, 0.9]), np.array([7.0, 7.0])   # a test choice: drift.py's Gulf Stream cell
FORCING = ConstantForcing(current=(0.3, 1.5), wind=(4.0, -2.0))   # a test choice: tests/rl/conftest.py's jet
LAT, LON, START = 25.1629, 279.9353, np.datetime64("2019-01-01T17:00", "us")


class TestCalculateDrift:
    def test_the_current_counts_twice_and_the_leeway_once(self):
        assert drift_c2.calculate_drift(WIND, CURRENT) == pytest.approx(2 * CURRENT + 0.02 * WIND)

    def test_it_differs_from_d002_by_exactly_one_current(self):
        gap = drift_c2.calculate_drift(WIND, CURRENT) - drift.calculate_drift(WIND, CURRENT)
        assert gap == pytest.approx(CURRENT)

    def test_out_matches_the_returned_array(self):
        buffer = np.empty((3, 2))
        currents = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
        drift_c2.calculate_drift(WIND, currents, out=buffer)
        assert buffer == pytest.approx(drift_c2.calculate_drift(WIND, currents))

    def test_nan_propagates(self):
        assert np.isnan(drift_c2.calculate_drift(WIND, [np.nan, np.nan])).all()


class TestPipeline:
    def test_in_calm_air_a_step_goes_twice_as_far_as_d002s(self):
        calm = ConstantForcing(current=(0.3, 1.5), wind=(0.0, 0.0))
        start = np.array([[LAT, LON]])
        base = DriftPipeline(calm, 60.0, sigma=0.0, sigma_u=0.0).advance(start, START)[0] - start
        c2 = DriftPipelineC2(calm, 60.0, sigma=0.0, sigma_u=0.0).advance(start, START)[0] - start
        assert c2 == pytest.approx(2 * base, rel=1e-6)

    def test_describe_records_the_factor(self):
        described = DriftPipelineC2(FORCING, 60.0).describe(START, LAT, LON, 600.0)
        assert described["current_factor"] == 2.0


class TestEnsemble:
    def test_same_seed_without_noise_the_c2_cloud_moves_twice_as_far_in_calm_air(self):
        calm = ConstantForcing(current=(0.3, 1.5), wind=(0.0, 0.0))
        args = (calm, 5, START, LAT, LON, 600.0, 60.0, 0.0)
        base = ensemble.run_ensemble(*args, seed=1, sigma_u=0.0)
        c2 = ensemble_c2.run_ensemble(*args, seed=1, sigma_u=0.0)
        assert c2.lat[-1] - LAT == pytest.approx(2 * (base.lat[-1] - LAT), rel=1e-6)

    def test_the_sidecar_says_it_is_c2(self):
        run = ensemble_c2.describe_run(FORCING, 5, START, LAT, LON, 600.0, 60.0, 0.0, seed=1)
        assert run["pipeline"]["current_factor"] == 2.0


class TestDatumAndWindows:
    def test_the_marker_drifts_twice_as_far(self):
        base = datum.marker_track(LAT, LON, START, 600.0, FORCING)
        c2 = datum_c2.marker_track(LAT, LON, START, 600.0, FORCING)
        assert c2.lat[-1] - LAT == pytest.approx(2 * (base.lat[-1] - LAT), rel=1e-6)

    def test_the_drift_bearing_is_twice_the_current_plus_leeway(self):
        u, v = 2 * np.array([0.3, 1.5]) + 0.02 * np.array([4.0, -2.0])
        expected = np.degrees(np.arctan2(u, v)) % 360.0
        assert datum_c2.drift_bearing(FORCING, LAT, LON, START) == pytest.approx(expected)

    def test_windows_carry_the_c2_marker_and_bearing(self):
        run = {"lat": 25.16, "lon": -80.06, "start": "2019-01-01T17:00", "particles": 50,
               "timestep": 60.0, "duration": 6300.0, "seed": 7, "datum_lat": LAT, "datum_lon": LON}
        ens = ensemble_c2.run_ensemble(FORCING, 50, START, LAT, LON, 6300.0, 60.0, 0.0, seed=7,
                                       save_every=300.0, sigma_u=0.0)
        built = windows_c2.build(ens, run, FORCING, (1800.0,))
        base = windows.build(ens, run, FORCING, (1800.0,))
        assert built.lat == pytest.approx(base.lat)
        marker_moved = built.marker_lat[0, -1] - built.marker_lat[0, 0]
        assert marker_moved == pytest.approx(2 * (base.marker_lat[0, -1] - base.marker_lat[0, 0]), rel=1e-6)
        assert built.drift_bearing_deg[0] == pytest.approx(datum_c2.drift_bearing(
            FORCING, built.marker_lat[0, 0], built.marker_lon[0, 0], START + np.timedelta64(1800, "s")))
