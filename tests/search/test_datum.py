"""Tests for sar.search.datum. Hand-computed values only, no files."""

import numpy as np
import pytest

from sar.model.position import EARTH_RADIUS_M
from sar.pipeline.forcing import ConstantForcing
from sar.search.datum import _cli, datum_at, drift_track, marker_track

LAT, LON = 26.5, 281.0
START = "2019-06-01T06:00"

# calculate_position's own conversion, so the expected numbers use the step's arithmetic.
M_PER_DEG = EARTH_RADIUS_M * np.pi / 180.0


def east_m(lon1, lon0=LON, lat=LAT):
    return (lon1 - lon0) * M_PER_DEG * np.cos(np.radians(lat))


class TestDriftTrack:
    def test_ten_minutes_east_at_one_metre_a_second(self):
        track = drift_track(LAT, LON, START, 600.0, ConstantForcing(current=(1.0, 0.0)), 0.0)
        assert track.t_s == pytest.approx(np.arange(0.0, 601.0, 60.0))
        assert east_m(track.lon[-1]) == pytest.approx(600.0, rel=1e-6)
        assert track.lat[-1] == pytest.approx(LAT)

    def test_a_ragged_duration_ends_with_a_short_step(self):
        track = drift_track(LAT, LON, START, 90.0, ConstantForcing(current=(1.0, 0.0)), 0.0)
        assert track.t_s.tolist() == [0.0, 60.0, 90.0]
        assert east_m(track.lon[-1]) == pytest.approx(90.0, rel=1e-6)

    def test_under_one_step_is_one_short_step(self):
        track = drift_track(LAT, LON, START, 30.0, ConstantForcing(current=(1.0, 0.0)), 0.0)
        assert track.t_s.tolist() == [0.0, 30.0]

    def test_leeway_adds_a_fraction_of_the_wind(self):
        forcing = ConstantForcing(current=(0.0, 0.0), wind=(10.0, 0.0))
        track = drift_track(LAT, LON, START, 600.0, forcing, 0.02)
        assert east_m(track.lon[-1]) == pytest.approx(0.2 * 600.0, rel=1e-6)

    def test_display_longitude_goes_in_and_store_longitude_comes_out(self):
        track = drift_track(LAT, LON - 360.0, START, 60.0, ConstantForcing(), 0.0)
        assert track.lon[0] == pytest.approx(LON)

    def test_bad_inputs_raise(self):
        with pytest.raises(ValueError, match="positive number of seconds"):
            drift_track(LAT, LON, START, 0.0, ConstantForcing(), 0.0)
        with pytest.raises(ValueError, match="must not be negative"):
            drift_track(LAT, LON, START, 60.0, ConstantForcing(), -0.01)


class TestDatumAndMarker:
    def test_the_datum_moves_with_current_and_leeway(self):
        forcing = ConstantForcing(current=(1.8, 0.0), wind=(5.0, 5.0))
        lat, lon = datum_at(LAT, LON, START, 4620.0, forcing, leeway=0.02)
        assert east_m(lon) == pytest.approx(1.9 * 4620.0, rel=1e-4)
        assert (lat - LAT) * M_PER_DEG == pytest.approx(0.1 * 4620.0, rel=1e-6)

    def test_the_marker_ignores_the_wind(self):
        forcing = ConstantForcing(current=(1.8, 0.0), wind=(5.0, 5.0))
        marker = marker_track(LAT, LON, START, 2700.0, forcing)
        assert east_m(marker.lon[-1]) == pytest.approx(1.8 * 2700.0, rel=1e-4)
        assert marker.lat[-1] == pytest.approx(LAT)

    def test_a_datum_with_no_leeway_is_where_the_marker_goes(self):
        forcing = ConstantForcing(current=(0.7, -0.4), wind=(8.0, 3.0))
        lat, lon = datum_at(LAT, LON, START, 2700.0, forcing, leeway=0.0)
        marker = marker_track(LAT, LON, START, 2700.0, forcing)
        assert (lat, lon) == pytest.approx((marker.lat[-1], marker.lon[-1]))

    def test_a_person_leaves_a_marker_centred_pattern_at_the_leeway_rate(self):
        # The doctrinal gap: 2 % of a 10 m/s wind is 0.2 m/s, 540 m over 45 minutes.
        forcing = ConstantForcing(current=(1.8, 0.0), wind=(0.0, 10.0))
        person = drift_track(LAT, LON, START, 2700.0, forcing, 0.02)
        marker = marker_track(LAT, LON, START, 2700.0, forcing)
        gap_m = (person.lat[-1] - marker.lat[-1]) * M_PER_DEG
        assert gap_m == pytest.approx(540.0, rel=1e-6)

    def test_marker_track_needs_a_forcing(self):
        with pytest.raises(ValueError, match="needs a forcing"):
            marker_track(LAT, LON, START, 60.0)


class TestCLI:
    def test_prints_the_datum_and_the_marker(self):
        out = _cli(["--lat", "26.5", "--lon", "-79.0", "--start", START, "--elapsed-min", "77",
                    "--constant-current", "1.8", "0.0", "--constant-wind", "5.0", "5.0"])
        assert out["datum"]["lon"] == pytest.approx(-78.9118, abs=1e-4)
        assert out["marker_end"]["lat"] == pytest.approx(out["datum"]["lat"])
