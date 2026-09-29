"""Tests for sar.search.sweep. Hand-computed geometry about one point, no files."""

import math

import numpy as np
import pytest

from sar.search.platform import SEARCH_SPEED_MS, STEP_S, SWEEP_WIDTH_M
from sar.search.sweep import _cli, closest_approach_m, relative_m, seen, sweep
from sar.utils.geo import M_PER_DEG_LAT, east_north, metres_per_degree_lon, offset_position

LAT0, LON0 = 26.5, 281.0
HALF = SWEEP_WIDTH_M / 2.0            # 92.6 m
LEG = SEARCH_SPEED_MS * STEP_S        # 2,778 m, one 60 s step at 90 kt
DRIFT = 1.8 * STEP_S                  # 108 m, one step of Gulf Stream


def at(east, north, lat0=LAT0, lon0=LON0):
    """Particles or a fix, (east, north) metres from the reference point, as (lat, lon)."""
    lat, lon = offset_position(lat0, lon0, np.asarray(east, float), np.asarray(north, float))
    return lat, lon


def fix(east, north, lat0=LAT0, lon0=LON0):
    lat, lon = at(east, north, lat0, lon0)
    return float(lat), float(lon)


def leg(bearing=90.0, length=LEG, lat0=LAT0, lon0=LON0):
    """A leg of `length` metres on `bearing`, centred on the reference point."""
    e, n = east_north(bearing, length / 2.0)
    return fix(-e, -n, lat0, lon0), fix(e, n, lat0, lon0)


EAST_LEG = leg()                      # x from -1389 to +1389 m, along y = 0


class TestRelativeM:
    def test_inverts_offset_position(self):
        lat, lon = offset_position(LAT0, LON0, 1234.5, -678.9)
        east, north = relative_m(lat, lon, LAT0, LON0)
        assert (east, north) == pytest.approx((1234.5, -678.9), rel=1e-9)

    def test_one_degree_of_latitude(self):
        assert relative_m(LAT0 + 1.0, LON0, LAT0, LON0) == pytest.approx((0.0, M_PER_DEG_LAT))

    def test_cos_lat_is_taken_at_the_reference(self):
        east, _ = relative_m(36.0, LON0 + 0.01, 36.0, LON0)
        assert east == pytest.approx(0.01 * metres_per_degree_lon(36.0))

    def test_longitude_wraps_across_zero(self):
        east, _ = relative_m(LAT0, 0.1, LAT0, 359.9)
        assert east == pytest.approx(0.2 * metres_per_degree_lon(LAT0))

    def test_either_longitude_convention(self):
        assert relative_m(LAT0, -78.99, LAT0, 281.0) == pytest.approx(
            relative_m(LAT0, 281.01, LAT0, 281.0))

    def test_arrays(self):
        east, north = relative_m(np.array([LAT0, LAT0 + 1.0]), np.array([LON0, LON0]), LAT0, LON0)
        assert north == pytest.approx([0.0, M_PER_DEG_LAT]) and east == pytest.approx([0.0, 0.0])


class TestClosestApproach:
    def test_beside_the_middle_of_a_segment(self):
        assert closest_approach_m(-1000.0, 50.0, 1000.0, 50.0) == pytest.approx(50.0)

    def test_past_the_end_measures_to_the_endpoint(self):
        # The segment (30, 40) -> (60, 80) points away from the origin: nearest is (30, 40).
        assert closest_approach_m(30.0, 40.0, 60.0, 80.0) == pytest.approx(50.0)
        assert closest_approach_m(60.0, 80.0, 30.0, 40.0) == pytest.approx(50.0)

    def test_a_zero_length_segment_is_the_distance_to_a_point(self):
        assert closest_approach_m(3.0, 4.0, 3.0, 4.0) == pytest.approx(5.0)

    def test_a_segment_through_the_origin_is_zero(self):
        assert closest_approach_m(-10.0, -10.0, 10.0, 10.0) == pytest.approx(0.0, abs=1e-12)

    def test_arrays(self):
        got = closest_approach_m(np.array([-1000.0, 30.0, 3.0]), np.array([50.0, 40.0, 4.0]),
                                 np.array([1000.0, 60.0, 3.0]), np.array([50.0, 80.0, 4.0]))
        assert got == pytest.approx([50.0, 50.0, 5.0])


class TestSeen:
    def test_on_the_track(self):
        lat, lon = at([0.0, -1000.0, 1300.0], [0.0, 0.0, 0.0])
        assert seen(lat, lon, *EAST_LEG).all()

    def test_edge_is_half_the_sweep_width(self):
        lat, lon = at([0.0, 0.0, 0.0, 0.0], [HALF - 1.0, HALF + 1.0, -(HALF - 1.0), -(HALF + 1.0)])
        assert seen(lat, lon, *EAST_LEG).tolist() == [True, False, True, False]

    def test_past_the_end_is_a_round_cap_not_a_strip(self):
        # 50 m beyond the east end on the line: 50 m away. 80 m beyond and 80 m north:
        # 113 m away, although only 80 m from the infinite line through the leg.
        lat, lon = at([LEG / 2 + 50.0, LEG / 2 + 80.0, -LEG / 2 - 50.0], [0.0, 80.0, 0.0])
        assert seen(lat, lon, *EAST_LEG).tolist() == [True, False, True]

    @pytest.mark.parametrize("bearing", [0.0, 45.0, 90.0, 200.0, 315.0])
    def test_a_zero_length_leg_sweeps_a_disc(self, bearing):
        centre = fix(0.0, 0.0)
        inside, outside = east_north(bearing, HALF - 1.0), east_north(bearing, HALF + 1.0)
        lat, lon = at([inside[0], outside[0]], [inside[1], outside[1]])
        assert seen(lat, lon, centre, centre).tolist() == [True, False]

    @pytest.mark.parametrize("bearing", [90.0, 0.0, 45.0])
    def test_the_same_leg_in_metres_sweeps_alike_at_17_and_36_north(self, bearing):
        # A lattice of offsets, kept only where the flat-earth answer is clearly in or out,
        # so centimetre differences in cos(lat) across the leg cannot flip a particle.
        e, n = np.meshgrid(np.arange(-1600.0, 1601.0, 37.0), np.arange(-300.0, 301.0, 11.0))
        e, n = e.ravel(), n.ravel()
        s, t = east_north(bearing, LEG / 2.0)
        truth = closest_approach_m(-s - e, -t - n, s - e, t - n)
        keep = np.abs(truth - HALF) > 0.5
        expected = truth[keep] <= HALF
        assert 0 < expected.sum() < expected.size
        for lat0 in (17.0, 36.0):
            lat, lon = at(e[keep], n[keep], lat0)
            assert np.array_equal(seen(lat, lon, *leg(bearing, lat0=lat0)), expected)

    def test_a_nan_particle_is_not_seen(self):
        assert seen(np.array([np.nan]), np.array([LON0]), *EAST_LEG).tolist() == [False]

    def test_mismatched_lat_and_lon_raise(self):
        with pytest.raises(ValueError, match="same shape"):
            seen(np.zeros(3), np.zeros(2), *EAST_LEG)

    def test_one_end_position_without_the_other_raises(self):
        lat, lon = at([0.0], [0.0])
        with pytest.raises(ValueError, match="both or neither"):
            seen(lat, lon, *EAST_LEG, lat_end=lat)
        with pytest.raises(ValueError, match="both or neither"):
            seen(lat, lon, *EAST_LEG, lon_end=lon)

    def test_end_positions_of_another_shape_raise(self):
        lat, lon = at([0.0, 1.0], [0.0, 1.0])
        with pytest.raises(ValueError, match="must match"):
            seen(lat, lon, *EAST_LEG, lat_end=lat[:1], lon_end=lon[:1])

    @pytest.mark.parametrize("width", [0.0, -185.2, np.nan, np.inf])
    def test_a_width_that_is_not_positive_raises(self, width):
        with pytest.raises(ValueError, match="sweep_width_m"):
            seen(*at([0.0], [0.0]), *EAST_LEG, sweep_width_m=width)

    @pytest.mark.parametrize("bad", [(np.nan, LON0), (LAT0,), (LAT0, LON0, 0.0), (LAT0, np.inf)])
    def test_a_helicopter_fix_that_is_not_one_finite_pair_raises(self, bad):
        with pytest.raises(ValueError, match="start"):
            seen(*at([0.0], [0.0]), bad, EAST_LEG[1])
        with pytest.raises(ValueError, match="end"):
            seen(*at([0.0], [0.0]), EAST_LEG[0], bad)


class TestSeenMoving:
    def test_end_positions_equal_to_the_start_change_nothing(self):
        rng = np.random.default_rng(3)
        lat, lon = at(rng.normal(0.0, 400.0, 500), rng.normal(0.0, 150.0, 500))
        assert np.array_equal(seen(lat, lon, *EAST_LEG, lat_end=lat, lon_end=lon),
                              seen(lat, lon, *EAST_LEG))

    def test_drifting_into_the_path_before_the_helicopter_arrives_is_seen(self):
        # Starts 150 m south of the east end and drifts north 108 m: 42 m off when the
        # helicopter gets there. Held still, it is 150 m off and missed (ADR003 section 3).
        lat, lon = at([LEG / 2], [-150.0])
        lat_end, lon_end = at([LEG / 2], [-150.0 + DRIFT])
        assert seen(lat, lon, *EAST_LEG).tolist() == [False]
        assert seen(lat, lon, *EAST_LEG, lat_end=lat_end, lon_end=lon_end).tolist() == [True]

    def test_drifting_out_before_the_helicopter_arrives_is_not_seen(self):
        # Starts 80 m south of x = +1000 and drifts south: when the helicopter passes x = +1000
        # at 86 % of the leg, it is 173 m off. Held still, it would count.
        lat, lon = at([1000.0], [-80.0])
        lat_end, lon_end = at([1000.0], [-80.0 - DRIFT])
        assert seen(lat, lon, *EAST_LEG).tolist() == [True]
        assert seen(lat, lon, *EAST_LEG, lat_end=lat_end, lon_end=lon_end).tolist() == [False]

    def test_keeping_pace_alongside_is_seen_only_within_the_half_width(self):
        lat, lon = at([-LEG / 2, -LEG / 2], [50.0, 150.0])
        lat_end, lon_end = at([LEG / 2, LEG / 2], [50.0, 150.0])
        assert seen(lat, lon, *EAST_LEG, lat_end=lat_end, lon_end=lon_end).tolist() == [True, False]


class TestSweep:
    def particles(self):
        """On the track, just inside, just outside, and far away."""
        lat, lon = at([0.0, 0.0, 0.0, 5000.0], [0.0, HALF - 1.0, HALF + 1.0, 0.0])
        return lat, lon, np.full(4, 0.25)

    def test_a_seen_particle_loses_pod_of_its_weight(self):
        lat, lon, w = self.particles()
        after, _ = sweep(lat, lon, w, *EAST_LEG, pod=0.8)
        assert after == pytest.approx([0.05, 0.05, 0.25, 0.25])

    def test_pod_one_by_default_clears_what_is_seen(self):
        lat, lon, w = self.particles()
        after, removed = sweep(lat, lon, w, *EAST_LEG)
        assert after.tolist() == [0.0, 0.0, 0.25, 0.25] and removed == pytest.approx(0.5)

    def test_pod_zero_changes_nothing(self):
        lat, lon, w = self.particles()
        after, removed = sweep(lat, lon, w, *EAST_LEG, pod=0.0)
        assert np.array_equal(after, w) and removed == 0.0

    def test_mass_removed_is_the_fall_in_total_weight_and_nothing_is_renormalised(self):
        rng = np.random.default_rng(7)
        lat, lon = at(rng.normal(0.0, 800.0, 2000), rng.normal(0.0, 200.0, 2000))
        w = rng.random(2000)
        w /= w.sum()
        after, removed = sweep(lat, lon, w, *EAST_LEG, pod=0.9)
        assert removed == pytest.approx(w.sum() - after.sum(), abs=1e-15)
        assert 0.0 < removed < 1.0 and after.sum() == pytest.approx(1.0 - removed)

    def test_the_input_weights_are_not_changed(self):
        lat, lon, w = self.particles()
        before = w.copy()
        sweep(lat, lon, w, *EAST_LEG)
        assert np.array_equal(w, before)

    def test_end_positions_reach_the_weights(self):
        lat, lon = at([LEG / 2], [-150.0])
        lat_end, lon_end = at([LEG / 2], [-150.0 + DRIFT])
        after, removed = sweep(lat, lon, np.ones(1), *EAST_LEG, lat_end=lat_end, lon_end=lon_end)
        assert after.tolist() == [0.0] and removed == 1.0

    def test_mismatched_weights_raise(self):
        lat, lon, _ = self.particles()
        with pytest.raises(ValueError, match="weight"):
            sweep(lat, lon, np.ones(3), *EAST_LEG)

    @pytest.mark.parametrize("pod", [-0.1, 1.1, np.nan])
    def test_pod_outside_zero_to_one_raises(self, pod):
        lat, lon, w = self.particles()
        with pytest.raises(ValueError, match="pod"):
            sweep(lat, lon, w, *EAST_LEG, pod=pod)

    def test_a_negative_width_raises(self):
        lat, lon, w = self.particles()
        with pytest.raises(ValueError, match="sweep_width_m"):
            sweep(lat, lon, w, *EAST_LEG, sweep_width_m=-1.0)


class TestCLI:
    def test_sweeps_a_synthetic_cloud(self):
        out = _cli(["--lat", "26.5", "--lon", "-79", "--particles", "2000", "--seed", "1",
                    "--bearing", "90"])
        assert out["particles"] == 2000 and out["particles_seen"] > 0
        assert 0.0 < out["mass_removed"] < 1.0
        assert out["mass_removed"] == pytest.approx(out["particles_seen"] / 2000)
        assert out["mass_left"] == pytest.approx(1.0 - out["mass_removed"])

    def test_matches_the_capsule_over_a_gaussian_cloud(self):
        # The strip |y| < W/2, |x| < L/2 over a 2 km Gaussian, plus the two half-disc caps
        # at the density found at the leg's ends. 0.0198 of the mass; 4 sigma at N = 2e5.
        s, a, h = 2000.0, HALF, LEG / 2

        def pdf(x):
            return math.exp(-0.5 * (x / s) ** 2) / (s * math.sqrt(2 * math.pi))

        strip = math.erf(a / (s * math.sqrt(2))) * math.erf(h / (s * math.sqrt(2)))
        expected = strip + math.pi * a * a * pdf(h) * pdf(0.0)
        n = 200_000
        out = _cli(["--lat", "26.5", "--lon", "-79", "--particles", str(n), "--seed", "2",
                    "--spread-km", "2", "--bearing", "90"])
        tolerance = 4.0 * math.sqrt(expected * (1 - expected) / n)
        assert out["mass_removed"] == pytest.approx(expected, abs=tolerance)

    def test_a_leg_length_that_is_not_positive_raises(self):
        with pytest.raises(ValueError, match="length_m"):
            _cli(["--lat", "26.5", "--lon", "-79", "--length-m", "0"])
