"""Tests for sar.validate.region: the 90 % region found from the particles (#89, #52).

The load-bearing check is calibration on a known answer: when the truth is drawn from the
same distribution as the particles, it must fall inside the level-L region a fraction L
of the time, whatever the cloud's shape and whatever the kernel bandwidth.
"""

import numpy as np
import pytest

from sar.utils.geo import M_PER_DEG_LAT, metres_per_degree_lon
from sar.validate.region import densities, energy_score, local_metres, score_cloud


def coverage(sample, trials=300, n=500, levels=(0.5, 0.9), bandwidth=1.0, seed=0):
    rng = np.random.default_rng(seed)
    hits = {level: 0 for level in levels}
    for _ in range(trials):
        points, truth = sample(rng, n), sample(rng, 1)[0]
        s = score_cloud(points, truth, levels=levels, bandwidth=bandwidth)
        for level in levels:
            hits[level] += s.inside(level)
    return {level: hits[level] / trials for level in levels}


def gaussian(rng, n, spread=1000.0):
    return spread * rng.standard_normal((n, 2))


def stretched(rng, n):
    """5:1 and turned 30 degrees, the shape shear gives a cloud."""
    z = rng.standard_normal((n, 2)) * [5000.0, 1000.0]
    a = np.radians(30.0)
    return z @ np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])


class TestCalibration:
    # 300 trials: one standard error at 0.9 is 1.7 points, so 0.05 is about 3 of them.
    def test_a_round_cloud_covers_its_own_draws_at_the_stated_rate(self):
        got = coverage(gaussian)
        assert got[0.9] == pytest.approx(0.9, abs=0.05)
        assert got[0.5] == pytest.approx(0.5, abs=0.09)

    def test_a_stretched_cloud_does_too(self):
        got = coverage(stretched, seed=1)
        assert got[0.9] == pytest.approx(0.9, abs=0.05)

    @pytest.mark.parametrize("bandwidth", [0.5, 2.0])
    def test_the_answer_does_not_hang_on_the_kernel_width(self, bandwidth):
        assert coverage(gaussian, bandwidth=bandwidth, seed=2)[0.9] == pytest.approx(0.9, abs=0.05)


class TestTheRegion:
    def test_the_90_percent_area_of_a_round_gaussian_is_2_pi_s2_ln10(self):
        points = gaussian(np.random.default_rng(3), 2000)
        area = score_cloud(points, [0.0, 0.0]).area_m2[0.9]
        assert area == pytest.approx(2 * np.pi * 1000.0 ** 2 * np.log(10.0), rel=0.1)

    def test_a_curved_cloud_leaves_its_empty_middle_out(self):
        """A quarter ring 10 km out: its centre of curvature is far from every particle,
        so it is outside, while a point on the ring is inside."""
        rng = np.random.default_rng(4)
        r = 10_000.0 + 300.0 * rng.standard_normal(1000)
        theta = rng.uniform(0.0, np.pi / 2, 1000)
        points = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        assert not score_cloud(points, [0.0, 0.0]).inside(0.9)
        on_ring = 10_000.0 * np.array([np.cos(np.pi / 4), np.sin(np.pi / 4)])
        assert score_cloud(points, on_ring).inside(0.9)

    def test_a_cloud_with_no_spread_still_scores(self):
        points = np.zeros((50, 2))
        assert score_cloud(points, [0.0, 0.0]).inside(0.5)
        assert not score_cloud(points, [10_000.0, 0.0]).inside(0.95)

    def test_too_few_particles_raises(self):
        with pytest.raises(ValueError, match="at least 10"):
            densities(np.zeros((5, 2)), [0.0, 0.0])


class TestEnergyScore:
    def test_it_prefers_the_right_width_to_too_narrow_or_too_wide(self):
        rng = np.random.default_rng(5)
        mean = {0.5: 0.0, 1.0: 0.0, 2.0: 0.0}
        for _ in range(200):
            truth = 1000.0 * rng.standard_normal(2)
            for k in mean:
                mean[k] += energy_score(gaussian(rng, 300, 1000.0 * k), truth)
        assert mean[1.0] < mean[0.5] and mean[1.0] < mean[2.0]

    def test_a_point_ensemble_scores_its_distance(self):
        assert energy_score(np.full((20, 2), 3.0), [0.0, 4.0 + 3.0]) == pytest.approx(5.0)


class TestLocalMetres:
    def test_a_degree_north_and_a_degree_east(self):
        out = local_metres([27.0, 26.0], [281.0, 282.0], 26.0, 281.0)
        assert out[0] == pytest.approx([0.0, M_PER_DEG_LAT])
        assert out[1] == pytest.approx([metres_per_degree_lon(26.0), 0.0])

    def test_longitude_wraps_rather_than_going_the_long_way_round(self):
        out = local_metres([26.0], [0.5], 26.0, 359.5)
        assert out[0, 0] == pytest.approx(metres_per_degree_lon(26.0))
