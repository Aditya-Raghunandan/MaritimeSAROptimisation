"""Unit tests for sar.utils.interpolation.

Pure numeric code: no fixtures, no network, no monkeypatching needed.
"""

import numpy as np
import pytest

from sar.utils.interpolation import interpolate_grid_series, interpolate_series


class TestInterpolateSeries:
    def test_returns_array_of_length_pieces(self):
        result = interpolate_series(1.0, 3.0, pieces=5)
        assert len(result) == 5

    def test_index_zero_equals_value_t1(self):
        result = interpolate_series(1.0, 3.0, pieces=5)
        assert result[0] == pytest.approx(1.0)

    def test_does_not_include_value_t2(self):
        result = interpolate_series(1.0, 3.0, pieces=5)
        assert not np.any(np.isclose(result, 3.0))

    def test_step_size_matches_expected(self):
        result = interpolate_series(1.0, 3.0, pieces=5)
        expected_step = (3.0 - 1.0) / 5
        np.testing.assert_allclose(np.diff(result), expected_step)

    def test_scalar_input_returns_1d_array(self):
        result = interpolate_series(1.0, 3.0, pieces=4)
        assert result.shape == (4,)

    def test_vector_input_returns_pieces_by_2(self):
        result = interpolate_series([1.0, 2.0], [3.0, 4.0], pieces=5)
        assert result.shape == (5, 2)
        np.testing.assert_allclose(result[0], [1.0, 2.0])
        expected_step = np.array([3.0 - 1.0, 4.0 - 2.0]) / 5
        np.testing.assert_allclose(result[1] - result[0], expected_step)

    def test_zero_pieces_raises(self):
        with pytest.raises(ValueError, match="pieces"):
            interpolate_series(1.0, 3.0, pieces=0)


class TestInterpolateGridSeries:
    def test_returns_pieces_by_n_by_2(self):
        points_t1 = [[1.0, 2.0], [10.0, 20.0], [-1.0, -2.0]]
        points_t2 = [[3.0, 4.0], [12.0, 22.0], [1.0, 2.0]]
        result = interpolate_grid_series(points_t1, points_t2, pieces=5)
        assert result.shape == (5, 3, 2)

    def test_index_zero_equals_points_t1(self):
        points_t1 = [[1.0, 2.0], [10.0, 20.0]]
        points_t2 = [[3.0, 4.0], [12.0, 22.0]]
        result = interpolate_grid_series(points_t1, points_t2, pieces=5)
        np.testing.assert_allclose(result[0], points_t1)

    def test_each_point_interpolated_independently(self):
        points_t1 = [[1.0, 2.0], [10.0, 20.0]]
        points_t2 = [[3.0, 4.0], [12.0, 22.0]]
        pieces = 5
        result = interpolate_grid_series(points_t1, points_t2, pieces)

        expected_point_0 = interpolate_series(points_t1[0], points_t2[0], pieces)
        expected_point_1 = interpolate_series(points_t1[1], points_t2[1], pieces)
        np.testing.assert_allclose(result[:, 0, :], expected_point_0)
        np.testing.assert_allclose(result[:, 1, :], expected_point_1)

    def test_mismatched_shapes_raises(self):
        points_t1 = [[1.0, 2.0], [10.0, 20.0]]
        points_t2 = [[3.0, 4.0]]
        with pytest.raises(ValueError, match="same shape"):
            interpolate_grid_series(points_t1, points_t2, pieces=5)

    def test_wrong_inner_dimension_raises(self):
        points_t1 = [[1.0, 2.0, 3.0]]
        points_t2 = [[4.0, 5.0, 6.0]]
        with pytest.raises(ValueError, match=r"\(N, 2\)"):
            interpolate_grid_series(points_t1, points_t2, pieces=5)
