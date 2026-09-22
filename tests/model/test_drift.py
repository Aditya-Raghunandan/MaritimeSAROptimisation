"""Tests for sar.model.drift (issue #5). Hand-computed values only, no files."""

import json

import numpy as np
import pytest

from sar.model.drift import LEEWAY_COEFFICIENT, _cli, as_vector, calculate_drift

# A Gulf Stream cell: current running north-east at about 1.5 m/s under a 10 m/s wind
# blowing from the south-west, which is the regime where leeway stops being a rounding
# error on the drift budget.
CURRENT = np.array([1.2, 0.9])
WIND = np.array([7.0, 7.0])


class TestAsVector:
    def test_accepts_a_list_a_tuple_and_an_array(self):
        for value in ([1.0, 2.0], (1.0, 2.0), np.array([1.0, 2.0])):
            assert as_vector(value, "wind").tolist() == [1.0, 2.0]

    def test_accepts_a_stack_of_pairs(self):
        assert as_vector([[1.0, 2.0], [3.0, 4.0]], "current").shape == (2, 2)

    def test_integers_become_floats(self):
        assert as_vector([1, 2], "wind").dtype == np.dtype(float)

    def test_three_components_raise(self):
        with pytest.raises(ValueError, match="wind must be \\[u, v\\]"):
            as_vector([1.0, 2.0, 3.0], "wind")

    def test_a_scalar_raises(self):
        with pytest.raises(ValueError, match="current must be"):
            as_vector(3.0, "current")

    def test_the_message_names_the_offending_argument(self):
        with pytest.raises(ValueError, match="got shape \\(3,\\)"):
            as_vector([1.0, 2.0, 3.0], "wind")


class TestAcceptanceCriteria:
    def test_zero_wind_returns_the_current_unchanged(self):
        """Exact, not approximate: the leeway term is exactly zero, so nothing is added."""
        assert calculate_drift([0.0, 0.0], CURRENT).tolist() == CURRENT.tolist()

    def test_zero_current_returns_leeway_times_wind(self):
        drift = calculate_drift(WIND, [0.0, 0.0])
        assert drift.tolist() == (LEEWAY_COEFFICIENT * WIND).tolist()

    def test_the_default_leeway_is_two_percent(self):
        assert LEEWAY_COEFFICIENT == 0.02
        assert calculate_drift([10.0, 0.0], [0.0, 0.0]) == pytest.approx([0.2, 0.0])

    def test_the_leeway_argument_overrides_the_default(self):
        """0.01 and 0.04 are the ends of R1c's range, which the sensitivity sweep walks."""
        assert calculate_drift([10.0, 0.0], [0.0, 0.0], leeway=0.01) == pytest.approx([0.1, 0.0])
        assert calculate_drift([10.0, 0.0], [0.0, 0.0], leeway=0.04) == pytest.approx([0.4, 0.0])

    def test_takes_u_v_for_both_and_returns_u_v(self):
        drift = calculate_drift([7.0, 7.0], [1.2, 0.9])
        assert isinstance(drift, np.ndarray)
        assert drift.shape == (2,)
        assert drift == pytest.approx([1.34, 1.04])


class TestTheSumItself:
    def test_the_components_are_added_separately(self):
        """The whole point of vectors: a headwind partially cancels a following current."""
        drift = calculate_drift([-10.0, 0.0], [1.0, 0.0])
        assert drift == pytest.approx([0.8, 0.0])

    def test_a_crosswind_moves_the_target_across_the_stream(self):
        """Leeway acts along the wind, the current along the stream, and v is untouched."""
        drift = calculate_drift([10.0, 0.0], [0.0, 1.5])
        assert drift == pytest.approx([0.2, 1.5])

    def test_the_inputs_are_not_modified(self):
        wind, current = np.array([7.0, 7.0]), np.array([1.2, 0.9])
        calculate_drift(wind, current)
        assert wind.tolist() == [7.0, 7.0]
        assert current.tolist() == [1.2, 0.9]


class TestNaNAndEnsembles:
    def test_a_missing_current_gives_a_missing_drift(self):
        """HYCOM writes NaN for land, and land must not read as still water (D016)."""
        assert np.isnan(calculate_drift(WIND, [np.nan, np.nan])).all()

    def test_one_missing_component_does_not_contaminate_the_other(self):
        drift = calculate_drift([10.0, 10.0], [np.nan, 1.0])
        assert np.isnan(drift[0])
        assert drift[1] == pytest.approx(1.2)

    def test_one_wind_applies_to_a_whole_ensemble(self):
        """D009 vectorises over particles, so the same call serves N of them."""
        currents = np.array([[1.2, 0.9], [0.0, 0.0], [-1.0, 0.5]])
        drift = calculate_drift([10.0, 0.0], currents)
        assert drift.shape == (3, 2)
        assert drift == pytest.approx(np.array([[1.4, 0.9], [0.2, 0.0], [-0.8, 0.5]]))

    def test_a_wind_per_particle_pairs_row_by_row(self):
        winds = np.array([[10.0, 0.0], [0.0, 10.0]])
        currents = np.array([[1.0, 0.0], [1.0, 0.0]])
        assert calculate_drift(winds, currents) == pytest.approx(np.array([[1.2, 0.0], [1.0, 0.2]]))


class TestTheEnsembleBuffer:
    def test_writes_into_the_buffer_it_is_given(self):
        """The ensemble loop owns its arrays, so a step must not allocate a new one."""
        currents = np.array([[1.2, 0.9], [0.0, 0.0]])
        buffer = np.zeros_like(currents)
        drift = calculate_drift([10.0, 0.0], currents, out=buffer)
        assert drift is buffer
        assert buffer == pytest.approx(np.array([[1.4, 0.9], [0.2, 0.0]]))

    def test_the_buffered_answer_matches_the_allocating_one(self):
        currents = np.array([[1.2, 0.9], [-1.0, 0.5], [0.3, -2.0]])
        buffer = np.empty_like(currents)
        calculate_drift(WIND, currents, out=buffer)
        assert buffer == pytest.approx(calculate_drift(WIND, currents))

    def test_a_buffer_of_the_wrong_shape_raises(self):
        with pytest.raises(ValueError):
            calculate_drift(WIND, np.zeros((3, 2)), out=np.zeros((2, 2)))


class TestCLI:
    def test_reports_the_drift_and_its_leeway_term(self):
        out = _cli(["--wind", "7", "7", "--current", "1.2", "0.9"])
        assert out["drift_ms"] == pytest.approx([1.34, 1.04])
        assert out["leeway_term_ms"] == pytest.approx([0.14, 0.14])
        assert out["leeway"] == LEEWAY_COEFFICIENT

    def test_the_leeway_flag_reaches_the_calculation(self):
        out = _cli(["--wind", "10", "0", "--current", "0", "0", "--leeway", "0.04"])
        assert out["drift_ms"] == pytest.approx([0.4, 0.0])

    def test_the_speed_is_the_magnitude_of_the_drift(self):
        out = _cli(["--wind", "0", "0", "--current", "3", "4"])
        assert out["drift_speed_ms"] == pytest.approx(5.0)

    def test_the_output_is_json_serialisable(self):
        json.dumps(_cli(["--wind", "7", "7", "--current", "1.2", "0.9"]))
