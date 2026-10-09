"""Tests for sar.model.position. Hand-computed values only, no files."""

import json

import numpy as np
import pytest

from sar.model.position import (
    DEFAULT_SIGMA,
    EARTH_RADIUS_M,
    MEMORY_TIME_S,
    evolve_velocity_error,
    start_velocity_error,
    velocity_memory,
    INTEGRATION_STEP_SECONDS,
    POLAR_LIMIT_DEG,
    _cli,
    as_position,
    calculate_position,
    check_latitude,
    random_displacement,
    step_displacement,
)

# The middle of the study box (D014), where cos(lat) is 0.8949 and so cannot hide a bug.
LAT, LON = 26.5, 281.4

DEG_PER_M = 180.0 / (np.pi * EARTH_RADIUS_M)


class TestAsPosition:
    def test_accepts_a_list_a_tuple_and_an_array(self):
        for value in ([LAT, LON], (LAT, LON), np.array([LAT, LON])):
            assert as_position(value).tolist() == [LAT, LON]

    def test_accepts_a_stack_of_pairs(self):
        assert as_position([[26.5, 281.4], [27.0, 281.0]]).shape == (2, 2)

    def test_integers_become_floats(self):
        assert as_position([26, 281]).dtype == np.dtype(float)

    def test_three_coordinates_raise(self):
        with pytest.raises(ValueError, match=r"position must be \[lat, lon\]"):
            as_position([26.5, 281.4, 0.0])

    def test_a_scalar_raises(self):
        with pytest.raises(ValueError, match="got shape"):
            as_position(26.5)

    def test_the_message_names_the_offending_argument(self):
        with pytest.raises(ValueError, match="start must be"):
            as_position([1.0, 2.0, 3.0], "start")


class TestAcceptanceCriteria:
    def test_zero_drift_leaves_the_position_unchanged(self):
        assert calculate_position([LAT, LON], [0.0, 0.0]).tolist() == [LAT, LON]

    def test_northward_drift_raises_the_latitude_and_leaves_the_longitude_alone(self):
        moved = calculate_position([LAT, LON], [0.0, 1.0])
        assert moved[0] > LAT
        assert moved[1] == LON

    def test_eastward_drift_raises_the_longitude_and_leaves_the_latitude_alone(self):
        moved = calculate_position([LAT, LON], [1.0, 0.0])
        assert moved[1] > LON
        assert moved[0] == LAT

    def test_the_longitude_step_scales_with_one_over_cos_latitude(self):
        drift = [1.0, 0.0]
        at_equator = calculate_position([0.0, LON], drift)[1] - LON
        for lat in (17.0, 26.5, 36.0, 60.0):
            here = calculate_position([lat, LON], drift)[1] - LON
            assert here == pytest.approx(at_equator / np.cos(np.radians(lat)))

    def test_the_default_timestep_is_sixty_seconds(self):
        assert INTEGRATION_STEP_SECONDS == 60.0
        assert calculate_position([LAT, LON], [1.0, 1.0]) == pytest.approx(
            calculate_position([LAT, LON], [1.0, 1.0], timestep=60.0)
        )


class TestTheArithmetic:
    def test_a_northward_step_is_the_distance_over_the_radius(self):
        moved = calculate_position([LAT, LON], [0.0, 1.0])
        assert moved[0] - LAT == pytest.approx(60.0 * DEG_PER_M)

    def test_an_eastward_step_is_that_divided_by_cos_latitude(self):
        moved = calculate_position([LAT, LON], [1.0, 0.0])
        assert moved[1] - LON == pytest.approx(60.0 * DEG_PER_M / np.cos(np.radians(LAT)))

    def test_one_degree_of_latitude_is_about_111_kilometres(self):
        """The figure a reader can check against a chart rather than against this code."""
        moved = calculate_position([LAT, LON], [0.0, 1.0], timestep=111195.0)
        assert moved[0] - LAT == pytest.approx(1.0, rel=1e-4)

    def test_the_step_scales_linearly_with_the_timestep(self):
        one = calculate_position([LAT, LON], [1.2, 0.9], timestep=60.0) - np.array([LAT, LON])
        ten = calculate_position([LAT, LON], [1.2, 0.9], timestep=600.0) - np.array([LAT, LON])
        assert ten == pytest.approx(10.0 * one)

    def test_a_negative_timestep_runs_the_step_backwards(self):
        forward = calculate_position([LAT, LON], [1.2, 0.9], timestep=60.0)
        assert calculate_position(forward, [1.2, 0.9], timestep=-60.0) == pytest.approx([LAT, LON])

    def test_the_components_do_not_leak_into_each_other(self):
        """Crossing [lat, lon] against [u, v] is the likeliest way to get this wrong."""
        east = calculate_position([LAT, LON], [1.0, 0.0])
        north = calculate_position([LAT, LON], [0.0, 1.0])
        assert east[0] == LAT and north[1] == LON
        assert calculate_position([LAT, LON], [1.0, 1.0]) == pytest.approx([north[0], east[1]])

    def test_the_inputs_are_not_modified(self):
        position, drift = np.array([LAT, LON]), np.array([1.2, 0.9])
        calculate_position(position, drift)
        assert position.tolist() == [LAT, LON]
        assert drift.tolist() == [1.2, 0.9]

    def test_a_non_finite_timestep_raises(self):
        with pytest.raises(ValueError, match="finite number of seconds"):
            calculate_position([LAT, LON], [1.0, 0.0], timestep=np.inf)


class TestTheStochasticTerm:
    def test_the_default_sigma_is_zero_so_the_step_is_deterministic(self):
        assert DEFAULT_SIGMA == 0.0
        repeated = [calculate_position([LAT, LON], [1.2, 0.9]).tolist() for _ in range(5)]
        assert repeated[1:] == repeated[:-1]

    def test_a_zero_sigma_draws_nothing_at_all(self):
        assert random_displacement((4, 2), 60.0, 0.0).tolist() == np.zeros((4, 2)).tolist()

    def test_a_non_zero_sigma_moves_the_particle_off_the_deterministic_path(self):
        deterministic = calculate_position([LAT, LON], [1.2, 0.9])
        noisy = calculate_position([LAT, LON], [1.2, 0.9], sigma=1.0, rng=7)
        assert noisy.tolist() != deterministic.tolist()

    def test_the_same_seed_gives_the_same_step(self):
        a = calculate_position([LAT, LON], [1.2, 0.9], sigma=1.0, rng=7)
        b = calculate_position([LAT, LON], [1.2, 0.9], sigma=1.0, rng=7)
        assert a.tolist() == b.tolist()

    def test_different_seeds_give_different_steps(self):
        a = calculate_position([LAT, LON], [1.2, 0.9], sigma=1.0, rng=7)
        b = calculate_position([LAT, LON], [1.2, 0.9], sigma=1.0, rng=8)
        assert a.tolist() != b.tolist()

    def test_the_displacement_has_standard_deviation_sigma_root_dt(self):
        """D009's sigma sqrt(dt) Z: 1 m/s^0.5 over 60 s is 7.746 m per component."""
        draws = random_displacement((200_000, 2), 60.0, 1.0, rng=0)
        assert draws.std(axis=0) == pytest.approx(np.sqrt(60.0), rel=0.02)
        assert draws.mean(axis=0) == pytest.approx([0.0, 0.0], abs=0.05)

    def test_it_scales_as_the_square_root_of_the_timestep(self):
        short = random_displacement((200_000, 2), 60.0, 1.0, rng=0).std()
        long = random_displacement((200_000, 2), 240.0, 1.0, rng=0).std()
        assert long / short == pytest.approx(2.0, rel=0.02)

    def test_the_noise_reaches_the_position_in_the_right_units(self):
        moved = calculate_position([LAT, LON], [0.0, 0.0], sigma=1.0, rng=0)
        draw = random_displacement((2,), 60.0, 1.0, rng=0)
        assert moved[0] - LAT == pytest.approx(draw[1] * DEG_PER_M)
        assert moved[1] - LON == pytest.approx(draw[0] * DEG_PER_M / np.cos(np.radians(LAT)))

    def test_each_particle_of_an_ensemble_gets_its_own_draw(self):
        moved = calculate_position(np.tile([LAT, LON], (50, 1)), [0.0, 0.0], sigma=1.0, rng=0)
        assert len(np.unique(moved[:, 0])) == 50

    def test_a_negative_sigma_raises(self):
        with pytest.raises(ValueError, match="sigma must not be negative"):
            calculate_position([LAT, LON], [1.0, 0.0], sigma=-0.1)


class TestLongitudeConvention:
    def test_longitude_comes_back_in_the_stored_zero_to_360_range(self):
        """D020 stores 0 to 360, so a position given as -78.6 comes back as 281.4."""
        assert calculate_position([LAT, -78.6], [0.0, 0.0])[1] == pytest.approx(281.4)

    def test_a_westward_step_across_the_prime_meridian_wraps_rather_than_going_negative(self):
        assert calculate_position([LAT, 0.0], [-1.0, 0.0], timestep=3600.0)[1] > 359.0

    def test_an_eastward_step_across_the_antimeridian_stays_in_range(self):
        assert 0.0 <= calculate_position([LAT, 359.999], [1.0, 0.0], timestep=3600.0)[1] < 1.0


class TestLatitudeLimits:
    def test_a_position_too_near_a_pole_raises(self):
        with pytest.raises(ValueError, match="the position given is at latitude"):
            calculate_position([89.5, LON], [1.0, 0.0])

    def test_a_step_that_would_carry_it_past_the_limit_raises(self):
        with pytest.raises(ValueError, match="the position after the step"):
            calculate_position([88.5, LON], [0.0, 100.0], timestep=86400.0)

    def test_the_limit_is_well_outside_the_study_box(self):
        assert POLAR_LIMIT_DEG == 89.0
        calculate_position([36.0, LON], [1.0, 1.0])

    def test_check_latitude_lets_nan_through(self):
        check_latitude(np.array([np.nan, 26.5]), "a position")


class TestNaNAndEnsembles:
    def test_a_missing_drift_gives_a_missing_position(self):
        """HYCOM writes NaN for land, and land must not read as still water (D016)."""
        assert np.isnan(calculate_position([LAT, LON], [np.nan, np.nan])).all()

    def test_one_missing_component_does_not_contaminate_the_other(self):
        moved = calculate_position([LAT, LON], [np.nan, 1.0])
        assert np.isnan(moved[1])
        assert moved[0] == pytest.approx(LAT + 60.0 * DEG_PER_M)

    def test_a_nan_position_stays_nan_instead_of_raising(self):
        assert np.isnan(calculate_position([np.nan, np.nan], [1.0, 1.0])).all()

    def test_one_drift_advances_a_whole_ensemble(self):
        positions = np.array([[26.5, 281.4], [27.0, 281.0], [36.0, 280.0]])
        moved = calculate_position(positions, [1.0, 0.0])
        assert moved.shape == (3, 2)
        assert moved[:, 0] == pytest.approx(positions[:, 0])
        expected = 60.0 * DEG_PER_M / np.cos(np.radians(positions[:, 0]))
        assert moved[:, 1] - positions[:, 1] == pytest.approx(expected)

    def test_a_drift_per_particle_pairs_row_by_row(self):
        positions = np.array([[0.0, 100.0], [0.0, 200.0]])
        moved = calculate_position(positions, np.array([[1.0, 0.0], [0.0, 1.0]]))
        assert moved[0, 1] - 100.0 == pytest.approx(60.0 * DEG_PER_M)
        assert moved[1, 0] == pytest.approx(60.0 * DEG_PER_M)


class TestTheEnsembleBuffer:
    def test_writes_into_the_buffer_it_is_given(self):
        positions = np.array([[26.5, 281.4], [27.0, 281.0]])
        buffer = np.zeros_like(positions)
        moved = calculate_position(positions, [1.0, 0.0], out=buffer)
        assert moved is buffer
        assert buffer == pytest.approx(calculate_position(positions, [1.0, 0.0]))

    def test_a_buffer_of_the_wrong_shape_raises(self):
        with pytest.raises(ValueError, match="out has shape"):
            calculate_position(np.zeros((3, 2)), [1.0, 0.0], out=np.zeros((2, 2)))


class TestStepDisplacement:
    def test_is_the_drift_times_the_timestep(self):
        assert step_displacement([1.2, 0.9], 60.0) == pytest.approx([72.0, 54.0])

    def test_the_default_timestep_matches_the_position_step(self):
        assert step_displacement([1.0, 0.0]) == pytest.approx([60.0, 0.0])

    def test_it_agrees_with_the_degrees_the_position_step_produces(self):
        east, north = step_displacement([1.2, 0.9], 600.0)
        moved = calculate_position([LAT, LON], [1.2, 0.9], timestep=600.0)
        assert moved[0] - LAT == pytest.approx(north * DEG_PER_M)
        assert moved[1] - LON == pytest.approx(east * DEG_PER_M / np.cos(np.radians(LAT)))


class TestCLI:
    def test_reports_the_moved_position_and_the_distance(self):
        out = _cli(["--position", "26.5", "281.4", "--drift", "1.2", "0.9"])
        assert out["moved_deg"] == pytest.approx(
            calculate_position([26.5, 281.4], [1.2, 0.9]).tolist()
        )
        assert out["drift_distance_m"] == pytest.approx(np.hypot(72.0, 54.0))

    def test_the_timestep_flag_reaches_the_calculation(self):
        out = _cli(["--position", "26.5", "281.4", "--drift", "1.0", "0.0", "--timestep", "600"])
        assert out["drift_displacement_m"] == pytest.approx([600.0, 0.0])

    def test_the_default_timestep_is_the_integration_step(self):
        out = _cli(["--position", "26.5", "281.4", "--drift", "1.0", "0.0"])
        assert out["timestep_s"] == INTEGRATION_STEP_SECONDS
        assert out["sigma"] == DEFAULT_SIGMA

    def test_a_seeded_sigma_is_repeatable_from_the_command_line(self):
        args = ["--position", "26.5", "281.4", "--drift", "1.2", "0.9",
                "--sigma", "1.0", "--seed", "7"]
        assert _cli(args)["moved_deg"] == _cli(args)["moved_deg"]

    def test_the_output_is_json_serialisable(self):
        json.dumps(_cli(["--position", "26.5", "281.4", "--drift", "1.2", "0.9"]))


class TestRandomVelocity:
    """D030: a velocity error per particle that forgets itself over T_L (Ornstein-Uhlenbeck)."""

    def test_the_memory_is_exp_of_minus_dt_over_t_l(self):
        assert velocity_memory(60.0, MEMORY_TIME_S) == pytest.approx(np.exp(-60.0 / (25.7 * 3600)))
        assert velocity_memory(60.0, MEMORY_TIME_S) == pytest.approx(0.99935, abs=1e-5)

    def test_the_memory_time_is_the_fit(self):
        assert MEMORY_TIME_S == 25.7 * 3600.0

    def test_the_start_is_the_steady_state(self):
        u = start_velocity_error((200_000, 2), 0.2, np.random.default_rng(1))
        assert u.std(axis=0) == pytest.approx([0.2, 0.2], rel=0.01)
        assert u.mean(axis=0) == pytest.approx([0.0, 0.0], abs=0.002)

    def test_the_variance_stays_sigma_u_squared_whatever_the_step(self):
        rng = np.random.default_rng(2)
        for dt in (60.0, 600.0, 7200.0):
            u = start_velocity_error((100_000, 2), 0.2, rng)
            for _ in range(50):
                u = evolve_velocity_error(u, dt, 0.2, 3600.0, rng)
            assert u.std() == pytest.approx(0.2, rel=0.015)

    @pytest.mark.parametrize("lag_steps", [1, 45, 60])
    def test_the_correlation_falls_as_exp_of_minus_lag_over_t_l(self, lag_steps):
        # One minute: 0.99935. 45 minutes, one search: 0.971. One hour at T_L = 1 h: 0.37.
        memory = 3600.0 if lag_steps == 60 else MEMORY_TIME_S
        rng = np.random.default_rng(3)
        u0 = start_velocity_error((200_000,), 0.2, rng)
        u = u0
        for _ in range(lag_steps):
            u = evolve_velocity_error(u, 60.0, 0.2, memory, rng)
        expected = np.exp(-lag_steps * 60.0 / memory)
        assert np.corrcoef(u0, u)[0, 1] == pytest.approx(expected, abs=0.004)

    def test_zero_sigma_u_is_no_random_velocity(self):
        assert np.all(start_velocity_error((5, 2), 0.0) == 0.0)
        assert np.all(evolve_velocity_error(np.zeros((5, 2)), 60.0, 0.0) == 0.0)

    def test_the_same_generator_state_gives_the_same_draws(self):
        a = evolve_velocity_error(np.ones(4), 60.0, 0.2, rng=np.random.default_rng(9))
        b = evolve_velocity_error(np.ones(4), 60.0, 0.2, rng=np.random.default_rng(9))
        assert a.tolist() == b.tolist()

    @pytest.mark.parametrize("bad", [-0.1, np.nan, np.inf])
    def test_a_bad_sigma_u_raises(self, bad):
        with pytest.raises(ValueError, match="sigma_u"):
            start_velocity_error((3, 2), bad)

    @pytest.mark.parametrize("bad", [0.0, -1.0, np.nan])
    def test_a_bad_memory_time_raises(self, bad):
        with pytest.raises(ValueError, match="memory time"):
            velocity_memory(60.0, bad)


class TestSigmaUByTheCurrent:
    """D033: the random velocity sized by the model's current at the start."""

    def test_it_grows_with_the_current_and_stops_at_the_cap(self):
        from sar.model.position import SIGMA_U_RULE, sigma_u_for_current
        cap = SIGMA_U_RULE["cap_speed_ms"]
        got = sigma_u_for_current(np.array([0.0, 0.3, 0.8, cap, cap + 1.0]))
        assert np.all(np.diff(got[:4]) > 0) and got[4] == got[3]
        assert got[0] == pytest.approx(np.hypot(SIGMA_U_RULE["a"], SIGMA_U_RULE["slide_ms"]))

    def test_a_negative_or_missing_speed_is_refused(self):
        from sar.model.position import sigma_u_for_current
        for bad in (-0.1, float("nan")):
            with pytest.raises(ValueError):
                sigma_u_for_current(bad)

    def test_one_value_per_particle_draws_what_one_value_for_all_draws(self):
        from sar.model.position import evolve_velocity_error, start_velocity_error
        a = start_velocity_error((5, 2), 0.3, np.random.default_rng(1))
        b = start_velocity_error((5, 2), np.full(5, 0.3), np.random.default_rng(1))
        assert np.array_equal(a, b)
        ea = evolve_velocity_error(a, 60.0, 0.3, rng=np.random.default_rng(2))
        eb = evolve_velocity_error(b, 60.0, np.full(5, 0.3), rng=np.random.default_rng(2))
        assert np.array_equal(ea, eb)

    def test_each_particle_spreads_by_its_own_sigma_u(self):
        from sar.model.position import start_velocity_error
        u = start_velocity_error((20000, 2), np.repeat([0.1, 0.5], 10000), np.random.default_rng(3))
        assert np.std(u[:10000]) == pytest.approx(0.1, rel=0.03)
        assert np.std(u[10000:]) == pytest.approx(0.5, rel=0.03)
