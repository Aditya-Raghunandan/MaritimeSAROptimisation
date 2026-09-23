"""Tests for sar.pipeline.track. Steady analytic fields only, no files and no network."""

import itertools
import json

import numpy as np
import pytest

from sar.model.drift import LEEWAY_COEFFICIENT
from sar.model.position import EARTH_RADIUS_M, INTEGRATION_STEP_SECONDS, calculate_position
from sar.pipeline.forcing import ConstantForcing
from sar.pipeline.track import DriftPipeline, TrackState, _cli

START = "2021-01-05T07:30"
LAT, LON = 26.5, 281.4
HOUR = 3600.0
STEP = INTEGRATION_STEP_SECONDS

DEG_PER_M = 180.0 / (np.pi * EARTH_RADIUS_M)


@pytest.fixture
def still():
    """No current and no wind, so the particle should not move at all."""
    return DriftPipeline(ConstantForcing(), STEP)


@pytest.fixture
def eastward():
    """A 1 m/s eastward current and calm air, so the drift is exactly 1 m/s east."""
    return DriftPipeline(ConstantForcing(current=(1.0, 0.0)), STEP)


class TestStepCount:
    def test_an_hour_at_the_default_step_is_sixty_steps(self, eastward):
        assert eastward.step_count(HOUR) == 60

    def test_a_duration_of_zero_is_no_steps(self, still):
        assert still.step_count(0.0) == 0

    def test_the_step_length_is_respected(self):
        assert DriftPipeline(ConstantForcing(), 600.0).step_count(HOUR) == 6

    def test_a_ragged_duration_raises_rather_than_being_shortened(self, still):
        with pytest.raises(ValueError, match="not a whole number"):
            still.step_count(90.0)

    def test_a_negative_duration_raises(self, still):
        with pytest.raises(ValueError, match="non-negative"):
            still.step_count(-60.0)

    def test_a_non_positive_timestep_raises(self):
        with pytest.raises(ValueError, match="positive number of seconds"):
            DriftPipeline(ConstantForcing(), 0.0)

    def test_the_step_is_required_rather_than_defaulted(self):
        """Every error in a run is linear in the step, so no run may inherit one."""
        with pytest.raises(TypeError):
            DriftPipeline(ConstantForcing())


class TestWhatTrackYields:
    def test_the_first_state_is_the_start_position_at_t_zero(self, eastward):
        first = next(eastward.track(START, LAT, LON, HOUR))
        assert first.step == 0
        assert first.seconds == 0.0
        assert first.time == np.datetime64(START)
        assert first.positions == pytest.approx(np.array([[LAT, LON]]))

    def test_n_steps_yield_n_plus_one_states(self, eastward):
        states = list(eastward.track(START, LAT, LON, HOUR))
        assert len(states) == 61
        assert [s.step for s in states] == list(range(61))

    def test_the_clock_advances_by_one_timestep_per_state(self, eastward):
        states = list(eastward.track(START, LAT, LON, 600.0))
        gaps = {b.time - a.time for a, b in itertools.pairwise(states)}
        assert gaps == {np.timedelta64(60, "s").astype("timedelta64[us]")}
        assert states[-1].time == np.datetime64("2021-01-05T07:40")

    def test_the_last_state_is_at_the_requested_duration(self, eastward):
        last = list(eastward.track(START, LAT, LON, HOUR))[-1]
        assert last.seconds == HOUR
        assert last.time == np.datetime64("2021-01-05T08:30")

    def test_only_the_last_state_has_no_forcing_attached(self, eastward):
        states = list(eastward.track(START, LAT, LON, 600.0))
        assert all(s.drift is not None for s in states[:-1])
        assert states[-1].drift is None and states[-1].current is None

    def test_a_duration_of_zero_yields_only_the_start(self, eastward):
        states = list(eastward.track(START, LAT, LON, 0.0))
        assert len(states) == 1
        assert states[0].positions == pytest.approx(np.array([[LAT, LON]]))
        assert states[0].drift is None


class TestTheTrackItself:
    def test_a_still_field_leaves_the_particle_where_it_started(self, still):
        for state in still.track(START, LAT, LON, HOUR):
            assert state.positions == pytest.approx(np.array([[LAT, LON]]))

    def test_an_hour_at_one_metre_per_second_east_covers_3600_metres(self, eastward):
        last = list(eastward.track(START, LAT, LON, HOUR))[-1]
        assert last.positions[0, 1] == pytest.approx(
            LON + HOUR * DEG_PER_M / np.cos(np.radians(LAT))
        )
        assert last.positions[0, 0] == pytest.approx(LAT)

    def test_each_state_is_the_previous_one_advanced_by_one_step(self, eastward):
        states = list(eastward.track(START, LAT, LON, 600.0))
        for before, after in itertools.pairwise(states):
            step = calculate_position(before.positions, before.drift, eastward.timestep)
            assert after.positions == pytest.approx(step)

    def test_the_drift_carries_the_leeway_term_from_the_wind(self):
        first = next(DriftPipeline(ConstantForcing(wind=(10.0, 0.0)), STEP)
                     .track(START, LAT, LON, HOUR))
        assert first.wind == pytest.approx(np.array([[10.0, 0.0]]))
        assert first.drift == pytest.approx(np.array([[10.0 * LEEWAY_COEFFICIENT, 0.0]]))

    def test_the_leeway_argument_reaches_the_drift(self):
        first = next(DriftPipeline(ConstantForcing(wind=(10.0, 0.0)), STEP, leeway=0.04)
                     .track(START, LAT, LON, HOUR))
        assert first.drift == pytest.approx(np.array([[0.4, 0.0]]))

    def test_a_longer_step_covers_the_same_ground_in_a_steady_field(self, eastward):
        """Euler is exact when the velocity does not change, so the step cannot matter here."""
        slow = list(eastward.track(START, LAT, LON, HOUR))[-1]
        fast = list(DriftPipeline(ConstantForcing(current=(1.0, 0.0)), 600.0)
                    .track(START, LAT, LON, HOUR))[-1]
        assert slow.positions == pytest.approx(fast.positions, rel=1e-9)


class TestTheStochasticTerm:
    def test_the_default_run_is_deterministic(self, eastward):
        assert eastward.sigma == 0.0
        first = list(eastward.track(START, LAT, LON, HOUR))[-1].positions
        second = list(eastward.track(START, LAT, LON, HOUR))[-1].positions
        assert first.tolist() == second.tolist()

    def test_a_non_zero_sigma_moves_the_track_off_the_deterministic_path(self, eastward):
        noisy = DriftPipeline(ConstantForcing(current=(1.0, 0.0)), STEP, sigma=0.5, seed=3)
        plain = list(eastward.track(START, LAT, LON, HOUR))[-1].positions
        assert list(noisy.track(START, LAT, LON, HOUR))[-1].positions.tolist() != plain.tolist()

    def test_the_same_seed_reproduces_the_whole_track(self):
        def run():
            pipeline = DriftPipeline(ConstantForcing(current=(1.0, 0.0)), STEP, sigma=0.5, seed=3)
            return [s.positions.tolist() for s in pipeline.track(START, LAT, LON, HOUR)]
        assert run() == run()

    def test_different_seeds_give_different_tracks(self):
        def run(seed):
            pipeline = DriftPipeline(ConstantForcing(current=(1.0, 0.0)), STEP,
                                     sigma=0.5, seed=seed)
            return list(pipeline.track(START, LAT, LON, HOUR))[-1].positions.tolist()
        assert run(3) != run(4)

    def test_the_spread_grows_as_the_square_root_of_time(self):
        """A random walk of n steps has variance n, so an hour spreads twice a quarter hour."""
        pipeline = DriftPipeline(ConstantForcing(), STEP, sigma=1.0, seed=0)
        spread = {s.seconds: (s.positions[:, 0] - LAT).std()
                  for s in pipeline.track(START, [LAT] * 4000, [LON] * 4000, HOUR)}
        assert spread[HOUR] / spread[900.0] == pytest.approx(2.0, rel=0.1)

    def test_a_negative_sigma_raises(self):
        with pytest.raises(ValueError, match="sigma must not be negative"):
            DriftPipeline(ConstantForcing(), STEP, sigma=-1.0)


class TestTheEnsembleItIsBuiltFor:
    def test_a_scalar_start_is_one_particle(self, eastward):
        assert eastward.start_positions(LAT, LON).shape == (1, 2)

    def test_several_starts_are_tracked_together(self, eastward):
        states = list(eastward.track(START, [26.5, 27.0, 36.0], [281.4, 281.0, 280.0], HOUR))
        assert states[0].positions.shape == (3, 2)
        assert states[-1].positions.shape == (3, 2)

    def test_each_particle_follows_its_own_latitude(self, eastward):
        lats = [0.0, 26.5, 60.0]
        last = list(eastward.track(START, lats, [281.4] * 3, HOUR))[-1]
        assert last.positions[:, 1] - 281.4 == pytest.approx(
            HOUR * DEG_PER_M / np.cos(np.radians(lats))
        )

    def test_particles_do_not_interact(self, eastward):
        """What the Monte Carlo ticket relies on: an ensemble is N independent tracks."""
        together = list(eastward.track(START, [26.5, 36.0], [281.4, 280.0], HOUR))[-1]
        alone = [list(eastward.track(START, lat, lon, HOUR))[-1].positions[0]
                 for lat, lon in ((26.5, 281.4), (36.0, 280.0))]
        assert together.positions == pytest.approx(np.array(alone))


class TestTrackStateRows:
    def test_one_row_per_particle_with_the_position_and_the_forcing(self, eastward):
        rows = next(eastward.track(START, [26.5, 27.0], [281.4, 281.0], HOUR)).rows()
        assert [r["particle"] for r in rows] == [0, 1]
        assert rows[0]["lat"] == pytest.approx(26.5)
        assert rows[0]["drift_u"] == pytest.approx(1.0)

    def test_the_final_state_reports_no_forcing(self, eastward):
        row = list(eastward.track(START, LAT, LON, 600.0))[-1].rows()[0]
        assert row["drift_u"] is None and row["current_v"] is None

    def test_the_rows_are_json_serialisable(self, eastward):
        json.dumps([row
                    for state in eastward.track(START, LAT, LON, 600.0)
                    for row in state.rows()])

    def test_a_state_can_be_built_without_forcing(self):
        assert TrackState(0, 0.0, np.datetime64(START), np.array([[LAT, LON]])).drift is None


class TestDescribe:
    def test_reports_what_the_run_was_asked_to_do(self, eastward):
        run = eastward.describe(START, LAT, LON, HOUR)
        assert run["steps"] == 60
        assert run["timestep_s"] == 60.0
        assert run["particles"] == 1
        assert run["start_lon"] == pytest.approx([LON])
        assert run["forcing"]["backend"] == "constant"

    def test_reports_the_sigma_and_the_seed_that_produced_the_run(self):
        pipeline = DriftPipeline(ConstantForcing(), STEP, sigma=0.5, seed=3)
        run = pipeline.describe(START, LAT, LON, HOUR)
        assert run["sigma"] == 0.5
        assert run["seed"] == 3

    def test_a_display_longitude_is_reported_in_the_stored_convention(self, eastward):
        assert eastward.describe(START, LAT, -78.6, HOUR)["start_lon"] == pytest.approx([281.4])


BASE_ARGS = ["--constant-current", "1.0", "0.0", "--start", START,
             "--lat", str(LAT), "--lon", str(LON), "--timestep", "60"]


class TestCLI:
    def test_a_steady_field_gives_the_hand_computed_answer(self):
        out = _cli([*BASE_ARGS, "--duration", "3600"])
        assert out["run"]["steps"] == 60
        assert len(out["track"]) == 61
        assert out["track"][-1]["lon"] == pytest.approx(
            LON + HOUR * DEG_PER_M / np.cos(np.radians(LAT))
        )

    def test_every_thins_the_output_but_keeps_the_first_and_the_last(self):
        out = _cli([*BASE_ARGS, "--duration", "3600", "--every", "10"])
        assert [r["step"] for r in out["track"]] == [0, 10, 20, 30, 40, 50, 60]

    def test_the_constant_wind_flag_reaches_the_drift(self):
        out = _cli(["--constant-current", "0", "0", "--constant-wind", "10", "0",
                    "--start", START, "--lat", str(LAT), "--lon", str(LON),
                    "--timestep", "60", "--duration", "60"])
        assert out["track"][0]["drift_u"] == pytest.approx(10.0 * LEEWAY_COEFFICIENT)

    def test_the_step_must_be_given(self):
        with pytest.raises(SystemExit):
            _cli(["--constant-current", "1.0", "0.0", "--start", START, "--lat", str(LAT),
                  "--lon", str(LON), "--duration", "60"])

    def test_a_seeded_sigma_is_repeatable_from_the_command_line(self):
        args = [*BASE_ARGS, "--duration", "600", "--sigma", "0.5", "--seed", "3"]
        assert _cli(args)["track"] == _cli(args)["track"]

    def test_the_output_is_json_serialisable(self):
        json.dumps(_cli([*BASE_ARGS, "--duration", "600"]))
