"""Tests for sar.rl.env and sar.rl.features: the environment is the referee, paid found x time left."""

import numpy as np
import pytest

pytest.importorskip("gymnasium")

from sar.rl.env import SearchEnv, window_items
from sar.rl.features import (
    MAP_CELLS,
    NEAR_CELLS,
    SIZE,
    VECTOR,
    heading_of,
    reward,
    time_weighted_return,
)
from sar.search.episode import STEPS, run
from sar.search.scenario import search_episode


class TestReward:
    def test_a_find_in_the_first_minute_pays_in_full_and_in_the_last_one_45th(self):
        assert reward(0.2, 0, STEPS) == pytest.approx(0.2)
        assert reward(0.2, STEPS - 1, STEPS) == pytest.approx(0.2 / STEPS)

    def test_the_return_weights_each_minute_by_the_time_left(self):
        removed = [0.1] * STEPS
        assert time_weighted_return(removed) == pytest.approx(0.1 * (STEPS + 1) / 2)

    def test_decay_zero_pays_every_minute_in_full_and_a_half_decay_halfway(self):
        assert reward(0.2, STEPS - 1, STEPS, decay=0.0) == pytest.approx(0.2)
        assert reward(0.2, 0, STEPS, decay=0.0) == pytest.approx(0.2)
        assert reward(0.2, STEPS - 1, STEPS, decay=0.5) == pytest.approx(0.2 * (1 - 0.5 * (STEPS - 1) / STEPS))

    def test_headings_run_clockwise_from_north(self):
        assert [heading_of(a, 8) for a in (0, 2, 4, 6)] == [0.0, 90.0, 180.0, 270.0]
        assert heading_of(35, 36) == 350.0


class TestSearchEnv:
    def test_passes_gymnasiums_checker(self, windows):
        from gymnasium.utils.env_checker import check_env
        check_env(SearchEnv([windows], headings=36), skip_render_check=True)

    def test_an_episode_is_45_steps_and_the_observation_has_its_size(self, windows):
        env = SearchEnv([windows], headings=8)
        obs, _ = env.reset(seed=0)
        assert obs.shape == (SIZE,) == (MAP_CELLS ** 2 + NEAR_CELLS ** 2 + len(VECTOR),)
        steps, done = 0, False
        while not done:
            obs, _, done, truncated, _ = env.step(env.action_space.sample())
            steps += 1
            assert not truncated
        assert steps == STEPS

    def test_the_rewards_are_the_referees_finds_weighted_by_time_left(self, windows):
        env = SearchEnv([windows], headings=36)
        env.reset(seed=0, options={"item": 1})
        rng, total, removed, done = np.random.default_rng(3), 0.0, [], False
        actions = []
        while not done:
            a = int(rng.integers(36))
            actions.append(a)
            _, r, done, _, info = env.step(a)
            total += r
            removed.append(info["removed"])
        assert total == pytest.approx(time_weighted_return(removed), rel=1e-12)
        setup = windows.setup(1)
        flown = iter(actions)
        again = run(lambda ep: heading_of(next(flown), 36), search_episode(setup))
        assert info["pos"] == pytest.approx(again["pos"], abs=1e-12)
        assert again["removed_per_step"] == pytest.approx(removed, abs=1e-15)

    def test_with_decay_zero_the_return_is_the_pos_and_the_end_is_terminal(self, windows):
        env = SearchEnv([windows], headings=36, decay=0.0)
        env.reset(seed=0, options={"item": 1})
        rng, total, done, steps = np.random.default_rng(5), 0.0, False, 0
        while not done:
            _, r, done, truncated, info = env.step(int(rng.integers(36)))
            total += r
            steps += 1
            assert not truncated          # never bootstrapped past the window: gamma = 1 is safe
        assert steps == STEPS and done
        assert total == pytest.approx(info["pos"], rel=1e-12)

    def test_two_headings_a_minute_fly_each_half_and_the_referee_scores_the_minute(self, windows):
        from sar.rl.features import halves_of, two_halves
        env = SearchEnv([windows], headings=36, decay=0.0, halves=12)
        assert env.action_space.n == 144
        assert halves_of(12 * 3 + 6, 12) == (90.0, 180.0)
        env.reset(seed=0, options={"item": 0})
        wp = two_halves(env.episode, 90.0, 180.0)
        assert wp.t_s[-1] == pytest.approx(60.0) and np.any(np.isclose(wp.t_s, 30.0))
        assert wp.heading_deg[-1] == pytest.approx(180.0)
        _, r, done, truncated, info = env.step(12 * 3 + 6)          # east for 30 s, then south
        assert env.episode.heading_deg == pytest.approx(180.0)
        assert r == pytest.approx(info["pos"]) and not done and not truncated
        steps = 1
        while not done:
            _, _, done, _, _ = env.step(int(env.action_space.sample()))
            steps += 1
        assert steps == STEPS

    def test_two_halves_can_turn_further_than_one_heading_command_can(self, windows):
        """Two right turns of 150 deg in one minute: 300 deg, a loop or a teardrop. One heading a
        minute turns the short way, never more than 180 deg, so it can never hook back tightly."""
        from sar.rl.features import two_halves
        env = SearchEnv([windows], headings=36, halves=12)
        env.reset(seed=0, options={"item": 0})
        ep = env.episode
        h0 = ep.heading_deg
        wp = two_halves(ep, (h0 + 150.0) % 360, (h0 + 300.0) % 360)
        turned = np.sum(np.diff(np.unwrap(np.radians(np.concatenate(([h0], wp.heading_deg))))))
        assert np.degrees(turned) == pytest.approx(300.0, abs=1e-6)
        ep.fly(wp)                                   # the referee accepts it at 7 deg/s
        assert ep.heading_deg == pytest.approx((h0 + 300.0) % 360)

    def test_halves_must_be_zero_or_at_least_two(self, windows):
        with pytest.raises(ValueError, match="halves"):
            SearchEnv([windows], headings=8, halves=1)

    def test_a_decay_outside_0_to_1_is_refused(self, windows):
        with pytest.raises(ValueError, match="decay"):
            SearchEnv([windows], headings=8, decay=1.5)

    def test_flying_at_the_cloud_finds_something(self, windows):
        env = SearchEnv([windows], headings=8)
        env.reset(seed=0, options={"item": 0})
        _, r, *_ = env.step(0)
        assert r > 0.0

    def test_cycling_visits_every_window_once_per_lap(self, windows):
        env = SearchEnv([windows], headings=8, cycle=True)
        seen = [env.reset()[1]["arrival_s"] for _ in range(len(env.items))]
        assert sorted(seen) == [1800.0, 2700.0, 3600.0]

    def test_transits_can_be_chosen(self, windows_dir):
        paths = sorted(windows_dir.glob("windows_*.npz"))
        assert len(window_items(paths)) == 3 * len(paths)
        assert len(window_items(paths, [45])) == len(paths)

    def test_the_maps_are_scaled_to_their_fullest_cell(self, windows):
        env = SearchEnv([windows], headings=8)
        obs, _ = env.reset(seed=0, options={"item": 0})
        far = obs[:MAP_CELLS ** 2]
        assert far.max() == pytest.approx(1.0) and far.min() >= 0.0
        vector = dict(zip(VECTOR, obs[-len(VECTOR):]))
        assert vector["time_left"] == 1.0 and vector["remaining"] == pytest.approx(1.0)
        assert vector["arrival_h"] == pytest.approx(0.5)

    def test_the_heading_inputs_start_along_the_drift_and_follow_the_turns(self, windows):
        env = SearchEnv([windows], headings=8)
        obs, _ = env.reset(seed=0, options={"item": 0})
        vector = dict(zip(VECTOR, obs[-len(VECTOR):]))
        start = np.radians(env.context.drift_bearing_deg or 0.0)
        assert (vector["heading_sin"], vector["heading_cos"]) == pytest.approx(
            (np.sin(start), np.cos(start)), abs=1e-6)
        obs, *_ = env.step(2)
        vector = dict(zip(VECTOR, obs[-len(VECTOR):]))
        assert env.episode.heading_deg == pytest.approx(90.0, abs=1e-6)
        assert (vector["heading_sin"], vector["heading_cos"]) == pytest.approx((1.0, 0.0), abs=1e-6)


class TestFindTheRealBuoy:
    """Telemachus 3's reward (13 Oct): paid for coming within W/2 of the real buoy, then it ends."""

    @staticmethod
    def with_buoy(windows, k, off_deg=0.0):
        from dataclasses import replace
        a = float(windows.arrivals_s[k])
        return replace(windows, truth_t_s=np.array([a, a + STEPS * 60.0]),
                       truth_lat=windows.marker_lat[k][[0, -1]] + off_deg,
                       truth_lon=windows.marker_lon[k][[0, -1]])

    def test_a_buoy_at_the_marker_is_found_in_the_first_minute_and_the_search_ends(self, windows):
        env = SearchEnv([self.with_buoy(windows, 0)], headings=36, find=1.0, pos_scale=0.0)
        env.reset(seed=0, options={"item": 0})
        _, paid, done, truncated, info = env.step(0)
        assert paid == 1.0 and done and not truncated and info["found"]

    def test_a_buoy_out_of_reach_pays_only_the_scaled_probability(self, windows):
        env = SearchEnv([self.with_buoy(windows, 0, off_deg=1.0)], headings=36, find=1.0,
                        pos_scale=0.5)
        env.reset(seed=0, options={"item": 0})
        steps, done = 0, False
        while not done:
            _, paid, done, _, info = env.step(env.action_space.sample())
            assert not info["found"]
            assert paid == pytest.approx(0.5 * reward(info["removed"], steps, STEPS))
            steps += 1
        assert steps == STEPS

    def test_paying_for_finds_needs_windows_with_a_buoy(self, windows):
        env = SearchEnv([windows], headings=36, find=1.0)
        with pytest.raises(ValueError, match="no real buoy"):
            env.reset(seed=0)

    def test_negative_weights_are_refused(self, windows):
        with pytest.raises(ValueError):
            SearchEnv([windows], headings=36, find=-1.0)
