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
from sar.search.episode import STEPS, SearchEpisode, run


class TestReward:
    def test_a_find_in_the_first_minute_pays_in_full_and_in_the_last_one_45th(self):
        assert reward(0.2, 0, STEPS) == pytest.approx(0.2)
        assert reward(0.2, STEPS - 1, STEPS) == pytest.approx(0.2 / STEPS)

    def test_the_return_weights_each_minute_by_the_time_left(self):
        removed = [0.1] * STEPS
        assert time_weighted_return(removed) == pytest.approx(0.1 * (STEPS + 1) / 2)

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
        again = run(lambda ep: heading_of(next(flown), 36), SearchEpisode(setup.window, setup.marker))
        assert info["pos"] == pytest.approx(again["pos"], abs=1e-12)
        assert again["removed_per_step"] == pytest.approx(removed, abs=1e-15)

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
        assert vector["transit_h"] == pytest.approx(0.5)
