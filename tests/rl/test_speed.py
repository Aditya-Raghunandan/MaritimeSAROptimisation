"""Tests for Telemachus S (15 Oct, L43): an agent that also picks its speed each minute, and the
referee that lets it, with every other searcher's flight unchanged."""

import numpy as np
import pytest

pytest.importorskip("gymnasium")

from sar.rl.env import SearchEnv
from sar.rl.features import SIZE, actions, speed_share, split_speed, two_halves
from sar.rl.fly import ppo_policy
from sar.search.platform import KNOT_MS, SEARCH_SPEED_MS, TURN_RATE_DEG_S, turn_rate_deg_s

SPEEDS = (45.0, 60.0, 90.0)


class TestTheReferee:
    def test_a_slower_helicopter_turns_tighter_and_flies_less_far(self, windows):
        env = SearchEnv([windows], headings=36, halves=12)
        env.reset(seed=0, options={"item": 0})
        ep = env.episode
        assert ep.speed_ms == SEARCH_SPEED_MS and ep.turn_rate_deg_s == TURN_RATE_DEG_S
        ep.set_speed(45 * KNOT_MS)
        assert ep.turn_rate_deg_s == pytest.approx(turn_rate_deg_s(30.0, 45 * KNOT_MS))
        assert ep.turn_rate_deg_s == pytest.approx(2 * TURN_RATE_DEG_S)        # half the speed
        ep.step((ep.heading_deg + 90.0) % 360)
        assert ep.metrics()["distance_m"] == pytest.approx(45 * KNOT_MS * 60.0)

    def test_a_path_flown_too_fast_for_the_speed_set_is_still_refused(self, windows):
        env = SearchEnv([windows], headings=36, halves=12)
        env.reset(seed=0, options={"item": 0})
        ep = env.episode
        fast = two_halves(ep, ep.heading_deg, ep.heading_deg)      # planned at 90 kt
        ep.set_speed(45 * KNOT_MS)
        with pytest.raises(ValueError, match="faster than"):
            ep.fly(fast)

    def test_a_speed_must_be_positive(self, windows):
        env = SearchEnv([windows], headings=36)
        env.reset(seed=0, options={"item": 0})
        for bad in (0.0, -1.0, float("nan")):
            with pytest.raises(ValueError, match="speed_ms"):
                env.episode.set_speed(bad)


class TestTheActions:
    def test_actions_are_speed_major(self):
        assert actions(36, 12, SPEEDS) == 432 and actions(36, 12) == 144 and actions(36) == 36
        assert split_speed(5, 36, 12) == (5, None)
        assert split_speed(5, 36, 12, SPEEDS) == (5, pytest.approx(45 * KNOT_MS))
        assert split_speed(144 + 5, 36, 12, SPEEDS) == (5, pytest.approx(60 * KNOT_MS))
        assert split_speed(431, 36, 12, SPEEDS) == (143, pytest.approx(90 * KNOT_MS))


class TestTheEnvironment:
    def test_passes_gymnasiums_checker(self, windows):
        from gymnasium.utils.env_checker import check_env
        check_env(SearchEnv([windows], headings=36, halves=12, speeds=SPEEDS), skip_render_check=True)

    def test_it_sees_its_speed_and_flies_the_one_it_picks(self, windows):
        env = SearchEnv([windows], headings=36, halves=12, decay=0.0, speeds=SPEEDS)
        assert env.action_space.n == 432
        obs, _ = env.reset(seed=0, options={"item": 0})
        assert obs.shape == (SIZE + 1,) and obs[-1] == pytest.approx(1.0)      # it arrives at 90 kt
        obs, _, _, _, info = env.step(0)                                       # 45 kt, north and north
        assert info["speed_ms"] == pytest.approx(45 * KNOT_MS) and obs[-1] == pytest.approx(0.5)
        assert env.episode.metrics()["distance_m"] == pytest.approx(45 * KNOT_MS * 60.0)
        _, _, _, _, info = env.step(2 * 144 + 3 * 12 + 3)                       # 90 kt, east and east
        assert info["speed_ms"] == pytest.approx(SEARCH_SPEED_MS)
        done = False
        while not done:
            _, _, done, _, _ = env.step(int(env.action_space.sample()))

    def test_without_speeds_it_is_the_environment_it_was(self, windows):
        env = SearchEnv([windows], headings=36, halves=12)
        obs, _ = env.reset(seed=0, options={"item": 0})
        assert env.action_space.n == 144 and obs.shape == (SIZE,)
        _, _, _, _, info = env.step(7)
        assert info["speed_ms"] == SEARCH_SPEED_MS

    def test_a_speed_must_be_positive_knots(self, windows):
        with pytest.raises(ValueError, match="speeds"):
            SearchEnv([windows], headings=36, halves=12, speeds=(0.0, 90.0))


def test_two_halves_never_repeats_a_time_at_the_join(windows):
    """13 Oct: a first half ending a few rounding units off the second command made a ~1e-15 s
    turn at 30 s, two waypoints at one time, and Waypoints refused it (t3_find died). Random start
    headings, both halves the same command: 420 failures in 400,000 before the fix."""
    env = SearchEnv([windows], headings=36, halves=12)
    env.reset(seed=0, options={"item": 0})
    ep = env.episode
    rng = np.random.default_rng(13)
    for h in rng.uniform(0.0, 360.0, 20_000):
        ep._heading = float(h)
        wp = two_halves(ep, 30.0, 30.0)
        assert np.all(np.diff(wp.t_s) > 0.0)


def test_a_speed_agent_is_flown_as_it_was_trained(windows):
    """The benchmark's flight of a speed agent (fly.ppo_policy) sets the speed it picks."""

    class Fixed:
        def __init__(self, action):
            self.action, self.seen = action, []

        def predict(self, obs, deterministic=True):
            self.seen.append(obs)
            return self.action, None

    env = SearchEnv([windows], headings=36, halves=12)
    env.reset(seed=0, options={"item": 0})
    model = Fixed(144 + 5)                                     # 60 kt
    policy = ppo_policy(model, 36, env.context, halves=12, speeds=SPEEDS)
    env.episode.fly(policy(env.episode))
    assert env.episode.speed_ms == pytest.approx(60 * KNOT_MS)
    assert model.seen[0].shape == (SIZE + 1,) and model.seen[0][-1] == pytest.approx(1.0)
    assert speed_share(env.episode) == pytest.approx(60 / 90)
