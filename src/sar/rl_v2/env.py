"""env.py: v1's SearchEnv with 2D observations and the final POS as its only reward."""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from sar.rl.env import SearchEnv
from sar.rl_v2.features import FAR_SHAPE, NEAR_SHAPE, VECTOR, reward, split


class SearchEnvV2(SearchEnv):
    """Same windows, referee and headings as v1; only what it sees and what it is paid change."""

    def __init__(self, sources, headings: int, arrivals_min=None, cycle: bool = False):
        super().__init__(sources, headings, arrivals_min, cycle)
        self.observation_space = gym.spaces.Dict({
            "far": gym.spaces.Box(0.0, 1.0, FAR_SHAPE, np.float32),
            "near": gym.spaces.Box(0.0, 1.0, NEAR_SHAPE, np.float32),
            "vec": gym.spaces.Box(-np.inf, np.inf, (len(VECTOR),), np.float32),
        })

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        return split(obs), info

    def step(self, action):
        obs, _, done, truncated, info = super().step(action)
        return split(obs), reward(info["pos"], done), done, truncated, info
