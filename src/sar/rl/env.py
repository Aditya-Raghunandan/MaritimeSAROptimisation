"""env.py: the search as a Gymnasium environment wrapped round the shared referee, one heading a minute."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import gymnasium as gym
import numpy as np

from sar.rl.features import SIZE, Context, heading_of, observe, reward
from sar.rl.windows import SearchWindows, load, parse_windows_name
from sar.search.episode import SearchEpisode


def window_items(sources, arrivals_min=None) -> list[tuple[object, int]]:
    """Every (source, arrival index) to train on, read from file names so no file is opened."""
    items = []
    for source in sources:
        minutes = ([round(a / 60.0) for a in source.arrivals_s] if isinstance(source, SearchWindows)
                   else parse_windows_name(source)["arrivals_min"])
        items += [(source, k) for k, m in enumerate(minutes)
                  if arrivals_min is None or m in arrivals_min]
    return items


class SearchEnv(gym.Env):
    """45 one-minute legs over one window's cloud; the reward is particles found x time left."""

    metadata: ClassVar[dict] = {"render_modes": []}

    def __init__(self, sources, headings: int, arrivals_min=None, cycle: bool = False):
        self.items = window_items(sources, arrivals_min)
        if not self.items:
            raise ValueError("SearchEnv needs at least one window")
        if isinstance(headings, bool) or not isinstance(headings, (int, np.integer)) or headings < 2:
            raise ValueError(f"headings must be an integer of at least 2, got {headings!r}")
        self.headings, self.cycle = int(headings), bool(cycle)
        self.action_space = gym.spaces.Discrete(self.headings)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (SIZE,), np.float32)
        self._next, self._cached = 0, (None, None)
        self.episode: SearchEpisode | None = None

    def _windows(self, source) -> SearchWindows:
        if isinstance(source, SearchWindows):
            return source
        if self._cached[0] != source:
            self._cached = (source, load(Path(source)))
        return self._cached[1]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        pick = (options or {}).get("item")
        if pick is None and self.cycle:
            pick, self._next = self._next, (self._next + 1) % len(self.items)
        elif pick is None:
            pick = int(self.np_random.integers(len(self.items)))
        source, k = self.items[pick]
        windows = self._windows(source)
        setup = windows.setup(k)
        self.context = Context(windows.lkp, setup.arrival_s, setup.drift_bearing_deg)
        self.episode = SearchEpisode(setup.window, setup.marker)
        self.pos = 0.0
        return observe(self.episode, self.context), {"window": windows.name,
                                                     "arrival_s": setup.arrival_s}

    def step(self, action):
        k = self.episode.k
        removed = self.episode.step(heading_of(int(action), self.headings))
        self.pos += removed
        done = self.episode.done
        return (observe(self.episode, self.context), reward(removed, k, self.episode.steps), done,
                False, {"pos": self.pos, "removed": removed})
