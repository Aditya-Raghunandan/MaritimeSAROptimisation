"""env.py: the search as a Gymnasium environment wrapped round the shared referee, one heading a minute."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import gymnasium as gym
import numpy as np

from sar.rl.features import SIZE, Context, command, observe, reward
from sar.rl.windows import SearchWindows, load, parse_windows_name
from sar.search.episode import SearchEpisode
from sar.search.scenario import search_episode


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
    """45 one-minute legs over one window's cloud; the reward is particles found x time left.

    `decay` sets how much time left weighs (features.reward): 1 is the first agent's reward,
    0 pays plain POS. The window's end is `terminated`, never `truncated`: the search really
    stops at 45 minutes, so nothing is bootstrapped past it (safe with gamma = 1).

    FINDING THE REAL BUOY (13 Oct, Telemachus 3). `find` > 0 pays that much the minute the
    helicopter first comes within W/2 of the windows' real buoy (the referee's closest approach),
    and the search ends there, terminated, as a real one would. `pos_scale` multiplies the
    probability reward above, so find = 1 with pos_scale = 0 pays only for finding the buoy and
    an episode's return is 1 or 0. Needs windows that carry a buoy (`windows.build_truth`).
    """

    metadata: ClassVar[dict] = {"render_modes": []}

    def __init__(self, sources, headings: int, arrivals_min=None, cycle: bool = False,
                 decay: float = 1.0, halves: int = 0, find: float = 0.0, pos_scale: float = 1.0):
        self.items = window_items(sources, arrivals_min)
        if not self.items:
            raise ValueError("SearchEnv needs at least one window")
        if isinstance(headings, bool) or not isinstance(headings, (int, np.integer)) or headings < 2:
            raise ValueError(f"headings must be an integer of at least 2, got {headings!r}")
        if not 0.0 <= float(decay) <= 1.0:
            raise ValueError(f"decay must be between 0 and 1, got {decay!r}")
        if isinstance(halves, bool) or not isinstance(halves, (int, np.integer)) or halves < 0 \
                or halves == 1:
            raise ValueError(f"halves must be 0 (one heading a minute) or at least 2, got {halves!r}")
        if float(find) < 0.0 or float(pos_scale) < 0.0:
            raise ValueError(f"find and pos_scale must not be negative, got {find!r} and {pos_scale!r}")
        self.headings, self.cycle, self.decay = int(headings), bool(cycle), float(decay)
        self.halves = int(halves)
        self.find, self.pos_scale = float(find), float(pos_scale)
        self.action_space = gym.spaces.Discrete(self.halves ** 2 if self.halves else self.headings)
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
        if self.find and setup.target is None:
            raise ValueError(f"{windows.name} has no real buoy to find; build it with build_truth")
        self.context = Context(windows.lkp, setup.arrival_s, setup.drift_bearing_deg)
        self.episode = search_episode(setup)
        self.pos = 0.0
        return observe(self.episode, self.context), {"window": windows.name,
                                                     "arrival_s": setup.arrival_s}

    def step(self, action):
        k = self.episode.k
        order = command(self.episode, int(action), self.headings, self.halves)
        seen = self.episode.found_s is not None
        removed = self.episode.fly(order) if self.halves else self.episode.step(order)
        self.pos += removed
        done = self.episode.done
        paid = self.pos_scale * reward(removed, k, self.episode.steps, self.decay)
        found = self.find > 0.0 and not seen and self.episode.found_s is not None
        if found:
            paid, done = paid + self.find, True
        return (observe(self.episode, self.context), paid, done, False,
                {"pos": self.pos, "removed": removed, "found": found})
