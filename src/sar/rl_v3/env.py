"""env.py: v1's SearchEnv, paid each minute what it found divided by the Expanding Square's POS."""

from __future__ import annotations

from sar.rl.env import SearchEnv
from sar.rl_v3.square import key

# A window the square barely scores on would make every find there enormous; below this POS the
# divisor is held here. Project choice: 0.02 is under every per-arrival mean on the test weeks.
SQUARE_FLOOR = 0.02


class SearchEnvV3(SearchEnv):
    """Same inputs, windows, referee and headings as v1; a search's rewards add up to POS / square's POS."""

    def __init__(self, sources, headings: int, square: dict[str, float], arrivals_min=None,
                 cycle: bool = False):
        super().__init__(sources, headings, arrivals_min, cycle)
        self.square = square
        self.scale = 1.0

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        name, minutes = info["window"], round(info["arrival_s"] / 60.0)
        self.scale = max(self.square[key(name, minutes)], SQUARE_FLOOR)
        return obs, {**info, "square_pos": self.scale}

    def step(self, action):
        obs, _, done, truncated, info = super().step(action)
        return obs, info["removed"] / self.scale, done, truncated, info
