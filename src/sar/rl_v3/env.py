"""env.py: v1's SearchEnv, paid each minute what it found divided by the Expanding Square's POS."""

from __future__ import annotations

import sys
import time
import zipfile

from sar.rl.env import SearchEnv
from sar.rl_v3.square import key

# A window the square barely scores on would make every find there enormous; below this POS the
# divisor is held here. Project choice: 0.02 is under every per-arrival mean on the test weeks.
SQUARE_FLOOR = 0.02
# A windows file that reads back corrupt (bad CRC) under load on the shared disk is read again
# this many times, a few seconds apart, before the run is stopped with the file's name.
READ_TRIES, READ_PAUSE_S = 4, 5.0


class SearchEnvV3(SearchEnv):
    """Same inputs, windows, referee and headings as v1; a search's rewards add up to POS / square's POS."""

    def __init__(self, sources, headings: int, square: dict[str, float], arrivals_min=None,
                 cycle: bool = False):
        super().__init__(sources, headings, arrivals_min, cycle)
        self.square = square
        self.scale = 1.0

    def _windows(self, source):
        for attempt in range(1, READ_TRIES + 1):
            try:
                return super()._windows(source)
            except (zipfile.BadZipFile, OSError, ValueError) as error:
                if attempt == READ_TRIES:
                    raise RuntimeError(f"{source} still unreadable after {READ_TRIES} reads: {error}") from error
                print(f"reading {source} failed ({error}); trying again in {READ_PAUSE_S:g} s",
                      file=sys.stderr, flush=True)
                time.sleep(READ_PAUSE_S)

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        name, minutes = info["window"], round(info["arrival_s"] / 60.0)
        self.scale = max(self.square[key(name, minutes)], SQUARE_FLOOR)
        return obs, {**info, "square_pos": self.scale}

    def step(self, action):
        obs, _, done, truncated, info = super().step(action)
        return obs, info["removed"] / self.scale, done, truncated, info
