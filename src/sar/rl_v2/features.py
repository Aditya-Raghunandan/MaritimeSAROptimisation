"""features.py: v1's observation reshaped into 2D maps, and a reward paid only at the end."""

from __future__ import annotations

import numpy as np

from sar.rl.features import MAP_CELLS, NEAR_CELLS, VECTOR, Context
from sar.rl.features import observe as observe_flat

FAR_SHAPE = (1, MAP_CELLS, MAP_CELLS)      # one channel, as a CNN reads an image
NEAR_SHAPE = (1, NEAR_CELLS, NEAR_CELLS)


def split(flat: np.ndarray) -> dict[str, np.ndarray]:
    """v1's 1,291 numbers as {"far": 1x32x32, "near": 1x16x16, "vec": 11}; the same values, regrouped."""
    nf, nn = MAP_CELLS ** 2, NEAR_CELLS ** 2
    return {"far": flat[:nf].reshape(FAR_SHAPE), "near": flat[nf:nf + nn].reshape(NEAR_SHAPE),
            "vec": flat[nf + nn:]}


def observe(episode, context: Context) -> dict[str, np.ndarray]:
    """The far map, the near map and VECTOR, from the referee's episode exactly as v1 reads them."""
    return split(observe_flat(episode, context))


def reward(pos: float, done: bool) -> float:
    """The probability found over the whole search, paid once at minute 45; nothing before."""
    return float(pos) if done else 0.0


__all__ = ["FAR_SHAPE", "NEAR_SHAPE", "VECTOR", "Context", "observe", "reward", "split"]
