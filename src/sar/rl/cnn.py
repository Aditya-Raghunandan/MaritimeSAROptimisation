"""cnn.py: a small convolutional reader for the agent's two maps, so it sees them as pictures.

The observation (features.observe) is one flat vector: the 32 x 32 far map about the marker,
the 16 x 16 near map about the helicopter, then VECTOR. The first agent read all 1,291 numbers
through a 64 x 64 MLP, which treats neighbouring cells as unrelated inputs. Here each map goes
through two stride-2 convolutions, which learn local shapes (a dense patch, its edge, a strip
already swept) wherever they are, and is then flattened, not pooled, so where things are on the
map survives. The two readings and the eleven numbers are joined into `features_dim` numbers
for PPO's policy and value heads.
"""

from __future__ import annotations

import gymnasium as gym
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from sar.rl.features import MAP_CELLS, NEAR_CELLS, SIZE, VECTOR

FAR = MAP_CELLS * MAP_CELLS
NEAR = NEAR_CELLS * NEAR_CELLS


def _reader(cells: int, out: int) -> nn.Sequential:
    """1 x cells x cells -> two stride-2 3 x 3 convolutions -> flatten -> `out` numbers."""
    return nn.Sequential(
        nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),
        nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
        nn.Flatten(),
        nn.Linear(32 * (cells // 4) ** 2, out), nn.ReLU(),
    )


class SearchCNN(BaseFeaturesExtractor):
    """The far and near maps through their own small CNNs, joined with the vector."""

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256,
                 far_out: int = 128, near_out: int = 64):
        if observation_space.shape != (SIZE,):
            raise ValueError(f"SearchCNN reads a ({SIZE},) observation, got {observation_space.shape}")
        super().__init__(observation_space, features_dim)
        self.far = _reader(MAP_CELLS, far_out)
        self.near = _reader(NEAR_CELLS, near_out)
        self.join = nn.Sequential(nn.Linear(far_out + near_out + len(VECTOR), features_dim), nn.ReLU())

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        b = obs.shape[0]
        far = obs[:, :FAR].reshape(b, 1, MAP_CELLS, MAP_CELLS)
        near = obs[:, FAR:FAR + NEAR].reshape(b, 1, NEAR_CELLS, NEAR_CELLS)
        vector = obs[:, FAR + NEAR:]
        return self.join(torch.cat((self.far(far), self.near(near), vector), dim=1))
