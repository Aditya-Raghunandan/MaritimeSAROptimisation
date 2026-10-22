"""cnn.py: the small CNN that reads the two maps for both the actor and the critic.

Stable-Baselines3's own NatureCNN is built for 84x84 screens: its 8x8 stride-4 first layer
leaves nothing to convolve by the third layer on a 16x16 map, so these maps get their own.
"""

from __future__ import annotations

import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

FAR_FEATURES, NEAR_FEATURES = 128, 64


def _branch(cells: int, out: int) -> nn.Sequential:
    """Two 3x3 conv layers, each halving the grid, then one dense layer: 32x32 -> 8x8, 16x16 -> 4x4."""
    return nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(32 * (cells // 4) ** 2, out), nn.ReLU())


class MapsCNN(BaseFeaturesExtractor):
    """far map -> 128 features, near map -> 64, joined with the 11 numbers: 203 features."""

    def __init__(self, observation_space):
        far, near, vec = (observation_space[k].shape for k in ("far", "near", "vec"))
        super().__init__(observation_space, FAR_FEATURES + NEAR_FEATURES + vec[0])
        self.far = _branch(far[-1], FAR_FEATURES)
        self.near = _branch(near[-1], NEAR_FEATURES)

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat((self.far(obs["far"]), self.near(obs["near"]), obs["vec"]), dim=1)
