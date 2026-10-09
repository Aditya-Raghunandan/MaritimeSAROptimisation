"""features.py: what PPO sees of a search and what it is paid, read from the referee's own episode."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sar.search.sweep import relative_m

MAP_CELLS, MAP_CELL_M = 32, 1000.0      # project choice: the plan's 32 x 32 map (R5a), 32 km about the marker
NEAR_CELLS, NEAR_CELL_M = 16, 250.0     # D007's 250 m episode cell, 4 km about the helicopter
KM10 = 10_000.0                         # project choice: distances in tens of km keep the inputs near 1
VECTOR = ("heli_east", "heli_north", "lkp_east", "lkp_north", "time_left", "remaining",
          "transit_h", "drift_sin", "drift_cos")
SIZE = MAP_CELLS ** 2 + NEAR_CELLS ** 2 + len(VECTOR)


@dataclass(frozen=True)
class Context:
    """What the call and the drop tell a searcher beyond the episode itself."""

    lkp: tuple[float, float]           # (lat, lon) of the last known position
    arrival_s: float                   # seconds from the call to arriving on scene
    drift_bearing_deg: float | None    # the target's drift at the datum, degrees true


def heading_of(action: int, headings: int) -> float:
    """Action a of `headings` evenly spaced compass headings, north first, clockwise."""
    return float(int(action) * 360.0 / headings)


def time_left_weight(k: int, steps: int) -> float:
    """The reward's weight for step k: the share of the window still to come when it starts."""
    return (steps - int(k)) / steps


def reward(removed: float, k: int, steps: int) -> float:
    """Particles found x time remaining, scaled by N x window so a perfect first minute pays 1."""
    return float(removed) * time_left_weight(k, steps)


def time_weighted_return(removed_per_step) -> float:
    """The episode's total reward from the referee's removed_per_step."""
    steps = len(removed_per_step)
    return float(sum(reward(r, k, steps) for k, r in enumerate(removed_per_step)))


def binned(east, north, weight, cells: int, cell_m: float) -> np.ndarray:
    """Probability in a cells x cells grid centred on the origin, the fullest cell scaled to 1."""
    half = cells * cell_m / 2.0
    i = np.floor((east + half) / cell_m).astype(np.int64)
    j = np.floor((north + half) / cell_m).astype(np.int64)
    inside = (i >= 0) & (i < cells) & (j >= 0) & (j < cells)
    grid = np.bincount(i[inside] * cells + j[inside], weights=weight[inside],
                       minlength=cells * cells)
    top = grid.max()
    return grid / top if top > 0 else grid


def observe(episode, context: Context) -> np.ndarray:
    """The far map about the marker, the near map about the helicopter, then VECTOR, as float32."""
    lat, lon = episode.particles()
    weight = np.asarray(episode.weight, dtype=float)
    live = weight > 0.0
    east, north = relative_m(lat[live], lon[live], *episode.marker_position)
    ox, oy = episode.offset
    far = binned(east, north, weight[live], MAP_CELLS, MAP_CELL_M)
    near = binned(east - ox, north - oy, weight[live], NEAR_CELLS, NEAR_CELL_M)
    lkp_east, lkp_north = relative_m(*context.lkp, *episode.position)
    b = context.drift_bearing_deg
    drift = (0.0, 0.0) if b is None else (np.sin(np.radians(b)), np.cos(np.radians(b)))
    vector = [ox / KM10, oy / KM10, float(lkp_east) / KM10, float(lkp_north) / KM10,
              1.0 - episode.t_s / episode.duration_s, episode.remaining,
              context.arrival_s / 3600.0, *drift]
    return np.concatenate((far, near, vector)).astype(np.float32)
