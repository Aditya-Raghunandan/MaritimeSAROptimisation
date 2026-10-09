"""benchmark.py: the benchmark's searchers and noise models, in one place (docs/benchmark.md).

The scorer (scripts/score_scenarios.py), the site's scenario bundles
(scripts/export_scenario_bundles.py) and the browser's golden fixture all build their
searchers here, so a searcher is the same searcher wherever it is flown, and the site's
replay is the paper's flight.

`recorded` keeps every step a searcher flew as the `Waypoints` the referee was given, which
is what the browser replays: a heading becomes the one straight sub-leg the referee would
have flown for it, bit for bit.
"""

from __future__ import annotations

import numpy as np

from sar.model.position import BY_CURRENT, CALIBRATED_SIGMA, CALIBRATED_SIGMA_U, SIGMA_U_RULE
from sar.search.episode import Waypoints, pattern_policy
from sar.search.greedy import greedy_policy, random_heading_policy
from sar.search.platform import STEP_S
from sar.search.scenario import DOCTRINAL, doctrinal_searcher
from sar.utils.geo import east_north

SEARCHERS = DOCTRINAL + ("greedy", "random")
# The drift engine's random term, by name. rvc is the calibrated one since D033: the random
# velocity sized by the current at the start. rv is D030's pooled size, kept as the
# comparison; rw the old random walk (D028).
NOISE = {"rvc": {"sigma_u": BY_CURRENT, "sigma": 0.0},
         "rv": {"sigma_u": CALIBRATED_SIGMA_U, "sigma": 0.0},
         "rw": {"sigma_u": 0.0, "sigma": CALIBRATED_SIGMA}}
# What a results folder's manifest records for each, so a rule change can't hide in a name.
NOISE_RECORD = {"rvc": {**NOISE["rvc"], "rule": SIGMA_U_RULE}, "rv": NOISE["rv"],
                "rw": NOISE["rw"]}
BASELINE = "expanding-square"          # D004: what every other searcher is compared with
# Greedy's settings, chosen on the 55 dev rows (D031), frozen before any test set is opened.
GREEDY = {"headings": 36, "decide_s": 60.0}


def searcher(name: str, setup, row, arrival_h: float, greedy: dict | None = None):
    """A searcher for one scenario: (policy, first bearing or None, how it was laid out)."""
    if name in DOCTRINAL:
        lkp = (float(row["lat"]), float(row["lon"]))
        pattern, bearing, why = doctrinal_searcher(name, setup, lkp)
        return pattern_policy(pattern), bearing, why
    if name == "greedy":
        g = GREEDY if greedy is None else greedy
        return (greedy_policy(**g), None,
                f"map, {g['headings']} headings every {g['decide_s']:g} s")
    if name == "random":
        return (random_heading_policy([int(row["seed"]), int(round(arrival_h * 60))]), None,
                "random heading each minute")
    raise ValueError(f"unknown searcher {name!r}")


def recorded(policy):
    """The policy, and the list it fills with every step it flies, as `Waypoints`."""
    steps: list[Waypoints] = []

    def flying(episode):
        action = policy(episode)
        if not isinstance(action, Waypoints):
            de, dn = east_north(float(action), episode.speed_ms * STEP_S)
            x, y = episode.offset
            action = Waypoints(np.array([STEP_S]), np.array([x + float(de)]),
                               np.array([y + float(dn)]))
        steps.append(action)
        return action

    return flying, steps
