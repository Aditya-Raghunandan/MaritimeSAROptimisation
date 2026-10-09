"""benchmark.py: the benchmark's searchers and noise models, in one place (docs/benchmark.md).

The scorer (scripts/score_scenarios.py), the site's scenario bundles
(scripts/export_scenario_bundles.py) and the browser's golden fixture all build their
searchers here, so a searcher is the same searcher wherever it is flown, and the site's
replay is the paper's flight.

`recorded` keeps every step a searcher flew as the `Waypoints` the referee was given, which
is what the browser replays: a heading becomes the path the referee flies for it (the turn
at the turn rate and the straight leg after, with the heading at every waypoint; D032), bit
for bit.
"""

from __future__ import annotations

from sar.model.position import CALIBRATED_SIGMA, CALIBRATED_SIGMA_U
from sar.search.episode import Turn, Waypoints, pattern_policy
from sar.search.greedy import greedy_policy, random_heading_policy
from sar.search.scenario import DOCTRINAL, doctrinal_searcher

SEARCHERS = DOCTRINAL + ("greedy", "random")
NOISE = {"rv": {"sigma_u": CALIBRATED_SIGMA_U, "sigma": 0.0},
         "rw": {"sigma_u": 0.0, "sigma": CALIBRATED_SIGMA}}
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
    """The policy, and the list it fills with every step it flies, as `Waypoints`.

    A heading or a `Turn` is recorded as the path the referee flies for it, with the
    heading at every waypoint, so a replay of the steps is the same flight (D032).
    """
    steps: list[Waypoints] = []

    def flying(episode):
        action = policy(episode)
        if isinstance(action, Turn):
            action = episode.waypoints_for(turn_deg=action.deg)
        elif not isinstance(action, Waypoints):
            action = episode.waypoints_for(float(action))
        steps.append(action)
        return action

    return flying, steps
