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

from sar.model.position import BY_CURRENT, CALIBRATED_SIGMA, CALIBRATED_SIGMA_U, SIGMA_U_RULE
from sar.search.episode import Turn, Waypoints, pattern_policy
from sar.search.greedy import greedy_policy, random_heading_policy
from sar.search.scenario import DOCTRINAL, REORDERED, doctrinal_searcher

SEARCHERS = DOCTRINAL + ("greedy", "random")
# Flown by the scorer on request but not yet by the site, its bundles or its golden fixture:
# the Parallel Track's legs, middle first with turns a helicopter can make (13 Oct).
EXTRA_SEARCHERS = REORDERED
# The drift engine's random term, by name. rvc is the calibrated one since D033: the random
# velocity sized by the current at the start. rv is D030's pooled size, kept as the
# comparison; rw the old random walk (D028).
# rvs (18 Oct, vault L41) is rvc recalibrated on the 1,448 TRAINING truth windows
# (scripts/cloud_shape.py fit, derived/shape/fit_train.json; 80 groups, fixes at 1, 2, 3 h):
# the whole drift scaled by 0.695 (95 % CI 0.61-0.79), then the random velocity scaled 0.891 along
# each particle's drift and 0.851 across it, so 90 % of the training buoys sit inside the central
# 90 % on each axis. Once the cloud is where the buoys go it can be smaller, and nearly round.
# A forecast recalibration, not a physical claim: a noisy, unbiased drift measured along its own
# direction shows the same lag and stretch.
SHAPE = {"current_scale": 0.695, "wind_scale": 0.695, "along": 0.891, "across": 0.851}
# rvsc (20 Oct, vault D035): rvs's drift scale, and the random velocity's size along and across
# each particle's drift set per speed band by the model's current at the start, so 90 % of the
# training truth buoys in each band sit inside the central 90 % on each axis (cloud_shape.py
# fit-rule, derived/shape/fit_rule_train.json; bands of >= 60 rows, made non-decreasing in speed,
# straight between, flat past the ends). Round below ~0.35 m/s, ~1.3-1.5x longer along the flow
# in faster water.
SHAPE_BY_CURRENT = {"table_s": (0.0724, 0.1243, 0.1736, 0.2281, 0.2748, 0.3401, 0.4521, 0.6179, 0.9866),
                    "table_along": (0.1661, 0.1661, 0.1711, 0.1711, 0.1831, 0.1831, 0.2575, 0.2952, 0.5354),
                    "table_across": (0.1522, 0.1762, 0.1762, 0.1762, 0.1762, 0.1771, 0.1915, 0.2203, 0.3636)}
NOISE = {"rvc": {"sigma_u": BY_CURRENT, "sigma": 0.0},
         "rv": {"sigma_u": CALIBRATED_SIGMA_U, "sigma": 0.0},
         "rw": {"sigma_u": 0.0, "sigma": CALIBRATED_SIGMA},
         "rvs": {"sigma_u": BY_CURRENT, "sigma": 0.0, "current_scale": SHAPE["current_scale"],
                 "wind_scale": SHAPE["wind_scale"], "shape": (SHAPE["along"], SHAPE["across"])},
         "rvsc": {"sigma_u": BY_CURRENT, "sigma": 0.0, "current_scale": SHAPE["current_scale"],
                  "wind_scale": SHAPE["wind_scale"], "shape": SHAPE_BY_CURRENT}}
# What a results folder's manifest records for each, so a rule change can't hide in a name.
NOISE_RECORD = {"rvc": {**NOISE["rvc"], "rule": SIGMA_U_RULE}, "rv": NOISE["rv"],
                "rw": NOISE["rw"], "rvs": {**NOISE["rvs"], "rule": SIGMA_U_RULE},
                "rvsc": {**NOISE["rvsc"], "rule": SIGMA_U_RULE}}
BASELINE = "expanding-square"          # D004: what every other searcher is compared with
# Greedy's settings, chosen on the 55 dev rows (D031), frozen before any test set is opened.
GREEDY = {"headings": 36, "decide_s": 60.0}


def searcher(name: str, setup, row, arrival_h: float, greedy: dict | None = None):
    """A searcher for one scenario: (policy, first bearing or None, how it was laid out)."""
    if name in DOCTRINAL + REORDERED:
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
