"""Tests for sar.search.benchmark: a recorded flight replays to the same score, bit for bit."""

import numpy as np
import pandas as pd
import pytest

from sar.pipeline.forcing import ConstantForcing
from sar.search.benchmark import SEARCHERS, recorded, searcher
from sar.search.episode import SearchEpisode, Waypoints, run
from sar.search.scenario import scenario_search

FORCING = ConstantForcing(current=(1.0, 0.5), wind=(5.0, 2.0))


def row():
    base = {"scenario": "S01", "start": "2019-06-01T06:00", "lat": 26.5, "lon": -79.0,
            "seed": 5}
    for h in range(1, 7):
        base[f"lat_{h}h"], base[f"lon_{h}h"] = 26.5 + 0.0045 * h, -79.0 + 0.01 * h
    return pd.Series(base)


@pytest.fixture(scope="module")
def setup():
    return scenario_search(row(), FORCING, 3600, 400)


@pytest.mark.parametrize("name", SEARCHERS)
def test_a_recorded_flight_replays_to_the_same_score(setup, name):
    policy, _, _ = searcher(name, setup, row(), 1.0)
    flying, steps = recorded(policy)
    first = run(flying, SearchEpisode(setup.window, setup.marker, target=setup.target))
    assert len(steps) == 45 and all(isinstance(s, Waypoints) for s in steps)

    again = iter(steps)
    replay = run(lambda ep: next(again),
                 SearchEpisode(setup.window, setup.marker, target=setup.target))
    assert replay["removed_per_step"] == first["removed_per_step"]
    assert replay["target"] == first["target"]
    assert first["pos"] > 0.0 or name == "random"


def test_an_unknown_searcher_is_refused(setup):
    with pytest.raises(ValueError, match="unknown searcher"):
        searcher("ppo", setup, row(), 1.0)


def test_headings_become_one_sub_leg_at_full_speed(setup):
    flying, steps = recorded(lambda ep: 90.0)
    ep = SearchEpisode(setup.window, setup.marker)
    ep.fly(flying(ep))
    assert list(steps[0].t_s) == [60.0]
    assert steps[0].east_m[0] == pytest.approx(ep.speed_ms * 60.0)
    assert abs(steps[0].north_m[0]) < 1e-9
    assert np.isclose(ep.offset[0], steps[0].east_m[0])
