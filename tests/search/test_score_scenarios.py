"""Tests for scripts/score_scenarios.py, imported by path. Constant forcing, synthetic rows."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sar.pipeline.forcing import ConstantForcing

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
START = "2019-06-01T06:00"
LAT, LON = 26.5, -79.0
FORCING = ConstantForcing(current=(1.0, 0.5), wind=(5.0, 2.0))


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


score = load("score_scenarios")


def row(name="S01", seed=5):
    base = {"scenario": name, "start": START, "lat": LAT, "lon": LON, "seed": seed,
            "set": "train", "stratum": "jet"}
    for h in range(1, 7):
        base[f"lat_{h}h"] = LAT + 0.0045 * h      # the buoy rides the 0.5 m/s north current
        base[f"lon_{h}h"] = LON + 0.0100 * h
    return pd.Series(base)


@pytest.fixture(scope="module")
def flights():
    return score.score_row(row(), FORCING, ["rv", "rw"], [1.0, 2.0], list(score.SEARCHERS), 300)


class TestRows:
    def test_specs_become_names(self):
        table = pd.DataFrame({"scenario": [f"S{k:02d}" for k in range(1, 6)]})
        assert score.row_names(["2-4", "s05", "1"], table) == ["S02", "S03", "S04", "S05", "S01"]

    def test_an_unknown_row_is_refused(self):
        with pytest.raises(ValueError, match="S09"):
            score.row_names(["9"], pd.DataFrame({"scenario": ["S01"]}))


class TestScore:
    def test_one_flight_per_noise_arrival_and_searcher(self, flights):
        assert len(flights) == 2 * 2 * 2
        assert {(f["noise"], f["arrival_h"], f["searcher"]) for f in flights} == {
            (n, a, s) for n in ("rv", "rw") for a in (1.0, 2.0) for s in score.SEARCHERS}

    def test_the_rate_adds_up_to_pos(self, flights):
        for f in flights:
            assert len(f["removed_per_step"]) == 45
            assert sum(f["removed_per_step"]) == pytest.approx(f["pos"])
            assert f["pos_15m"] <= f["pos_30m"] <= f["pos"] + 1e-12
            assert 0.0 < f["pos"] < 1.0

    def test_the_buoy_is_scored(self, flights):
        for f in flights:
            assert f["gaps"] == 0 and f["closest_m"] is not None and isinstance(f["found"], bool)

    def test_the_noise_models_are_the_ones_named(self, flights):
        rv = next(f for f in flights if f["noise"] == "rv")
        rw = next(f for f in flights if f["noise"] == "rw")
        assert rv["sigma"] == 0.0 and rv["sigma_u"] == pytest.approx(0.226)
        assert rw["sigma_u"] == 0.0 and rw["sigma"] == pytest.approx(26.3)

    def test_it_is_repeatable(self, flights):
        again = score.score_row(row(), FORCING, ["rv"], [1.0], ["expanding-square"], 300)
        first = next(f for f in flights if (f["noise"], f["arrival_h"], f["searcher"])
                     == ("rv", 1.0, "expanding-square"))
        assert again[0]["pos"] == first["pos"]


class TestSummary:
    def test_the_pairs_are_rv_minus_rw_on_the_same_row(self, flights, tmp_path):
        other = score.score_row(row("S02", seed=6), FORCING, ["rv", "rw"], [1.0, 2.0],
                                list(score.SEARCHERS), 300)
        for name, fl in (("S01", flights), ("S02", other)):
            (tmp_path / f"{name}.jsonl").write_text("".join(json.dumps(f) + "\n" for f in fl))
        means, pairs = score.summarise(score.read_flights(tmp_path))
        assert len(means) == 2 * 2 * 2 and set(means["n"]) == {2}
        one = pairs[(pairs["searcher"] == "sector") & (pairs["arrival_h"] == 2.0)].iloc[0]
        expect = np.mean([next(f["pos"] for f in fl if f["noise"] == "rv"
                               and f["searcher"] == "sector" and f["arrival_h"] == 2.0)
                          - next(f["pos"] for f in fl if f["noise"] == "rw"
                                 and f["searcher"] == "sector" and f["arrival_h"] == 2.0)
                          for fl in (flights, other)])
        assert one["mean"] == pytest.approx(expect) and one["n"] == 2
