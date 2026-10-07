"""Tests for scripts/pick_scenario_buoys.py's benchmark table (--every), on synthetic starts."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


pick = load("pick_scenario_buoys")


def starts(unit, hours, buoy="1", group=0):
    t0 = pd.Timestamp("2020-03-01T00:00")
    return pd.DataFrame({"ID": buoy, "unit": unit, "group": group,
                         "start": [t0 + pd.Timedelta(hours=h) for h in hours],
                         "stratum": "moderate"})


class TestTile:
    def test_a_new_start_every_48_h_until_the_unit_ends(self):
        tiled = pick.tile(starts("a", range(0, 200)), 48)
        hours = (tiled["start"] - tiled["start"].iloc[0]).dt.total_seconds() / 3600
        assert list(hours) == [0, 48, 96, 144, 192]

    def test_a_gap_is_skipped_not_the_end(self):
        valid = list(range(0, 10)) + list(range(60, 70)) + list(range(130, 140))
        tiled = pick.tile(starts("a", valid), 48)
        hours = (tiled["start"] - tiled["start"].iloc[0]).dt.total_seconds() / 3600
        assert list(hours) == [0, 60, 130]

    def test_every_unit_is_walked_and_rows_are_named_in_time_order(self):
        cands = pd.concat([starts("b", range(5, 100), buoy="2", group=1),
                           starts("a", range(0, 100))], ignore_index=True)
        tiled = pick.tile(cands, 48)
        assert list(tiled["unit"]) == ["a", "b", "a", "b", "a"]          # 0, 5, 48, 53, 96 h
        assert list(tiled["scenario"]) == ["D0001", "D0002", "D0003", "D0004", "D0005"]
        assert tiled["start"].is_monotonic_increasing
        assert tiled["seed"].is_unique and set(tiled["set"]) == {"dev"}

    def test_back_to_back(self):
        assert len(pick.tile(starts("a", range(0, 24)), 4)) == 6

    def test_the_interval_must_be_positive(self):
        with pytest.raises(ValueError, match="every_h"):
            pick.tile(starts("a", range(3)), 0)

    @pytest.mark.parametrize("text, h", [("48h", 48.0), ("2d", 48.0), ("4", 4.0)])
    def test_hours(self, text, h):
        assert pick.hours(text) == h

    def test_names_widen_past_9999(self):
        many = starts("a", np.arange(0, 10_001 * 4, 4))
        assert pick.tile(many, 4)["scenario"].iloc[-1] == "D10001"
