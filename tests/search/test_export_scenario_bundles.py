"""Tests for scripts/export_scenario_bundles.py on constant forcing: the files decode back."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sar.pipeline.forcing import ConstantForcing
from sar.search.benchmark import GREEDY, NOISE, SEARCHERS
from sar.search.scenario import scenario_search
from sar.search.sweep import relative_m
from sar.utils.geo import offset_position

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
FORCING = ConstantForcing(current=(1.0, 0.5), wind=(5.0, 2.0))


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


bundles = load("export_scenario_bundles")


def row(name="S01"):
    base = {"scenario": name, "start": "2019-06-01T06:00", "lat": 26.5, "lon": -79.0,
            "seed": 5, "set": "train", "stratum": "jet", "group": 3, "speed_4h_ms": 1.1}
    for h in range(1, 7):
        base[f"lat_{h}h"], base[f"lon_{h}h"] = 26.5 + 0.0045 * h, -79.0 + 0.01 * h
    return pd.Series(base)


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    out = tmp_path_factory.mktemp("bundles")
    entry = bundles.export_row(row(), FORCING, ["rv"], [1.0], 300, out, GREEDY)
    index = bundles.write_index(out, [entry], {"particles": 300})
    return out, entry, index


def test_the_files(exported):
    out, entry, index = exported
    assert entry["windows"] == ["rv_1h"]
    assert {p.name for p in (out / "S01").iterdir()} == {"rv_1h.json", "rv_1h.f32",
                                                         "field_1h.f32"}
    assert index["scenarios"][0]["scenario"] == "S01" and index["particles"] == 300
    assert entry["group"] == 3 and entry["straightness_4h"] == pytest.approx(1.0, abs=1e-3)


def test_the_cloud_decodes_to_the_engines_cloud(exported):
    out, _, _ = exported
    meta = json.loads((out / "S01" / "rv_1h.json").read_text())
    cloud = np.fromfile(out / "S01" / "rv_1h.f32", dtype="<f4").reshape(46, 300, 2)
    setup = scenario_search(row(), FORCING, 3600, 300, **NOISE["rv"])
    mlat, mlon = np.array(meta["marker"]["lat"]), np.array(meta["marker"]["lon"])
    lat, lon = offset_position(mlat[:, None], mlon[:, None], cloud[..., 0].astype(float),
                               cloud[..., 1].astype(float))
    east, north = relative_m(lat, lon, setup.window.lat, setup.window.lon)
    assert np.max(np.hypot(east, north)) < 0.01
    assert meta["rebuild_error_m"] < 0.01 and meta["weight"] == pytest.approx(1 / 300)


def test_every_searcher_is_recorded_with_the_papers_numbers(exported):
    out, _, _ = exported
    meta = json.loads((out / "S01" / "rv_1h.json").read_text())
    assert set(meta["flights"]) == set(SEARCHERS)
    for f in meta["flights"].values():
        assert len(f["steps"]) == 45
        assert sum(f["python"]["drain_rate"]) == pytest.approx(f["python"]["pos"])
        assert f["python"]["closest_m"] is not None


def test_the_field_grid(exported):
    out, _, _ = exported
    meta = json.loads((out / "S01" / "rv_1h.json").read_text())
    setup = scenario_search(row(), FORCING, 3600, 300, **NOISE["rv"])
    assert (meta["field_centre"]["lat"], meta["field_centre"]["lon"]) == pytest.approx(setup.datum)
    grid = np.fromfile(out / "S01" / "field_1h.f32", dtype="<f4").reshape(10, 21, 21, 4)
    assert np.allclose(grid[..., 0], 1.0) and np.allclose(grid[..., 1], 0.5)
    assert np.allclose(grid[..., 2], 5.0) and np.allclose(grid[..., 3], 2.0)


def test_index_merges_every_process(exported, tmp_path):
    for k, name in enumerate(("S02", "S01")):
        (tmp_path / f".index.host.{k}.json").write_text(json.dumps(
            {"settings": {"particles": 9}, "rows": [{"scenario": name}]}))
    index = bundles.write_index(tmp_path, None, None)
    assert [e["scenario"] for e in index["scenarios"]] == ["S01", "S02"]
    assert index["particles"] == 9
