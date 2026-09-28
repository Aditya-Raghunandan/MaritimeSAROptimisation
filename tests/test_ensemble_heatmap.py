"""Tests for scripts/ensemble_heatmap.py and scripts/converge_ensemble.py, imported by path."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from sar.pipeline.ensemble import Ensemble, synthetic_cloud, write_csv

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


heatmap = load("ensemble_heatmap")
converge = load("converge_ensemble")

START = "2019-06-01T06:00"
LAT, LON, SPREAD_KM, CELL_M = 26.5, -79.0, 2.0, 250.0
RUN = {"lat": LAT, "lon": LON, "start": START, "particles": 20_000, "timestep": 60.0,
       "duration": 3600.0, "seed": 9}


@pytest.fixture
def cloud_csv(tmp_path):
    """A synthetic cloud of known centre and spread, written and read back through the CSV."""
    cloud = synthetic_cloud((LAT, LON), SPREAD_KM, 20_000, np.random.default_rng(9), time=START)
    beached = np.zeros((1, 20_000), dtype=bool)
    beached[0, :500] = True
    cloud = Ensemble(cloud.times, cloud.lat, cloud.lon, cloud.weight, beached)
    return write_csv(cloud, RUN, tmp_path)


class TestSelectTime:
    def test_an_offset_and_an_instant_find_the_same_state(self, cloud_csv):
        e = heatmap.read_csv(cloud_csv)
        assert heatmap.select_time(e, "0h") == heatmap.select_time(e, START) == 0

    def test_a_time_that_was_not_saved_raises(self, cloud_csv):
        with pytest.raises(ValueError, match="not a saved time"):
            heatmap.select_time(heatmap.read_csv(cloud_csv), "24h")


class TestReduceCloud:
    def test_the_map_sums_to_one(self, cloud_csv):
        e = heatmap.read_csv(cloud_csv)
        assert heatmap.reduce_cloud(e.lat[0], e.lon[0], CELL_M)["p"].sum() == pytest.approx(1.0)

    def test_the_centre_and_spread_are_recovered_within_one_cell(self, cloud_csv):
        e = heatmap.read_csv(cloud_csv)
        r = heatmap.reduce_cloud(e.lat[0], e.lon[0], CELL_M)
        ns, ew = r["grid"].cell_size_m
        north_m = (r["centroid_lat"] - LAT) * ns / r["grid"].dlat
        east_m = (r["centroid_lon"] - LON) * ew / r["grid"].dlon
        assert abs(north_m) < CELL_M and abs(east_m) < CELL_M
        assert r["spread_km"] * 1000.0 == pytest.approx(SPREAD_KM * 1000.0, abs=CELL_M)


class TestMain:
    def test_writes_the_png_and_the_spec_and_reports_lost_and_beached(self, cloud_csv, tmp_path,
                                                                     capsys):
        record = heatmap.main(["--csv", str(cloud_csv), "--at", "0h", "--cell-m", str(CELL_M),
                               "--out", str(tmp_path / "fig")])
        png = Path(record["figure"])
        assert png.exists()
        written = json.loads(png.with_suffix(".json").read_text())
        assert set(written["spec"]) == {"lat0", "dlat", "nlat", "lon0", "dlon", "nlon"}
        assert written["lost"] == 0
        assert written["beached_mass"] == pytest.approx(500 / 20_000)
        assert "beached mass 0.0250" in capsys.readouterr().out


class TestConvergeLadder:
    ARGS = ("--lat", "26.5", "--lon", "-79", "--datum-sigma-km", "2", "--start", START,
            "--timestep", "60", "--duration", "10m", "--sigma", "0.1", "--seed", "4",
            "--constant-current", "1.8", "0", "--cell-m", "250")

    def test_a_short_ladder_runs_and_its_error_falls_as_one_over_root_n(self, tmp_path):
        result = converge.main([*self.ARGS, "--rungs", "100", "1000", "10000",
                                "--repeats", "20", "--out", str(tmp_path)])
        assert Path(result["figure"]).exists()
        assert Path(result["figure"]).with_suffix(".json").exists()
        assert result["slopes"]["centroid_error_km"] == pytest.approx(-0.5, abs=0.1)
        assert result["slopes"]["spread_error_km"] == pytest.approx(-0.5, abs=0.15)

    def test_a_single_repeat_is_refused(self, tmp_path):
        with pytest.raises(SystemExit):
            converge.main([*self.ARGS, "--rungs", "100", "--repeats", "1", "--out", str(tmp_path)])
