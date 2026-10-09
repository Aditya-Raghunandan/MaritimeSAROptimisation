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


def row(name="S01", seed=5, group=1, water="jet"):
    base = {"scenario": name, "start": START, "lat": LAT, "lon": LON, "seed": seed,
            "set": "train", "stratum": water, "group": group, "ID": "300234000000001",
            "speed_4h_ms": 1.1}
    for h in range(1, 7):
        base[f"lat_{h}h"] = LAT + 0.0045 * h      # the buoy rides the 0.5 m/s north current
        base[f"lon_{h}h"] = LON + 0.0100 * h
    return pd.Series(base)


@pytest.fixture(scope="module")
def flights():
    return score.score_row(row(), FORCING, ["rv", "rw"], [1.0, 2.0], list(score.SEARCHERS), 300)


def flight(flights, noise, arrival, searcher):
    return next(f for f in flights if (f["noise"], f["arrival_h"], f["searcher"])
                == (noise, arrival, searcher))


class TestRows:
    def test_specs_become_names(self):
        table = pd.DataFrame({"scenario": [f"S{k:02d}" for k in range(1, 6)]})
        assert score.row_names(["2-4", "s05", "1"], table) == ["S02", "S03", "S04", "S05", "S01"]

    def test_positions_work_for_any_naming(self):
        table = pd.DataFrame({"scenario": [f"D{k:04d}" for k in range(1, 101)]})
        assert score.row_names(["98-100", "D0007"], table) == ["D0098", "D0099", "D0100",
                                                               "D0007"]

    def test_an_unknown_row_is_refused(self):
        with pytest.raises(ValueError, match="row 9"):
            score.row_names(["9"], pd.DataFrame({"scenario": ["S01"]}))
        with pytest.raises(ValueError, match="S09"):
            score.row_names(["S09"], pd.DataFrame({"scenario": ["S01"]}))


class TestScore:
    def test_one_flight_per_noise_arrival_and_searcher(self, flights):
        assert len(flights) == 2 * 2 * 6
        assert {(f["noise"], f["arrival_h"], f["searcher"]) for f in flights} == {
            (n, a, s) for n in ("rv", "rw") for a in (1.0, 2.0) for s in score.SEARCHERS}

    def test_the_six_searchers(self):
        assert score.SEARCHERS == ("expanding-square", "sector", "parallel", "trackline",
                                   "greedy", "random")

    def test_the_drain_rate_adds_up_to_pos(self, flights):
        for f in flights:
            assert len(f["drain_rate"]) == 45
            assert sum(f["drain_rate"]) == pytest.approx(f["pos"])
            assert f["peak_drain_rate"] == max(f["drain_rate"])
            assert f["pos_15m"] <= f["pos_30m"] + 1e-12 <= f["pos"] + 2e-12
            assert 0.0 <= f["pos"] < 1.0

    def test_the_buoy_is_scored(self, flights):
        for f in flights:
            assert f["gaps"] == 0 and f["closest_m"] is not None and isinstance(f["found"], bool)

    def test_the_scenario_rides_along(self, flights):
        for f in flights:
            assert f["group"] == 1 and f["water"] == "jet" and f["speed_4h_ms"] == 1.1
            assert f["straightness_4h"] == pytest.approx(1.0, abs=1e-3)   # a straight buoy

    def test_patterns_are_laid_out_as_the_site_lays_them(self, flights):
        assert flight(flights, "rv", 2.0, "expanding-square")["layout"] == "along the drift"
        for name in ("parallel", "trackline"):
            assert flight(flights, "rv", 2.0, name)["layout"] == "along the datum line"
        assert flight(flights, "rv", 2.0, "greedy")["first_bearing_deg"] is None
        assert flight(flights, "rv", 2.0, "greedy")["layout"] == "map, 36 headings every 60 s"

    def test_greedy_takes_its_settings(self):
        fl = score.score_row(row(), FORCING, ["rv"], [1.0], ["greedy"], 300,
                             {"headings": 8, "decide_s": 10.0})
        assert fl[0]["layout"] == "map, 8 headings every 10 s"

    def test_the_noise_models_are_the_ones_named(self, flights):
        rv = next(f for f in flights if f["noise"] == "rv")
        rw = next(f for f in flights if f["noise"] == "rw")
        assert rv["sigma"] == 0.0 and rv["sigma_u"] == pytest.approx(0.226)
        assert rw["sigma_u"] == 0.0 and rw["sigma"] == pytest.approx(26.3)

    def test_it_is_repeatable_random_floor_included(self, flights):
        again = score.score_row(row(), FORCING, ["rv"], [1.0], ["expanding-square", "random"],
                                300)
        for f in again:
            assert f["pos"] == flight(flights, "rv", 1.0, f["searcher"])["pos"]

    def test_lines_are_json(self, flights):
        assert json.loads(json.dumps(flights[0]))["searcher"] == "expanding-square"


class TestManifest:
    def settings(self, **change):
        return {"commit": "abc", "table_sha256": "123", "particles": 300, **change}

    def test_the_first_claim_writes_it_and_the_same_settings_pass(self, tmp_path):
        score.claim(tmp_path, self.settings())
        held = score.claim(tmp_path, self.settings())
        assert held["commit"] == "abc"
        assert json.loads((tmp_path / "manifest.json").read_text())["particles"] == 300

    def test_many_claims_at_once_never_read_half_a_manifest(self, tmp_path):
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(8) as pool:
            held = list(pool.map(lambda _: score.claim(tmp_path, self.settings()), range(32)))
        assert {h["commit"] for h in held} == {"abc"}
        assert [p.name for p in tmp_path.iterdir()] == ["manifest.json"]

    def test_different_settings_are_refused(self, tmp_path):
        score.claim(tmp_path, self.settings())
        with pytest.raises(ValueError, match="commit, particles"):
            score.claim(tmp_path, self.settings(commit="def", particles=1000))


def write(folder, name, fl):
    (folder / "flights").mkdir(exist_ok=True)
    (folder / "flights" / f"{name}.jsonl").write_text("".join(json.dumps(f) + "\n" for f in fl))


@pytest.fixture(scope="module")
def table(flights, tmp_path_factory):
    # S01 and S02 share group 1; S03 is group 2 on its own.
    folder = tmp_path_factory.mktemp("bench")
    others = {"S02": row("S02", seed=6, group=1), "S03": row("S03", seed=7, group=2,
                                                             water="quiet")}
    write(folder, "S01", flights)
    for name, r in others.items():
        write(folder, name, score.score_row(r, FORCING, ["rv", "rw"], [1.0, 2.0],
                                            list(score.SEARCHERS), 300))
    f = score.read_flights(folder)
    return f, score.summarise(f, straight_at=0.5, resamples=200)


class TestSummary:
    def test_groups_are_averaged_first(self, table):
        f, t = table
        sel = (f["noise"] == "rv") & (f["searcher"] == "sector") & (f["arrival_h"] == 2.0)
        pos = f[sel].set_index("scenario")["pos"]
        expect = np.mean([np.mean([pos["S01"], pos["S02"]]), pos["S03"]])
        got = t[(t["measure"] == "pos") & (t["slice"] == "all") & (t["noise"] == "rv")
                & (t["searcher"] == "sector") & (t["arrival_h"] == 2.0)].iloc[0]
        assert got["mean"] == pytest.approx(expect)
        assert got["n_rows"] == 3 and got["n_groups"] == 2
        assert got["lo"] <= got["mean"] <= got["hi"]

    def test_the_pairs_are_on_the_same_row(self, table):
        f, t = table
        d = t[(t["measure"] == "pos difference") & (t["slice"] == "all")]
        assert set(d["noise"]) >= {"rv - rw"}
        assert {"greedy - expanding-square", "random - expanding-square"} <= set(d["searcher"])
        one = d[(d["noise"] == "rv - rw") & (d["searcher"] == "sector")
                & (d["arrival_h"] == 1.0)].iloc[0]
        by = f[(f["searcher"] == "sector") & (f["arrival_h"] == 1.0)].pivot(
            index="scenario", columns="noise", values="pos")
        diff = by["rv"] - by["rw"]
        assert one["mean"] == pytest.approx(np.mean([np.mean([diff["S01"], diff["S02"]]),
                                                     diff["S03"]]))

    def test_the_slices(self, table):
        _, t = table
        assert {"all", "water: jet", "water: quiet", "path: straight (>= 0.500)"} <= set(t["slice"])

    def test_old_flights_are_read_with_the_new_name(self, flights, tmp_path):
        old = [{**{k: v for k, v in f.items() if k != "drain_rate"},
                "removed_per_step": f["drain_rate"]} for f in flights[:2]]
        (tmp_path / "S01.jsonl").write_text("".join(json.dumps(f) + "\n" for f in old))
        assert "drain_rate" in score.read_flights(tmp_path)


def test_a_row_across_a_forcing_gap_is_skipped_and_says_why(tmp_path, monkeypatch):
    from sar.pipeline.gridded import ForcingGapError
    table = pd.DataFrame([row("S01"), row("S02", seed=6)])
    csv = tmp_path / "t.csv"
    table.to_csv(csv, index=False)

    def fake(r, *args, **kwargs):
        if r["scenario"] == "S01":
            raise ForcingGapError("current has a 12 h gap")
        return [{"scenario": r["scenario"], "pos": 0.5, "searcher": "x", "noise": "rv",
                 "arrival_h": 1.0}]

    monkeypatch.setattr(score, "score_row", fake)
    score.main(["score", "--csv", str(csv), "--forcing-dir", "x", "--rows", "1-2",
                "--out", str(tmp_path / "out")])
    flights = tmp_path / "out" / "flights"
    assert json.loads((flights / "S01.skipped.json").read_text())["reason"].startswith("current")
    assert (flights / "S02.jsonl").exists() and not (flights / "S01.jsonl").exists()


def test_the_batch_jobs_commit_wins(monkeypatch):
    monkeypatch.setenv("SAR_COMMIT", "abc123")
    assert score.git_commit() == "abc123"
