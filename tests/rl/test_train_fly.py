"""Tests for sar.rl.train and sar.rl.fly: the split, a tiny run, and flights in the site's bundle format."""

import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("stable_baselines3")

from sar.rl import fly, train
from sar.rl.env import SearchEnv
from sar.rl.windows import is_test, load

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "plot_training.py"
spec = importlib.util.spec_from_file_location("plot_training", SCRIPT)
plot_training = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plot_training)

FIXTURE = Path(__file__).resolve().parents[2] / "frontend/tests/e2e/fixtures/scenarios/v1/S01"


@pytest.fixture(scope="module")
def trained(windows_dir, tmp_path_factory):
    out = tmp_path_factory.mktemp("runs")
    return train._cli(["--windows", str(windows_dir), "--validation-every", "3",
                       "--arrival-min", "30", "45", "60", "--headings", "8", "--timesteps", "128",
                       "--envs", "1", "--seed", "1", "--eval-every", "64", "--patience", "0",
                       "--improve-threshold", "0", "--out", str(out)])


class TestSplit:
    def test_month_end_and_validation_weeks_are_kept_out_of_training(self, windows_dir):
        groups = train.split(windows_dir, 3)
        assert len(groups["test"]) == 1 and "20190129" in groups["test"][0].name
        assert len(groups["train"]) + len(groups["validation"]) == 3
        assert len(groups["validation"]) == 1
        assert not set(groups["train"]) & set(groups["validation"])

    def test_the_last_seven_days_of_a_month_are_test(self, windows_dir):
        name = min(windows_dir.glob("windows_*20190101*.npz")).name
        days = {"20190121": False, "20190125": True, "20190131": True, "20190221": False,
                "20190222": True, "20200222": False, "20200223": True}
        assert {d: is_test(name.replace("20190101", d)) for d in days} == days


class TestTrain:
    def test_a_tiny_run_writes_both_models_and_its_config(self, trained):
        assert (trained / "model.zip").exists() and (trained / "best_model.zip").exists()
        config = json.loads((trained / "config.json").read_text())
        assert config["arguments"]["headings"] == 8 and config["test_files"] == 1
        assert not any("20190129" in name for name in config["train"])
        assert config["trained_timesteps"] >= 128

    def test_patience_stops_a_run_that_stops_improving(self, windows_dir, tmp_path):
        run = train._cli(["--windows", str(windows_dir), "--validation-every", "3",
                          "--arrival-min", "30", "--headings", "8", "--timesteps", "4096",
                          "--envs", "1", "--seed", "1", "--eval-every", "64", "--patience", "1",
                          "--improve-threshold", "1", "--out", str(tmp_path)])
        assert json.loads((run / "config.json").read_text())["trained_timesteps"] == 128

    def test_the_curves_mark_the_best_validation(self, trained):
        data = plot_training.draw(trained, patience=1, threshold=1.0)
        assert (trained / "training_curves.png").exists()
        assert data["last_gain_steps"] == 64 and data["would_stop_steps"] == 128
        assert data["best"]["reward"] == max(v["reward"] for v in data["validation"])


class TestPlateau:
    def test_a_gain_under_the_threshold_does_not_count(self):
        assert train.plateau([0.5, 0.6, 0.6005, 0.6008, 0.6009], 3, 0.001) == (1, 4)

    def test_a_real_gain_resets_the_count(self):
        assert train.plateau([0.5, 0.6, 0.6, 0.65, 0.65, 0.65], 3, 0.001) == (3, None)

    def test_patience_zero_never_stops(self):
        assert train.plateau([0.7] * 50, 0, 0.001) == (0, None)


class TestFly:
    def test_the_flown_ppo_scores_what_the_environment_paid(self, trained, windows_dir):
        model, headings = fly.load_policy(trained / "model.zip")
        path = min(windows_dir.glob("windows_*20190129*.npz"))
        flights = fly.score_windows([path], [fly.PPO, "expanding-square", "greedy", "random"],
                                    model, headings, [45])
        ppo = flights[flights["searcher"] == fly.PPO].iloc[0]
        env = SearchEnv([path], headings=headings, arrivals_min=[45])
        obs, _ = env.reset(options={"item": 0})
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, info = env.step(action)
        assert ppo["pos"] == pytest.approx(info["pos"], abs=1e-12)
        assert set(flights["searcher"]) == {fly.PPO, "expanding-square", "greedy", "random"}
        summary = fly.summarise(flights)
        assert {"ppo_minus_square", "ppo_wins"} <= set(summary.columns)

    def test_a_window_record_has_the_bundles_flight_format(self, windows):
        bundle = json.loads((FIXTURE / "rv_2h.json").read_text())
        setup = windows.setup(0)
        from sar.rl.features import Context
        context = Context(windows.lkp, setup.arrival_s, setup.drift_bearing_deg)
        flights = fly.fly_window(setup, context, {"lat": windows.lkp[0], "lon": windows.lkp[1],
                                                  "seed": windows.seed}, ["greedy"])
        ours, theirs = flights["greedy"], bundle["flights"]["greedy"]
        assert set(ours) == set(theirs)
        assert set(ours["python"]) == set(theirs["python"])
        assert set(ours["steps"][0]) == set(theirs["steps"][0])
        assert len(ours["steps"]) == len(theirs["steps"]) == 45

    def test_a_published_window_rebuilds_to_the_papers_own_scores(self):
        record = json.loads((FIXTURE / "rv_2h.json").read_text())
        setup, context = fly.bundle_setup(record, FIXTURE)
        row = {"lat": record["lkp"]["lat"], "lon": record["lkp"]["lon"], "seed": 0}
        flights = fly.fly_window(setup, context, row, ["expanding-square", "greedy"])
        for name in ("expanding-square", "greedy"):
            assert flights[name]["python"]["pos"] == pytest.approx(
                record["flights"][name]["python"]["pos"], abs=1e-6)

    def test_bundle_mode_adds_a_ppo_flight_the_site_can_replay(self, trained, tmp_path):
        folder = tmp_path / "S01"
        shutil.copytree(FIXTURE, folder)
        fly._cli(["bundle", "--model", str(trained / "model.zip"), "--bundle",
                  str(folder / "rv_2h.json")])
        record = json.loads((folder / "rv_2h.json").read_text())
        ppo = record["flights"][fly.PPO]
        assert len(ppo["steps"]) == 45 and 0.0 <= ppo["python"]["pos"] <= 1.0
        assert np.sum(ppo["python"]["drain_rate"]) == pytest.approx(ppo["python"]["pos"])
        assert set(record["flights"]) >= {"expanding-square", "greedy", fly.PPO}

    def test_windows_mode_writes_flights_and_a_summary(self, trained, windows_dir, tmp_path):
        fly._cli(["windows", "--windows", str(windows_dir), "--arrival-min",
                  "30", "--searchers", "ppo", "expanding-square", "--model",
                  str(trained / "model.zip"), "--out", str(tmp_path)])
        assert (tmp_path / "summary.csv").exists() and (tmp_path / "flights.csv").exists()
        (record,) = [json.loads(p.read_text()) for p in (tmp_path / "flights").glob("*.json")]
        assert set(record["flights"]) == {"ppo", "expanding-square"}
        name = load(min(windows_dir.glob("windows_*20190129*.npz"))).name
        assert record["scenario"] == name and record["arrival_s"] == 1800.0
