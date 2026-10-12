"""Tests for sar.rl.train and sar.rl.fly: the split, a tiny run, and flights in the site's bundle format."""

import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("stable_baselines3")

from sar.rl import fly, train
from sar.rl.env import SearchEnv, window_items
from sar.rl.windows import is_test, load

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


plot_training, produce_maps, compare_runs = (script(n) for n in ("plot_training", "produce_maps", "compare_runs"))
BASELINE_PPO = ["--learning-rate", "0.0003", "--ent-coef", "0", "--clip-range", "0.2"]

FIXTURE = Path(__file__).resolve().parents[2] / "frontend/tests/e2e/fixtures/scenarios/v1/S01"


@pytest.fixture(scope="module")
def trained(windows_dir, tmp_path_factory):
    out = tmp_path_factory.mktemp("runs")
    return train._cli(["--windows", str(windows_dir), "--validation-every", "3",
                       "--arrival-min", "30", "45", "60", "--headings", "8", "--timesteps", "128",
                       "--envs", "1", "--seed", "1", "--eval-every", "64", "--patience", "0",
                       "--improve-threshold", "0", *BASELINE_PPO, "--out", str(out)])


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
                          "--improve-threshold", "1", "--learning-rate", "0.0001", "--ent-coef", "0.005",
                          "--clip-range", "0.1", "--name", "run_entropy0.005", "--out", str(tmp_path)])
        config = json.loads((run / "config.json").read_text())
        assert run.name == "run_entropy0.005" and config["trained_timesteps"] == 128
        assert config["ppo"]["learning_rate"] == 0.0001 and config["ppo"]["ent_coef"] == 0.005
        model, _ = fly.load_policy(run / "model.zip")
        assert model.ent_coef == 0.005 and model.clip_range(1.0) == 0.1 and model.learning_rate == 0.0001

    def test_the_baseline_ppo_settings_are_recorded(self, trained):
        ppo = json.loads((trained / "config.json").read_text())["ppo"]
        assert (ppo["learning_rate"], ppo["ent_coef"], ppo["clip_range"]) == (0.0003, 0.0, 0.2)
        assert (ppo["n_steps"], ppo["batch_size"], ppo["n_epochs"]) == (2048, 64, 10)

    def test_the_curves_mark_the_best_validation(self, trained):
        data = plot_training.draw(trained, patience=1, threshold=1.0)
        assert (trained / "training_curves.png").exists()
        assert data["last_gain_steps"] == 64 and data["would_stop_steps"] == 128
        assert data["best"]["reward"] == max(v["reward"] for v in data["validation"])


class TestTelemachus2:
    """The second agent's settings (12 Oct, L40): plain POS, gamma 1, a CNN over the maps."""

    def test_the_cnn_reads_a_batch_of_observations(self):
        torch = pytest.importorskip("torch")
        import gymnasium as gym
        from sar.rl.cnn import SearchCNN
        from sar.rl.features import SIZE
        cnn = SearchCNN(gym.spaces.Box(-np.inf, np.inf, (SIZE,), np.float32))
        out = cnn(torch.zeros(5, SIZE))
        assert tuple(out.shape) == (5, 256)

    def test_a_cnn_run_on_plain_pos_trains_saves_and_flies(self, windows_dir, tmp_path):
        run = train._cli(["--windows", str(windows_dir), "--validation-every", "3",
                          "--arrival-min", "30", "45", "--headings", "8", "--timesteps", "128",
                          "--envs", "1", "--seed", "1", "--eval-every", "64", "--patience", "0",
                          "--improve-threshold", "0", *BASELINE_PPO, "--reward-decay", "0",
                          "--gamma", "1", "--policy", "cnn", "--batch-size", "32", "--target-kl", "0.03",
                          "--name", "tele_cnn_test", "--out", str(tmp_path)])
        config = json.loads((run / "config.json").read_text())
        assert config["reward_decay"] == 0.0 and config["eval_decay"] == 0.0
        assert config["ppo"]["gamma"] == 1.0 and config["ppo"]["policy"] == "cnn"
        assert config["ppo"]["batch_size"] == 32 and config["ppo"]["target_kl"] == 0.03
        model, headings = fly.load_policy(run / "best_model.zip")
        assert type(model.policy.features_extractor).__name__ == "SearchCNN"
        path = min(windows_dir.glob("windows_*20190129*.npz"))
        flights = fly.score_windows([path], [fly.PPO], model, headings, [45])
        assert 0.0 <= flights.iloc[0]["pos"] <= 1.0

    def test_the_validation_score_is_pos_when_eval_decay_is_zero(self, windows_dir, tmp_path):
        run = train._cli(["--windows", str(windows_dir), "--validation-every", "3",
                          "--arrival-min", "30", "--headings", "8", "--timesteps", "64",
                          "--envs", "1", "--seed", "1", "--eval-every", "64", "--patience", "0",
                          "--improve-threshold", "0", *BASELINE_PPO, "--reward-decay", "0.5",
                          "--eval-decay", "0", "--name", "tele_half_test", "--out", str(tmp_path)])
        held = np.load(run / "eval" / "evaluations.npz")
        model, headings = fly.load_policy(run / "best_model.zip")
        val = train.split(windows_dir, 3)["validation"]
        env = SearchEnv([str(p) for p in val], headings=headings, arrivals_min=[30], cycle=True)
        pos = []
        for _ in range(len(window_items([str(p) for p in val], [30]))):
            obs, _ = env.reset()
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, done, _, info = env.step(action)
            pos.append(info["pos"])
        # SB3's Monitor rounds an episode's return to 6 decimals
        assert held["results"][-1].mean() == pytest.approx(np.mean(pos), abs=1e-6)


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


class TestMaps:
    def test_a_map_lands_in_its_model_place_season_date_and_arrival_folder(self, trained, windows_dir, tmp_path):
        model, headings = fly.load_policy(trained / "model.zip")
        lat, lon = produce_maps.read_locations()["florida-current"]
        window = produce_maps.find_window(windows_dir, lat, lon, "2019-01-29")
        path = produce_maps.draw_map(trained, model, headings, window, 30, "florida-current", "2019-01-29",
                                     ["expanding-square", "greedy"], True, tmp_path)
        name = trained.name
        assert path == (tmp_path / name / "florida-current" / "winter" / "2019-01-29" / "30min"
                        / f"{name}_florida-current_2019-01-29_30min.png")
        assert path.stat().st_size > 50_000

    def test_the_model_rows_list_every_setting_once(self, trained):
        model, _ = fly.load_policy(trained / "model.zip")
        rows = dict(produce_maps.model_rows(trained, model))
        assert rows["headings"] == "8" and rows["learning rate"] == "0.0003" and rows["clip range"] == "0.2"
        assert rows["network"] == "64 × 64" and rows["arrival times"] == "30 45 60"

    def test_an_arrival_time_the_windows_lack_is_refused(self, trained, windows_dir, tmp_path):
        model, headings = fly.load_policy(trained / "model.zip")
        window = min(windows_dir.glob("windows_*20190129*.npz"))
        with pytest.raises(ValueError, match="arrival times"):
            produce_maps.draw_map(trained, model, headings, window, 120, "florida-current", "2019-01-29",
                                  ["expanding-square"], False, tmp_path)

    def test_the_six_days_are_test_weeks(self, windows_dir):
        name = min(windows_dir.glob("windows_*20190101*.npz")).name
        assert all(is_test(name.replace("20190101", d.replace("-", ""))) for d in produce_maps.SEASON)


class TestCompare:
    def test_each_run_is_set_against_the_baseline(self, trained, tmp_path):
        for name, ent in (("run_baseline", 0.0), ("run_entropy0.005", 0.005)):
            run = tmp_path / name
            shutil.copytree(trained, run)
            config = json.loads((run / "config.json").read_text())
            config["arguments"]["ent_coef"] = ent
            (run / "config.json").write_text(json.dumps(config))
            (run / "test").mkdir()
            (run / "test" / "summary.csv").write_text(
                "arrival_min,searcher,pos,time_weighted,ppo_minus_square,ppo_wins\n"
                "30,expanding-square,0.40,0.3,0.05,0.6\n30,ppo,0.45,0.35,0.05,0.6\n")
        table = compare_runs.compare(tmp_path)
        assert list(table["run"]) == ["run_baseline", "run_entropy0.005"]
        assert list(table["changed"]) == ["—", "ent_coef 0.005"]
        assert table["lead_30"].tolist() == [0.05, 0.05] and (tmp_path / "comparison.csv").exists()
