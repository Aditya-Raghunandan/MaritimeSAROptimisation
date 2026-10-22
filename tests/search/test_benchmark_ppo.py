"""The trained agent as a benchmark searcher: scripts/score_scenarios.py and
scripts/export_scenario_bundles.py with --ppo-model. Skipped without the rl extra.

The model is an untrained Stable-Baselines3 PPO built on the observation and action spaces
sar.rl trains on: what is tested is the plumbing (the referee flies it, the manifest names
it, the site's bundle replays it), not whether it searches well.
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("stable_baselines3")
pytest.importorskip("sar.rl.fly")

import gymnasium as gym  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

from sar.pipeline.forcing import ConstantForcing  # noqa: E402
from sar.rl.features import SIZE  # noqa: E402
from sar.search.benchmark import GREEDY  # noqa: E402

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
FORCING = ConstantForcing(current=(1.0, 0.5), wind=(5.0, 2.0))
HEADINGS = 12


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


score = load("score_scenarios")
bundles = load("export_scenario_bundles")


def row(name="S01"):
    base = {"scenario": name, "start": "2019-06-01T06:00", "lat": 26.5, "lon": -79.0,
            "seed": 5, "set": "train", "stratum": "jet", "group": 3, "speed_4h_ms": 1.1}
    for h in range(1, 7):
        base[f"lat_{h}h"], base[f"lon_{h}h"] = 26.5 + 0.0045 * h, -79.0 + 0.01 * h
    return pd.Series(base)


class Spaces(gym.Env):
    """Only the spaces: enough to build and save a policy, never stepped."""

    observation_space = gym.spaces.Box(-np.inf, np.inf, (SIZE,), np.float32)
    action_space = gym.spaces.Discrete(HEADINGS)


def save_model(folder: Path, size: int = SIZE) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    PPO("MlpPolicy", Spaces(), seed=1, device="cpu").save(folder / "best_model.zip")
    (folder / "config.json").write_text(json.dumps({
        "arguments": {"headings": HEADINGS, "arrival_min": [30, 60, 120, 180]},
        "git_commit": "trained-here", "observation": {"size": size}}))
    return folder / "best_model.zip"


@pytest.fixture(scope="module")
def ppo(tmp_path_factory):
    return score.PPOSearcher(save_model(tmp_path_factory.mktemp("runs") / "run_test"))


@pytest.fixture(scope="module")
def flights(ppo):
    return score.score_row(row(), FORCING, ["rv"], [1.0], ["expanding-square", "ppo"], 300,
                           ppo=ppo)


def test_the_agent_is_flown_by_the_same_referee(flights):
    agent = next(f for f in flights if f["searcher"] == "ppo")
    assert len(agent["drain_rate"]) == 45
    assert sum(agent["drain_rate"]) == pytest.approx(agent["pos"])
    assert 0.0 <= agent["pos"] <= 1.0
    assert agent["turn_rate_deg_s"] == pytest.approx(
        next(f for f in flights if f["searcher"] == "expanding-square")["turn_rate_deg_s"])
    assert agent["layout"].startswith("trained agent (run_test)")


def test_it_is_repeatable(ppo, flights):
    again = score.score_row(row(), FORCING, ["rv"], [1.0], ["ppo"], 300, ppo=ppo)
    assert again[0]["pos"] == next(f for f in flights if f["searcher"] == "ppo")["pos"]


def test_the_manifest_names_the_model_byte_for_byte(ppo, tmp_path):
    csv = tmp_path / "t.csv"
    csv.write_text("scenario\nS01\n")
    rec = score.experiment(csv, ["rv"], [1.0], ["ppo"], 300, ppo=ppo)
    assert rec["ppo"]["run"] == "run_test" and rec["ppo"]["headings"] == HEADINGS
    assert rec["ppo"]["sha256"] == score.sha256(ppo.path)
    assert rec["ppo"]["trained_commit"] == "trained-here"
    assert score.experiment(csv, ["rv"], [1.0], ["random"], 300)["ppo"] is None


def test_a_model_trained_on_another_observation_is_refused(tmp_path):
    with pytest.raises(ValueError, match="observation"):
        score.PPOSearcher(save_model(tmp_path / "old", size=SIZE - 2))


def test_ppo_and_its_model_go_together(tmp_path):
    with pytest.raises(SystemExit):
        score.main(["score", "--csv", "x.csv", "--forcing-dir", ".", "--rows", "1",
                    "--searchers", "ppo", "--out", str(tmp_path)])


def test_the_bundle_carries_the_agent_as_the_site_replays_it(ppo, flights, tmp_path):
    entry = bundles.export_row(row(), FORCING, ["rv"], [1.0], 300, tmp_path, GREEDY, ppo)
    meta = json.loads((tmp_path / "S01" / "rv_1h.json").read_text())
    agent = meta["flights"]["ppo"]
    assert len(agent["steps"]) == 45
    # Format 2: the heading at every waypoint, so the browser's referee checks the turns.
    assert all(s["heading_deg"] is not None for s in agent["steps"])
    assert agent["python"]["pos"] == next(f for f in flights if f["searcher"] == "ppo")["pos"]
    assert entry["windows"] == ["rv_1h"]
