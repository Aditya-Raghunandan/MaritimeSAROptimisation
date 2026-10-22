"""Tests for scripts/pick_ppo_run.py: the agent is chosen on validation, finished runs only."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
spec = importlib.util.spec_from_file_location("pick_ppo_run", SCRIPTS / "pick_ppo_run.py")
pick = importlib.util.module_from_spec(spec)
sys.modules["pick_ppo_run"] = pick
spec.loader.exec_module(pick)


def make_run(runs: Path, name: str, curve, finished: bool = True) -> Path:
    run = runs / name
    (run / "eval").mkdir(parents=True)
    steps = 100_000 * np.arange(1, len(curve) + 1)
    # EvalCallback's layout: one row per evaluation, one column per validation episode.
    results = np.array(curve, dtype=float)[:, None] + np.array([[-0.01, 0.0, 0.01]])
    np.savez(run / "eval" / "evaluations.npz", timesteps=steps, results=results)
    (run / "best_model.zip").write_bytes(b"best")
    if finished:
        (run / "model.zip").write_bytes(b"last")
    return run


@pytest.fixture
def runs(tmp_path):
    runs = tmp_path / "runs"
    make_run(runs, "run_baseline", [0.30, 0.34, 0.33])
    make_run(runs, "run_entropy0.007", [0.31, 0.35, 0.36, 0.35])
    make_run(runs, "run_clip0.5", [0.50], finished=False)       # still training: left out
    return runs


def test_the_best_validation_among_finished_runs_wins(runs):
    chosen, table, unfinished = pick.pick(runs)
    assert chosen["run"] == "run_entropy0.007"
    assert chosen["best"] == pytest.approx(0.36) and chosen["at_timesteps"] == 300_000
    assert [v["run"] for v in table] == ["run_entropy0.007", "run_baseline"]
    assert unfinished == ["run_clip0.5"]


def test_the_record(runs, tmp_path):
    out = tmp_path / "chosen.json"
    pick.main(["--runs-dir", str(runs), "--out", str(out)])
    record = json.loads(out.read_text())
    assert record["chosen"] == "run_entropy0.007" and record["unfinished"] == ["run_clip0.5"]
    assert record["model"].endswith("best_model.zip")


@pytest.mark.skipif(sys.platform == "win32", reason="a symlink needs an administrator on Windows")
def test_the_link(runs, tmp_path):
    link = tmp_path / "chosen"
    pick.main(["--runs-dir", str(runs), "--link", str(link)])
    assert (link / "best_model.zip").read_bytes() == b"best"
    assert link.resolve() == (runs / "run_entropy0.007").resolve()
    # A second choice moves the link rather than failing on it.
    make_run(runs, "run_heading72", [0.40])
    pick.main(["--runs-dir", str(runs), "--link", str(link)])
    assert link.resolve() == (runs / "run_heading72").resolve()


def test_nothing_finished_is_an_error(tmp_path):
    make_run(tmp_path / "runs", "run_a", [0.3], finished=False)
    with pytest.raises(SystemExit):
        pick.pick(tmp_path / "runs")
