"""train.py: PPO (D005) on the training windows, keeping the best on held-out validation weeks."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from sar.rl.features import MAP_CELL_M, MAP_CELLS, NEAR_CELL_M, NEAR_CELLS, SIZE, VECTOR
from sar.rl.windows import is_test, parse_windows_name


def split(windows_dir, validation_every: int) -> dict:
    """Windows files by role: test is the last week of each month, validation every k-th other week."""
    groups = {"train": [], "validation": [], "test": []}
    for path in sorted(Path(windows_dir).glob("windows_*.npz")):
        start = np.datetime64(parse_windows_name(path)["start"], "D")
        # first Monday after the Unix epoch: 1970-01-05
        week = int((start - np.datetime64("1970-01-05", "D")).astype(int)) // 7
        if is_test(path):
            groups["test"].append(path)
        elif validation_every and week % validation_every == validation_every - 1:
            groups["validation"].append(path)
        else:
            groups["train"].append(path)
    return groups


def git_commit() -> str:
    """The checkout's commit, so a run can be traced to its code; 'unknown' outside a checkout."""
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _cli(argv=None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--windows", required=True, help="directory of windows_*.npz files")
    parser.add_argument("--validation-every", type=int, required=True,
                        help="hold out every k-th week outside the test weeks to pick the best "
                             "model; 0 for none")
    parser.add_argument("--arrival-min", type=int, nargs="+", required=True,
                        help="transit times to train on, such as 30 45 60")
    parser.add_argument("--headings", type=int, required=True,
                        help="compass headings to choose from each minute; greedy flies 36")
    parser.add_argument("--timesteps", type=int, required=True, help="environment steps to train")
    parser.add_argument("--envs", type=int, required=True, help="environments run in parallel")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--eval-every", type=int, required=True,
                        help="environment steps between scores on the validation weeks")
    parser.add_argument("--out", required=True, help="directory the run's folder goes in")
    parser.add_argument("--force", action="store_true", help="replace an existing run folder")
    args = parser.parse_args(argv)

    import stable_baselines3
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import EvalCallback
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    from sar.rl.env import SearchEnv, window_items

    groups = split(args.windows, args.validation_every)
    if not groups["train"]:
        parser.error(f"no training windows in {args.windows}")
    run = Path(args.out) / f"ppo_h{args.headings}_seed{args.seed}_T{args.timesteps}"
    if run.exists() and not args.force:
        parser.error(f"{run} exists; pass --force to replace it")
    run.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)

    kwargs = {"headings": args.headings, "arrivals_min": args.arrival_min}
    vec = SubprocVecEnv if args.envs > 1 else DummyVecEnv
    env = make_vec_env(SearchEnv, n_envs=args.envs, seed=args.seed, vec_env_cls=vec,
                       monitor_dir=str(run / "monitor"),
                       env_kwargs={"sources": [str(p) for p in groups["train"]], **kwargs})
    model = PPO("MlpPolicy", env, seed=args.seed, device="cpu", verbose=1)

    callback = None
    if groups["validation"]:
        sources = [str(p) for p in groups["validation"]]
        evaluator = make_vec_env(SearchEnv, n_envs=1, seed=args.seed,
                                 env_kwargs={"sources": sources, "cycle": True, **kwargs})
        callback = EvalCallback(evaluator, best_model_save_path=str(run),
                                log_path=str(run / "eval"),
                                eval_freq=max(args.eval_every // args.envs, 1),
                                n_eval_episodes=len(window_items(sources, args.arrival_min)),
                                deterministic=True)
    else:
        print("no validation windows: best_model.zip will not be written", file=sys.stderr)

    (run / "config.json").write_text(json.dumps({
        "arguments": vars(args), "git_commit": git_commit(),
        "stable_baselines3": stable_baselines3.__version__,
        "observation": {"size": SIZE, "map": [MAP_CELLS, MAP_CELL_M],
                        "near": [NEAR_CELLS, NEAR_CELL_M], "vector": list(VECTOR)},
        "reward": "removed x (steps - k) / steps: particles found x time remaining, / (N x window)",
        "train": [p.name for p in groups["train"]],
        "validation": [p.name for p in groups["validation"]],
        "test_files": len(groups["test"]),
    }, indent=1))
    model.learn(total_timesteps=args.timesteps, callback=callback)
    model.save(run / "model.zip")
    env.close()
    print(run)
    return run


if __name__ == "__main__":
    _cli()
