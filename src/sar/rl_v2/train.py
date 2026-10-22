"""train.py: PPO v2 on the same split as v1: 2D maps through a CNN, paid the final POS only.

Run from the repository root, as scripts/v2/train_ppo.sbatch does:
    PYTHONPATH=src python -m sar.rl_v2.train --windows <windows dir> --validation-every 8 \
        --arrival-min 30 60 120 180 --headings 36 --timesteps 3000000 --envs 4 --seed 1 \
        --eval-every 100000 --patience 10 --improve-threshold 0.001 --learning-rate 0.0003 \
        --ent-coef 0 --clip-range 0.2 --gamma 1.0 --out <runs_v2 dir> --name run_baseline_v2

The split, the early stop and the run record are v1's (sar.rl.train), imported, not copied.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sar.rl.features import MAP_CELL_M, MAP_CELLS, NEAR_CELL_M, NEAR_CELLS, VECTOR
from sar.rl.train import git_commit, split, stop_on_plateau


def _cli(argv=None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--windows", required=True, help="directory of windows_*.npz files; only read")
    parser.add_argument("--validation-every", type=int, required=True,
                        help="hold out every k-th week outside the test weeks; 0 for none")
    parser.add_argument("--arrival-min", type=int, nargs="+", required=True,
                        help="arrival times to train on, minutes from the call to on scene")
    parser.add_argument("--headings", type=int, required=True, help="compass headings to choose from")
    parser.add_argument("--timesteps", type=int, required=True, help="environment steps to train")
    parser.add_argument("--envs", type=int, required=True, help="environments run in parallel")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--eval-every", type=int, required=True,
                        help="environment steps between scores on the validation weeks")
    parser.add_argument("--patience", type=int, required=True,
                        help="validations in a row without beating the best by --improve-threshold "
                             "before stopping; 0 never stops early")
    parser.add_argument("--improve-threshold", type=float, required=True,
                        help="how much a validation must beat the best by to count")
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--ent-coef", type=float, required=True)
    parser.add_argument("--clip-range", type=float, required=True)
    parser.add_argument("--gamma", type=float, required=True,
                        help="discount; 1.0 weighs every minute's heading equally against the final POS")
    parser.add_argument("--out", required=True, help="directory the run's folder goes in")
    parser.add_argument("--name", help="the run folder's name; ppo_v2_h<headings>_seed<seed>_T<timesteps> "
                                       "if not given")
    parser.add_argument("--force", action="store_true", help="replace an existing run folder")
    args = parser.parse_args(argv)

    import stable_baselines3
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import EvalCallback
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    from sar.rl.env import window_items
    from sar.rl_v2.cnn import MapsCNN
    from sar.rl_v2.env import SearchEnvV2
    from sar.rl_v2.features import FAR_SHAPE, NEAR_SHAPE

    groups = split(args.windows, args.validation_every)
    if not groups["train"]:
        parser.error(f"no training windows in {args.windows}")
    if args.patience and not groups["validation"]:
        parser.error("--patience needs validation windows; pass --patience 0 with --validation-every 0")
    run = Path(args.out) / (args.name or f"ppo_v2_h{args.headings}_seed{args.seed}_T{args.timesteps}")
    if run.exists() and not args.force:
        parser.error(f"{run} exists; pass --force to replace it")
    run.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)

    kwargs = {"headings": args.headings, "arrivals_min": args.arrival_min}
    vec = SubprocVecEnv if args.envs > 1 else DummyVecEnv
    env = make_vec_env(SearchEnvV2, n_envs=args.envs, seed=args.seed, vec_env_cls=vec,
                       monitor_dir=str(run / "monitor"),
                       env_kwargs={"sources": [str(p) for p in groups["train"]], **kwargs})
    model = PPO("MultiInputPolicy", env, learning_rate=args.learning_rate, ent_coef=args.ent_coef,
                clip_range=args.clip_range, gamma=args.gamma, seed=args.seed, device="cpu", verbose=1,
                policy_kwargs={"features_extractor_class": MapsCNN})

    callback = None
    if groups["validation"]:
        sources = [str(p) for p in groups["validation"]]
        evaluator = make_vec_env(SearchEnvV2, n_envs=1, seed=args.seed,
                                 env_kwargs={"sources": sources, "cycle": True, **kwargs})
        callback = EvalCallback(evaluator, best_model_save_path=str(run), log_path=str(run / "eval"),
                                eval_freq=max(args.eval_every // args.envs, 1),
                                n_eval_episodes=len(window_items(sources, args.arrival_min)),
                                deterministic=True,
                                callback_after_eval=(stop_on_plateau(args.patience, args.improve_threshold)
                                                     if args.patience else None))
    else:
        print("no validation windows: best_model.zip will not be written", file=sys.stderr)

    config = {
        "version": "v2", "arguments": vars(args), "git_commit": git_commit(),
        "stable_baselines3": stable_baselines3.__version__,
        "observation": {"far": [list(FAR_SHAPE), MAP_CELL_M], "near": [list(NEAR_SHAPE), NEAR_CELL_M],
                        "vector": list(VECTOR), "map_cells": [MAP_CELLS, NEAR_CELLS]},
        "policy": "MultiInputPolicy, features from sar.rl_v2.cnn.MapsCNN",
        "ppo": {"learning_rate": args.learning_rate, "n_steps": model.n_steps,
                "batch_size": model.batch_size, "n_epochs": model.n_epochs, "gamma": model.gamma,
                "gae_lambda": model.gae_lambda, "clip_range": args.clip_range,
                "ent_coef": args.ent_coef, "vf_coef": model.vf_coef,
                "max_grad_norm": model.max_grad_norm, "net_arch": model.policy.net_arch},
        "reward": "the episode's POS, paid once at the last step; 0 at every other step",
        "train": [p.name for p in groups["train"]],
        "validation": [p.name for p in groups["validation"]],
        "test_files": len(groups["test"]),
    }
    (run / "config.json").write_text(json.dumps(config, indent=1))
    model.learn(total_timesteps=args.timesteps, callback=callback)
    model.save(run / "model.zip")
    config["trained_timesteps"] = int(model.num_timesteps)
    (run / "config.json").write_text(json.dumps(config, indent=1))
    env.close()
    print(run)
    return run


if __name__ == "__main__":
    _cli()
