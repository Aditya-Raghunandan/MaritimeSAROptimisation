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


def plateau(scores, patience: int, threshold: float) -> tuple[int | None, int | None]:
    """(last validation that beat the best by more than threshold, the one `patience` later that stops training)."""
    best, improved, waited = -np.inf, None, 0
    for i, score in enumerate(scores):
        if score > best + threshold:
            best, improved, waited = score, i, 0
        else:
            waited += 1
            if patience and waited >= patience:
                return improved, i
    return improved, None


def stop_on_plateau(patience: int, threshold: float):
    """An EvalCallback callback_after_eval that ends training once `plateau` says stop."""
    from stable_baselines3.common.callbacks import BaseCallback

    class StopOnPlateau(BaseCallback):
        def __init__(self):
            super().__init__()
            self.scores = []

        def _on_step(self) -> bool:
            self.scores.append(float(self.parent.last_mean_reward))
            if plateau(self.scores, patience, threshold)[1] is None:
                return True
            print(f"validation has not beaten its best by {threshold} in {patience} checks: "
                  f"stopping at {self.num_timesteps:,} steps")
            return False

    return StopOnPlateau()


def _cli(argv=None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--windows", required=True, help="directory of windows_*.npz files")
    parser.add_argument("--validation-every", type=int, required=True,
                        help="hold out every k-th week outside the test weeks to pick the best "
                             "model; 0 for none")
    parser.add_argument("--arrival-min", type=int, nargs="+", required=True,
                        help="arrival times to train on, minutes from the call to on scene, "
                             "such as 30 60 120 180")
    parser.add_argument("--headings", type=int, required=True,
                        help="compass headings to choose from each minute; greedy flies 36")
    parser.add_argument("--timesteps", type=int, required=True, help="environment steps to train")
    parser.add_argument("--envs", type=int, required=True, help="environments run in parallel")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--eval-every", type=int, required=True,
                        help="environment steps between scores on the validation weeks")
    parser.add_argument("--patience", type=int, required=True,
                        help="stop after this many validations in a row without beating the best "
                             "by --improve-threshold; 0 never stops early")
    parser.add_argument("--improve-threshold", type=float, required=True,
                        help="how much a validation must beat the best by to count, such as 0.001")
    parser.add_argument("--learning-rate", type=float, required=True,
                        help="PPO's step size; Stable-Baselines3's default is 0.0003")
    parser.add_argument("--ent-coef", type=float, required=True,
                        help="entropy bonus, which pays the policy to keep exploring; default 0")
    parser.add_argument("--clip-range", type=float, required=True,
                        help="largest change in a heading's probability per update; default 0.2")
    # Telemachus 2 (12 Oct, L40). Optional, and every default is the first agent's setting.
    parser.add_argument("--reward-decay", type=float, default=1.0,
                        help="how much time left weighs a find: 1 pays found x time left (the "
                             "first agent), 0 pays plain POS, what the benchmark judges")
    parser.add_argument("--eval-decay", type=float,
                        help="the same for the validation score that picks best_model.zip and stops "
                             "training; defaults to --reward-decay; 0 picks on validation POS")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="how much the next minute counts against this one; 1 counts the whole "
                             "45-minute window alike (safe: the window's end is terminal)")
    parser.add_argument("--policy", choices=("mlp", "cnn"), default="mlp",
                        help="mlp: Stable-Baselines3's 64 x 64 MLP; cnn: sar.rl.cnn.SearchCNN reads "
                             "the two maps as pictures")
    parser.add_argument("--batch-size", type=int, default=64, help="PPO minibatch; SB3's default is 64")
    parser.add_argument("--n-epochs", type=int, default=10, help="passes over each batch; SB3's default is 10")
    parser.add_argument("--target-kl", type=float,
                        help="stop a batch's passes early once the policy has moved this far (KL); "
                             "none by default")
    parser.add_argument("--halves", type=int, default=0,
                        help="0: one heading a minute (the first agent); N: two headings a minute, "
                             "30 s each, each from N, so N x N actions")
    parser.add_argument("--net-arch", type=int, nargs="+",
                        help="hidden layers of the policy and value heads, such as 256 256; SB3's "
                             "64 64 if not given")
    # Telemachus 3 (13 Oct): paid for finding the real buoy, on real buoys' starts. Optional.
    parser.add_argument("--validation-windows",
                        help="a directory of validation windows (such as other buoys' starts); then "
                             "every file in --windows trains and none is set aside by date")
    parser.add_argument("--find-reward", type=float, default=0.0,
                        help="paid the minute the real buoy is first within W/2; the search ends "
                             "there (windows must carry a buoy, sar.rl.windows.build_truth)")
    parser.add_argument("--pos-scale", type=float, default=1.0,
                        help="multiplies the probability reward; 0 with --find-reward 1 pays finds only")
    parser.add_argument("--eval-find", type=float,
                        help="--find-reward for the validation score; defaults to --find-reward")
    parser.add_argument("--eval-pos-scale", type=float,
                        help="--pos-scale for the validation score; defaults to --pos-scale. With "
                             "--eval-find 1 and 0 here, validation is the share of buoys found")
    parser.add_argument("--init-from",
                        help="a trained model.zip to start from (same observation and actions); "
                             "its network and PPO settings, with this run's rate, entropy, clip, "
                             "KL brake, gamma, batch and epochs")
    # Telemachus S (15 Oct, L43): it also picks its speed each minute. Optional.
    parser.add_argument("--speeds", type=float, nargs="+",
                        help="speeds in knots to choose from each minute, such as 45 60 90; the "
                             "turn rate follows (30 degree bank) and the observation gains the "
                             "speed. Not given: always 90 kt, as every agent before")
    parser.add_argument("--torch-threads", type=int, default=1,
                        help="threads for the network's update; a CNN wants the node's cores")
    parser.add_argument("--out", required=True, help="directory the run's folder goes in")
    parser.add_argument("--name", help="the run folder's name; ppo_h<headings>_seed<seed>_T<timesteps> "
                                       "if not given")
    parser.add_argument("--force", action="store_true", help="replace an existing run folder")
    args = parser.parse_args(argv)

    import stable_baselines3
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import EvalCallback
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    from sar.rl.env import SearchEnv, window_items

    if args.validation_windows:
        groups = {"train": sorted(Path(args.windows).glob("windows_*.npz")),
                  "validation": sorted(Path(args.validation_windows).glob("windows_*.npz")),
                  "test": []}
    else:
        groups = split(args.windows, args.validation_every)
    if not groups["train"]:
        parser.error(f"no training windows in {args.windows}")
    if args.patience and not groups["validation"]:
        parser.error("--patience needs validation windows; pass --patience 0 with --validation-every 0")
    run = Path(args.out) / (args.name or f"ppo_h{args.headings}_seed{args.seed}_T{args.timesteps}")
    if run.exists() and not args.force:
        parser.error(f"{run} exists; pass --force to replace it")
    run.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, args.torch_threads))
    eval_decay = args.reward_decay if args.eval_decay is None else args.eval_decay
    for name, value in (("--reward-decay", args.reward_decay), ("--eval-decay", eval_decay)):
        if not 0.0 <= value <= 1.0:
            parser.error(f"{name} must be between 0 and 1")
    eval_find = args.find_reward if args.eval_find is None else args.eval_find
    eval_pos_scale = args.pos_scale if args.eval_pos_scale is None else args.eval_pos_scale

    kwargs = {"headings": args.headings, "arrivals_min": args.arrival_min, "halves": args.halves,
              "speeds": tuple(args.speeds or ())}
    vec = SubprocVecEnv if args.envs > 1 else DummyVecEnv
    env = make_vec_env(SearchEnv, n_envs=args.envs, seed=args.seed, vec_env_cls=vec,
                       monitor_dir=str(run / "monitor"),
                       env_kwargs={"sources": [str(p) for p in groups["train"]],
                                   "decay": args.reward_decay, "find": args.find_reward,
                                   "pos_scale": args.pos_scale, **kwargs})
    policy_kwargs = None
    if args.net_arch:
        policy_kwargs = {"net_arch": {"pi": list(args.net_arch), "vf": list(args.net_arch)}}
    if args.policy == "cnn":
        from sar.rl.cnn import SearchCNN

        policy_kwargs = {"features_extractor_class": SearchCNN,
                         "features_extractor_kwargs": {"features_dim": 256},
                         "net_arch": {"pi": list(args.net_arch or [128]),
                                      "vf": list(args.net_arch or [128])}}
    if args.init_from:
        # Loaded with this run's settings in place of the saved ones; the network, its size and
        # n_steps come from the file (custom_objects are applied before the model is set up).
        model = PPO.load(args.init_from, env=env, device="cpu", custom_objects={
            "learning_rate": args.learning_rate, "ent_coef": args.ent_coef,
            "clip_range": args.clip_range, "target_kl": args.target_kl, "gamma": args.gamma,
            "batch_size": args.batch_size, "n_epochs": args.n_epochs})
        model.set_random_seed(args.seed)
        model.verbose = 1
    else:
        model = PPO("MlpPolicy", env, learning_rate=args.learning_rate, ent_coef=args.ent_coef,
                    clip_range=args.clip_range, gamma=args.gamma, batch_size=args.batch_size,
                    n_epochs=args.n_epochs, target_kl=args.target_kl, policy_kwargs=policy_kwargs,
                    seed=args.seed, device="cpu", verbose=1)

    callback = None
    if groups["validation"]:
        sources = [str(p) for p in groups["validation"]]
        evaluator = make_vec_env(SearchEnv, n_envs=1, seed=args.seed,
                                 env_kwargs={"sources": sources, "cycle": True,
                                             "decay": eval_decay, "find": eval_find,
                                             "pos_scale": eval_pos_scale, **kwargs})
        callback = EvalCallback(evaluator, best_model_save_path=str(run),
                                log_path=str(run / "eval"),
                                eval_freq=max(args.eval_every // args.envs, 1),
                                n_eval_episodes=len(window_items(sources, args.arrival_min)),
                                deterministic=True,
                                callback_after_eval=(
                                    stop_on_plateau(args.patience, args.improve_threshold)
                                    if args.patience else None))
    else:
        print("no validation windows: best_model.zip will not be written", file=sys.stderr)

    config = {
        "arguments": vars(args), "git_commit": git_commit(),
        "stable_baselines3": stable_baselines3.__version__,
        "observation": {"size": SIZE + (1 if args.speeds else 0), "map": [MAP_CELLS, MAP_CELL_M],
                        "near": [NEAR_CELLS, NEAR_CELL_M],
                        "vector": list(VECTOR) + (["speed_share"] if args.speeds else [])},
        "speeds_kt": list(args.speeds or []),
        "ppo": {"learning_rate": args.learning_rate, "n_steps": model.n_steps,
                "batch_size": model.batch_size, "n_epochs": model.n_epochs, "gamma": model.gamma,
                "gae_lambda": model.gae_lambda, "clip_range": args.clip_range,
                "ent_coef": args.ent_coef, "vf_coef": model.vf_coef,
                "max_grad_norm": model.max_grad_norm, "net_arch": model.policy.net_arch,
                "target_kl": args.target_kl, "policy": args.policy},
        "reward": (f"removed x (1 - {args.reward_decay:g} x k / steps): particles found x time-left "
                   f"weight, / (N x window); decay 1 = time remaining, 0 = plain POS"),
        "reward_decay": args.reward_decay,
        "eval_decay": eval_decay,
        "find_reward": args.find_reward, "pos_scale": args.pos_scale,
        "eval_find": eval_find, "eval_pos_scale": eval_pos_scale,
        "init_from": args.init_from,
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
