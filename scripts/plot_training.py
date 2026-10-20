"""plot_training.py: a PPO run's learning curves, where validation stopped improving, and when --patience stops it.

Reads what sar.rl.train wrote into the run folder: monitor/*.monitor.csv (every training
episode's reward) and eval/evaluations.npz (every validation pass). Draws both against steps,
marks the last validation that beat the best by more than --improve-threshold, and the one --patience
validations later where an early-stopping run would have ended. Writes training_curves.png and
training_curves.json into the run folder, and prints the validation table.

    python scripts/plot_training.py --run /home/26p67/data/derived/rl/runs/ppo_h36_seed1_T3000000 \
        --patience 5 --improve-threshold 0.001
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sar.rl.train import plateau

# The dataviz reference palette, as scripts/plot_sigma_calibration.py uses it.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
ROLLING = 500   # training episodes per point of the training curve, about 22,500 steps

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED,
    "ytick.color": MUTED, "text.color": INK, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2.0, "lines.solid_capstyle": "round", "font.size": 10,
    "legend.frameon": False, "axes.titlesize": 11, "axes.titleweight": "semibold",
})


def training_episodes(run: Path) -> pd.DataFrame:
    """Every training episode from all envs, in the order they finished, with the steps flown by then."""
    paths = sorted((run / "monitor").glob("*.monitor.csv"))
    if not paths:
        raise SystemExit(f"no monitor/*.monitor.csv in {run}")
    episodes = pd.concat([pd.read_csv(p, skiprows=1) for p in paths]).sort_values("t")
    return episodes.assign(steps=episodes["l"].cumsum()).reset_index(drop=True)


def validations(run: Path) -> tuple[np.ndarray, np.ndarray]:
    """(steps, mean reward over the validation windows) at every validation pass."""
    path = run / "eval" / "evaluations.npz"
    if not path.exists():
        raise SystemExit(f"no {path}: the run had no validation windows")
    with np.load(path) as e:
        return e["timesteps"], e["results"].mean(axis=1)


def draw(run: Path, patience: int, threshold: float) -> dict:
    episodes = training_episodes(run)
    steps, scores = validations(run)
    improved, stop = plateau(scores, patience, threshold)
    best = int(np.argmax(scores))
    rolling = episodes["r"].rolling(ROLLING, min_periods=ROLLING // 5).mean()

    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.plot(episodes["steps"] / 1e6, rolling, color=BLUE, linewidth=1.4,
            label=f"training, mean of {ROLLING} episodes")
    ax.plot(steps / 1e6, scores, color=ORANGE, marker="o", markersize=3.5, label="validation")
    ax.plot(steps[best] / 1e6, scores[best], marker="*", markersize=14, color=ORANGE,
            markeredgecolor=INK, linestyle="none", label=f"best {scores[best]:.3f}, best_model.zip")
    if improved is not None:
        ax.axvline(steps[improved] / 1e6, color=AQUA, linestyle="--", linewidth=1.4)
        ax.annotate(f"last gain over {threshold:g}\n{steps[improved] / 1e6:.2f}M steps",
                    (steps[improved] / 1e6, ax.get_ylim()[0]), xytext=(4, 8),
                    textcoords="offset points", fontsize=9, color=INK2)
    if stop is not None:
        end = episodes["steps"].iloc[-1] / 1e6
        ax.axvspan(steps[stop] / 1e6, end, color=MUTED, alpha=0.12, linewidth=0)
        ax.annotate(f"patience {patience} stops here\n{end - steps[stop] / 1e6:.2f}M steps saved",
                    (steps[stop] / 1e6, ax.get_ylim()[1]), xytext=(4, -24),
                    textcoords="offset points", fontsize=9, color=INK2)
    ax.set(xlabel="training steps (millions)", ylabel="reward per episode (time-weighted POS)",
           title=run.name)
    ax.legend(loc="lower right")
    fig.savefig(run / "training_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    data = {"run": run.name, "patience": patience, "improve_threshold": threshold,
            "validation": [{"steps": int(s), "reward": float(r)} for s, r in zip(steps, scores)],
            "best": {"steps": int(steps[best]), "reward": float(scores[best])},
            "last_gain_steps": None if improved is None else int(steps[improved]),
            "would_stop_steps": None if stop is None else int(steps[stop]),
            "trained_steps": int(episodes["steps"].iloc[-1])}
    (run / "training_curves.json").write_text(json.dumps(data, indent=2))
    return data


def _cli(argv=None):
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--run", required=True, help="a run folder, such as <out>/ppo_h36_seed1_T3000000")
    p.add_argument("--patience", type=int, default=5, help="validations without a gain before stopping")
    p.add_argument("--improve-threshold", type=float, default=0.001, help="smallest gain that counts")
    args = p.parse_args(argv)
    data = draw(Path(args.run), args.patience, args.improve_threshold)

    print(f"{'steps':>11}  {'validation':>10}")
    for i, row in enumerate(data["validation"]):
        mark = ("  best" if row["steps"] == data["best"]["steps"] else "") + \
               ("  last gain" if row["steps"] == data["last_gain_steps"] else "") + \
               ("  would stop" if row["steps"] == data["would_stop_steps"] else "")
        print(f"{row['steps']:>11,}  {row['reward']:>10.4f}{mark}")
    stop = data["would_stop_steps"]
    print(f"\ntrained {data['trained_steps']:,} steps; with --patience {args.patience} "
          f"--improve-threshold {args.improve_threshold:g} it "
          + ("would not have stopped early" if stop is None
             else f"would have stopped at {stop:,}, saving {data['trained_steps'] - stop:,} steps"))
    print("wrote", Path(args.run) / "training_curves.png")


if __name__ == "__main__":
    _cli()
