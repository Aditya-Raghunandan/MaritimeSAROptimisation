"""compare_runs.py: one row per experiment, set against run_baseline, from what each run and its wrap-up wrote.

Reads each run's config.json, eval/evaluations.npz and test/summary.csv (sar.rl.fly windows) and
writes <runs-dir>/comparison.csv: the setting that differs from run_baseline, the steps trained,
the best validation score and when, and per arrival time the model's mean POS on the test weeks,
the Expanding Square's, the model's lead and the share of windows it won. A run whose wrap-up
has not finished gets blanks rather than stopping the table.

    PYTHONPATH=src python scripts/compare_runs.py --runs-dir /home/26p67/data/derived/rl/runs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASELINE = "run_baseline"
SETTINGS = ("headings", "learning_rate", "ent_coef", "clip_range")


def row(run: Path, baseline: dict | None) -> dict:
    config = json.loads((run / "config.json").read_text())
    args = config["arguments"]
    changed = [f"{k} {args[k]:g}" for k in SETTINGS
               if baseline is not None and k in args and args[k] != baseline.get(k)]
    out = {"run": run.name, "changed": "; ".join(changed) or ("—" if run.name == BASELINE else ""),
           "trained_steps": config.get("trained_timesteps")}
    evals = run / "eval" / "evaluations.npz"
    if evals.exists():
        with np.load(evals) as e:
            means = e["results"].mean(axis=1)
            best = int(np.argmax(means))
            out.update(best_validation=round(float(means[best]), 4), best_at=int(e["timesteps"][best]))
    summary = run / "test" / "summary.csv"
    if summary.exists():
        table = pd.read_csv(summary)
        for minutes, group in table.groupby("arrival_min"):
            pos = group.set_index("searcher")["pos"]
            out[f"pos_{minutes}"] = round(float(pos.get("ppo", np.nan)), 4)
            out[f"square_{minutes}"] = round(float(pos.get("expanding-square", np.nan)), 4)
            if "ppo_minus_square" in group:
                out[f"lead_{minutes}"] = round(float(group["ppo_minus_square"].iloc[0]), 4)
                out[f"wins_{minutes}"] = round(float(group["ppo_wins"].iloc[0]), 3)
    return out


def compare(runs_dir, names=None) -> pd.DataFrame:
    runs_dir = Path(runs_dir)
    names = names or sorted(d.name for d in runs_dir.glob("run_*") if (d / "config.json").exists())
    base = runs_dir / BASELINE / "config.json"
    baseline = json.loads(base.read_text())["arguments"] if base.exists() else None
    order = sorted(names, key=lambda n: (n != BASELINE, n))
    table = pd.DataFrame([row(runs_dir / n, baseline) for n in order])
    table.to_csv(runs_dir / "comparison.csv", index=False)
    return table


def _cli(argv=None):
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--runs-dir", required=True, help="the folder holding the run_* folders")
    p.add_argument("--models", nargs="+", help="run folders to compare (default: every run_*)")
    args = p.parse_args(argv)
    table = compare(args.runs_dir, args.models)
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(table.to_string(index=False))
    print("wrote", Path(args.runs_dir) / "comparison.csv")


if __name__ == "__main__":
    _cli()
