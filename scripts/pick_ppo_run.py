"""pick_ppo_run.py: which trained agent the benchmark flies, chosen on validation alone.

Every run folder under --runs-dir that finished training (it has model.zip) is read from its
eval/evaluations.npz: Stable-Baselines3's EvalCallback scores the policy on the validation
windows every --eval-every steps and keeps the best as best_model.zip. The run whose best
validation return is highest is the one flown. The benchmark's buoys and the test weeks play
no part in the choice, so none of them is used twice.

--link makes a symlink to the chosen run (score_scenarios.py --ppo-model LINK/best_model.zip
then always means the chosen one), and --out writes the table the choice was made from.

    python scripts/pick_ppo_run.py --runs-dir $DATA/derived/rl/runs \\
        --link $DATA/derived/rl/chosen --out $DATA/derived/rl/chosen.json
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def validation(run: Path) -> dict | None:
    """A finished run's best validation return and when, or None if it is not finished."""
    curve = run / "eval" / "evaluations.npz"
    if not all(p.exists() for p in (run / "model.zip", run / "best_model.zip", curve)):
        return None
    held = np.load(curve)
    means = held["results"].mean(axis=1)
    k = int(np.argmax(means))
    return {"run": run.name, "best": float(means[k]), "at_timesteps": int(held["timesteps"][k]),
            "evaluations": int(means.size), "last_timesteps": int(held["timesteps"][-1])}


def pick(runs_dir: Path) -> tuple[dict, list[dict], list[str]]:
    table, unfinished = [], []
    for run in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        v = validation(run)
        if v:
            table.append(v)
        else:
            unfinished.append(run.name)
    if not table:
        raise SystemExit(f"no finished run under {runs_dir}")
    table.sort(key=lambda v: v["best"], reverse=True)
    return table[0], table, unfinished


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--runs-dir", required=True)
    parser.add_argument("--link", help="symlink to point at the chosen run")
    parser.add_argument("--out", help="JSON of the choice and every run's validation")
    args = parser.parse_args(argv)

    runs_dir = Path(args.runs_dir)
    chosen, table, unfinished = pick(runs_dir)
    for v in table:
        mark = "*" if v is chosen else " "
        print(f"{mark} {v['run']:<24} best validation {v['best']:.4f} at {v['at_timesteps']:>9,} "
              f"steps ({v['evaluations']} evaluations, to {v['last_timesteps']:,})")
    for name in unfinished:
        print(f"  {name:<24} not finished, left out")
    model = runs_dir / chosen["run"] / "best_model.zip"
    if args.link:
        link = Path(args.link)
        tmp = link.with_name(link.name + f".{os.getpid()}.tmp")
        tmp.symlink_to((runs_dir / chosen["run"]).resolve(), target_is_directory=True)
        os.replace(tmp, link)
        model = link / "best_model.zip"
    if args.out:
        Path(args.out).write_text(json.dumps({
            "chosen": chosen["run"], "model": str(model),
            "rule": "highest best validation return (eval/evaluations.npz) among finished runs",
            "when_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "runs": table, "unfinished": unfinished}, indent=1))
    print(f"chosen: {chosen['run']} -> {model}")


if __name__ == "__main__":
    main()
