#!/usr/bin/env bash
#SBATCH --job-name=collect
#SBATCH --partition=jaguarcluster
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/collect_%j.out
#SBATCH --error=logs/collect_%j.out

# Gathers every trained run's config, validation, training and test results into CSVs and one tar.gz; only reads the runs.
# From the repository root: sbatch scripts/collect_results.sh   (or: bash scripts/collect_results.sh)

set -euo pipefail

PY="${SAR_PYTHON:-/home/26p67/envs/sar-rl/bin/python}"
DATA="${SAR_DATA:-/home/26p67/data}"
STAMP="$(date +%Y%m%d_%H%M)"
OUT="${OUT:-$HOME/results_export_$STAMP}"
# version=runs directory pairs; add more with EXTRA="v3=/path/to/runs_v3"
SOURCES="v1=$DATA/derived/rl/runs v2=$DATA/derived/rl/runs_v2 c2=$DATA/derived/rl_c2/runs ${EXTRA:-}"

mkdir -p "$OUT" logs
"$PY" - "$OUT" $SOURCES <<'PY'
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

out = Path(sys.argv[1])
sources = [s.split("=", 1) for s in sys.argv[2:] if "=" in s]
runs, vals, episodes, flights, summaries, notes = [], [], [], [], [], []
STEP_BIN = 100_000   # project choice: training episodes averaged per 100k steps, the validation interval


def tag(frame, version, run):
    frame.insert(0, "run", run)
    frame.insert(0, "version", version)
    return frame


for version, root in sources:
    root = Path(root)
    if not root.is_dir():
        notes.append(f"{version}: {root} missing")
        continue
    for extra in ("comparison.csv", "performance_breakdown.csv"):
        if (root / extra).exists():
            pd.read_csv(root / extra).to_csv(out / f"{version}_{extra}", index=False)
    for run in sorted(d for d in root.iterdir() if (d / "config.json").exists()):
        name = run.name
        config = json.loads((run / "config.json").read_text())
        row = pd.json_normalize(config, sep=".").iloc[0].to_dict()
        row = {k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in row.items()}
        row.update(version=version, run=name, dir=str(run),
                   has_model=(run / "model.zip").exists(), has_best_model=(run / "best_model.zip").exists(),
                   has_test=(run / "test" / "flights.csv").exists())

        evals = run / "eval" / "evaluations.npz"
        if evals.exists():
            npz = np.load(evals)
            res = np.asarray(npz["results"], dtype=float).reshape(len(npz["timesteps"]), -1)
            means = res.mean(axis=1)
            best = int(np.argmax(means))
            row.update(best_val=float(means[best]), best_val_step=int(npz["timesteps"][best]),
                       last_val_step=int(npz["timesteps"][-1]), n_validations=len(means))
            vals.append(tag(pd.DataFrame({"timesteps": npz["timesteps"], "val_mean": means,
                                          "val_std": res.std(axis=1), "n_episodes": res.shape[1],
                                          "is_best": np.arange(len(means)) == best}), version, name))
        else:
            notes.append(f"{version}/{name}: no eval/evaluations.npz")

        monitors = sorted((run / "monitor").glob("*.monitor.csv"))
        if monitors:
            ep = pd.concat([pd.read_csv(p, skiprows=1) for p in monitors]).sort_values("t")
            ep["step"] = ep["l"].cumsum()
            ep["step_bin"] = (ep["step"] // STEP_BIN + 1) * STEP_BIN
            agg = ep.groupby("step_bin").agg(episodes=("r", "size"), reward_mean=("r", "mean"),
                                            reward_std=("r", "std"), length_mean=("l", "mean")).reset_index()
            episodes.append(tag(agg, version, name))
            row["monitor_steps"] = int(ep["step"].iloc[-1])
        else:
            notes.append(f"{version}/{name}: no monitor/*.monitor.csv")

        test = run / "test"
        if (test / "flights.csv").exists():
            f = pd.read_csv(test / "flights.csv")
            row["test_flights"] = len(f)
            row["test_windows"] = f[["window", "arrival_min"]].drop_duplicates().shape[0]
            row["test_searchers"] = " ".join(sorted(f["searcher"].unique()))
            flights.append(tag(f, version, name))
        else:
            notes.append(f"{version}/{name}: no test/flights.csv")
        if (test / "summary.csv").exists():
            summaries.append(tag(pd.read_csv(test / "summary.csv"), version, name))
        runs.append(row)

if not runs:
    sys.exit("no run folders with a config.json were found: " + " ".join(r for _, r in sources))
first = ["version", "run", "has_model", "has_best_model", "has_test", "best_val", "best_val_step",
         "last_val_step", "n_validations", "monitor_steps", "test_flights", "test_windows", "test_searchers"]
table = pd.DataFrame(runs)
table = table[[c for c in first if c in table] + [c for c in table if c not in first]]
table.to_csv(out / "runs.csv", index=False)
for frames, file in ((vals, "validation.csv"), (episodes, "training_episodes.csv"),
                     (flights, "flights.csv"), (summaries, "summary.csv")):
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(out / file, index=False)
(out / "notes.txt").write_text("\n".join(notes) + "\n")
print(f"{len(runs)} runs; {sum(len(f) for f in flights)} test flights; notes: {len(notes)}")
print("\n".join(notes))
PY

cp scripts/rl_locations.txt "$OUT/" 2>/dev/null || true
git rev-parse HEAD > "$OUT/collected_at_commit.txt" 2>/dev/null || true
tar -czf "$OUT.tar.gz" -C "$(dirname "$OUT")" "$(basename "$OUT")"
echo "wrote $OUT.tar.gz ($(du -h "$OUT.tar.gz" | cut -f1)); copy it off with:"
echo "scp $USER@<cluster-login>:$OUT.tar.gz ."
