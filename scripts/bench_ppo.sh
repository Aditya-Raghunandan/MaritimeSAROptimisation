#!/usr/bin/env bash
# The trained agent through the benchmark (D031) and onto the site, submitted to Slurm in one
# go. Run on jaguar1 from the checkout; it submits the jobs and returns, and Slurm chains them:
#     AFTER=59519:59520 bash scripts/bench_ppo.sh      # AFTER: the training jobs, if running
#
#   1. pick      the finished run with the best validation score (scripts/pick_ppo_run.py);
#                $CHOSEN becomes a link to it, $CHOSEN.json says why
#   2. score     the six searchers and the agent over the 55 and the dev table, each into a
#                folder of its own (scripts/score_scenarios.sbatch)
#   3. bundles   the site's files with the agent's flight, as scenarios/v3
#                (scripts/export_scenario_bundles.sbatch)
#   4. finish    both summaries, v3's index.json and benchmark.json
#
# Nothing is uploaded: publishing v3 is a person's decision, and the command is printed at
# the end. Needs an environment with the rl extra and a NetCDF reader: SAR_PYTHON.

set -euo pipefail

PY="${SAR_PYTHON:-/home/26p67/envs/sar-bench/bin/python}"
DATA="${SAR_DATA:-/home/26p67/data}"
RUNS="${RUNS:-$DATA/derived/rl/runs}"
CHOSEN="${CHOSEN:-$DATA/derived/rl/chosen}"
TAG="${TAG:-d032d033_ppo}"
R55="${R55:-$DATA/derived/results/scenarios55_N10000_$TAG}"
RDEV="${RDEV:-$DATA/derived/results/bench_dev_48h_N10000_$TAG}"
V3="${V3:-$DATA/published/scenarios/v3}"
AFTER="${AFTER:-}"
SEARCHERS="expanding-square sector parallel trackline greedy random ppo"
MODEL="$CHOSEN/best_model.zip"

[ -f scripts/score_scenarios.py ] || { echo "run this from the repository root"; exit 1; }
PYTHONPATH=src "$PY" -c "import stable_baselines3, netCDF4, sar.rl.fly" \
    || { echo "$PY needs stable_baselines3, netCDF4 and sar.rl"; exit 1; }
for d in "$R55" "$RDEV" "$V3"; do
    [ ! -e "$d" ] || { echo "$d exists: one folder is one experiment; move it or set TAG/V3"; exit 1; }
done
mkdir -p logs
export SAR_PYTHON="$PY" SAR_DATA="$DATA"
export SAR_COMMIT="$(git rev-parse HEAD)$(git -c core.checkStat=minimal diff --quiet HEAD -- || echo -dirty)"

echo "== 1/4 pick"
DEP=()
[ -z "$AFTER" ] || DEP=(--dependency="afterany:$AFTER")
PICK=$(sbatch --parsable --job-name=pick_ppo --partition=jaguarcluster --time=00:10:00 --mem=2G \
    "${DEP[@]}" --output=logs/pick_ppo_%j.out \
    --wrap="PYTHONPATH=src $PY scripts/pick_ppo_run.py --runs-dir $RUNS --link $CHOSEN --out $CHOSEN.json")
echo "   job $PICK${AFTER:+, after $AFTER}"

echo "== 2/4 score"
S55=$(SEARCHERS="$SEARCHERS" PPO_MODEL="$MODEL" OUT="$R55" \
    sbatch --parsable --array=1-14 --dependency="afterok:$PICK" --kill-on-invalid-dep=yes \
    scripts/score_scenarios.sbatch)
SDEV=$(SEARCHERS="$SEARCHERS" PPO_MODEL="$MODEL" OUT="$RDEV" PER=25 \
    CSV="$DATA/derived/scenarios/bench_dev_48h.csv" \
    sbatch --parsable --array=1-96 --dependency="afterok:$PICK" --kill-on-invalid-dep=yes \
    scripts/score_scenarios.sbatch)
echo "   the 55: $S55 -> $R55"
echo "   dev:    $SDEV -> $RDEV"

echo "== 3/4 bundles"
B3=$(PPO_MODEL="$MODEL" OUT="$V3" \
    sbatch --parsable --array=1-14 --dependency="afterok:$PICK" --kill-on-invalid-dep=yes \
    scripts/export_scenario_bundles.sbatch)
echo "   $B3 -> $V3"

echo "== 4/4 finish"
FIN=$(sbatch --parsable --job-name=bench_ppo_finish --partition=jaguarcluster --time=01:00:00 \
    --mem=16G --dependency="afterok:$S55:$SDEV:$B3" --kill-on-invalid-dep=yes \
    --output=logs/bench_ppo_finish_%j.out \
    --wrap="set -e; export PYTHONPATH=src
$PY scripts/score_scenarios.py summary $R55
$PY scripts/score_scenarios.py summary $RDEV
$PY scripts/export_scenario_bundles.py index --out $V3
$PY scripts/export_scenario_bundles.py benchmark --summary scenarios55 $R55/summary.csv --summary dev $RDEV/summary.csv --out $V3/benchmark.json")
echo "   $FIN"

echo
echo "Submitted. Watch with: squeue -u $USER"
echo "When $FIN is done, and only when someone says publish (the pipeline's environment has huggingface_hub):"
echo "  PYTHONPATH=src /home/26p67/envs/sar/bin/python scripts/export_scenario_bundles.py upload --src $V3 --repo AdityaRugs/MaritimeSARoperations --path-in-repo scenarios/v3"
