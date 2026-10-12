#!/usr/bin/env bash
# telemachus_tonight.sh -- train the second agent's six runs (12 Oct, L40), then benchmark the
# best of them beside the first agent, all in one submission. Run on jaguar1 from the repository
# root:
#
#     bash scripts/telemachus_tonight.sh
#
# 1. Six training jobs, one node each (--exclusive, never jaguar11, 4x slower), into
#    $DATA/derived/rl/runs_tele. Every one is paid plain POS (or half the old time weighting)
#    and judged on validation POS (--eval-decay 0), so their best models compare like for like.
# 2. After all six end (afterany): pick the best on validation POS (scripts/pick_ppo_run.py),
#    then score the 55 and the dev table with all seven searchers plus the second agent as
#    "telemachus" (the first stays "ppo", from $DATA/derived/rl/chosen), and build the site's
#    scenarios/v4. The 55 and v4 finish on their own; the dev table finishes after.
# No upload: that waits for Aditya.
set -euo pipefail

DATA="${SAR_DATA:-/home/26p67/data}"
RL_PY="${RL_PY:-/home/26p67/envs/sar-rl/bin/python}"
BENCH_PY="${BENCH_PY:-/home/26p67/envs/sar-bench/bin/python}"
RUNS="${RUNS:-$DATA/derived/rl/runs_tele}"
WINDOWS="${WINDOWS:-$DATA/derived/rl/windows_arr30-60-120-180}"
FIRST="${FIRST:-$DATA/derived/rl/chosen/best_model.zip}"     # the first agent, picked 12 Oct
CHOSEN="${CHOSEN:-$DATA/derived/rl/chosen_tele}"
TAG="${TAG:-d032d033_tele}"
R55="$DATA/derived/results/scenarios55_N10000_$TAG"
RDEV="$DATA/derived/results/bench_dev_48h_N10000_$TAG"
V4="${V4:-$DATA/published/scenarios/v4}"
TIMESTEPS="${TIMESTEPS:-3000000}"

[ -f src/sar/rl/train.py ] || { echo "run this from the repository root"; exit 1; }
for d in "$R55" "$RDEV" "$V4"; do
    [ ! -e "$d" ] || { echo "$d exists: one folder is one experiment"; exit 1; }
done
[ -f "$FIRST" ] || { echo "no first agent at $FIRST"; exit 1; }
mkdir -p logs "$RUNS"
export SAR_DATA="$DATA"
export SAR_COMMIT="$(git rev-parse HEAD)$(git -c core.checkStat=minimal diff --quiet HEAD -- || echo -dirty)"

COMMON=(--windows "$WINDOWS" --validation-every 8 --arrival-min 30 60 120 180 --headings 36
        --timesteps "$TIMESTEPS" --envs 4 --eval-every 100000 --patience 10 --improve-threshold 0.001
        --learning-rate 0.0003 --clip-range 0.2 --batch-size 256 --n-epochs 10 --target-kl 0.03
        --eval-decay 0 --out "$RUNS")
# name            policy  decay  gamma  ent    seed  threads
RUNS_TABLE=(
  "tele_mlp_pos     mlp     0      1.0    0.005  1     1"
  "tele_cnn_pos     cnn     0      1.0    0.005  1     4"
  "tele_cnn_pos_s2  cnn     0      1.0    0.005  2     4"
  "tele_cnn_ent01   cnn     0      1.0    0.01   1     4"
  "tele_cnn_g0999   cnn     0      0.999  0.005  1     4"
  "tele_cnn_half    cnn     0.5    1.0    0.005  1     4"
)
echo "== 1/3 training -> $RUNS"
JOBS=()
for line in "${RUNS_TABLE[@]}"; do
    read -r NAME POLICY DECAY GAMMA ENT SEED THREADS <<< "$line"
    FORCE=()
    [ -d "$RUNS/$NAME" ] && FORCE=(--force)
    J=$(SAR_PYTHON="$RL_PY" sbatch --parsable --job-name="$NAME" --exclusive --exclude=jaguar11 \
        scripts/train_ppo.sbatch "${COMMON[@]}" --name "$NAME" --policy "$POLICY" \
        --reward-decay "$DECAY" --gamma "$GAMMA" --ent-coef "$ENT" --seed "$SEED" \
        --torch-threads "$THREADS" "${FORCE[@]}")
    JOBS+=("$J")
    echo "   $NAME: job $J"
done
AFTER=$(IFS=:; echo "${JOBS[*]}")

echo "== 2/3 pick and score"
PICK=$(sbatch --parsable --job-name=pick_tele --partition=jaguarcluster --time=00:10:00 --mem=2G \
    --dependency="afterany:$AFTER" --output=logs/pick_tele_%j.out \
    --wrap="PYTHONPATH=src $BENCH_PY scripts/pick_ppo_run.py --runs-dir $RUNS --link $CHOSEN --out $CHOSEN.json")
SEARCHERS="expanding-square sector parallel trackline greedy random ppo telemachus"
AGENTS=(SAR_PYTHON="$BENCH_PY" SEARCHERS="$SEARCHERS" PPO_MODEL="$FIRST" TELEMACHUS_MODEL="$CHOSEN/best_model.zip")
S55=$(env "${AGENTS[@]}" OUT="$R55" sbatch --parsable --array=1-14 --dependency="afterok:$PICK" \
    --kill-on-invalid-dep=yes scripts/score_scenarios.sbatch)
SDEV=$(env "${AGENTS[@]}" OUT="$RDEV" PER=25 CSV="$DATA/derived/scenarios/bench_dev_48h.csv" \
    sbatch --parsable --array=1-96 --dependency="afterok:$PICK" --kill-on-invalid-dep=yes \
    scripts/score_scenarios.sbatch)
B4=$(env SAR_PYTHON="$BENCH_PY" PPO_MODEL="$FIRST" TELEMACHUS_MODEL="$CHOSEN/best_model.zip" OUT="$V4" \
    sbatch --parsable --array=1-14 --dependency="afterok:$PICK" --kill-on-invalid-dep=yes \
    scripts/export_scenario_bundles.sbatch)
echo "   pick $PICK; the 55 $S55 -> $R55; dev $SDEV -> $RDEV; v4 $B4 -> $V4"

echo "== 3/3 finish"
F55=$(sbatch --parsable --job-name=tele_finish55 --partition=jaguarcluster --time=01:00:00 --mem=16G \
    --dependency="afterok:$S55:$B4" --kill-on-invalid-dep=yes --output=logs/tele_finish55_%j.out \
    --wrap="set -e; export PYTHONPATH=src
$BENCH_PY scripts/score_scenarios.py summary $R55
$BENCH_PY scripts/export_scenario_bundles.py index --out $V4
$BENCH_PY scripts/export_scenario_bundles.py benchmark --summary scenarios55 $R55/summary.csv --out $V4/benchmark.json")
FDEV=$(sbatch --parsable --job-name=tele_finishdev --partition=jaguarcluster --time=01:00:00 --mem=16G \
    --dependency="afterok:$SDEV:$F55" --kill-on-invalid-dep=yes --output=logs/tele_finishdev_%j.out \
    --wrap="set -e; export PYTHONPATH=src
$BENCH_PY scripts/score_scenarios.py summary $RDEV
$BENCH_PY scripts/export_scenario_bundles.py benchmark --summary scenarios55 $R55/summary.csv --summary dev $RDEV/summary.csv --out $V4/benchmark.json")
echo "   the 55 and v4: $F55; dev: $FDEV"
echo "Submitted. Watch: squeue -u \$USER; curves: $RUNS/*/eval/evaluations.npz"
