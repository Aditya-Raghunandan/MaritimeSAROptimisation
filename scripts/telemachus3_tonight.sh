#!/usr/bin/env bash
# telemachus3_tonight.sh -- Telemachus 3, paid for finding the real buoy (13 Oct), in one
# submission. Run on jaguar1 from the repository root:
#
#     bash scripts/telemachus3_tonight.sh
#
# 1. Rows (scripts/truth_windows.py select, on the login node: a 9,538-row CSV): dev buoys only,
#    every 5th shared-water group held out for validation. Then their windows, with each buoy's
#    real track, on 12 nodes.
# 2. Five runs, each starting from Telemachus 2 (rl/chosen_tele, two headings a minute) and
#    scored on validation by the share of held-out buoys FOUND (--eval-find 1 --eval-pos-scale 0):
#    paid finds only, finds + a quarter or a half of the POS reward, a second seed, and the
#    control paid POS alone on the same buoys (did the reward help, or the real buoys' clouds?).
# 3. Pick on validation finds (scripts/pick_ppo_run.py), then the 55 at every arrival 15 min to
#    3 h and the held-out validation buoys, flown by Telemachus 2 (as `ppo`) and 3 (as
#    `telemachus`) beside the Square, the Parallel Track and greedy, and both summaries.
# No upload, no site change: that waits for Aditya.
set -euo pipefail

DATA="${SAR_DATA:-/home/26p67/data}"
RL_PY="${RL_PY:-/home/26p67/envs/sar-rl/bin/python}"
BENCH_PY="${BENCH_PY:-/home/26p67/envs/sar-bench/bin/python}"
WINDOWS="${WINDOWS:-$DATA/derived/rl/windows_truth}"
RUNS="${RUNS:-$DATA/derived/rl/runs_t3}"
START_FROM="${START_FROM:-$DATA/derived/rl/chosen_tele/best_model.zip}"
CHOSEN="${CHOSEN:-$DATA/derived/rl/chosen_t3}"
R55="$DATA/derived/results/scenarios55_N10000_d032d033_t3"
RVAL="$DATA/derived/results/truthval_N10000_d032d033_t3"
TIMESTEPS="${TIMESTEPS:-3000000}"

[ -f scripts/truth_windows.py ] || { echo "run this from the repository root"; exit 1; }
for d in "$R55" "$RVAL"; do [ ! -e "$d" ] || { echo "$d exists: one folder is one experiment"; exit 1; }; done
[ -f "$START_FROM" ] || { echo "no Telemachus 2 at $START_FROM"; exit 1; }
mkdir -p logs "$RUNS"
export SAR_DATA="$DATA"
export SAR_COMMIT="$(git rev-parse HEAD)$(git -c core.checkStat=minimal diff --quiet HEAD -- || echo -dirty)"

echo "== 1/3 rows and windows -> $WINDOWS"
[ -f "$WINDOWS/rows_train.csv" ] || nice PYTHONPATH=src "$BENCH_PY" scripts/truth_windows.py select \
    --csv "$DATA/derived/scenarios/bench_dev_48h.csv" --out "$WINDOWS"
W=$(env OUT="$WINDOWS" TASKS=12 SAR_PYTHON="$BENCH_PY" sbatch --parsable --array=1-12 scripts/truth_windows.sbatch)
echo "   windows $W"

echo "== 2/3 training -> $RUNS"
COMMON=(--windows "$WINDOWS/train" --validation-windows "$WINDOWS/validation" --validation-every 0
        --arrival-min 30 60 120 180 --headings 36 --halves 12 --timesteps "$TIMESTEPS" --envs 4
        --eval-every 150000 --patience 10 --improve-threshold 0.002 --learning-rate 0.0001
        --clip-range 0.2 --gamma 1.0 --batch-size 256 --n-epochs 10 --target-kl 0.03
        --eval-find 1 --eval-pos-scale 0 --init-from "$START_FROM" --out "$RUNS")
# name               find  pos_scale  ent    seed
RUNS_TABLE=(
  "t3_find           1     0          0.005  1"
  "t3_find_pos25     1     0.25       0.005  1"
  "t3_find_pos50     1     0.5        0.005  1"
  "t3_find_pos25_s2  1     0.25       0.005  2"
  "t3_pos_control    0     1          0.005  1"
)
JOBS=()
for line in "${RUNS_TABLE[@]}"; do
    read -r NAME FIND POS ENT SEED <<< "$line"
    J=$(SAR_PYTHON="$RL_PY" sbatch --parsable --job-name="$NAME" --exclusive --exclude=jaguar11 \
        --dependency="afterok:$W" --kill-on-invalid-dep=yes scripts/train_ppo.sbatch \
        "${COMMON[@]}" --name "$NAME" --find-reward "$FIND" --pos-scale "$POS" --ent-coef "$ENT" \
        --seed "$SEED")
    JOBS+=("$J")
    echo "   $NAME: job $J"
done
AFTER=$(IFS=:; echo "${JOBS[*]}")

echo "== 3/3 pick and score"
PICK=$(sbatch --parsable --job-name=pick_t3 --partition=jaguarcluster --time=00:10:00 --mem=2G \
    --dependency="afterany:$AFTER" --output=logs/pick_t3_%j.out \
    --wrap="PYTHONPATH=src $BENCH_PY scripts/pick_ppo_run.py --runs-dir $RUNS --link $CHOSEN --out $CHOSEN.json")
AGENTS=(SAR_PYTHON="$BENCH_PY" NOISE=rvc SEARCHERS="expanding-square parallel greedy ppo telemachus"
        PPO_MODEL="$START_FROM" TELEMACHUS_MODEL="$CHOSEN/best_model.zip")
S55=$(env "${AGENTS[@]}" ARRIVALS="0.25 0.5 0.75 1 2 3" OUT="$R55" sbatch --parsable --array=1-14 \
    --exclude=jaguar11 --dependency="afterok:$PICK" --kill-on-invalid-dep=yes scripts/score_scenarios.sbatch)
SVAL=$(env "${AGENTS[@]}" ARRIVALS="0.5 1 2 3" OUT="$RVAL" CSV="$WINDOWS/rows_validation.csv" PER=4 \
    sbatch --parsable --array=1-12 --exclude=jaguar11 --dependency="afterok:$PICK" \
    --kill-on-invalid-dep=yes scripts/score_scenarios.sbatch)
F=$(sbatch --parsable --job-name=t3_finish --partition=jaguarcluster --time=00:30:00 --mem=8G \
    --dependency="afterok:$S55:$SVAL" --kill-on-invalid-dep=yes --output=logs/t3_finish_%j.out \
    --wrap="set -e; export PYTHONPATH=src
$BENCH_PY scripts/score_scenarios.py summary $R55
$BENCH_PY scripts/score_scenarios.py summary $RVAL")
echo "   pick $PICK; the 55 $S55 -> $R55; validation buoys $SVAL -> $RVAL; summaries $F"
