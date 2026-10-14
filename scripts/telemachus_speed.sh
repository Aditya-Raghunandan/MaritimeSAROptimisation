#!/usr/bin/env bash
# telemachus_speed.sh -- Telemachus S (15 Oct, L43): Telemachus 2 exactly, except that each
# minute it also picks its speed, 45, 60 or 90 kt, and turns at the rate a 30 degree bank gives
# at that speed (14.0, 10.5, 7.0 deg/s). One experiment on its own branch (exp/telemachus-speed)
# and in its own folders; nothing here touches the site or Telemachus 2's results. Run on jaguar1
# from the repository root:
#
#     bash scripts/telemachus_speed.sh
#
# 1. One training job (one node), Telemachus 2's arguments from its config.json word for word,
#    plus --speeds 45 60 90, into $DATA/derived/rl/runs_speed/tele_speed.
# 2. After it (afterok): the 55 at all six arrivals, and the 181 held-out real buoys (Telemachus
#    3's validation rows) at 30 min to 3 h, flying the Expanding Square, the Parallel Track,
#    greedy, Telemachus 2 (as "ppo") and Telemachus S (as "telemachus") over the same clouds,
#    then both summaries. Same folder, same referee: the comparison is like for like.
set -euo pipefail

DATA="${SAR_DATA:-/home/26p67/data}"
RL_PY="${RL_PY:-/home/26p67/envs/sar-rl/bin/python}"
BENCH_PY="${BENCH_PY:-/home/26p67/envs/sar-bench/bin/python}"
WINDOWS="${WINDOWS:-$DATA/derived/rl/windows_arr30-60-120-180}"
TRUTH="${TRUTH:-$DATA/derived/rl/windows_truth}"
TELE2="${TELE2:-$DATA/derived/rl/chosen_tele/best_model.zip}"
RUNS="${RUNS:-$DATA/derived/rl/runs_speed}"
NAME="${NAME:-tele_speed}"
R55="$DATA/derived/results/scenarios55_N10000_d032d033_speed"
RVAL="$DATA/derived/results/truthval_N10000_d032d033_speed"

[ -f src/sar/rl/train.py ] || { echo "run this from the repository root"; exit 1; }
for d in "$RUNS/$NAME" "$R55" "$RVAL"; do
    [ ! -e "$d" ] || { echo "$d exists: one folder is one experiment"; exit 1; }
done
[ -f "$TELE2" ] || { echo "no Telemachus 2 at $TELE2"; exit 1; }
[ -f "$TRUTH/rows_validation.csv" ] || { echo "no held-out buoys at $TRUTH/rows_validation.csv"; exit 1; }
mkdir -p logs "$RUNS"
export SAR_DATA="$DATA"
export SAR_COMMIT="$(git rev-parse HEAD)$(git -c core.checkStat=minimal diff --quiet HEAD -- || echo -dirty)"

echo "== 1/2 training -> $RUNS/$NAME"
T=$(SAR_PYTHON="$RL_PY" sbatch --parsable --job-name="$NAME" --exclusive --exclude=jaguar11 \
    scripts/train_ppo.sbatch \
    --windows "$WINDOWS" --validation-every 8 --arrival-min 30 60 120 180 --headings 36 --halves 12 \
    --timesteps 3000000 --envs 4 --seed 1 --eval-every 100000 --patience 10 --improve-threshold 0.001 \
    --learning-rate 0.0003 --ent-coef 0.005 --clip-range 0.2 --reward-decay 0 --eval-decay 0 \
    --gamma 1.0 --policy mlp --batch-size 256 --n-epochs 10 --target-kl 0.03 --torch-threads 1 \
    --speeds 45 60 90 --out "$RUNS" --name "$NAME")
echo "   $NAME: job $T"

echo "== 2/2 score, after training"
AGENTS=(SAR_PYTHON="$BENCH_PY" NOISE=rvc SEARCHERS="expanding-square parallel greedy ppo telemachus"
        PPO_MODEL="$TELE2" TELEMACHUS_MODEL="$RUNS/$NAME/best_model.zip")
S55=$(env "${AGENTS[@]}" ARRIVALS="0.25 0.5 0.75 1 2 3" OUT="$R55" sbatch --parsable --array=1-14 \
    --exclude=jaguar11 --dependency="afterok:$T" --kill-on-invalid-dep=yes scripts/score_scenarios.sbatch)
SVAL=$(env "${AGENTS[@]}" ARRIVALS="0.5 1 2 3" OUT="$RVAL" CSV="$TRUTH/rows_validation.csv" PER=4 \
    sbatch --parsable --array=1-12 --exclude=jaguar11 --dependency="afterok:$T" \
    --kill-on-invalid-dep=yes scripts/score_scenarios.sbatch)
F=$(sbatch --parsable --job-name=speed_finish --partition=jaguarcluster --time=00:30:00 --mem=8G \
    --dependency="afterok:$S55:$SVAL" --kill-on-invalid-dep=yes --output=logs/speed_finish_%j.out \
    --wrap="set -e; export PYTHONPATH=src
$BENCH_PY scripts/score_scenarios.py summary $R55
$BENCH_PY scripts/score_scenarios.py summary $RVAL")
echo "   the 55 $S55 -> $R55; held-out buoys $SVAL -> $RVAL; summaries $F"
