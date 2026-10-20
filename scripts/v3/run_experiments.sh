#!/usr/bin/env bash
# PPO v3, run_baseline_v3 only: v1's run_baseline with one change, each minute paid what it found
# divided by the Expanding Square's POS on that window, gamma 1.0. Run on jaguar1 from the
# repository root; it submits two jobs and returns:
#     bash scripts/v3/run_experiments.sh
#
#   1. training   scripts/v3/train_ppo.sbatch  -> /home/26p67/data/derived/rl/runs_v3/run_baseline_v3
#                 (first flies the square over the training and validation windows, about 4,000,
#                 into runs_v3/square_pos_arr30-60-120-180.csv; a rerun reads it back)
#   2. wrap-up    scripts/v3/wrap_run.sbatch   -> curves, <run>/test/, figures/produced_maps_v3/, and
#                 runs_v3/performance_breakdown.{csv,xlsx}: v3 beside v1's run_baseline and v2's
#                 run_baseline_v2, by arrival time and place
#
# It only reads the existing windows (windows_arr30-60-120-180, cut from the Monte Carlos already
# on the cluster) and the v1 and v2 test results, and writes only to runs_v3/,
# figures/produced_maps_v3/ and logs/v3/; no Monte Carlo is rerun and nothing of v1 or v2 is touched. Safe to rerun: a trained run is not trained
# again, a wrapped one is not wrapped again. Watch it with: squeue -u "$USER"

set -euo pipefail

PY="${SAR_PYTHON:-/home/26p67/envs/sar-rl/bin/python}"
DATA="${SAR_DATA:-/home/26p67/data}"
RUNS="${RUNS:-$DATA/derived/rl/runs_v3}"
ARRIVALS="30 60 120 180"
WINDOWS="${WINDOWS:-$DATA/derived/rl/windows_arr30-60-120-180}"
SQUARE_CACHE="$RUNS/square_pos_arr30-60-120-180.csv"
FIGURES="${FIGURES:-figures/produced_maps_v3}"
COMPARE_RUNS="${COMPARE_RUNS:-$DATA/derived/rl/runs/run_baseline $DATA/derived/rl/runs_v2/run_baseline_v2}"
EXCLUDE="${EXCLUDE:-jaguar11}"   # 4x slower than the others for training
EXCLUSIVE="${EXCLUSIVE:-1}"      # 1 gives the training run a node to itself
EXPECTED_WINDOWS=1305
PATIENCE=10
THRESHOLD=0.001
NAME=run_baseline_v3

[ -f src/sar/rl_v3/train.py ] || { echo "run this from the repository root, after git pull"; exit 1; }
"$PY" -c "import stable_baselines3, matplotlib" 2>/dev/null \
    || { echo "the sar-rl environment needs stable_baselines3 and matplotlib: bash scripts/setup_rl_env.sh"; exit 1; }
HAVE=0
[ -d "$WINDOWS" ] && HAVE=$(find "$WINDOWS" -maxdepth 1 -name 'windows_*.npz' | wc -l)
[ "$HAVE" -ge "$EXPECTED_WINDOWS" ] \
    || { echo "only $HAVE of $EXPECTED_WINDOWS windows in $WINDOWS; v3 reuses them and does not make them"; exit 1; }
mkdir -p logs/v3 "$RUNS"

echo "== 1/2 training $NAME"
TRAIN_JOB=""
if [ -f "$RUNS/$NAME/model.zip" ]; then
    echo "   already trained"
else
    FORCE=""
    [ -d "$RUNS/$NAME" ] && FORCE="--force"   # a run that died partway starts again
    NODE_OPTS=(--exclude="$EXCLUDE")
    [ "$EXCLUSIVE" = 1 ] && NODE_OPTS+=(--exclusive)
    # shellcheck disable=SC2086
    TRAIN_JOB=$(sbatch --parsable --job-name="ppo_v3_$NAME" "${NODE_OPTS[@]}" scripts/v3/train_ppo.sbatch \
        --windows "$WINDOWS" --validation-every 8 --arrival-min $ARRIVALS --timesteps 3000000 --envs 4 \
        --seed 1 --eval-every 100000 --patience "$PATIENCE" --improve-threshold "$THRESHOLD" \
        --headings 36 --learning-rate 0.0003 --ent-coef 0 --clip-range 0.2 --gamma 1.0 \
        --square-cache "$SQUARE_CACHE" --out "$RUNS" --name "$NAME" $FORCE)
    echo "   job $TRAIN_JOB"
fi

echo "== 2/2 wrap-up"
if [ -f "$RUNS/$NAME/test/summary.csv" ] && [ -f "$RUNS/$NAME/maps.txt" ] && [ -z "$TRAIN_JOB" ]; then
    echo "   already wrapped"
else
    AFTER=""
    [ -n "$TRAIN_JOB" ] && AFTER="--dependency=afterok:$TRAIN_JOB"
    # shellcheck disable=SC2086
    JOB=$(sbatch --parsable --job-name="wrap_v3_$NAME" --exclude="$EXCLUDE" --kill-on-invalid-dep=yes $AFTER \
        --export=ALL,NAME="$NAME",RUNS="$RUNS",WINDOWS="$WINDOWS",ARRIVALS="$ARRIVALS",PATIENCE="$PATIENCE",THRESHOLD="$THRESHOLD",FIGURES="$FIGURES",COMPARE_RUNS="$COMPARE_RUNS" \
        scripts/v3/wrap_run.sbatch)
    echo "   job $JOB"
fi

echo
echo "Submitted. Watch with: squeue -u $USER"
echo "Logs: logs/v3/ppo_<job>.out (training), logs/v3/wrap_<job>.out"
echo "Results: $RUNS/$NAME/test/summary.csv; $RUNS/performance_breakdown.{csv,xlsx}; maps in $FIGURES/$NAME/"
