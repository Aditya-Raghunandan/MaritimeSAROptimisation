#!/usr/bin/env bash
# PPO v2, run_baseline_v2 only: 2D maps through a CNN, the final POS as the only reward, gamma 1.0;
# everything else is run_baseline's. Run on jaguar1 from the repository root; it submits two jobs
# and returns:
#     bash scripts/v2/run_experiments.sh
#
#   1. training   scripts/v2/train_ppo.sbatch  -> /home/26p67/data/derived/rl/runs_v2/run_baseline_v2
#   2. wrap-up    scripts/v2/wrap_run.sbatch   -> curves, <run>/test/, figures/produced_maps_v2/
#
# It only reads the existing windows (windows_arr30-60-120-180, cut from the Monte Carlos already
# on the cluster) and writes only to runs_v2/, figures/produced_maps_v2/ and logs/v2/, so nothing
# of the first version, running or finished, is touched. Safe to rerun: a trained run is not
# trained again, a wrapped one is not wrapped again. Watch it with: squeue -u "$USER"

set -euo pipefail

PY="${SAR_PYTHON:-/home/26p67/envs/sar-rl/bin/python}"
DATA="${SAR_DATA:-/home/26p67/data}"
RUNS="${RUNS:-$DATA/derived/rl/runs_v2}"
ARRIVALS="30 60 120 180"
WINDOWS="${WINDOWS:-$DATA/derived/rl/windows_arr30-60-120-180}"
FIGURES="${FIGURES:-figures/produced_maps_v2}"
EXCLUDE="${EXCLUDE:-jaguar11}"   # 4x slower than the others for training
EXCLUSIVE="${EXCLUSIVE:-1}"      # 1 gives the training run a node to itself
EXPECTED_WINDOWS=1305
PATIENCE=10
THRESHOLD=0.001
NAME=run_baseline_v2

[ -f src/sar/rl_v2/train.py ] || { echo "run this from the repository root, after git pull"; exit 1; }
"$PY" -c "import stable_baselines3, matplotlib" 2>/dev/null \
    || { echo "the sar-rl environment needs stable_baselines3 and matplotlib: bash scripts/setup_rl_env.sh"; exit 1; }
HAVE=0
[ -d "$WINDOWS" ] && HAVE=$(find "$WINDOWS" -maxdepth 1 -name 'windows_*.npz' | wc -l)
[ "$HAVE" -ge "$EXPECTED_WINDOWS" ] \
    || { echo "only $HAVE of $EXPECTED_WINDOWS windows in $WINDOWS; v2 reuses them and does not make them"; exit 1; }
mkdir -p logs/v2 "$RUNS"

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
    TRAIN_JOB=$(sbatch --parsable --job-name="ppo_v2_$NAME" "${NODE_OPTS[@]}" scripts/v2/train_ppo.sbatch \
        --windows "$WINDOWS" --validation-every 8 --arrival-min $ARRIVALS --timesteps 3000000 --envs 4 \
        --seed 1 --eval-every 100000 --patience "$PATIENCE" --improve-threshold "$THRESHOLD" \
        --headings 36 --learning-rate 0.0003 --ent-coef 0 --clip-range 0.2 --gamma 1.0 \
        --out "$RUNS" --name "$NAME" $FORCE)
    echo "   job $TRAIN_JOB"
fi

echo "== 2/2 wrap-up"
if [ -f "$RUNS/$NAME/test/summary.csv" ] && [ -f "$RUNS/$NAME/maps.txt" ] && [ -z "$TRAIN_JOB" ]; then
    echo "   already wrapped"
else
    AFTER=""
    [ -n "$TRAIN_JOB" ] && AFTER="--dependency=afterok:$TRAIN_JOB"
    # shellcheck disable=SC2086
    JOB=$(sbatch --parsable --job-name="wrap_v2_$NAME" --exclude="$EXCLUDE" --kill-on-invalid-dep=yes $AFTER \
        --export=ALL,NAME="$NAME",RUNS="$RUNS",WINDOWS="$WINDOWS",ARRIVALS="$ARRIVALS",PATIENCE="$PATIENCE",THRESHOLD="$THRESHOLD",FIGURES="$FIGURES" \
        scripts/v2/wrap_run.sbatch)
    echo "   job $JOB"
fi

echo
echo "Submitted. Watch with: squeue -u $USER"
echo "Logs: logs/v2/ppo_<job>.out (training), logs/v2/wrap_<job>.out"
echo "Results: $RUNS/$NAME/test/summary.csv; maps in $FIGURES/$NAME/"
