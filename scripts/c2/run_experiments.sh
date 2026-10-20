#!/usr/bin/env bash
# The c2 experiment, from scratch: v1's run_baseline trained on Monte Carlos whose drift doubles the
# current (sar.model.drift_c2). Run on jaguar1 from the repository root; it submits three jobs and returns:
#     bash scripts/c2/run_experiments.sh
#
#   1. montes     scripts/c2/training_montes.sh -> derived/rl_c2/windows_arr30-60-120-180, 1,305 files
#   2. training   scripts/c2/train_ppo.sbatch   -> derived/rl_c2/runs/run_baseline_c2
#   3. wrap-up    scripts/c2/wrap_run.sbatch    -> curves, <run>/test/, figures/produced_maps_c2/, and
#                 runs/performance_breakdown.{csv,xlsx} beside v1's run_baseline
#
# Writes only to derived/rl_c2/, figures/produced_maps_c2/ and logs/c2/; nothing of v1, v2 or v3 is touched.
# Safe to rerun: existing windows, a trained run and a wrapped run are each skipped. Watch it with: squeue -u "$USER"

set -euo pipefail

PY="${SAR_PYTHON:-/home/26p67/envs/sar-rl/bin/python}"
DATA="${SAR_DATA:-/home/26p67/data}"
OUT="${OUT:-$DATA/derived/rl_c2}"
RUNS="${RUNS:-$OUT/runs}"
ARRIVALS="30 60 120 180"
WINDOWS="${WINDOWS:-$OUT/windows_arr${ARRIVALS// /-}}"
FIGURES="${FIGURES:-figures/produced_maps_c2}"
COMPARE_RUNS="${COMPARE_RUNS:-$DATA/derived/rl/runs/run_baseline}"
EXCLUDE="${EXCLUDE:-jaguar11}"   # 4x slower than the others for training
EXCLUSIVE="${EXCLUSIVE:-1}"      # 1 gives the training run a node to itself
EXPECTED_WINDOWS=1305            # 5 places x 261 weeks
PATIENCE=10
THRESHOLD=0.001
NAME=run_baseline_c2

[ -f src/sar/rl/windows_c2.py ] || { echo "run this from the repository root, after git pull"; exit 1; }
"$PY" -c "import stable_baselines3, matplotlib" 2>/dev/null \
    || { echo "the sar-rl environment needs stable_baselines3 and matplotlib: bash scripts/setup_rl_env.sh"; exit 1; }
mkdir -p logs/c2 "$RUNS"

echo "== 1/3 c2 Monte Carlos and windows into $WINDOWS"
HAVE=0
[ -d "$WINDOWS" ] && HAVE=$(find "$WINDOWS" -maxdepth 1 -name 'windows_*.npz' | wc -l)
AFTER_WINDOWS=""
if [ "$HAVE" -lt "$EXPECTED_WINDOWS" ]; then
    JOB=$(sbatch --parsable --export=ALL,ARRIVALS="$ARRIVALS",OUT="$OUT",WINDOWS="$WINDOWS",KEEP_CSV=0 scripts/c2/training_montes.sh)
    AFTER_WINDOWS="--dependency=afterok:$JOB"
    echo "   $HAVE of $EXPECTED_WINDOWS there; job $JOB"
else
    echo "   all $HAVE there"
fi

echo "== 2/3 training $NAME"
TRAIN_JOB=""
if [ -f "$RUNS/$NAME/model.zip" ]; then
    echo "   already trained"
else
    FORCE=""
    [ -d "$RUNS/$NAME" ] && FORCE="--force"   # a run that died partway starts again
    NODE_OPTS=(--exclude="$EXCLUDE")
    [ "$EXCLUSIVE" = 1 ] && NODE_OPTS+=(--exclusive)
    # shellcheck disable=SC2086
    TRAIN_JOB=$(sbatch --parsable --job-name="ppo_c2_$NAME" "${NODE_OPTS[@]}" $AFTER_WINDOWS scripts/c2/train_ppo.sbatch \
        --windows "$WINDOWS" --validation-every 8 --arrival-min $ARRIVALS --timesteps 3000000 --envs 4 \
        --seed 1 --eval-every 100000 --patience "$PATIENCE" --improve-threshold "$THRESHOLD" \
        --headings 36 --learning-rate 0.0003 --ent-coef 0 --clip-range 0.2 \
        --out "$RUNS" --name "$NAME" $FORCE)
    echo "   job $TRAIN_JOB"
fi

echo "== 3/3 wrap-up"
if [ -f "$RUNS/$NAME/test/summary.csv" ] && [ -f "$RUNS/$NAME/maps.txt" ] && [ -z "$TRAIN_JOB" ]; then
    echo "   already wrapped"
else
    AFTER=""
    [ -n "$TRAIN_JOB" ] && AFTER="--dependency=afterok:$TRAIN_JOB"
    # shellcheck disable=SC2086
    JOB=$(sbatch --parsable --job-name="wrap_c2_$NAME" --exclude="$EXCLUDE" --kill-on-invalid-dep=yes $AFTER \
        --export=ALL,NAME="$NAME",RUNS="$RUNS",WINDOWS="$WINDOWS",ARRIVALS="$ARRIVALS",PATIENCE="$PATIENCE",THRESHOLD="$THRESHOLD",FIGURES="$FIGURES",COMPARE_RUNS="$COMPARE_RUNS" \
        scripts/c2/wrap_run.sbatch)
    echo "   job $JOB"
fi

echo
echo "Submitted. Watch with: squeue -u $USER"
echo "Logs: logs/c2/mc-training-c2_<job>_<task>.out, logs/c2/ppo_<job>.out, logs/c2/wrap_<job>.out"
echo "Results: $RUNS/$NAME/test/summary.csv; $RUNS/performance_breakdown.{csv,xlsx}; maps in $FIGURES/$NAME/"
