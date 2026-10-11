#!/usr/bin/env bash
#SBATCH --job-name=wrap
#SBATCH --partition=jaguarcluster
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=logs/wrap_%j.out
#SBATCH --error=logs/wrap_%j.out

# The wrap-up for one finished training run: its learning curves, its scores on every test window
# against the Expanding Square, and its maps. scripts/run_experiments.sh submits one per run,
# after that run's training; to rerun one by hand, from the repository root:
#     sbatch --export=ALL,NAME=run_heading72,RUNS=/home/26p67/data/derived/rl/runs,\
#WINDOWS=/home/26p67/data/derived/rl/windows_arr30-60-120-180,ARRIVALS="30 60 120 180",\
#PATIENCE=10,THRESHOLD=0.001,FIGURES=figures/produced_maps scripts/wrap_run.sbatch

set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

PY="${SAR_PYTHON:-/home/26p67/envs/sar-rl/bin/python}"
: "${NAME:?pass NAME, the run folder}" "${RUNS:?pass RUNS}" "${WINDOWS:?pass WINDOWS}"
: "${ARRIVALS:?pass ARRIVALS}" "${PATIENCE:?pass PATIENCE}" "${THRESHOLD:?pass THRESHOLD}"
FIGURES="${FIGURES:-figures/produced_maps}"
RUN="$RUNS/$NAME"
MODEL="$RUN/best_model.zip"
[ -f "$MODEL" ] || MODEL="$RUN/model.zip"
[ -f "$MODEL" ] || { echo "no model in $RUN: did its training finish?"; exit 1; }

echo "== $NAME 1/3 learning curves"
PYTHONPATH=src "$PY" scripts/plot_training.py --run "$RUN" --patience "$PATIENCE" --improve-threshold "$THRESHOLD"

echo "== $NAME 2/3 every test window, against the Expanding Square"
# shellcheck disable=SC2086
PYTHONPATH=src "$PY" -m sar.rl.fly windows --windows "$WINDOWS" --arrival-min $ARRIVALS \
    --searchers ppo expanding-square --model "$MODEL" --out "$RUN/test"

echo "== $NAME 3/3 maps into $FIGURES/$NAME"
PYTHONPATH=src "$PY" scripts/produce_maps.py --runs-dir "$RUNS" --windows "$WINDOWS" --models "$NAME" \
    --out "$FIGURES" > "$RUN/maps.txt"
echo "$(wc -l < "$RUN/maps.txt") maps written; their paths are in $RUN/maps.txt"
