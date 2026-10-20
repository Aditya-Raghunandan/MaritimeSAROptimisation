#!/usr/bin/env bash
#SBATCH --job-name=wrap_v2
#SBATCH --partition=jaguarcluster
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=logs/v2/wrap_%j.out
#SBATCH --error=logs/v2/wrap_%j.out

# The wrap-up for one finished v2 run: learning curves, every test window against the Expanding
# Square, and its maps into figures/produced_maps_v2. scripts/v2/run_experiments.sh submits it
# after the run's training, passing NAME, RUNS, WINDOWS, ARRIVALS, PATIENCE, THRESHOLD, FIGURES.

set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

PY="${SAR_PYTHON:-/home/26p67/envs/sar-rl/bin/python}"
: "${NAME:?pass NAME, the run folder}" "${RUNS:?pass RUNS}" "${WINDOWS:?pass WINDOWS}"
: "${ARRIVALS:?pass ARRIVALS}" "${PATIENCE:?pass PATIENCE}" "${THRESHOLD:?pass THRESHOLD}"
FIGURES="${FIGURES:-figures/produced_maps_v2}"
RUN="$RUNS/$NAME"
MODEL="$RUN/best_model.zip"
[ -f "$MODEL" ] || MODEL="$RUN/model.zip"
[ -f "$MODEL" ] || { echo "no model in $RUN: did its training finish?"; exit 1; }

echo "== $NAME 1/3 learning curves"
PYTHONPATH=src "$PY" scripts/plot_training.py --run "$RUN" --patience "$PATIENCE" --improve-threshold "$THRESHOLD"

echo "== $NAME 2/3 every test window, against the Expanding Square"
# shellcheck disable=SC2086
PYTHONPATH=src "$PY" -m sar.rl_v2.fly windows --windows "$WINDOWS" --arrival-min $ARRIVALS \
    --searchers ppo expanding-square --model "$MODEL" --out "$RUN/test"

echo "== $NAME 3/3 maps into $FIGURES/$NAME"
PYTHONPATH=src "$PY" scripts/v2/produce_maps.py --runs-dir "$RUNS" --windows "$WINDOWS" --models "$NAME" \
    --out "$FIGURES" > "$RUN/maps.txt"
echo "$(wc -l < "$RUN/maps.txt") maps written; their paths are in $RUN/maps.txt"
