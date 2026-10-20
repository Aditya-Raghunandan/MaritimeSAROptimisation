#!/usr/bin/env bash
#SBATCH --job-name=wrap_c2
#SBATCH --partition=jaguarcluster
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=logs/c2/wrap_%j.out
#SBATCH --error=logs/c2/wrap_%j.out

# The wrap-up for the c2 run: curves, every c2 test window against the Expanding Square, maps, and
# the breakdown by arrival time and place. scripts/c2/run_experiments.sh submits it, passing NAME,
# RUNS, WINDOWS, ARRIVALS, PATIENCE, THRESHOLD, FIGURES and COMPARE_RUNS (other runs, only read).

set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

PY="${SAR_PYTHON:-/home/26p67/envs/sar-rl/bin/python}"
: "${NAME:?pass NAME, the run folder}" "${RUNS:?pass RUNS}" "${WINDOWS:?pass WINDOWS}"
: "${ARRIVALS:?pass ARRIVALS}" "${PATIENCE:?pass PATIENCE}" "${THRESHOLD:?pass THRESHOLD}"
FIGURES="${FIGURES:-figures/produced_maps_c2}"
RUN="$RUNS/$NAME"
MODEL="$RUN/best_model.zip"
[ -f "$MODEL" ] || MODEL="$RUN/model.zip"
[ -f "$MODEL" ] || { echo "no model in $RUN: did its training finish?"; exit 1; }

echo "== $NAME 1/4 learning curves"
PYTHONPATH=src "$PY" scripts/plot_training.py --run "$RUN" --patience "$PATIENCE" --improve-threshold "$THRESHOLD"

echo "== $NAME 2/4 every c2 test window, against the Expanding Square"
# shellcheck disable=SC2086
PYTHONPATH=src "$PY" -m sar.rl.fly windows --windows "$WINDOWS" --arrival-min $ARRIVALS \
    --searchers ppo expanding-square --model "$MODEL" --out "$RUN/test"

echo "== $NAME 3/4 maps into $FIGURES/$NAME"
PYTHONPATH=src "$PY" scripts/produce_maps.py --runs-dir "$RUNS" --windows "$WINDOWS" --models "$NAME" \
    --out "$FIGURES" > "$RUN/maps.txt"
echo "$(wc -l < "$RUN/maps.txt") maps written; their paths are in $RUN/maps.txt"

echo "== $NAME 4/4 breakdown by arrival time and place into $RUNS/performance_breakdown.{csv,xlsx}"
# Each run is scored on its own windows, so c2 beside v1 compares leads over the square, not raw POS.
BREAKDOWN=("$RUN")
for other in ${COMPARE_RUNS:-}; do
    if [ -f "$other/test/flights.csv" ]; then BREAKDOWN+=("$other"); else echo "   skipping $other: no test/flights.csv yet"; fi
done
PYTHONPATH=src "$PY" scripts/performance_breakdown.py --runs "${BREAKDOWN[@]}" --out "$RUNS/performance_breakdown"
