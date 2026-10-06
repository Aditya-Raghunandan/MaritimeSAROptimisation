#!/usr/bin/env bash
#SBATCH --job-name=score
#SBATCH --partition=jaguarcluster
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/score_%A_%a.out
#SBATCH --error=logs/score_%A_%a.out

# The doctrinal patterns over the scenario table, scored by the referee
# (scripts/score_scenarios.py). Slurm gives a whole node per task, so each task scores four
# rows at once: task k takes rows 4k-3 to 4k. Submit from the repo checkout:
#
#     sbatch --array=1-14 scripts/score_scenarios.sbatch                 # rows 1-55, both noises
#     NOISE=rv ARRIVALS="2" sbatch --array=1-13 scripts/score_scenarios.sbatch

set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

PY="${SAR_PYTHON:-/home/26p67/envs/sar/bin/python}"
DATA="${SAR_DATA:-/home/26p67/data}"
CSV="${CSV:-$DATA/derived/scenarios/scenarios.csv}"
N="${N:-10000}"
NOISE="${NOISE:-rv rw}"
ARRIVALS="${ARRIVALS:-1 2 3}"
LAST="${LAST:-55}"
OUT="${OUT:-$DATA/derived/scenarios/scores_N${N}}"
K="${SLURM_ARRAY_TASK_ID:?submit with --array}"

for R in $(seq $((4 * K - 3)) $((4 * K))); do
    [ "$R" -le "$LAST" ] || continue
    PYTHONPATH=src "$PY" scripts/score_scenarios.py score --csv "$CSV" --forcing-dir "$DATA" \
        --rows "$R" --noise $NOISE --arrival-h $ARRIVALS --particles "$N" --out "$OUT" &
done
wait
echo "task $K done: $OUT"
