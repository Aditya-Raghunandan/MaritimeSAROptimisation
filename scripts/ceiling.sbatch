#!/usr/bin/env bash
#SBATCH --job-name=ceiling
#SBATCH --partition=jaguarcluster
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --exclude=jaguar11
#SBATCH --output=logs/ceiling_%A_%a.out
#SBATCH --error=logs/ceiling_%A_%a.out

# The ceiling over a scenario table (scripts/ceiling.py run). Slurm gives a whole node per
# task, so each task runs four processes of PER rows: task k takes rows 4*PER*(k-1)+1 to
# 4*PER*k. Submit from the checkout:
#
#     sbatch --array=1-4 scripts/ceiling.sbatch                  # the 55, PER=4
#     ARRIVALS="0.25 0.5 0.75 1 2 3" N=100000 OUT=... sbatch --array=1-4 scripts/ceiling.sbatch
#     CSV=... NOISE=rvsc EFFORTS="1 2 4" LEAN=1 PER=8 OUT=... sbatch --array=1-N scripts/ceiling.sbatch

set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

PY="${SAR_PYTHON:-/home/26p67/envs/sar-bench/bin/python}"
DATA="${SAR_DATA:-/home/26p67/data}"
CSV="${CSV:-$DATA/derived/scenarios/scenarios.csv}"
N="${N:-100000}"
ARRIVALS="${ARRIVALS:-1 2 3}"
NOISE="${NOISE:-rvc}"
EFFORTS="${EFFORTS:-1}"
EXTRA=()
[ "${LEAN:-0}" = 1 ] && EXTRA=(--lean)
PER="${PER:-4}"
LAST="${LAST:-$(( $(wc -l < "$CSV") - 1 ))}"
OUT="${OUT:?pass OUT, the results folder}"
K="${SLURM_ARRAY_TASK_ID:?submit with --array}"

for P in 0 1 2 3; do
    A=$(( 4 * PER * (K - 1) + P * PER + 1 ))
    B=$(( A + PER - 1 ))
    [ "$A" -le "$LAST" ] || continue
    [ "$B" -le "$LAST" ] || B=$LAST
    PYTHONPATH=src "$PY" scripts/ceiling.py run --csv "$CSV" --forcing-dir "$DATA" \
        --rows "$A-$B" --arrival-h $ARRIVALS --particles "$N" --noise "$NOISE" \
        --efforts $EFFORTS "${EXTRA[@]}" --out "$OUT" &
done
wait
echo "task $K done: $OUT"
