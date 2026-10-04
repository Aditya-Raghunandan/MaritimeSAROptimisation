#!/usr/bin/env bash
#SBATCH --job-name=mc-training
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.out

set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

PY="${SAR_PYTHON:-/home/26p67/envs/sar/bin/python}"
DATA="${SAR_DATA:-/home/26p67/data}"
BUOYS="${BUOYS:-scripts/scenario_buoys.txt}"
FIRST="${FIRST:-2019-01-01}"
LAST="${LAST:-2024-12-31}"
N="${N:-10000}"
SIGMA="${SIGMA:-26.3}"
BASE_SEED="${BASE_SEED:-20261004}"
K="${SLURM_ARRAY_TASK_ID:?submit with --array}"

IFS=';' read -r LAT LON < <(sed -n "${K}p" "$BUOYS" | tr -d '\r') || true
[ -n "${LAT:-}" ] || { echo "no line $K in $BUOYS"; exit 1; }

DAY="$FIRST"
WEEK=0
while [[ ! "$DAY" > "$LAST" ]]; do
    START="${DAY}T17:00:00"
    SEED=$((BASE_SEED + K * 1000 + WEEK))
    OUT="data/training/montes/ensemble/${DAY:0:4}"
    mkdir -p "$OUT"
    echo "========= buoy $K at $LAT;$LON, start $START UTC, seed $SEED ========="
    PYTHONPATH=src "$PY" -m sar.pipeline.ensemble --forcing-dir "$DATA" \
        --lat "$LAT" --lon "$LON" --datum-sigma-km 0 --start "$START" \
        --timestep 60 --duration 4h --particles "$N" --sigma "$SIGMA" \
        --seed "$SEED" --save-every 5m --out "$OUT" \
        || echo "buoy $K $START failed or already done, moving on"
    DAY=$(date -d "$DAY + 7 days" +%F)
    WEEK=$((WEEK + 1))
    echo "DONE ========= buoy $K at $LAT;$LON, start $START UTC, seed $SEED ========="
done
echo "buoy $K done: $WEEK weeks"
