#!/usr/bin/env bash
#SBATCH --job-name=truthwin
#SBATCH --partition=jaguarcluster
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --exclude=jaguar11
#SBATCH --output=logs/truthwin_%A_%a.out
#SBATCH --error=logs/truthwin_%A_%a.out

# Real buoys' starts as training windows (scripts/truth_windows.py build): four processes a
# node, process p of task k builds every (4 * TASKS)-th row from 4 * (k - 1) + p.
#     OUT=$DATA/derived/rl/windows_truth TASKS=12 sbatch --array=1-12 scripts/truth_windows.sbatch

set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY="${SAR_PYTHON:-/home/26p67/envs/sar-bench/bin/python}"
DATA="${SAR_DATA:-/home/26p67/data}"
OUT="${OUT:?pass OUT, the windows folder holding rows_train.csv and rows_validation.csv}"
TASKS="${TASKS:?pass TASKS, the array size}"
K="${SLURM_ARRAY_TASK_ID:?submit with --array}"
for P in 0 1 2 3; do
    PYTHONPATH=src "$PY" scripts/truth_windows.py build --forcing-dir "$DATA" --out "$OUT" \
        --part $(( 4 * (K - 1) + P )) --parts $(( 4 * TASKS )) --particles "${N:-5000}" &
done
wait
echo "task $K done"
