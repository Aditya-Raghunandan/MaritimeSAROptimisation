#!/usr/bin/env bash
#SBATCH --job-name=sigma89
#SBATCH --partition=jaguarcluster
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=06:00:00
#SBATCH --output=logs/sigma89_%A_%a.out
#SBATCH --error=logs/sigma89_%A_%a.out

# The sigma calibration (#89, sar.validate.calibrate_sigma) on a compute node.
#
# WHY A SLURM ARRAY. Stage 1 is ~1,400 engine runs, one per start day, and the ladder is
# five times that with 1,000 particles per window. Each array task takes every M-th start
# day (task_batches), so every task sees every season and they finish together. Nothing
# heavy runs on the head node (D017).
#
# The first argument is the stage; the rest go to it. The run stages get --data and their
# task number from the array:
#     sbatch scripts/calibrate_sigma.sbatch windows --out $D/windows
#     sbatch --array=0-31%24 scripts/calibrate_sigma.sbatch stage1 --windows $D/windows --out $D/stage1
#     sbatch scripts/calibrate_sigma.sbatch fit --windows $D/windows --stage1 $D/stage1 --out $D/fit.json
#     sbatch --array=0-31%24 scripts/calibrate_sigma.sbatch ladder --windows $D/windows \
#         --sigmas 25 35 50 70 100 --out $D/ladder
#     sbatch --array=0-31%24 scripts/calibrate_sigma.sbatch twin --windows $D/windows --out $D/twin
#     sbatch scripts/calibrate_sigma.sbatch calibrate --windows $D/windows --fit $D/fit.json \
#         --ladder $D/ladder --twin $D/twin --out $D/calibration.json

set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

PY="${SAR_PYTHON:-/home/26p67/envs/sar/bin/python}"
DATA="${SAR_DATA:-/home/26p67/data}"
[ -x "$PY" ] || { echo "no interpreter at $PY -- see D017 on micromamba"; exit 1; }

STAGE="$1"; shift
TASK=()
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    TASK=(--task "$SLURM_ARRAY_TASK_ID" --tasks "${SIGMA_TASKS:-$SLURM_ARRAY_TASK_COUNT}")
fi

echo "stage $STAGE on $(hostname), task ${SLURM_ARRAY_TASK_ID:-none}, commit $(git rev-parse --short HEAD)"
case "$STAGE" in
    windows)
        PYTHONPATH=src "$PY" -m sar.validate.drift_windows --data "$DATA" "$@" ;;
    stage1|ladder|twin)
        PYTHONPATH=src "$PY" -m sar.validate.calibrate_sigma "$STAGE" --data "$DATA" "${TASK[@]}" "$@" ;;
    *)
        PYTHONPATH=src "$PY" -m sar.validate.calibrate_sigma "$STAGE" "$@" ;;
esac
