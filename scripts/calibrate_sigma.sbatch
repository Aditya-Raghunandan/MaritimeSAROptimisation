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
#     SIGMA_PROCS=4 sbatch --array=0-11 --cpus-per-task=4 scripts/calibrate_sigma.sbatch \
#         ladder --windows $D/windows --sigmas 25 35 50 70 100 --out $D/ladder
#     SIGMA_PROCS=4 sbatch --array=0-11 --cpus-per-task=4 scripts/calibrate_sigma.sbatch \
#         twin --windows $D/windows --out $D/twin
#     sbatch scripts/calibrate_sigma.sbatch calibrate --windows $D/windows --fit $D/fit.json \
#         --ladder $D/ladder --twin $D/twin --out $D/calibration.json
#
# At another horizon (D028's amendment, 4 Oct: 4 h), stage 1 is reused. fit, the ladder and
# calibrate take the hour, and calibrate refuses a fit made at a different one:
#     sbatch scripts/calibrate_sigma.sbatch fit --windows $D/windows --stage1 $D/stage1 \
#         --hour 4 --horizons 1 2 3 6 12 24 --beta-hours 1 6 --out $D/h4/fit.json
#     SIGMA_PROCS=4 sbatch --array=0-11 --cpus-per-task=4 scripts/calibrate_sigma.sbatch \
#         ladder --windows $D/windows --sigmas 12 15 19 22 25 28 32 40 --leads 1 2 3 4 6 \
#         --out $D/h4/ladder
#     sbatch scripts/calibrate_sigma.sbatch calibrate --windows $D/windows --hour 4 \
#         --fit $D/h4/fit.json --ladder $D/h4/ladder --out $D/h4/calibration.json
#
# The random velocity (D030, 6 Oct): the same ladder with --sigmas-u (m/s) and T_L fixed at
# 25.7 h; calibrate reads the model from the rows and adds the slide as a velocity. Then the
# confirmation at sigma_u final, both tiers, every horizon:
#     SIGMA_PROCS=4 sbatch --array=0-11 --cpus-per-task=4 scripts/calibrate_sigma.sbatch \
#         ladder --windows $D/windows --sigmas-u 0.12 0.15 0.18 0.21 0.24 0.27 0.32 0.40 \
#         --leads 1 2 3 4 6 --out $D/rv/ladder
#     sbatch scripts/calibrate_sigma.sbatch calibrate --windows $D/windows --hour 4 \
#         --fit $D/h4/fit.json --ladder $D/rv/ladder --out $D/rv/calibration.json
#     SIGMA_PROCS=4 sbatch --array=0-11 --cpus-per-task=4 scripts/calibrate_sigma.sbatch \
#         ladder --windows $D/windows --sigmas-u <final> --tiers undrogued drogued \
#         --leads 1 2 3 4 6 12 24 --out $D/rv/confirm
#     sbatch scripts/calibrate_sigma.sbatch calibrate --windows $D/windows --hour 4 \
#         --fit $D/h4/fit.json --ladder $D/rv/ladder --confirm $D/rv/confirm \
#         --out $D/rv/calibration.json

set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

PY="${SAR_PYTHON:-/home/26p67/envs/sar/bin/python}"
DATA="${SAR_DATA:-/home/26p67/data}"
[ -x "$PY" ] || { echo "no interpreter at $PY -- see D017 on micromamba"; exit 1; }

STAGE="$1"; shift
# SIGMA_PROCS processes per array task. This partition gives each task a whole node
# (measured 2 Oct: 32 one-CPU tasks ran 12 at a time, one per node), so a task that
# runs one process leaves three of the node's four cores idle. Submit with
# --cpus-per-task=$SIGMA_PROCS.
PROCS="${SIGMA_PROCS:-1}"
ID="${SLURM_ARRAY_TASK_ID:-0}"
COUNT="${SIGMA_TASKS:-${SLURM_ARRAY_TASK_COUNT:-1}}"

echo "stage $STAGE on $(hostname), task $ID of $COUNT x $PROCS, commit $(git rev-parse --short HEAD)"
case "$STAGE" in
    windows)
        PYTHONPATH=src "$PY" -m sar.validate.drift_windows --data "$DATA" "$@" ;;
    stage1|ladder|twin)
        for p in $(seq 0 $((PROCS - 1))); do
            PYTHONPATH=src "$PY" -m sar.validate.calibrate_sigma "$STAGE" --data "$DATA" \
                --task $((ID * PROCS + p)) --tasks $((COUNT * PROCS)) "$@" &
        done
        wait ;;
    *)
        PYTHONPATH=src "$PY" -m sar.validate.calibrate_sigma "$STAGE" "$@" ;;
esac
