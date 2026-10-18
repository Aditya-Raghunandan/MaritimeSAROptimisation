#!/usr/bin/env bash
# run_parts.sh CMD... -- one Slurm task as four processes of CMD, parts 4K to 4K+3 of 4 x TASKS.
#     TASKS=12 sbatch --array=1-12 --cpus-per-task=4 scripts/run_parts.sh python script.py ...
# Under bash on purpose: the same loop in sbatch --wrap runs under /bin/sh, whose `wait` left the
# finished processes unreaped and the task hanging until its time limit (jobs 59970, 59998, 18 Oct).
set -uo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src
: "${TASKS:?pass TASKS, the array size}"
K=${K:-$((SLURM_ARRAY_TASK_ID - 1))}
for P in 0 1 2 3; do "$@" --part $((4 * K + P)) --parts $((4 * TASKS)) & done
wait
echo "task $K done"
