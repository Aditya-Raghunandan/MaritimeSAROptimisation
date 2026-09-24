#!/usr/bin/env bash
#SBATCH --job-name=benchstep
#SBATCH --partition=jaguarcluster
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=logs/benchstep_%j.out
#SBATCH --error=logs/benchstep_%j.out

# Time one ensemble step on a compute node (scripts/bench_ensemble_step.py).
#
# WHY A SLURM JOB. The compute nodes are 2012 i7-3770s and the laptop is not, so a laptop
# number says nothing about cluster wall-clock. And nothing heavy runs on the head node
# (D017).
#
# WHY BENCH_TASKS. At 1e6 particles the step is memory-bound. BENCH_TASKS=4 runs four
# copies at once on ONE node, which is what a Slurm array packing four tasks per node
# does -- measured 24 Sep at 2.7x slower each, so a node is worth ~1.5 cores, not 4.
#
# Submit from the repo checkout on the cluster:
#     sbatch scripts/bench_ensemble_step.sbatch
#     BENCH_TASKS=4 sbatch --ntasks=4 scripts/bench_ensemble_step.sbatch

set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

PY="${SAR_PYTHON:-/home/26p67/envs/sar/bin/python}"
[ -x "$PY" ] || { echo "no interpreter at $PY -- see D017 on micromamba"; exit 1; }
export PY
TASKS="${BENCH_TASKS:-1}"

lscpu | grep -E "Model name|^CPU\(s\)"
if [ "$TASKS" -eq 1 ]; then
    PYTHONPATH=src "$PY" scripts/bench_ensemble_step.py
else
    # $PY and $SLURM_PROCID are expanded inside each task, not here.
    srun --ntasks="$TASKS" --cpus-per-task=1 bash -c \
        'PYTHONPATH=src "$PY" scripts/bench_ensemble_step.py --particles 1000000 | tail -1 | sed "s/^/task $SLURM_PROCID: /"'
fi
