#!/usr/bin/env bash
#SBATCH --job-name=benchmc
#SBATCH --partition=jaguarcluster
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=logs/benchmc_%j.out
#SBATCH --error=logs/benchmc_%j.out

# Time full Monte Carlo runs on a compute node (scripts/bench_monte_carlo.py); every argument is passed through.
# The forcing is fetched live from HYCOM and ERA5, and whether compute nodes reach the internet is unmeasured (D017).
#
# Submit from the repo checkout on the cluster:
#     sbatch scripts/bench_monte_carlo.sbatch --lat 26.5 --lon -79 --datum-sigma-km 2 \
#         --start 2019-06-01T06:00 --timestep 60 --duration 4h --save-every 30m \
#         --sigma 0.1 --seed 20260927 --out /home/26p67/data

set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

PY="${SAR_PYTHON:-/home/26p67/envs/sar/bin/python}"
[ -x "$PY" ] || { echo "no interpreter at $PY -- see D017 on micromamba"; exit 1; }

lscpu | grep -E "Model name|^CPU\(s\)"
PYTHONPATH=src "$PY" scripts/bench_monte_carlo.py "$@"
