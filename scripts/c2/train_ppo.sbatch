#!/usr/bin/env bash
#SBATCH --job-name=ppo_c2
#SBATCH --partition=jaguarcluster
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=logs/c2/ppo_%j.out
#SBATCH --error=logs/c2/ppo_%j.out

# PPO v1 training (sar.rl.train) on the c2 windows; every argument is passed through. scripts/c2/run_experiments.sh submits it.

set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

PY="${SAR_PYTHON:-/home/26p67/envs/sar-rl/bin/python}"
[ -x "$PY" ] || { echo "no interpreter at $PY -- run: bash scripts/setup_rl_env.sh"; exit 1; }

PYTHONPATH=src "$PY" -m sar.rl.train "$@"
