#!/usr/bin/env bash
#SBATCH --job-name=ppo_v3
#SBATCH --partition=jaguarcluster
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=logs/v3/ppo_%j.out
#SBATCH --error=logs/v3/ppo_%j.out

# PPO v3 training (sar.rl_v3.train); every argument is passed through, --envs should match
# --cpus-per-task. scripts/v3/run_experiments.sh submits it for run_baseline_v3.

set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

PY="${SAR_PYTHON:-/home/26p67/envs/sar-rl/bin/python}"
[ -x "$PY" ] || { echo "no interpreter at $PY -- run: bash scripts/setup_rl_env.sh"; exit 1; }

PYTHONPATH=src "$PY" -m sar.rl_v3.train "$@"
