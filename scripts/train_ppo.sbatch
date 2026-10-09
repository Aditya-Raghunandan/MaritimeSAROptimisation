#!/usr/bin/env bash
#SBATCH --job-name=ppo
#SBATCH --partition=jaguarcluster
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=logs/ppo_%j.out
#SBATCH --error=logs/ppo_%j.out

# PPO training (sar.rl.train); every argument is passed through, and --envs should match --cpus-per-task.
#     sbatch scripts/train_ppo.sbatch --windows /home/26p67/data/derived/rl/windows \
#         --validation-every 8 \
#         --arrival-min 30 45 60 --headings 36 --timesteps 2000000 --envs 4 --seed 1 \
#         --eval-every 100000 --out /home/26p67/data/derived/rl/runs

set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

PY="${SAR_PYTHON:-/home/26p67/envs/sar-rl/bin/python}"   # PPO's own environment, not the pipeline's
[ -x "$PY" ] || { echo "no interpreter at $PY -- run: bash scripts/setup_rl_env.sh"; exit 1; }
"$PY" -c "import stable_baselines3" 2>/dev/null \
    || { echo "stable_baselines3 is not installed in $PY -- run: bash scripts/setup_rl_env.sh"; exit 1; }

PYTHONPATH=src "$PY" -m sar.rl.train "$@"
