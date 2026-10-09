#!/usr/bin/env bash
# Adds PPO's packages (sar.rl) to the cluster's micromamba environment, then proves a training run
# starts on a compute node. No sudo: the environment is in the home folder. Run from the repo root:
#     bash scripts/setup_rl_env.sh
# Safe to rerun. --freeze-installed leaves every package the pipeline already uses as it is.

set -euo pipefail

ENV="${SAR_ENV:-/home/26p67/envs/sar}"
WINDOWS="${WINDOWS:-/home/26p67/data/derived/rl/windows}"
PARTITION="${PARTITION:-jaguarcluster}"
TORCH=2.13.0      # the newest CPU build on conda-forge on 9 Oct 2026; pip has 2.14.1
SB3=2.9.0
GYM=1.3.0

fail() { echo; echo "FAILED: $*"; exit 1; }

[ -f src/sar/rl/train.py ] || fail "run this from the repository root (the folder with src/ in it)"
[ -x "$ENV/bin/python" ] || fail "no environment at $ENV"

MAMBA="${MAMBA_EXE:-}"
for candidate in "$(command -v micromamba 2>/dev/null || true)" "$HOME/bin/micromamba" \
                 "$HOME/.local/bin/micromamba" "$HOME/micromamba/bin/micromamba"; do
    [ -n "$MAMBA" ] && break
    [ -n "$candidate" ] && [ -x "$candidate" ] && MAMBA="$candidate"
done
[ -n "$MAMBA" ] || fail "micromamba not found; set MAMBA_EXE=/path/to/micromamba and rerun"

echo "== 1/4 installing pytorch-cpu $TORCH, stable-baselines3 $SB3, gymnasium $GYM into $ENV"
"$MAMBA" install -y -p "$ENV" --override-channels -c conda-forge --freeze-installed \
    "pytorch-cpu=$TORCH" "stable-baselines3=$SB3" "gymnasium=$GYM" \
    || fail "the install did not solve without changing packages already in $ENV; nothing was changed. Send this output to Claude"

CHECK='import torch, gymnasium, stable_baselines3 as sb3
assert torch.version.cuda is None, "a CUDA build of torch was installed"
print("torch", torch.__version__, "| gymnasium", gymnasium.__version__, "| stable-baselines3", sb3.__version__)'

echo "== 2/4 importing on this node ($(hostname))"
"$ENV/bin/python" -c "$CHECK" || fail "the packages do not import on $(hostname)"

echo "== 3/4 importing on a compute node (waits for a slot in $PARTITION)"
srun --partition="$PARTITION" --time=00:05:00 --mem=2G "$ENV/bin/python" -c "$CHECK" \
    || fail "the packages do not import on a compute node"

echo "== 4/4 a 256-step training run on a compute node, on the real windows"
SMOKE="$(dirname "$WINDOWS")/smoke_$$"
trap 'rm -rf "$SMOKE"' EXIT
PYTHONPATH=src srun --partition="$PARTITION" --time=00:15:00 --mem=8G "$ENV/bin/python" -m sar.rl.train \
    --windows "$WINDOWS" --validation-every 8 --arrival-min 30 45 60 --headings 36 \
    --timesteps 256 --envs 1 --seed 1 --eval-every 100000000 --out "$SMOKE" \
    || fail "the training run did not start; the traceback above says why"

echo
echo "ALL GOOD. Submit training with:"
echo "  sbatch scripts/train_ppo.sbatch --windows $WINDOWS --validation-every 8 --arrival-min 30 45 60 --headings 36 --timesteps 3000000 --envs 4 --seed 1 --eval-every 100000 --out $(dirname "$WINDOWS")/runs"
