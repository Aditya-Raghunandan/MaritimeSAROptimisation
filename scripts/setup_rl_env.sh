#!/usr/bin/env bash
# Builds PPO's own micromamba environment (sar.rl) and proves a training run starts on a compute node.
# Separate from the pipeline's /home/26p67/envs/sar, which it never touches: installing torch there
# would replace that environment's pip-installed numpy and fsspec. No sudo. Run from the repo root:
#     bash scripts/setup_rl_env.sh
# Safe to rerun.

set -euo pipefail

ENV="${SAR_RL_ENV:-/home/26p67/envs/sar-rl}"
WINDOWS="${WINDOWS:-/home/26p67/data/derived/rl/windows}"
PARTITION="${PARTITION:-jaguarcluster}"
# torch 2.13.0 is the newest CPU build on conda-forge on 9 Oct 2026; numpy matches the pipeline's.
SPECS=(python=3.12 pytorch-cpu=2.13.0 stable-baselines3=2.9.0 gymnasium=1.3.0 numpy=2.5.3
       pandas xarray cftime pyarrow scipy)

fail() { echo; echo "FAILED: $*"; exit 1; }

[ -f src/sar/rl/train.py ] || fail "run this from the repository root (the folder with src/ in it)"
[ -d "$WINDOWS" ] || fail "no windows at $WINDOWS"

MAMBA="${MAMBA_EXE:-}"
for candidate in "$(command -v micromamba 2>/dev/null || true)" "$HOME/bin/micromamba" \
                 "$HOME/.local/bin/micromamba" "$HOME/micromamba/bin/micromamba"; do
    [ -n "$MAMBA" ] && break
    [ -n "$candidate" ] && [ -x "$candidate" ] && MAMBA="$candidate"
done
[ -n "$MAMBA" ] || fail "micromamba not found; set MAMBA_EXE=/path/to/micromamba and rerun"

if [ -x "$ENV/bin/python" ]; then
    echo "== 1/4 $ENV exists; making sure it has ${SPECS[*]}"
    "$MAMBA" install -y -p "$ENV" --override-channels -c conda-forge "${SPECS[@]}" \
        || fail "the install into $ENV did not solve; send the output above to Claude"
else
    echo "== 1/4 creating $ENV with ${SPECS[*]} (about 300 MB)"
    "$MAMBA" create -y -p "$ENV" --override-channels -c conda-forge "${SPECS[@]}" \
        || fail "creating $ENV did not solve; send the output above to Claude"
fi

CHECK='import sys, torch, gymnasium, stable_baselines3 as sb3, numpy, pandas, xarray
import sar.rl.train, sar.rl.env, sar.rl.fly
assert torch.version.cuda is None, "a CUDA build of torch was installed"
print("python", sys.version.split()[0], "| torch", torch.__version__, "| gymnasium", gymnasium.__version__,
      "| stable-baselines3", sb3.__version__, "| numpy", numpy.__version__, "| pandas", pandas.__version__,
      "| xarray", xarray.__version__)'

echo "== 2/4 importing on this node ($(hostname))"
PYTHONPATH=src "$ENV/bin/python" -c "$CHECK" || fail "the packages do not import on $(hostname)"

echo "== 3/4 importing on a compute node (waits for a slot in $PARTITION)"
PYTHONPATH=src srun --partition="$PARTITION" --time=00:05:00 --mem=2G "$ENV/bin/python" -c "$CHECK" \
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
