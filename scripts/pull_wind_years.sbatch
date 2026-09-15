#!/usr/bin/env bash
#SBATCH --job-name=era5pull
#SBATCH --partition=jaguarcluster
#SBATCH --array=0-4%3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --output=logs/era5_%A_%a.out
#SBATCH --error=logs/era5_%A_%a.out

# Pull the five study years of ERA5 wind, one calendar year per array task.
#
# WHY AN ARRAY AND NOT ONE LONG JOB: a failure costs one year, not nine hours.
# It is also the shape every later workload takes -- one scenario per task for
# the drift ensembles, one seed per task for PPO (see the vault's
# `Parallelism - where it actually is`), so it is worth establishing now on a
# job we can afford to get wrong.
#
# WHY IT CAN RUN HERE AT ALL: measured 2026-09-15 on jaguar11 -- the private
# compute nodes DO have outbound HTTPS to storage.googleapis.com and
# tds.hycom.org. That was the last open question in D017 and it had been
# assumed either way for five days.
#
# %3 caps it at three tasks at once. Each task opens 32 concurrent connections
# (the pull is latency-bound, not bandwidth-bound -- one ARCO chunk is a whole
# global timestep, so subsetting our box does not reduce what crosses the wire).
# Five tasks unthrottled is 160 in-flight requests over a shared cluster uplink,
# which is rude rather than fast.
#
#   cd ~/SAR/MaritimeSAROptimisation && mkdir -p logs && sbatch scripts/pull_wind_years.sbatch
#
# Watch:  squeue -u $USER        Cancel: scancel <jobid>
# Output: /home/26p67/data/raw/era5_17-36N_82-63W_<year>0101-<year+1>0101.nc

set -euo pipefail

YEARS=(2019 2020 2021 2022 2023)
Y=${YEARS[$SLURM_ARRAY_TASK_ID]}
NEXT=$((Y + 1))

# NOT /data1. It exists but is root-owned and unwritable -- nothing is mounted
# on it. See the 2026-09-14 correction in D017.
OUT=/home/26p67/data

# NOT $(dirname "$0") -- Slurm copies the script to /var/spool/slurmd/job<N>/
# and runs it from there, so $0 points at the spool copy and the repo is nowhere
# near it. SLURM_SUBMIT_DIR is the directory sbatch was invoked from.
REPO="${SLURM_SUBMIT_DIR:-$HOME/SAR/MaritimeSAROptimisation}"
cd "$REPO"

# Call the interpreter by absolute path. `source .venv/bin/activate` does NOT
# reliably take effect under srun -- task 0 of job 58618 ran the system
# /usr/lib/python3.8 and died on `import dask`, while the same activate works
# fine over ssh on the head node. An absolute path has no PATH semantics to
# get wrong, and 3.8 is end-of-life and cannot install these pins anyway.
# NOT $REPO/.venv/bin/python. A venv does not contain an interpreter -- its
# bin/python is a SYMLINK to the system one, which is 3.12 on the head node and
# 3.8 on every compute node, so the same venv silently runs two different and
# incompatible Pythons depending on where the task lands. Three array attempts
# died on that before adding `python -V` made it visible.
#
# The micromamba environment carries its own 31 MB interpreter, and its packages
# come from conda-forge, which builds against glibc 2.17 -- the compute nodes are
# Ubuntu 20.04 with glibc 2.31, and modern pip wheels (cryptography's Rust
# extension in particular) require 2.33+. See the second correction in D017.
PY="${SAR_PYTHON:-/home/26p67/envs/sar/bin/python}"
[ -x "$PY" ] || { echo "no interpreter at $PY -- see D017 on micromamba"; exit 1; }

# The editable install resolves on the head node and NOT on the compute nodes,
# even though /home is the same NFS export on both -- `import sar` raises
# ModuleNotFoundError under srun while working fine under ssh. Rather than
# debug pip's editable-install machinery across NFS, put src on the path
# explicitly. One line, and it removes the dependency entirely.
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

echo "host      : $(hostname)"
echo "task      : ${SLURM_ARRAY_TASK_ID} of ${SLURM_ARRAY_TASK_COUNT:-?}"
echo "year      : ${Y}-01-01 .. ${NEXT}-01-01  (end exclusive)"
echo "out       : ${OUT}"
echo "started   : $(date -Is)"
# NOT df / -- on a compute node that is its own 7.3 G root, 95% full. The
# output goes to /home, which is 192.168.1.1:/export/home, 340 G free.
df -h "$OUT" | tail -1
echo

"$PY" -V
"$PY" -m sar.fetch.wind --start "${Y}-01-01" --end "${NEXT}-01-01" --out "${OUT}"

echo
echo "finished  : $(date -Is)"
ls -lh "${OUT}/raw/" | grep "${Y}" || true
