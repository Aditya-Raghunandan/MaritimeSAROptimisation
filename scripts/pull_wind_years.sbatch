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

cd "$(dirname "$0")/.."
source .venv/bin/activate

echo "host      : $(hostname)"
echo "task      : ${SLURM_ARRAY_TASK_ID} of ${SLURM_ARRAY_TASK_COUNT:-?}"
echo "year      : ${Y}-01-01 .. ${NEXT}-01-01  (end exclusive)"
echo "out       : ${OUT}"
echo "started   : $(date -Is)"
df -h / | tail -1
echo

python -m sar.fetch.wind --start "${Y}-01-01" --end "${NEXT}-01-01" --out "${OUT}"

echo
echo "finished  : $(date -Is)"
ls -lh "${OUT}/raw/" | grep "${Y}" || true
