#!/usr/bin/env bash
#SBATCH --job-name=hycompull
#SBATCH --partition=jaguarcluster
#SBATCH --array=0-59%3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --output=logs/hycom_%A_%a.out
#SBATCH --error=logs/hycom_%A_%a.out

# Pull the five study years of HYCOM surface current as gridded NetCDF,
# ONE MONTH per array task. Sixty tasks, three at a time.
#
# THE FIRST TWO ATTEMPTS FAILED, AND THE REASON IS WORTH KEEPING.
#
# Attempt 1 (array 58644) asked for a HALF-YEAR in one OPeNDAP request -- 1,448
# timesteps -- and six tasks died after ~36 min each having written nothing:
#
#     500 java.net.SocketTimeoutException: Read timed out;
#         water_u -- 8981:10428,0:0,2425:2900,3475:3712
#
# The throughput measured beforehand (4.0 s/timestep over a 4-day request) was
# correct and gave no warning, because the RATE and the LARGEST ANSWERABLE
# REQUEST are independent limits and only the rate had been measured. A scan of
# 1,600 individual days found ZERO unreadable, which is what proves the archive
# is complete and the fault was entirely in how it was asked for.
#
# Attempt 2 chunked at 8 days (64 steps, 29 MB/variable) and STILL timed out;
# 4 days (32 steps, 14.5 MB) had gone through in 128 s. Requests are now 2 days
# -- half the largest size seen to work -- because throughput from this
# endpoint varies by almost an order of magnitude between runs (4.0 s/timestep
# clean, 35 s on the first 3-day pull), and a size that only just works on a
# good day will not survive a bad one.
#
# HENCE MONTHS, AND HENCE RESUME. A month is ~248 timesteps and ~220 MB, so a
# lost task costs one month rather than six. write_current_netcdf SKIPS a
# window whose file already exists, so this array can simply be resubmitted
# until the archive is complete and it will only fetch what is missing. Against
# a server this erratic that matters more than picking the right size.
#
# %3 because tds.hycom.org is a public server run for everyone, not an
# allocation we hold -- and because we have already leaned on it hard today.
#
# Submit from ~/SAR/viz-staging, and resubmit freely:
#     sbatch scripts/pull_current_years.sbatch

set -euo pipefail

PY="${SAR_PYTHON:-/home/26p67/envs/sar/bin/python}"
[ -x "$PY" ] || { echo "no interpreter at $PY -- see D017 on micromamba"; exit 1; }

STAGE="${SAR_STAGE:-/home/26p67/SAR/viz-staging}"
OUT="${SAR_DATA:-/home/26p67/data}"

# One month per task across the D014 window, 2019-01-01 -> 2024-01-01.
# Start inclusive, end exclusive, so consecutive months never overlap.
i="${SLURM_ARRAY_TASK_ID:-0}"
YEAR=$(( 2019 + i / 12 ))
MONTH=$(( i % 12 + 1 ))
START=$(printf "%04d-%02d-01" "$YEAR" "$MONTH")
if [ "$MONTH" -eq 12 ]; then
    END=$(printf "%04d-01-01" $(( YEAR + 1 )))
else
    END=$(printf "%04d-%02d-01" "$YEAR" $(( MONTH + 1 )))
fi

mkdir -p logs

echo "host      $(hostname)"
echo "task      $i  of 60"
echo "window    $START -> $END  (end exclusive)"
echo "writing   $OUT/raw"
echo "started   $(date -Is)"
echo

cd "$STAGE"
PYTHONPATH="$STAGE/src" "$PY" -u -m sar.fetch.current \
    --start "$START" --end "$END" --out "$OUT"

echo
echo "finished  $(date -Is)"
