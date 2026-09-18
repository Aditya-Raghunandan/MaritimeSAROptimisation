#!/usr/bin/env bash
#SBATCH --job-name=hycompull
#SBATCH --partition=jaguarcluster
#SBATCH --array=0-9%3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=logs/hycom_%A_%a.out
#SBATCH --error=logs/hycom_%A_%a.out

# Pull the five study years of HYCOM surface current as gridded NetCDF,
# one HALF-YEAR per array task. Closes the gap that #12 opened the path for.
#
# WHY HALF-YEARS AND NOT YEARS. Measured on jaguar11, 2026-09-18, against the
# live OPeNDAP server rather than estimated:
#
#     1 day   (8 steps)   46.9 s   5.9 s/step   (includes connection setup)
#     4 days  (32 steps)  128.7 s  4.0 s/step   <- the rate that matters
#
# At 4.0 s/step the whole archive is 14,608 steps = 16.3 HOURS serial. A year
# is 3.3 h, a half-year 1.6 h. Half-years because a failed task then costs
# 1.6 h rather than 3.3, and because 10 tasks at %3 finishes in about 5.4 h
# against 6.6 h for 5 tasks -- the array is bound by the concurrency cap, so
# smaller pieces pack better.
#
# WHY %3 AND NOT ALL TEN AT ONCE. tds.hycom.org is a public THREDDS server
# run for everyone, not a resource we have been allocated. Three concurrent
# readers is a fair share; ten is a way to get rate-limited and to deserve it.
# The wind pull settled on the same cap for the same reason.
#
# SIZE. 0.885 MB per timestep measured, so a half-year is about 1.3 GB and the
# whole archive 12.9 GB. That sits under write_current_netcdf's 3 GB refuse-
# without---force threshold, so no task needs --force -- which is the point of
# choosing a piece size below the guard rather than switching the guard off.
# /home has ~473 GB free.
#
# Submit from ~/SAR/viz-staging:
#     sbatch scripts/pull_current_years.sbatch

set -euo pipefail

PY="${SAR_PYTHON:-/home/26p67/envs/sar/bin/python}"
[ -x "$PY" ] || { echo "no interpreter at $PY -- see D017 on micromamba"; exit 1; }

STAGE="${SAR_STAGE:-/home/26p67/SAR/viz-staging}"
OUT="${SAR_DATA:-/home/26p67/data}"

# Half-year boundaries across the D014 window, 2019-01-01 -> 2024-01-01.
# Start inclusive, end exclusive, so consecutive pieces do not overlap and the
# concatenated archive has no duplicate timestamps.
BOUNDS=(2019-01-01 2019-07-01 2020-01-01 2020-07-01 2021-01-01 2021-07-01 \
        2022-01-01 2022-07-01 2023-01-01 2023-07-01 2024-01-01)

i="${SLURM_ARRAY_TASK_ID:-0}"
START="${BOUNDS[$i]}"
END="${BOUNDS[$((i + 1))]}"

mkdir -p logs

echo "host      $(hostname)"
echo "task      $i  of ${#BOUNDS[@]}"
echo "window    $START -> $END  (end exclusive)"
echo "writing   $OUT/raw"
echo "started   $(date -Is)"
echo

cd "$STAGE"
PYTHONPATH="$STAGE/src" "$PY" -u -m sar.fetch.current \
    --start "$START" --end "$END" --out "$OUT"

echo
echo "finished  $(date -Is)"
