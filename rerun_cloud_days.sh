#!/usr/bin/env bash
#SBATCH --job-name=cloud_days
#SBATCH --partition=jaguarcluster
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=logs/cloud_days_%j.out
#SBATCH --error=logs/cloud_days_%j.out
# Re-runs the 50 Monte Carlo runs of Table D.clouds (5 sites x 10 days) with the study's settings and seed,
# into $OUT, then measures them with cloud_stats.py. From the repository root: sbatch rerun_cloud_days.sh
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY="${SAR_PYTHON:-/home/26p67/envs/sar/bin/python}"
DATA="${SAR_DATA:-/home/26p67/data}"
OUT="${OUT:-$HOME/cloud_days}"
DAYS="2019-01-29 2019-07-30 2020-01-28 2020-07-28 2021-01-26 2021-07-27 2022-01-25 2022-07-26 2023-01-31 2023-07-25"
mkdir -p "$OUT" logs

while IFS=';' read -r LAT LON NAME; do
    [ -n "$LAT" ] || continue
    for DAY in $DAYS; do
        echo "== $NAME $DAY"
        PYTHONPATH=src "$PY" -m sar.pipeline.ensemble --forcing-dir "$DATA" \
            --lat "$LAT" --lon "$LON" --datum-sigma-km 0 --start "${DAY}T17:00:00" \
            --timestep 60 --duration 4h --particles 10000 \
            --seed 20261008 --save-every 5m --out "$OUT" \
            || echo "$NAME $DAY: the Monte Carlo failed, moving on"
    done
done < <(tr -d '\r' < scripts/rl_locations.txt)

"$PY" cloud_stats.py --montes "$OUT" --windows "$DATA/derived/rl/windows_arr30-60-120-180" \
    --out "$OUT/cloud_stats.csv"
