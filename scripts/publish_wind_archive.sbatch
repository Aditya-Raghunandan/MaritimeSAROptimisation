#!/usr/bin/env bash
#SBATCH --job-name=windzarr
#SBATCH --partition=jaguarcluster
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/windzarr_%j.out
#SBATCH --error=logs/windzarr_%j.out

# Publish the five-year ERA5 archive as multi-resolution Zarr (D021 step 4).
# Reads /home/26p67/data/raw/era5_*.nc, writes /home/26p67/data/published/.
#
# WHY THIS IS A SLURM JOB AND NOT A HEAD-NODE COMMAND. It reads 3.1 GB and
# zstd-19-compresses about 2 GB of it. That is only a few minutes of one core,
# so the temptation is to just run it -- which is exactly what happened on
# 2026-09-15, when a fetch on jaguar1 ran at 483 % CPU and made the login node
# unusable for the people whose login node it is. The rule earned that day is
# that shared head nodes run nothing but ssh and sbatch. See D017.
#
# WHY 16 GB. The exporter holds a whole variable in memory while writing it:
# 43,824 hourly steps x 77 x 77 x float32 is 1.04 GB per variable, and u10 and
# v10 are both live during the hourly tier. Measured peak should sit near 4 GB;
# 16 GB is slack, not a requirement, and the nodes have 31 GB.
#
# WHY NO NETWORK. Unlike the fetch jobs this one never leaves the filesystem --
# it reads local NetCDF and writes local Zarr. The upload to Hugging Face is a
# separate step (scripts/publish_archive.py) and is deliberately NOT here: it
# needs a credential, and a credential does not belong in a batch script.
#
# Submit from ~/SAR/viz-staging:
#     sbatch scripts/publish_wind_archive.sbatch

set -euo pipefail

PY="${SAR_PYTHON:-/home/26p67/envs/sar/bin/python}"
[ -x "$PY" ] || { echo "no interpreter at $PY -- see D017 on micromamba"; exit 1; }

STAGE="${SAR_STAGE:-/home/26p67/SAR/viz-staging}"
DATA="${SAR_DATA:-/home/26p67/data}"
OUT="${SAR_OUT:-$DATA/published}"

mkdir -p logs "$OUT"

echo "host      $(hostname)"
echo "python    $("$PY" -V)"
echo "reading   $DATA/raw"
echo "writing   $OUT"
echo "started   $(date -Is)"
echo

cd "$STAGE"
PYTHONPATH="$STAGE/src" "$PY" -u -m sar.viz.archive \
    --data "$DATA" --out "$OUT" --product wind

echo
echo "finished  $(date -Is)"
du -sh "$OUT"
