#!/usr/bin/env bash
#SBATCH --job-name=fieldzarr
#SBATCH --partition=jaguarcluster
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=logs/fieldzarr_%j.out
#SBATCH --error=logs/fieldzarr_%j.out

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
#     sbatch scripts/publish_wind_archive.sbatch                    # wind
#     SAR_PRODUCT=current sbatch scripts/publish_wind_archive.sbatch  # current
#
# WHY 24 GB AND NOT 16. The current grid is 476 x 238 against wind's 77 x 77,
# so the 3-hourly tier is 6.6 GB per variable and 13.2 GB with water_u and
# water_v both live. write_zarr_tier chunks with dask so the write streams and
# peak memory does not scale with the archive length, but the nodes only have
# 31 GB and the source is being read at the same time -- 24 GB is headroom, not
# a requirement.

set -euo pipefail

PY="${SAR_PYTHON:-/home/26p67/envs/sar/bin/python}"
[ -x "$PY" ] || { echo "no interpreter at $PY -- see D017 on micromamba"; exit 1; }

STAGE="${SAR_STAGE:-/home/26p67/SAR/viz-staging}"
DATA="${SAR_DATA:-/home/26p67/data}"
OUT="${SAR_OUT:-$DATA/published}"
PRODUCT="${SAR_PRODUCT:-wind}"
# HYCOM really is missing 7 timesteps in 14,608 (D011). --fill-gaps puts them
# back as NaN so the axis is regular; nothing is interpolated. Harmless for
# wind, whose axis is already gap-free.
FILL="${SAR_FILL_GAPS:---fill-gaps}"

mkdir -p logs "$OUT"

echo "host      $(hostname)"
echo "python    $("$PY" -V)"
echo "product   $PRODUCT"
echo "reading   $DATA/raw"
echo "writing   $OUT"
echo "started   $(date -Is)"
echo

cd "$STAGE"
PYTHONPATH="$STAGE/src" "$PY" -u -m sar.viz.archive \
    --data "$DATA" --out "$OUT" --product "$PRODUCT" $FILL

echo
echo "finished  $(date -Is)"
du -sh "$OUT"
