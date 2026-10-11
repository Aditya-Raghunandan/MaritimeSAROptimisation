#!/usr/bin/env bash
#SBATCH --job-name=bundles
#SBATCH --partition=jaguarcluster
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/bundles_%A_%a.out
#SBATCH --error=logs/bundles_%A_%a.out

# The site's scenario bundles (scripts/export_scenario_bundles.py), four rows per node:
# task k exports rows 4k-3 to 4k. About 22 MB and 6 s a row. Submit from the checkout:
#
#     sbatch --array=1-14 scripts/export_scenario_bundles.sbatch
#     # when the array has finished:
#     python scripts/export_scenario_bundles.py index --out $OUT

set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

PY="${SAR_PYTHON:-/home/26p67/envs/sar/bin/python}"
DATA="${SAR_DATA:-/home/26p67/data}"
CSV="${CSV:-$DATA/derived/scenarios/scenarios.csv}"
OUT="${OUT:-$DATA/published/scenarios/v2}"     # format 2: the D032 helicopter
LAST="${LAST:-$(( $(wc -l < "$CSV") - 1 ))}"
K="${SLURM_ARRAY_TASK_ID:?submit with --array}"

for R in $(seq $((4 * K - 3)) $((4 * K))); do
    [ "$R" -le "$LAST" ] || continue
    PYTHONPATH=src "$PY" scripts/export_scenario_bundles.py export --csv "$CSV" \
        --forcing-dir "$DATA" --rows "$R" --out "$OUT" &
done
wait
echo "task $K done: $OUT"
