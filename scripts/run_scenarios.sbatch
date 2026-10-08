#!/usr/bin/env bash
#SBATCH --job-name=scenarios
#SBATCH --partition=jaguarcluster
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/scenarios_%A_%a.out
#SBATCH --error=logs/scenarios_%A_%a.out

# One Monte Carlo per row of the scenario table (scripts/pick_scenario_buoys.py), then its
# maps. The array index is the row: S01 is 1. Submit from the repo checkout:
#
#     sbatch --array=1-50 scripts/run_scenarios.sbatch              # the 50, N = 10,000
#     N=1000 sbatch --array=1-40 scripts/run_scenarios.sbatch       # training rows at N = 1,000
#     sbatch --array=51 scripts/run_scenarios.sbatch                # a spare, if a row failed
#
# Every run starts all particles at the buoy's real fix (D026), uses the engine's calibrated
# random term (D030: a random velocity with memory, matched at 4 h) and the row's own seed,
# so every strategy can search the same cloud. SIGMA_U overrides sigma_u; SIGMA=26.3 runs
# the old random walk instead (D028), for comparisons.

set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

PY="${SAR_PYTHON:-/home/26p67/envs/sar/bin/python}"
DATA="${SAR_DATA:-/home/26p67/data}"
CSV="${CSV:-$DATA/derived/scenarios/scenarios.csv}"
N="${N:-10000}"                  # particles
SAVE="${SAVE:-5m}"               # how often the cloud is written
DURATION="${DURATION:-4h}"       # the search is done by 3.7 h; the cloud holds ~90 % to 24 h (ADR005)
SIGMA_U="${SIGMA_U:-}"           # empty: the engine's default, CALIBRATED_SIGMA_U (D030)
SIGMA="${SIGMA:-}"               # set only to run the old random walk (D028)
if [ -n "$SIGMA" ]; then NOISE=(--sigma "$SIGMA"); elif [ -n "$SIGMA_U" ]; then NOISE=(--sigma-u "$SIGMA_U"); else NOISE=(); fi
MAPS="${MAPS:-1h 2h 3h 4h}"      # heatmaps to draw; empty for none
OUT="${OUT:-$DATA/derived/scenarios/runs_N${N}}"
K="${SLURM_ARRAY_TASK_ID:?submit with --array}"

read -r SCEN START LAT LON SEED < <(awk -F, -v k="$K" '
    { sub(/\r$/, "") }
    NR == 1 { for (i = 1; i <= NF; i++) c[$i] = i; next }
    NR == k + 1 { print $c["scenario"], $c["start"], $c["lat"], $c["lon"], $c["seed"] }' "$CSV")
[ -n "${SCEN:-}" ] || { echo "no row $K in $CSV"; exit 1; }
echo "$SCEN: start $START at $LAT, $LON; seed $SEED; N $N; every $SAVE for $DURATION"

PYTHONPATH=src "$PY" -m sar.pipeline.ensemble --forcing-dir "$DATA" \
    --lat "$LAT" --lon "$LON" --datum-sigma-km 0 --start "$START" \
    --timestep 60 --duration "$DURATION" --particles "$N" ${NOISE[@]+"${NOISE[@]}"} \
    --seed "$SEED" --save-every "$SAVE" --out "$OUT"

RUN=$(ls "$OUT"/derived/ensemble_*_N"${N}"_*_seed"${SEED}".csv | head -1)
for AT in $MAPS; do
    PYTHONPATH=src "$PY" scripts/ensemble_heatmap.py --csv "$RUN" --at "$AT" --cell-m 250 \
        --out "$OUT/maps/$SCEN"
done
echo "$SCEN done: $RUN"
