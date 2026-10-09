#!/usr/bin/env bash
#SBATCH --job-name=mc-training
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --array=1-5
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.out

# One location per array task (scripts/rl_locations.txt): a Monte Carlo every 7 days at 17:00 UTC,
# then its search windows for each transit time (sar.rl.windows). docs/ppo-search.md has the why.
#     sbatch scripts/training_montes.sh

set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

PY="${SAR_PYTHON:-/home/26p67/envs/sar/bin/python}"
DATA="${SAR_DATA:-/home/26p67/data}"
LOCATIONS="${LOCATIONS:-scripts/rl_locations.txt}"
FIRST="${FIRST:-2019-01-01}"     # five whole years inside HYCOM's 2018-12-04 to 2024-09-05
LAST="${LAST:-2023-12-31}"
N="${N:-10000}"
SEED="${SEED:-20261008}"         # one seed for every run, by choice
ARRIVALS="${ARRIVALS:-30 45 60}" # minutes from the call to arriving on scene
OUT="${OUT:-$DATA/derived/rl}"
KEEP_CSV="${KEEP_CSV:-1}"        # 0 deletes each run's CSV once its windows file is written
K="${SLURM_ARRAY_TASK_ID:?submit with --array}"

IFS=';' read -r LAT LON NAME < <(sed -n "${K}p" "$LOCATIONS" | tr -d '\r') || true
[ -n "${LAT:-}" ] || { echo "no line $K in $LOCATIONS"; exit 1; }
# The position as sar.pipeline.ensemble.run_name writes it, so the run's CSV can be found again.
POS=$(PYTHONPATH=src "$PY" -c 'import sys; from sar.pipeline.ensemble import run_name
print(run_name({"lat": round(float(sys.argv[1]), 2), "lon": round(float(sys.argv[2]), 2),
    "start": "2019-01-01T17:00", "particles": 1, "timestep": 60, "duration": 60,
    "seed": 0}).split("_")[1])' "$LAT" "$LON")

DAY="$FIRST"
WEEK=0
while [[ ! "$DAY" > "$LAST" ]]; do
    START="${DAY}T17:00:00"
    RUN="$OUT/montes/derived/ensemble_${POS}_${DAY//-/}T1700_N${N}_dt60s_T4h_seed${SEED}.csv"
    echo "========= $NAME ($LAT;$LON), start $START UTC ========="
    if [ ! -f "$RUN" ]; then
        PYTHONPATH=src "$PY" -m sar.pipeline.ensemble --forcing-dir "$DATA" \
            --lat "$LAT" --lon "$LON" --datum-sigma-km 0 --start "$START" \
            --timestep 60 --duration 4h --particles "$N" \
            --seed "$SEED" --save-every 5m --out "$OUT/montes" \
            || echo "$NAME $START: the Monte Carlo failed, moving on"
    fi
    if [ -f "$RUN" ]; then
        # shellcheck disable=SC2086
        PYTHONPATH=src "$PY" -m sar.rl.windows --run "$RUN" --forcing-dir "$DATA" \
            --arrival-min $ARRIVALS --out "$OUT/windows" \
            && { [ "$KEEP_CSV" = 1 ] || rm -f "$RUN" "${RUN%.csv}.json"; } \
            || echo "$NAME $START: no windows written (already there, or the forcing has a gap)"
    fi
    DAY=$(date -d "$DAY + 7 days" +%F)
    WEEK=$((WEEK + 1))
done
echo "$NAME done: $WEEK weeks"
