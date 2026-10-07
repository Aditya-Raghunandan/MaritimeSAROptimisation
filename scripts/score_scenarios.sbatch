#!/usr/bin/env bash
#SBATCH --job-name=score
#SBATCH --partition=jaguarcluster
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/score_%A_%a.out
#SBATCH --error=logs/score_%A_%a.out

# Every searcher over a scenario table, scored by the referee (scripts/score_scenarios.py).
# Slurm gives a whole node per task, so each task runs four processes, each scoring PER
# consecutive rows: task k takes rows 4*PER*(k-1)+1 to 4*PER*k. Submit from the checkout:
#
#     sbatch --array=1-14 scripts/score_scenarios.sbatch                  # the 55, PER=1
#     CSV=$DATA/derived/scenarios/bench_dev_48h.csv PER=25 OUT=... \
#         sbatch --array=1-96 scripts/score_scenarios.sbatch              # 9,538 rows
#     SEARCHERS=greedy GREEDY_HEADINGS=8 GREEDY_DECIDE_S=10 OUT=... sbatch ...
#
# One OUT folder is one experiment: the scorer refuses flights made with other settings.

set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

PY="${SAR_PYTHON:-/home/26p67/envs/sar/bin/python}"
DATA="${SAR_DATA:-/home/26p67/data}"
CSV="${CSV:-$DATA/derived/scenarios/scenarios.csv}"
N="${N:-10000}"
NOISE="${NOISE:-rv rw}"
ARRIVALS="${ARRIVALS:-1 2 3}"
SEARCHERS="${SEARCHERS:-expanding-square sector parallel trackline greedy random}"
GREEDY_HEADINGS="${GREEDY_HEADINGS:-36}"
GREEDY_DECIDE_S="${GREEDY_DECIDE_S:-60}"
PER="${PER:-1}"
LAST="${LAST:-$(( $(wc -l < "$CSV") - 1 ))}"
OUT="${OUT:-$DATA/derived/results/$(basename "$CSV" .csv)_N${N}}"
K="${SLURM_ARRAY_TASK_ID:?submit with --array}"
# The commit, asked of git once here rather than by every process at once (the scorer reads it).
# Best set at submission: SAR_COMMIT=$(git rev-parse HEAD) sbatch ...
export SAR_COMMIT="${SAR_COMMIT:-$(git rev-parse HEAD)$(git -c core.checkStat=minimal diff --quiet HEAD -- || echo -dirty)}"

for P in 0 1 2 3; do
    A=$(( 4 * PER * (K - 1) + P * PER + 1 ))
    B=$(( A + PER - 1 ))
    [ "$A" -le "$LAST" ] || continue
    [ "$B" -le "$LAST" ] || B=$LAST
    PYTHONPATH=src "$PY" scripts/score_scenarios.py score --csv "$CSV" --forcing-dir "$DATA" \
        --rows "$A-$B" --noise $NOISE --arrival-h $ARRIVALS --searchers $SEARCHERS \
        --greedy-headings "$GREEDY_HEADINGS" --greedy-decide-s "$GREEDY_DECIDE_S" \
        --particles "$N" --out "$OUT" &
done
wait
echo "task $K done: $OUT"
