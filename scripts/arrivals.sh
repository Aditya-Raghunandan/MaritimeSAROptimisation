#!/usr/bin/env bash
# arrivals.sh -- the 55 at every arrival from 15 min to 3 h, every searcher, and the ceiling at
# the same arrivals, in one submission (13 Oct). Run on jaguar1 from the repository root:
#
#     bash scripts/arrivals.sh
#
# 1. The 55 (rvc) at 0.25, 0.5, 0.75, 1, 2 and 3 h: the four Coast Guard patterns, the Parallel
#    Track reordered (parallel-skip), greedy, the random floor, the first agent (ppo) and the
#    second (telemachus), into one folder, then its summary.
# 2. The ceiling (scripts/ceiling.py) at the same arrivals on 10^5 particles, then its summary
#    against step 1's flights.
# The agents practised arrivals of 30, 60, 120 and 180 min only: 15 and 45 min are new to them.
set -euo pipefail

DATA="${SAR_DATA:-/home/26p67/data}"
BENCH_PY="${BENCH_PY:-/home/26p67/envs/sar-bench/bin/python}"
FIRST="${FIRST:-$DATA/derived/rl/chosen/best_model.zip}"
SECOND="${SECOND:-$DATA/derived/rl/chosen_tele/best_model.zip}"
TAG="${TAG:-d032d033_arrivals}"
ARRIVALS="${ARRIVALS:-0.25 0.5 0.75 1 2 3}"
OUT="$DATA/derived/results/scenarios55_N10000_$TAG"
CEIL="$DATA/derived/results/ceiling55_N100000_$TAG"

[ -f scripts/ceiling.py ] || { echo "run this from the repository root"; exit 1; }
for d in "$OUT" "$CEIL"; do
    [ ! -e "$d" ] || { echo "$d exists: one folder is one experiment"; exit 1; }
done
for m in "$FIRST" "$SECOND"; do
    [ -f "$m" ] || { echo "no agent at $m"; exit 1; }
done
mkdir -p logs
export SAR_COMMIT="$(git rev-parse HEAD)$(git -c core.checkStat=minimal diff --quiet HEAD -- || echo -dirty)"

S=$(env SAR_PYTHON="$BENCH_PY" NOISE=rvc ARRIVALS="$ARRIVALS" \
    SEARCHERS="expanding-square sector parallel parallel-skip trackline greedy random ppo telemachus" \
    PPO_MODEL="$FIRST" TELEMACHUS_MODEL="$SECOND" OUT="$OUT" \
    sbatch --parsable --array=1-14 --exclude=jaguar11 scripts/score_scenarios.sbatch)
F=$(sbatch --parsable --job-name=arrivals_sum --partition=jaguarcluster --time=00:30:00 --mem=8G \
    --dependency="afterok:$S" --kill-on-invalid-dep=yes --output=logs/arrivals_sum_%j.out \
    --wrap="PYTHONPATH=src $BENCH_PY scripts/score_scenarios.py summary $OUT")
C=$(env SAR_PYTHON="$BENCH_PY" ARRIVALS="$ARRIVALS" OUT="$CEIL" \
    sbatch --parsable --array=1-4 scripts/ceiling.sbatch)
X=$(sbatch --parsable --job-name=arrivals_ceil --partition=jaguarcluster --time=00:15:00 --mem=4G \
    --dependency="afterok:$F:$C" --kill-on-invalid-dep=yes --output=logs/arrivals_ceil_%j.out \
    --wrap="PYTHONPATH=src $BENCH_PY scripts/ceiling.py summary $CEIL --flights $OUT/flights.parquet")
echo "the 55 $S -> $OUT; summary $F; ceiling $C -> $CEIL; ceiling summary $X"
