#!/usr/bin/env bash
# sealed_run.sh -- the final test (D025): the frozen drift model and every searcher, on the sealed
# buoys, once. Run on jaguar1 from the repository root, in three stages:
#
#     bash scripts/sealed_run.sh dryrun           # the whole pipeline on the dev validation rows
#     FREEZE="<freeze list>" bash scripts/sealed_run.sh open    # cuts the sealed table: the opening
#     bash scripts/sealed_run.sh score            # after the table: flights, exam, ceiling, summaries
#
# What is frozen (the vault's D025 log holds the dated list): the drift model rvsc (D035), the four
# Coast Guard patterns, greedy (36 headings a minute, D031), the random floor, Telemachus S
# (`ppo`) and Telemachus 2 (`telemachus`), arrivals 0.5 / 1 / 2 / 3 h, 10^4 particles, turns at
# 7 deg/s (D032). Three outputs per table, one folder each:
#   score/    every searcher's flights (score_scenarios.py): POS and the real buoy found;
#   exam/     the drift model's exam (drift_exam.py): the buoy inside the 90 % region, the centre
#             against "stayed put", and the shape along and across the drift, at 1-4 h;
#   ceiling/  the teleport ceiling at 1x, 2x and 4x one window's sweep (ceiling.py --lean), 10^5
#             particles: what no searcher can clear, and what more effort would.
# The dry run puts the same three on the 181 dev validation rows, so the sealed numbers have a
# dev twin made by the same code, and a bug shows before the opening rather than after.
set -euo pipefail

DATA="${SAR_DATA:-/home/26p67/data}"
PY="${BENCH_PY:-/home/26p67/envs/sar-bench/bin/python}"
TAG="${TAG:-a}"                        # fresh names for every attempt (NFS, 18 Oct)
SEALED="$DATA/derived/sealed"
DRY="$DATA/derived/sealed_dryrun_$TAG"
TS_MODEL="$DATA/derived/rl/runs_speed/tele_speed/best_model.zip"
T2_MODEL="$DATA/derived/rl/chosen_tele/best_model.zip"
SEARCHERS="expanding-square sector parallel trackline greedy random ppo telemachus"
NOISE=rvsc
ARRIVALS="0.5 1 2 3"
[ -f src/sar/rl/train.py ] || { echo "run this from the repository root"; exit 1; }
mkdir -p logs
export SAR_DATA="$DATA" PYTHONPATH=src
export SAR_COMMIT="$(git rev-parse HEAD)$(git -c core.checkStat=minimal diff --quiet HEAD -- || echo -dirty)"
case "$SAR_COMMIT" in *-dirty) echo "commit first: the opening records the commit"; exit 1 ;; esac

tasks() { echo $(( ($(( $(wc -l < "$1") - 1 )) + 4 * $2 - 1) / (4 * $2) )); }   # table, rows per process

submit() {   # table, folder: flights, exam and ceiling over one table, then the summaries
    local csv="$1" out="$2" s e c
    [ ! -e "$out" ] || { echo "$out exists: one folder is one experiment"; exit 1; }
    mkdir -p "$out"
    s=$(SAR_PYTHON="$PY" CSV="$csv" NOISE="$NOISE" ARRIVALS="$ARRIVALS" SEARCHERS="$SEARCHERS" \
        PPO_MODEL="$TS_MODEL" TELEMACHUS_MODEL="$T2_MODEL" PER=8 OUT="$out/score" \
        sbatch --parsable --array=1-"$(tasks "$csv" 8)" --exclude=jaguar11 scripts/score_scenarios.sbatch)
    e=$(TASKS=12 sbatch --parsable --job-name=drift_exam --partition=jaguarcluster --array=1-12 \
        --exclude=jaguar11 --cpus-per-task=4 --mem=16G --time=02:00:00 \
        --output=logs/drift_exam_%A_%a.out scripts/run_parts.sh "$PY" scripts/drift_exam.py run \
        --csv "$csv" --forcing-dir "$DATA" --noise "$NOISE" --out "$out/exam")
    c=$(SAR_PYTHON="$PY" CSV="$csv" NOISE="$NOISE" ARRIVALS="$ARRIVALS" EFFORTS="1 2 4" LEAN=1 \
        N=100000 PER=4 OUT="$out/ceiling" \
        sbatch --parsable --array=1-"$(tasks "$csv" 4)" --mem=24G --time=03:00:00 scripts/ceiling.sbatch)
    sbatch --parsable --job-name=sealed_finish --partition=jaguarcluster --time=00:30:00 --mem=16G \
        --dependency="afterok:$s:$e:$c" --kill-on-invalid-dep=yes --output=logs/sealed_finish_%j.out \
        --wrap="set -e; export PYTHONPATH=src
$PY scripts/score_scenarios.py summary $out/score
$PY scripts/drift_exam.py summary $out/exam
$PY scripts/ceiling.py summary $out/ceiling --flights $out/score/flights.parquet" >/dev/null
    echo "$csv -> $out: flights $s, exam $e, ceiling $c"
}

case "${1:-}" in
dryrun)
    submit "$DATA/derived/rl/windows_truth/rows_validation.csv" "$DRY"
    ;;
open)
    : "${FREEZE:?pass FREEZE, the dated freeze list this opening belongs to (vault D025)}"
    [ ! -e "$SEALED/opened.json" ] || { echo "the sealed set was opened already: $SEALED/opened.json"; exit 1; }
    LAND=$(ls "$DATA"/raw/hycom_*.nc | head -1)
    sbatch --parsable --job-name=sealed_open --partition=jaguarcluster --time=01:00:00 --mem=16G \
        --exclude=jaguar11 --output=logs/sealed_open_%j.out \
        --wrap="set -e; export PYTHONPATH=src
$PY scripts/pick_scenario_buoys.py --data $DATA --land $LAND --every 48h --split sealed \
    --open-sealed '$FREEZE' --out $SEALED/bench_sealed_48h.csv"
    ;;
score)
    [ -f "$SEALED/opened.json" ] || { echo "no opening recorded: run the open stage first"; exit 1; }
    submit "$SEALED/bench_sealed_48h.csv" "$SEALED/rvsc"
    ;;
*) echo "usage: bash scripts/sealed_run.sh dryrun|open|score"; exit 1 ;;
esac
