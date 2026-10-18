#!/usr/bin/env bash
# cloud_shape_run.sh -- the cloud-shape experiment (L41, 18 Oct): does a drift forecast whose
# cloud is the right shape let the agent's lead on the forecast (POS) turn into real buoys found?
# Run on jaguar1 from the repository root, in two stages:
#
#     bash scripts/cloud_shape_run.sh prepare    # the check on held-out buoys, and the practice clouds
#     bash scripts/cloud_shape_run.sh train      # after reading the check: two runs, then the scores
#
# The correction is the noise model rvs (sar.search.benchmark.NOISE), fitted by
# scripts/cloud_shape.py on the TRAINING truth buoys only: the whole drift scaled, and the random
# velocity stretched along the drift and squeezed across it.
#
# prepare: (1) the 181 held-out buoys' clouds again with rvs, same seeds, and the same
#   measurement as before, so before and after are paired; (2) every practice windows file
#   remade with rvs, same names and seeds.
# train: (1) Telemachus S's exact arguments on the rvs practice clouds, from scratch, seed 1;
#   (2) the control: Telemachus 2's exact arguments, seed 2, on the round clouds, so seed luck can
#   be told from the speed choice; then each scored on the 55 (all six arrivals) and the 181
#   held-out buoys, with the patterns and greedy, under its own drift model.
set -euo pipefail

DATA="${SAR_DATA:-/home/26p67/data}"
PY="${BENCH_PY:-/home/26p67/envs/sar-bench/bin/python}"
RL_PY="${RL_PY:-/home/26p67/envs/sar-rl/bin/python}"
SHAPE="${SHAPE:-$DATA/derived/shape/shape.json}"
ROUND="$DATA/derived/rl/windows_arr30-60-120-180"
SHAPED="$DATA/derived/rl/windows_rvs_arr30-60-120-180"
TRUTH="$DATA/derived/rl/windows_truth"
TRUTH_RVS="$DATA/derived/rl/windows_truth_rvs"
RUNS="$DATA/derived/rl/runs_shape"
TS_MODEL="$DATA/derived/rl/runs_speed/tele_speed/best_model.zip"
T2_MODEL="$DATA/derived/rl/chosen_tele/best_model.zip"
[ -f src/sar/rl/train.py ] || { echo "run this from the repository root"; exit 1; }
mkdir -p logs
export SAR_DATA="$DATA" PYTHONPATH=src
export SAR_COMMIT="$(git rev-parse HEAD)$(git -c core.checkStat=minimal diff --quiet HEAD -- || echo -dirty)"

case "${1:-}" in
prepare)
    [ -f "$SHAPE" ] || { echo "no fit at $SHAPE"; exit 1; }
    [ ! -e "$TRUTH_RVS/validation" ] || { echo "$TRUTH_RVS/validation exists"; exit 1; }
    mkdir -p "$TRUTH_RVS" && cp "$TRUTH"/rows_*.csv "$TRUTH_RVS"/
    V=$(TASKS=12 sbatch --parsable --job-name=truth_rvs --partition=jaguarcluster --array=1-12         --exclude=jaguar11 --cpus-per-task=4 --mem=16G --time=01:00:00         --output=logs/truth_rvs_%A_%a.out scripts/run_parts.sh "$PY" scripts/truth_windows.py build         --forcing-dir "$DATA" --out "$TRUTH_RVS" --roles validation --shape-json "$SHAPE")
    M=$(sbatch --parsable --job-name=measure_rvs --partition=jaguarcluster --array=0-11         --exclude=jaguar11 --mem=8G --time=00:30:00 --dependency="afterok:$V"         --output=logs/measure_rvs_%A_%a.out --wrap="export OMP_NUM_THREADS=1
$PY scripts/cloud_shape.py measure --windows $TRUTH_RVS/validation --rows $TRUTH_RVS/rows_validation.csv     --forcing-dir $DATA --out $DATA/derived/shape/validation_rvs/part\$SLURM_ARRAY_TASK_ID.csv     --part \$SLURM_ARRAY_TASK_ID --parts 12")
    W=$(TASKS=12 sbatch --parsable --job-name=windows_rvs --partition=jaguarcluster --array=1-12         --exclude=jaguar11 --cpus-per-task=4 --mem=24G --time=02:00:00         --output=logs/windows_rvs_%A_%a.out scripts/run_parts.sh "$PY" scripts/shaped_windows.py         --src "$ROUND" --out "$SHAPED" --forcing-dir "$DATA" --shape-json "$SHAPE")
    echo "held-out clouds $V -> measured $M; practice clouds $W -> $SHAPED"
    ;;
train)
    for d in "$RUNS/tele_shape" "$RUNS/tele2_seed2"; do
        [ ! -e "$d" ] || { echo "$d exists: one folder is one experiment"; exit 1; }
    done
    [ "$(ls "$SHAPED" | wc -l)" -eq "$(ls "$ROUND" | wc -l)" ] || { echo "the shaped practice clouds are not all there"; exit 1; }
    mkdir -p "$RUNS"
    COMMON=(--validation-every 8 --arrival-min 30 60 120 180 --headings 36 --halves 12
        --timesteps 3000000 --envs 4 --eval-every 100000 --patience 10 --improve-threshold 0.001
        --learning-rate 0.0003 --ent-coef 0.005 --clip-range 0.2 --reward-decay 0 --eval-decay 0
        --gamma 1.0 --policy mlp --batch-size 256 --n-epochs 10 --target-kl 0.03 --torch-threads 1
        --out "$RUNS")
    A=$(SAR_PYTHON="$RL_PY" sbatch --parsable --job-name=tele_shape --exclusive --exclude=jaguar11 \
        scripts/train_ppo.sbatch --windows "$SHAPED" --seed 1 --speeds 45 60 90 "${COMMON[@]}" --name tele_shape)
    B=$(SAR_PYTHON="$RL_PY" sbatch --parsable --job-name=tele2_seed2 --exclusive --exclude=jaguar11 \
        scripts/train_ppo.sbatch --windows "$ROUND" --seed 2 "${COMMON[@]}" --name tele2_seed2)
    echo "training: Telemachus S on the shaped clouds $A; Telemachus 2 seed 2 on the round ones $B"
    score() {   # name, after-job, noise, ppo model, telemachus model
        local r55="$DATA/derived/results/scenarios55_N10000_$1" rval="$DATA/derived/results/truthval_N10000_$1"
        local env=(SAR_PYTHON="$PY" NOISE="$3" SEARCHERS="expanding-square parallel greedy ppo telemachus"
                   PPO_MODEL="$4" TELEMACHUS_MODEL="$5")
        local s55 sval
        s55=$(env "${env[@]}" ARRIVALS="0.25 0.5 0.75 1 2 3" OUT="$r55" sbatch --parsable --array=1-14 \
            --exclude=jaguar11 --dependency="afterok:$2" --kill-on-invalid-dep=yes scripts/score_scenarios.sbatch)
        sval=$(env "${env[@]}" ARRIVALS="0.5 1 2 3" OUT="$rval" CSV="$TRUTH/rows_validation.csv" PER=4 \
            sbatch --parsable --array=1-12 --exclude=jaguar11 --dependency="afterok:$2" \
            --kill-on-invalid-dep=yes scripts/score_scenarios.sbatch)
        sbatch --parsable --job-name="$1_finish" --partition=jaguarcluster --time=00:30:00 --mem=8G \
            --dependency="afterok:$s55:$sval" --kill-on-invalid-dep=yes --output="logs/$1_finish_%j.out" \
            --wrap="set -e; export PYTHONPATH=src; $PY scripts/score_scenarios.py summary $r55; $PY scripts/score_scenarios.py summary $rval"
        echo "   $1: the 55 $s55, held-out $sval"
    }
    score d032d033_rvs "$A" rvs "$TS_MODEL" "$RUNS/tele_shape/best_model.zip"
    score d032d033_seed2 "$B" rvc "$T2_MODEL" "$RUNS/tele2_seed2/best_model.zip"
    ;;
*) echo "usage: bash scripts/cloud_shape_run.sh prepare|train"; exit 1 ;;
esac
