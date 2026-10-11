#!/usr/bin/env bash
# The nine PPO experiments, end to end, submitted to Slurm in one go. Run on jaguar1 from the
# repository root; it submits the jobs and returns, and Slurm chains them:
#     bash scripts/run_experiments.sh
#
#   1. windows    stage 1 again with the four arrival times, into their own folder, unless all
#                 1,305 windows files are already there (scripts/training_montes.sh)
#   2. training   nine jobs at once, one node each, after step 1 (scripts/train_ppo.sbatch)
#   3. wrap-up    per run, after its own training: curves, test scores, maps (scripts/wrap_run.sbatch);
#                 if a training job fails, its wrap-up is cancelled rather than left waiting
#   4. compare    after every wrap-up: <runs>/comparison.csv (scripts/compare_runs.py)
#
# Every run is run_baseline's settings with one change. Safe to rerun: a run with a model.zip
# is not trained again, and a run with test/summary.csv and maps.txt is not wrapped again.
# Watch it with: squeue -u "$USER"

set -euo pipefail

PY="${SAR_PYTHON:-/home/26p67/envs/sar-rl/bin/python}"
DATA="${SAR_DATA:-/home/26p67/data}"
RUNS="${RUNS:-$DATA/derived/rl/runs}"
ARRIVALS="${ARRIVALS:-30 60 120 180}"
WINDOWS="${WINDOWS:-$DATA/derived/rl/windows_arr${ARRIVALS// /-}}"
FIGURES="${FIGURES:-figures/produced_maps}"
EXCLUDE="${EXCLUDE:-jaguar11}"   # 4x slower than the others for training
EXCLUSIVE="${EXCLUSIVE:-1}"      # 1 gives each training run a node to itself
EXPECTED_WINDOWS=1305            # 5 places x 261 weeks
PATIENCE=10
THRESHOLD=0.001

# The baseline: today's settings, Stable-Baselines3's defaults for the PPO knobs.
COMMON="--windows $WINDOWS --validation-every 8 --arrival-min $ARRIVALS --timesteps 3000000 --envs 4 --seed 1 --eval-every 100000 --patience $PATIENCE --improve-threshold $THRESHOLD --out $RUNS"
BASE_HEADINGS=36 BASE_LR=0.0003 BASE_ENT=0 BASE_CLIP=0.2

# name                headings         learning rate  entropy     clip
EXPERIMENTS=(
  "run_baseline        $BASE_HEADINGS  $BASE_LR       $BASE_ENT   $BASE_CLIP"
  "run_heading72       72              $BASE_LR       $BASE_ENT   $BASE_CLIP"
  "run_heading24       24              $BASE_LR       $BASE_ENT   $BASE_CLIP"
  "run_learnrate0.0004 $BASE_HEADINGS  0.0004         $BASE_ENT   $BASE_CLIP"
  "run_learnrate0.0002 $BASE_HEADINGS  0.0002         $BASE_ENT   $BASE_CLIP"
  "run_entropy0.007    $BASE_HEADINGS  $BASE_LR       0.007       $BASE_CLIP"
  "run_entropy0.005    $BASE_HEADINGS  $BASE_LR       0.005       $BASE_CLIP"
  "run_clip0.5         $BASE_HEADINGS  $BASE_LR       $BASE_ENT   0.5"
  "run_clip0.1         $BASE_HEADINGS  $BASE_LR       $BASE_ENT   0.1"
)

[ -f src/sar/rl/train.py ] || { echo "run this from the repository root"; exit 1; }
"$PY" -c "import stable_baselines3, matplotlib" 2>/dev/null \
    || { echo "the sar-rl environment needs stable_baselines3 and matplotlib: bash scripts/setup_rl_env.sh"; exit 1; }
mkdir -p logs "$RUNS"

echo "== 1/4 windows with arrival times $ARRIVALS in $WINDOWS"
HAVE=0
[ -d "$WINDOWS" ] && HAVE=$(find "$WINDOWS" -maxdepth 1 -name 'windows_*.npz' | wc -l)
AFTER_WINDOWS=""
if [ "$HAVE" -lt "$EXPECTED_WINDOWS" ]; then
    JOB=$(sbatch --parsable --export=ALL,ARRIVALS="$ARRIVALS",WINDOWS="$WINDOWS",KEEP_CSV=0 scripts/training_montes.sh)
    AFTER_WINDOWS="--dependency=afterok:$JOB"
    echo "   $HAVE of $EXPECTED_WINDOWS there; stage 1 submitted as job $JOB"
else
    echo "   all $HAVE there"
fi

echo "== 2/4 training"
NODE_OPTS=(--exclude="$EXCLUDE")
[ "$EXCLUSIVE" = 1 ] && NODE_OPTS+=(--exclusive)
declare -A TRAIN_JOB
for line in "${EXPERIMENTS[@]}"; do
    read -r NAME HEADINGS LR ENT CLIP <<< "$line"
    if [ -f "$RUNS/$NAME/model.zip" ]; then
        echo "   $NAME already trained"
        continue
    fi
    FORCE=""
    [ -d "$RUNS/$NAME" ] && FORCE="--force"   # a run that died partway starts again
    # shellcheck disable=SC2086
    TRAIN_JOB[$NAME]=$(sbatch --parsable --job-name="ppo_$NAME" "${NODE_OPTS[@]}" $AFTER_WINDOWS \
        scripts/train_ppo.sbatch $COMMON --headings "$HEADINGS" --learning-rate "$LR" \
        --ent-coef "$ENT" --clip-range "$CLIP" --name "$NAME" $FORCE)
    echo "   $NAME: job ${TRAIN_JOB[$NAME]}"
done

echo "== 3/4 wrap-up"
WRAP_JOBS=()
for line in "${EXPERIMENTS[@]}"; do
    read -r NAME _ <<< "$line"
    if [ -f "$RUNS/$NAME/test/summary.csv" ] && [ -f "$RUNS/$NAME/maps.txt" ] && [ -z "${TRAIN_JOB[$NAME]:-}" ]; then
        echo "   $NAME already wrapped"
        continue
    fi
    AFTER=""
    [ -n "${TRAIN_JOB[$NAME]:-}" ] && AFTER="--dependency=afterok:${TRAIN_JOB[$NAME]}"
    # shellcheck disable=SC2086
    JOB=$(sbatch --parsable --job-name="wrap_$NAME" --exclude="$EXCLUDE" --kill-on-invalid-dep=yes $AFTER \
        --export=ALL,NAME="$NAME",RUNS="$RUNS",WINDOWS="$WINDOWS",ARRIVALS="$ARRIVALS",PATIENCE="$PATIENCE",THRESHOLD="$THRESHOLD",FIGURES="$FIGURES" \
        scripts/wrap_run.sbatch)
    WRAP_JOBS+=("$JOB")
    echo "   $NAME: job $JOB"
done

echo "== 4/4 comparison"
AFTER=""
[ ${#WRAP_JOBS[@]} -gt 0 ] && AFTER="--dependency=afterany:$(IFS=:; echo "${WRAP_JOBS[*]}")"
# shellcheck disable=SC2086
JOB=$(sbatch --parsable --job-name=compare --partition=jaguarcluster --time=00:30:00 --mem=2G \
    --kill-on-invalid-dep=yes $AFTER \
    --output=logs/compare_%j.out \
    --wrap="PYTHONPATH=src $PY scripts/compare_runs.py --runs-dir $RUNS")
echo "   job $JOB writes $RUNS/comparison.csv"

echo
echo "Submitted. Watch with: squeue -u $USER"
echo "Logs: logs/ppo_<job>.out (training), logs/wrap_<job>.out, logs/compare_$JOB.out"
echo "Figures land in $FIGURES/<run>/... on the cluster; commit and push them from there when you are happy."
