#!/usr/bin/env bash
#SBATCH --job-name=eval_run
#SBATCH --partition=jaguarcluster
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=logs/eval/eval_%j.out
#SBATCH --error=logs/eval/eval_%j.out

# Curves, test scores against the Expanding Square, maps and breakdown for any trained run, written beside it and never into it.
# Usage, from the repository root: RUN=/home/26p67/data/derived/rl/runs_tele/tele2x_mlp sbatch scripts/eval_run.sbatch

set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

PY="${SAR_PYTHON:-/home/26p67/envs/sar-rl/bin/python}"
: "${RUN:?pass RUN, the trained run folder}"
RUN="${RUN%/}"
NAME="$(basename "$RUN")"
OUT="${OUT:-$(dirname "$RUN")/eval_$NAME}"
FIGURES="${FIGURES:-figures/produced_maps_eval}"
MIRROR="$OUT/runs/$NAME"
MODEL="$RUN/best_model.zip"
[ -f "$MODEL" ] || MODEL="$RUN/model.zip"
[ -f "$MODEL" ] || { echo "no model in $RUN"; exit 1; }
[ -f src/sar/rl/fly.py ] || { echo "run this from the repository root"; exit 1; }
mkdir -p logs/eval "$MIRROR"

# The run's files linked into a folder of our own, so every output lands there and the run is only read.
for f in config.json model.zip best_model.zip eval monitor; do
    if [ -e "$RUN/$f" ]; then ln -sfn "$RUN/$f" "$MIRROR/$f"; fi
done

echo "== 0/4 which code reads this model"
read -r ROUTE WINDOWS_DEFAULT ARRIVALS_DEFAULT < <(PYTHONPATH=src "$PY" - "$MODEL" "$RUN/config.json" <<'EOF'
import json, sys
from gymnasium import spaces
from stable_baselines3 import PPO
from sar.rl.features import SIZE
from sar.rl_v2.features import FAR_SHAPE, NEAR_SHAPE

model = PPO.load(sys.argv[1], device="cpu")
config = json.load(open(sys.argv[2]))
args, space = config["arguments"], model.observation_space
if isinstance(space, spaces.Dict) and space["far"].shape == FAR_SHAPE and space["near"].shape == NEAR_SHAPE:
    route = "v2"
elif isinstance(space, spaces.Box) and space.shape == (SIZE,):
    route = "v1"
else:
    sys.exit(f"this model reads {space}, which no code on this branch builds; check out its commit "
             f"{config.get('git_commit', 'unknown')} and rerun")
print(f"model {sys.argv[1]}: version {config.get('version', 'v1')}, headings {args['headings']}, "
      f"observation {space}, flown through {route}", file=sys.stderr)
print(route, args["windows"], ",".join(map(str, args["arrival_min"])))
EOF
)
[ -n "${ROUTE:-}" ] || { echo "stopped: the model does not fit this branch's code (reason above)"; exit 1; }
WINDOWS="${WINDOWS:-$WINDOWS_DEFAULT}"
ARRIVALS="${ARRIVALS:-${ARRIVALS_DEFAULT//,/ }}"
if [ "$ROUTE" = v2 ]; then FLY=sar.rl_v2.fly; MAPS=scripts/v2/produce_maps.py; else FLY=sar.rl.fly; MAPS=scripts/produce_maps.py; fi
echo "   windows $WINDOWS, arrivals $ARRIVALS, out $OUT"

echo "== 1/4 learning curves"
PYTHONPATH=src "$PY" scripts/plot_training.py --run "$MIRROR" \
    --patience "$(PYTHONPATH=src "$PY" -c "import json; print(json.load(open('$RUN/config.json'))['arguments'].get('patience', 5))")" \
    || echo "   no curves: the run has no monitor or eval logs"

echo "== 2/4 every test window, against the Expanding Square"
# shellcheck disable=SC2086
PYTHONPATH=src "$PY" -m "$FLY" windows --windows "$WINDOWS" --arrival-min $ARRIVALS \
    --searchers ppo expanding-square --model "$MODEL" --out "$MIRROR/test"

echo "== 3/4 maps into $FIGURES/$NAME"
# shellcheck disable=SC2086
PYTHONPATH=src "$PY" "$MAPS" --runs-dir "$OUT/runs" --windows "$WINDOWS" --models "$NAME" \
    --arrivals $ARRIVALS --out "$FIGURES" > "$OUT/maps.txt"
echo "   $(wc -l < "$OUT/maps.txt") maps; their paths are in $OUT/maps.txt"

echo "== 4/4 breakdown by arrival time and place"
PYTHONPATH=src "$PY" scripts/performance_breakdown.py --runs "$MIRROR" --out "$OUT/performance_breakdown"

echo
echo "Stats:     $MIRROR/test/summary.csv, $OUT/performance_breakdown.{csv,xlsx}"
echo "Curves:    $MIRROR/training_curves.png"
echo "Maps:      $FIGURES/$NAME/"
