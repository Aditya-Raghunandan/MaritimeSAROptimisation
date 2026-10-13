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
#     OUT=$DATA/published/scenarios/v3 PPO_MODEL=$DATA/derived/rl/chosen/best_model.zip \
#         SAR_PYTHON=/home/26p67/envs/sar-bench/bin/python sbatch --array=1-14 ...   # + the agent
#     # when the array has finished:
#     python scripts/export_scenario_bundles.py index --out $OUT

set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

PY="${SAR_PYTHON:-/home/26p67/envs/sar/bin/python}"
DATA="${SAR_DATA:-/home/26p67/data}"
CSV="${CSV:-$DATA/derived/scenarios/scenarios.csv}"
OUT="${OUT:-$DATA/published/scenarios/v2}"     # format 2: the D032 helicopter
LAST="${LAST:-$(( $(wc -l < "$CSV") - 1 ))}"
PPO_MODEL="${PPO_MODEL:-}"   # the trained agent's best_model.zip: adds its flight
PPO_ARGS=()
[ -z "$PPO_MODEL" ] || PPO_ARGS=(--ppo-model "$PPO_MODEL")
TELEMACHUS_MODEL="${TELEMACHUS_MODEL:-}"   # the second agent's best_model.zip
[ -z "$TELEMACHUS_MODEL" ] || PPO_ARGS+=(--telemachus-model "$TELEMACHUS_MODEL")
NOISE="${NOISE:-rvc rv}"     # the drift models' random terms to export
ARRIVALS="${ARRIVALS:-1 2 3}"   # hours from the call to arriving on scene, such as 0.25 0.5 0.75 1 2 3
K="${SLURM_ARRAY_TASK_ID:?submit with --array}"

for R in $(seq $((4 * K - 3)) $((4 * K))); do
    [ "$R" -le "$LAST" ] || continue
    PYTHONPATH=src "$PY" scripts/export_scenario_bundles.py export --csv "$CSV" \
        --forcing-dir "$DATA" --rows "$R" --noise $NOISE --arrival-h $ARRIVALS "${PPO_ARGS[@]}" \
        --out "$OUT" &
done
wait
echo "task $K done: $OUT"
