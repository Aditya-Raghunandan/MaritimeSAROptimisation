#!/usr/bin/env bash
#SBATCH --job-name=hfupload
#SBATCH --partition=jaguarcluster
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=logs/hfupload_%j.out
#SBATCH --error=logs/hfupload_%j.out

# Upload the published Zarr archive to Hugging Face (D021 step 4, last mile).
#
# WHY A JOB RATHER THAN A HEAD-NODE COMMAND: 1.6 GB across 2,238 files, hashed
# and uploaded. It is network-bound rather than CPU-bound, but it runs for a
# while and the head node is everyone's login node. Same rule as the export.
#
# MEASURED 2026-09-18: compute nodes reach huggingface.co on 443 (HTTP 200 from
# jaguar11). This was not a safe assumption -- D017 had only established that
# they reach storage.googleapis.com and tds.hycom.org -- so it was checked
# before this script was written rather than after it failed.
#
# THE CREDENTIAL IS NOT IN HERE AND MUST NOT BE. `huggingface-cli login` has
# already written it to ~/.cache/huggingface/token, which is on shared /home
# and therefore readable from the compute node. publish_archive.py picks it up
# from there and never prints it. A token in a batch script is a token in the
# repository, in the job's environment, and in `scontrol show job` output.
#
# Submit from ~/SAR/viz-staging:
#     sbatch scripts/upload_archive.sbatch

set -euo pipefail

PY="${SAR_PYTHON:-/home/26p67/envs/sar/bin/python}"
[ -x "$PY" ] || { echo "no interpreter at $PY -- see D017 on micromamba"; exit 1; }

STAGE="${SAR_STAGE:-/home/26p67/SAR/viz-staging}"
SRC="${SAR_SRC:-/home/26p67/data/published}"
REPO="${SAR_REPO:-AdityaRugs/MaritimeSARoperations}"

# tqdm keeps writing its bar even to a non-tty, separated by carriage returns
# rather than newlines. In the 58642 log that made the whole progress display
# ONE line, 25 kB of it by the time the first phase was 13 % done: `grep -c`
# counts it as a single match, `tail` cannot trim it, and the output that
# matters sits at the far end of it.
export HF_HUB_DISABLE_PROGRESS_BARS=1

mkdir -p logs

echo "host      $(hostname)"
echo "source    $SRC"
echo "repo      $REPO"
echo "started   $(date -Is)"
echo

cd "$STAGE"
"$PY" -u scripts/publish_archive.py --src "$SRC" --repo "$REPO"

echo
echo "finished  $(date -Is)"
