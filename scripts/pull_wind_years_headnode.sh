#!/usr/bin/env bash
# Five-year ERA5 pull, year by year, on the HEAD NODE.
#
# WHY NOT SLURM: the cluster is heterogeneous and nothing said so until
# 2026-09-15. jaguar1 runs Ubuntu 24.04 with Python 3.12.3; the compute nodes
# run Ubuntu 20.04 with Python 3.8.10 and no 3.1x at all. A venv created on the
# head node has bin/python -> /usr/bin/python3, which resolves to 3.12 there and
# 3.8 on a compute node -- the SAME venv silently runs a different, incompatible
# interpreter depending on where it executes. Our pins need >= 3.11, so the
# compute nodes cannot run this code without a self-contained interpreter
# (micromamba), which is what D017 originally advised for a reason nobody had
# correctly identified.
#
# Running here is defensible despite Ken's "do not use these two systems for
# computing": this is latency-bound I/O, not computation. One ARCO chunk is a
# whole global timestep, so the cost is per-request round trips, not CPU.
#
#   nohup bash scripts/pull_wind_years_headnode.sh > logs/pull.log 2>&1 &
#
# Watch:  tail -f logs/pull.log        Stop: pkill -f pull_wind_years_headnode

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$REPO/.venv/bin/python"
OUT=/home/26p67/data          # NOT /data1 -- unwritable, nothing mounted on it
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

cd "$REPO"
echo "host       : $(hostname)"
echo "interpreter: $("$PY" -V 2>&1)  at $PY"
echo "out        : $OUT"
df -h "$OUT" | tail -1
echo

FAILED=()
for Y in 2019 2020 2021 2022 2023; do
    NEXT=$((Y + 1))
    echo "=============================================================="
    echo "YEAR $Y  ($Y-01-01 .. $NEXT-01-01, end exclusive)   $(date -Is)"
    echo "=============================================================="
    # Year by year on purpose: a failure costs one year, not the whole run,
    # and the finished years stay on disk.
    if "$PY" -m sar.fetch.wind --start "$Y-01-01" --end "$NEXT-01-01" --out "$OUT"; then
        echo "YEAR $Y OK   $(date -Is)"
    else
        echo "YEAR $Y FAILED (rc=$?)   $(date -Is)"
        FAILED+=("$Y")
    fi
    echo
done

echo "=============================================================="
echo "finished $(date -Is)"
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "FAILED YEARS: ${FAILED[*]}  -- rerun just those"
else
    echo "all five years complete"
fi
ls -lh "$OUT/raw/"
df -h "$OUT" | tail -1
