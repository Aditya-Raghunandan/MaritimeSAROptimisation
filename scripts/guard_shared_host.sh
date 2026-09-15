#!/usr/bin/env bash
# Refuse to start heavy work on a shared login node. Source this at the top of
# any script that will run for more than a moment or use more than one core.
#
#   source "$(dirname "$0")/guard_shared_host.sh"
#
# WHY THIS EXISTS. On 2026-09-15 the five-year ERA5 pull was started directly on
# jaguar1 because the Slurm route was blocked. It ran for 30 minutes at 483 %
# CPU with 85 threads and 10.6 GB resident -- and achieved nothing, because the
# time was going on dask graph construction rather than downloading. jaguar1 is
# the login and file-serving node for EVERY user of the cluster, and Ken's own
# guidance is "do not use these two systems for computing". The person who
# started it could not get htop to draw in his own session.
#
# Falling back to the head node because the compute nodes were unusable did not
# make the job lighter. It moved the load somewhere more damaging.
#
# Override deliberately, never by habit:
#   ALLOW_HEAD_NODE=1 bash scripts/whatever.sh

set -uo pipefail

SHARED_HOSTS="jaguar1 jaguar2"
HOST="$(hostname -s 2>/dev/null || hostname)"

# Cap concurrency everywhere. 32 threads is right for a machine we own and
# antisocial on one we share; the fetch is latency-bound, so fewer threads costs
# wall-clock but not correctness.
export SAR_FETCH_THREADS="${SAR_FETCH_THREADS:-8}"

for h in $SHARED_HOSTS; do
    if [ "$HOST" = "$h" ]; then
        if [ "${ALLOW_HEAD_NODE:-0}" != "1" ]; then
            cat >&2 <<MSG

REFUSING TO RUN: $HOST is a shared login node.

  Ken's guidance is "do not use these two systems for computing", and on
  2026-09-15 a job here made the machine unusable for its actual users.

  Options, in order of preference:
    1. Submit through Slurm       sbatch scripts/<job>.sbatch
       (note: compute nodes run Python 3.8 -- they need a self-contained
        interpreter, not the venv. See the second correction in D017.)
    2. Run it on your own laptop and scp the result up.
    3. If it really is light -- seconds, one core -- override deliberately:
         ALLOW_HEAD_NODE=1 $0

MSG
            exit 3
        fi

        # Allowed, but stay out of everyone's way.
        echo "WARNING: running on shared login node $HOST (ALLOW_HEAD_NODE=1)."
        echo "         threads capped to $SAR_FETCH_THREADS, renicing to 19."
        renice -n 19 -p $$ >/dev/null 2>&1 || true

        read -r one _ _ <<<"$(cat /proc/loadavg)"
        cores="$(nproc)"
        # awk rather than bash arithmetic: load average is a float.
        if awk -v l="$one" -v c="$cores" 'BEGIN{exit !(l > c*0.6)}'; then
            echo "         load is already ${one} on ${cores} cores -- consider waiting." >&2
        fi
    fi
done
