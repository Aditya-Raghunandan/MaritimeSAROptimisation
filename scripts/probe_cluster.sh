#!/usr/bin/env bash
# probe_cluster.sh -- answer every unknown about the Jaguar cluster in one pass.
#
# Run ONCE on the head node, after the first interactive login has set up keys:
#     ssh 26p67@jaguar1.eie.wits.ac.za 'bash -s' < code/probe_cluster.sh | tee probe.txt
# then paste probe.txt into notebook/aditya/.  See D017.
#
# Why this exists: the official access guide
# (github.com/witseie/computecluster/blob/main/clusterAccessGuide.md) documents
# SSH and VS Code ONLY.  It states no storage paths, no quotas, no Slurm
# examples and no Python guidance.  Everything below is therefore unverified
# until this script has run.
#
# READ-ONLY.  It creates nothing and installs nothing.

echo "=== identity / host ==="
id; hostname; uname -a; lsb_release -d 2>/dev/null || cat /etc/os-release | head -2

echo; echo "=== STORAGE: the number Ken's email never stated is /data ==="
# Email gives /home = 1.7 GB SSD, /data1 = 4 TB 7200rpm.  /data size UNKNOWN.
df -h /home /data /data1 2>&1
echo "-- our usage --"; du -sh "$HOME" 2>/dev/null
for d in /data /data1; do echo "-- $d writable? --"; test -w "$d" && echo "yes" || echo "NO (may need a subdir made for us)"; done
echo "-- quota (blank output usually means no quota system) --"; quota -s 2>/dev/null || echo "quota: not available"

echo; echo "=== SCHEDULER ==="
sinfo -o "%P %a %D %c %m %l %G" 2>/dev/null || echo "sinfo: not found"
echo "-- queue --"; squeue 2>/dev/null | head -5 || echo "squeue: not found"
echo "-- array limit --"; scontrol show config 2>/dev/null | grep -iE 'MaxArraySize|MaxJobCount|DefaultTime|MaxTime' || echo "scontrol: not found"
qstat -Q 2>/dev/null || true   # in case it is PBS/Torque after all

echo; echo "=== PYTHON (Ubuntu 20.04 ships 3.8, which is EOL) ==="
for p in python3 python3.8 python3.10 python3.11 python3.12; do
  command -v $p >/dev/null && echo "$p -> $($p --version 2>&1)"
done
for m in conda mamba micromamba module; do command -v $m >/dev/null && echo "FOUND: $m ($(command -v $m))"; done
echo "-- matlab --"; command -v matlab >/dev/null && matlab -help 2>&1 | head -1 || echo "matlab: not on PATH"

echo; echo "=== OUTBOUND INTERNET FROM THE HEAD NODE ==="
# These three hosts are the entire data pipeline.  A 000 means blocked.
for h in storage.googleapis.com tds.hycom.org erddap.aoml.noaa.gov; do
  printf '  %-28s HTTP %s\n' "$h" "$(curl -sS -o /dev/null -m 20 -w '%{http_code}' "https://$h/" 2>/dev/null || echo 000)"
done
echo "-- proxy env --"; env | grep -i proxy || echo "  (no proxy variables set)"
echo "-- dns --"; getent hosts storage.googleapis.com || echo "  DNS FAILED"

echo; echo "=== OUTBOUND INTERNET FROM A PRIVATE COMPUTE NODE ==="
echo "  (jaguar3 is 192.168.1.3, private. This answers WHERE the bulk fetch can run.)"
if ssh -o BatchMode=yes -o ConnectTimeout=10 jaguar3 true 2>/dev/null; then
  for h in storage.googleapis.com tds.hycom.org erddap.aoml.noaa.gov; do
    printf '  jaguar3 -> %-24s HTTP %s\n' "$h" \
      "$(ssh -o BatchMode=yes jaguar3 "curl -sS -o /dev/null -m 20 -w '%{http_code}' https://$h/ 2>/dev/null || echo 000")"
  done
else
  echo "  passwordless ssh to jaguar3 NOT working yet -- do the ssh-copy-id step first (D017, H2)."
fi

echo; echo "=== GPU NODE (dica10, Tesla K40c) ==="
ssh -o BatchMode=yes -o ConnectTimeout=10 dica10 'nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv' 2>/dev/null \
  || echo "  dica10 unreachable or no key yet"

echo; echo "=== DONE -- paste this whole output into notebook/aditya/ ==="
