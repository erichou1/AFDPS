#!/usr/bin/env bash
# Lightweight GPU-utilization logger for the GB200 runs.
#
#   bash scripts/gpu_log.sh start <logfile>   # begin sampling nvidia-smi to CSV (bg)
#   bash scripts/gpu_log.sh stop  <logfile>    # stop sampling
#   bash scripts/gpu_log.sh summarize <logfile># print mean/median/p10 util + peak mem
#
# Sampling every 0.5s: utilization.gpu (%), memory.used (MiB). The PID of the sampler
# is stored next to the logfile (<logfile>.pid) so stop can kill it.
set -euo pipefail
CMD="${1:-}"; LOG="${2:-gpu_util.csv}"
PIDF="$LOG.pid"

case "$CMD" in
  start)
    echo "timestamp,util_gpu_pct,mem_used_mib" > "$LOG"
    ( while true; do
        nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used \
          --format=csv,noheader,nounits | tr -d ' ' >> "$LOG"
        sleep 0.5
      done ) &
    echo $! > "$PIDF"
    echo "gpu_log: sampling -> $LOG (pid $(cat "$PIDF"))"
    ;;
  stop)
    if [[ -f "$PIDF" ]]; then kill "$(cat "$PIDF")" 2>/dev/null || true; rm -f "$PIDF"; fi
    echo "gpu_log: stopped ($LOG)"
    ;;
  summarize)
    python3 - "$LOG" <<'PY'
import sys, statistics as st
rows=[l.strip().split(',') for l in open(sys.argv[1]) if ',' in l][1:]
u=[float(r[1]) for r in rows if r[1].replace('.','',1).isdigit()]
m=[float(r[2]) for r in rows if r[2].replace('.','',1).isdigit()]
if not u: print("no samples"); sys.exit(0)
u.sort()
p10=u[int(0.10*(len(u)-1))]
print(f"GPU util %%: mean={st.mean(u):.1f} median={st.median(u):.1f} p10={p10:.1f} "
      f"min={min(u):.0f} max={max(u):.0f} n={len(u)}")
print(f"GPU mem MiB: max={max(m):.0f} mean={st.mean(m):.0f}")
PY
    ;;
  *) echo "usage: bash scripts/gpu_log.sh [start|stop|summarize] <logfile>"; exit 1 ;;
esac
