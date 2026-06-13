#!/usr/bin/env bash
# Shard the 10 InverseBench FWI test cases across GPUs/processes.
#
# Each shard runs a disjoint, contiguous slice of the case ids in its own process
# (CUDA_VISIBLE_DEVICES=k -> each sees its GPU as cuda:0). Because main.py reseeds PER
# GLOBAL CASE ID, the split is bit-for-bit identical to a single unsharded sequential run;
# only wall-clock changes. Per-case results are saved as they finish, then aggregated.
#
# Usage:
#   bash scripts/run_fwi_afdps_sharded.sh [exp_name] [ngpu] [ncases] [extra hydra overrides...]
# Example:
#   bash scripts/run_fwi_afdps_sharded.sh final 4 10 algorithm.method.guidance_gamma=2.0
set -euo pipefail
EXP="${1:-final}"
NGPU="${2:-4}"
NCASES="${3:-10}"          # InverseBench FWI test set size
shift $(( $# < 3 ? $# : 3 )) || true
EXTRA=("$@")

FWI_DIR="$(cd "$(dirname "$0")/.." && pwd)"
NS_DIR="$(cd "$FWI_DIR/../navier_stokes" && pwd)"
cd "$NS_DIR"

# FWI case ids are 1..NCASES (the benchmark id_list is 1-10).
echo "Splitting $NCASES FWI cases across $NGPU GPU(s) | exp=${EXP}"
[ ${#EXTRA[@]} -gt 0 ] && echo "Extra overrides: ${EXTRA[*]}"
pids=()
for ((k=0; k<NGPU; k++)); do
  start=$(( 1 + k * NCASES / NGPU ))
  end=$(( (k+1) * NCASES / NGPU ))
  if (( start > end )); then continue; fi          # more GPUs than cases -> idle GPU
  range="${start}-${end}"
  log="${EXP}_shard${k}.log"
  echo "  GPU $k -> cases ${range}   (log: ${log})"
  CUDA_VISIBLE_DEVICES=$k nohup python "$FWI_DIR/main.py" \
      problem=fwi-afdps algorithm=afdps pretrain=fwi \
      problem.data.id_list="$range" num_samples=1 wandb=false \
      exp_name="${EXP}_shard${k}" "${EXTRA[@]}" > "$log" 2>&1 &
  pids+=($!)
done

echo "Launched ${#pids[@]} shard(s); waiting..."
fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
(( fail )) && echo "WARNING: a shard exited non-zero -- inspect ${EXP}_shard*.log before trusting the aggregate."

RESULTS="../full_waveform_inversion/results/fwi-afdps/AFDPS"
echo "== Aggregating Table-7 metrics over all cases =="
python "$FWI_DIR/scripts/aggregate_fwi_afdps_results.py" \
    "${RESULTS}/${EXP}_shard"*"/result_"*".pt" \
    --logs "${NS_DIR}/${EXP}_shard"*.log --label "AFDPS (${EXP})"
