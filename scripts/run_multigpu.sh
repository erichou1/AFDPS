#!/usr/bin/env bash
# Run the full Track-B Navier-Stokes benchmark, splitting the test cases across GPUs.
#
# Each GPU runs a disjoint, contiguous slice of the test cases in its own process
# (CUDA_VISIBLE_DEVICES=k, so each sees its GPU as cuda:0). Because main.py seeds
# PER GLOBAL CASE ID, the split is bit-for-bit equivalent to a single-GPU sequential
# run -- only the wall-clock changes. Per-case results are saved as they finish, then
# aggregated across all shards into one relative-L2 (mean +/- std).
#
# Usage:
#   bash scripts/run_multigpu.sh <guidance_gamma> [exp_name] [ngpu] [ncases] [extra hydra overrides...]
# Examples:
#   bash scripts/run_multigpu.sh 10                     # gamma=10, 4 GPUs, 10 cases, full config
#   bash scripts/run_multigpu.sh 10 final 4 10
#   bash scripts/run_multigpu.sh 10 ds4 4 10 problem.model.downsample_factor=4 problem.model.sigma_noise=1.0
set -euo pipefail
GAMMA="${1:?usage: bash scripts/run_multigpu.sh <guidance_gamma> [exp_name] [ngpu] [ncases] [extra overrides...]}"
EXP="${2:-final}"
NGPU="${3:-4}"
NCASES="${4:-10}"
shift $(( $# < 4 ? $# : 4 )) || true
EXTRA=("$@")   # any remaining args are passed verbatim as hydra overrides (e.g. grid settings)

REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
EXPBASE="${EXP}_g${GAMMA}"
EXPDIRROOT="exps/inference/navier-stokes-afdps-ds2/AFDPS"

echo "Splitting $NCASES cases across $NGPU GPU(s) | guidance_gamma=$GAMMA | exp=$EXPBASE"
[ ${#EXTRA[@]} -gt 0 ] && echo "Extra overrides: ${EXTRA[*]}"
pids=()
for ((k=0; k<NGPU; k++)); do
  start=$(( k * NCASES / NGPU ))
  end=$(( (k+1) * NCASES / NGPU - 1 ))
  if (( start > end )); then continue; fi          # more GPUs than cases -> idle GPU
  range="${start}-${end}"
  log="${EXPBASE}_shard${k}.log"
  echo "  GPU $k -> cases ${range}   (log: ${log})"
  CUDA_VISIBLE_DEVICES=$k nohup python main.py \
      problem=navier-stokes-afdps algorithm=afdps pretrain=navier-stokes \
      algorithm.method.guidance_gamma="$GAMMA" \
      problem.data.id_list="$range" num_samples=1 wandb=false \
      exp_name="${EXPBASE}_shard${k}" "${EXTRA[@]}" > "$log" 2>&1 &
  pids+=($!)
done

echo "Launched ${#pids[@]} shard(s); waiting..."
fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
(( fail )) && echo "WARNING: a shard exited non-zero -- inspect ${EXPBASE}_shard*.log before trusting the aggregate."

echo "== Aggregating relative-L2 over all cases =="
python scripts/aggregate.py "${EXPDIRROOT}/${EXPBASE}_shard"*"/result_"*".pt"
