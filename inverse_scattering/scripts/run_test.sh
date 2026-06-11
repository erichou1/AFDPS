#!/usr/bin/env bash
# Full test run for one receiver count, sharded across co-tenant processes on the GPU,
# with GPU-utilization logging, then aggregation into the Table-3 format.
#
# Usage (from inverse_scattering/):
#   bash scripts/run_test.sh [R] [NSHARD] [J] [STEPS]
#   e.g.  bash scripts/run_test.sh 360 2 1024 200
#
# Per-case seeding (main.py seeds seed+case_id BEFORE observation generation) makes the
# union of shards reproduce the same INPUTS as a single run; the aggregated mean PSNR is
# therefore shard-count-independent (CUDA nondeterminism means it is NOT bitwise identical
# unless torch.use_deterministic_algorithms(True)). Benchmark single-shard util first
# (scripts/smoke_gb200.sh) and only raise NSHARD if a real utilization gap remains.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"; cd "$HERE"
PY="${PYTHON:-python3}"
R="${1:-360}"; NSHARD="${2:-2}"; J="${3:-1024}"; STEPS="${4:-200}"
NCASES="${NCASES:-100}"   # full test set is 100 cases (ids 0-99)

echo "== precompute SVD (R=$R) =="
"$PY" scripts/precompute_svd.py --numTrans 20 --numRec "$R"

GLOGTAG="exps/gpu_util_R${R}.csv"
bash scripts/gpu_log.sh start "$GLOGTAG"
trap 'bash scripts/gpu_log.sh stop "$GLOGTAG" || true' EXIT

chunk=$(( (NCASES + NSHARD - 1) / NSHARD ))
pids=()
for ((k=0; k<NSHARD; k++)); do
  lo=$(( k * chunk )); hi=$(( lo + chunk - 1 ))
  (( hi > NCASES - 1 )) && hi=$(( NCASES - 1 ))
  (( lo > hi )) && break
  exp="final_R${R}_shard${k}"
  echo "  shard $k: ids ${lo}-${hi} -> exp_name=$exp"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PY" main.py \
    problem=inv-scatter-afdps algorithm=afdps pretrain=inv-scatter \
    num_samples=1 wandb=false exp_name="$exp" \
    problem.model.numRec="$R" problem.data.id_list="${lo}-${hi}" \
    algorithm.method.num_particles="$J" algorithm.method.num_steps="$STEPS" \
    algorithm.method.sampler_kwargs.progress=false &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done

bash scripts/gpu_log.sh stop "$GLOGTAG"; trap - EXIT
echo "== GPU utilization (R=$R) =="
bash scripts/gpu_log.sh summarize "$GLOGTAG"

echo "== aggregate (R=$R) =="
"$PY" scripts/aggregate_table3.py --numRec "$R" --numTrans 20 \
  "exps/inference/inverse-scatter-afdps/AFDPS/final_R${R}_shard*/result_*.pt"
