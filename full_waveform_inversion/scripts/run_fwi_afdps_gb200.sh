#!/usr/bin/env bash
# AFDPS x FWI tuned to saturate a GB200 node.
#
# FWI's bottleneck is NOT the GPU: the wave-equation forward/adjoint solves run on CPU via
# Devito's compiled C kernels + a dask LocalCluster. The GB200's GPU runs the EDM prior and
# the Hutchinson/weight algebra. So we drive BOTH:
#   * GPU  -> a large particle ensemble (batched score-net evaluations) + tf32 + torch.compile;
#   * CPU  -> Devito OpenMP across the Grace cores, with the 16 shots parallelized by dask
#            (the parent operator's LocalCluster already fans the shots out across cores).
# Static solver/geometry tensors are built once in the operator __init__ and reused.
#
# Usage:
#   bash scripts/run_fwi_afdps_gb200.sh [exp_name] [num_particles] [num_steps] [extra overrides...]
# Example:
#   bash scripts/run_fwi_afdps_gb200.sh gb200 48 200 algorithm.method.guidance_gamma=2.0
set -euo pipefail
EXP="${1:-gb200}"
NP="${2:-48}"
NSTEPS="${3:-200}"
shift $(( $# < 3 ? $# : 3 )) || true
EXTRA=("$@")

FWI_DIR="$(cd "$(dirname "$0")/.." && pwd)"
NS_DIR="$(cd "$FWI_DIR/../navier_stokes" && pwd)"
cd "$NS_DIR"

# ---- Devito CPU backend: OpenMP across the Grace cores ----
export DEVITO_LANGUAGE="${DEVITO_LANGUAGE:-openmp}"
export DEVITO_LOGGING="${DEVITO_LOGGING:-WARNING}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$(nproc 2>/dev/null || echo 16)}"
# ---- keep the GPU fed ----
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "AFDPS x FWI GB200 | exp=${EXP} | particles=${NP} steps=${NSTEPS} | OMP=${OMP_NUM_THREADS}"
[ ${#EXTRA[@]} -gt 0 ] && echo "Extra overrides: ${EXTRA[*]}"

python "$FWI_DIR/main.py" \
    problem=fwi-afdps algorithm=afdps pretrain=fwi \
    problem.data.id_list=1-10 \
    algorithm.method.num_particles="$NP" \
    algorithm.method.num_steps="$NSTEPS" \
    num_samples=1 wandb=false tf32=true compile=true \
    exp_name="$EXP" "${EXTRA[@]}"

RESULTS="../full_waveform_inversion/results/fwi-afdps/AFDPS"
echo "== Aggregating Table-7 metrics =="
python "$FWI_DIR/scripts/aggregate_fwi_afdps_results.py" \
    "${RESULTS}/${EXP}/result_"*".pt" --label "AFDPS (${EXP})" || true
