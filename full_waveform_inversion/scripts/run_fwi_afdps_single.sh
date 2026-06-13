#!/usr/bin/env bash
# Single-process AFDPS x FWI run over the InverseBench test cases (or a sub-range).
#
# Runs with cwd = navier_stokes/ so the relative checkpoint (checkpoints/fwi-5m.pt) and
# data (../data/fwi-test) paths resolve, and invokes this directory's main.py.
#
# Usage:
#   bash scripts/run_fwi_afdps_single.sh [exp_name] [id_range] [extra hydra overrides...]
# Examples:
#   bash scripts/run_fwi_afdps_single.sh baseline 1-10
#   bash scripts/run_fwi_afdps_single.sh sweep_g2 1-10 algorithm.method.guidance_gamma=2.0 problem.model.sigma_noise=0.5
set -euo pipefail
EXP="${1:-default}"
IDS="${2:-1-10}"
shift $(( $# < 2 ? $# : 2 )) || true
EXTRA=("$@")

FWI_DIR="$(cd "$(dirname "$0")/.." && pwd)"
NS_DIR="$(cd "$FWI_DIR/../navier_stokes" && pwd)"
cd "$NS_DIR"

echo "AFDPS x FWI | cases ${IDS} | exp=${EXP} | cwd=${NS_DIR}"
[ ${#EXTRA[@]} -gt 0 ] && echo "Extra overrides: ${EXTRA[*]}"

python "$FWI_DIR/main.py" \
    problem=fwi-afdps algorithm=afdps pretrain=fwi \
    problem.data.id_list="$IDS" \
    num_samples=1 wandb=false \
    exp_name="$EXP" "${EXTRA[@]}"
