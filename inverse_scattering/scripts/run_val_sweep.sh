#!/usr/bin/env bash
# Validation sweep for one receiver count: run a small grid of AFDPS configs on the
# 10 validation cases, then rank them with scripts/val_table.py. Use the winner per R
# for the full test run (scripts/run_test.sh).
#
# Usage (from inverse_scattering/):  bash scripts/run_val_sweep.sh [R] [J]
#   e.g.  bash scripts/run_val_sweep.sh 360 512
#
# The grid is a pragmatic coordinate descent around the promoted primary (exact anisotropic
# PiGDM + exact_linear + ensemble mean). Extend/trim the arrays as the budget allows.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"; cd "$HERE"
PY="${PYTHON:-python3}"
R="${1:-360}"; J="${2:-512}"
VAL_ROOT="${VAL_ROOT:-../data/inv-scatter-val}"   # 10-image validation LMDB
VAL_IDS="${VAL_IDS:-0-9}"

echo "== precompute SVD (R=$R) =="
"$PY" scripts/precompute_svd.py --numTrans 20 --numRec "$R"

run () {  # tag, extra-overrides...
  local tag="$1"; shift
  echo "  -- val run: $tag"
  "$PY" main.py problem=inv-scatter-afdps algorithm=afdps pretrain=inv-scatter \
    num_samples=1 wandb=false exp_name="val_${R}_${tag}" \
    problem.model.numRec="$R" problem.data.root="$VAL_ROOT" problem.data.id_list="$VAL_IDS" \
    algorithm.method.num_particles="$J" algorithm.method.sampler_kwargs.progress=false "$@"
}

# 1) guidance family x step type (primary + faithful ablation), at 200 steps, mean reduce
run "pigdm_exact_s200"  algorithm.method.num_steps=200 \
    algorithm.method.sampler_kwargs.guidance_mode=full algorithm.method.sampler_kwargs.guidance_step=exact_linear
run "auto_exact_s200"   algorithm.method.num_steps=200 \
    algorithm.method.sampler_kwargs.guidance_mode=auto algorithm.method.sampler_kwargs.guidance_step=exact_linear
for g in 1.0 3.0 10.0; do
  run "fixed_euler_g${g}_s200" algorithm.method.num_steps=200 algorithm.method.guidance_gamma=$g \
      algorithm.method.sampler_kwargs.guidance_mode=fixed algorithm.method.sampler_kwargs.guidance_step=euler
done

# 2) steps around the primary
for s in 100 400; do
  run "pigdm_exact_s${s}" algorithm.method.num_steps=$s \
      algorithm.method.sampler_kwargs.guidance_mode=full algorithm.method.sampler_kwargs.guidance_step=exact_linear
done

# 3) reduction + resampling on the primary
run "pigdm_exact_best"  algorithm.method.num_steps=200 algorithm.method.reduce=best \
    algorithm.method.sampler_kwargs.guidance_mode=full algorithm.method.sampler_kwargs.guidance_step=exact_linear
run "pigdm_exact_resample" algorithm.method.num_steps=200 \
    algorithm.method.sampler_kwargs.guidance_mode=full algorithm.method.sampler_kwargs.guidance_step=exact_linear \
    algorithm.method.sampler_kwargs.resample=true

echo "== validation ranking (R=$R) =="
"$PY" scripts/val_table.py "exps/inference/inverse-scatter-afdps/AFDPS/val_${R}_*"
